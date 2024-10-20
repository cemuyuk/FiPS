import os
import dotenv
from typing import Optional, Tuple, List, Dict, Callable

import wandb
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    Dataset,
    random_split,
)
import torch.distributed as dist

from torchvision import datasets, transforms

from timm import models
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup, create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation


from tqdm import tqdm

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from training.transformer import (
    GrayScale,
    GaussianBlur,
    Solarization,
    TransformDataset,
)
from data.custom_hf_dataset import FakeImageNetDataset

from sparsimony.distributions.base import (
    UniformDistribution,
)
from sparsimony.schedulers.base import (
    CosineDecayScheduler,
    AcceleratedCubicScheduler,
)
from sparsimony.dst.base import DSTMixin
from sparsimony.dst.static import (
    StaticMagnitudeSparsifier,
)
from sparsimony.dst.gmp import GMP
from sparsimony.dst.rigl import RigL

from sd2s.lrm import SMModel, SMLayer
from sd2s.utils import (
    DDPWrappedGetAttr,
    get_single_process_model_state_from_distributed_state,
    get_world_size,
)
from SETTINGS import CONFIG_DIR
from training.wandb_utils import (
    init_wandb,
    parse_wandb_run_id,
    wandb_finetune_log,
    wand_finetune_val_log,
)


@hydra.main(
    config_path=CONFIG_DIR, config_name="deit_base_config", version_base="1.3"
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.training.seed)
    local_rank, global_rank, world_size = _parse_dist_envs()
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=global_rank,
        )
    device = _get_device(cfg, local_rank)
    # DATA
    if not cfg.data.transforms.three_augment:
        transforms = _get_transforms_v1(cfg)
    else:
        transforms = _get_transforms(cfg)
    train_dataset, validation_dataset = _get_datasets(cfg, transforms)
    train_sampler, validation_sampler = _get_sampler(
        train_dataset, validation_dataset
    )
    train_dataloader, validation_dataloader = _get_dataloaders(
        cfg,
        train_dataset,
        validation_dataset,
        train_sampler,
        validation_sampler,
    )
    # MODEL, OPTIMIZER, SCHEDULER, SPARSIFIER
    c_model = _init_compressed_model(cfg)
    optimizer = _get_optimizer(cfg, c_model)
    total_steps = cfg.training.num_epochs * len(train_dataloader)
    scheduler = _get_scheduler(cfg, optimizer, total_steps)
    sparsifier = _get_sparsifier(
        cfg, optimizer, _compute_t_end(cfg, total_steps), total_steps
    )

    if cfg.data.mixup_active:
        mixup_fn = _get_mixup_fn(cfg)
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = nn.CrossEntropyLoss()

    c_model.to(device)  # To device before prepare for all_reduce on scores
    if sparsifier is not None:
        sparse_config = [
            {"tensor_fqn": f"{fqn}.V"}
            for fqn, module in c_model.named_modules()
            if isinstance(module, SMLayer)
        ]
        sparsifier.prepare(c_model, sparse_config)
    checkpoint_dict = torch.load(cfg.model.checkpoint_path)
    try:
        checkpoint_dict = _load_checkpoint(c_model, checkpoint_dict)
    except RuntimeError:
        checkpoint_dict = get_single_process_model_state_from_distributed_state(
            checkpoint_dict
        )
        checkpoint_dict = _load_checkpoint(c_model, checkpoint_dict)

    # INIT DDP
    if world_size > 1:
        c_model = DDPWrappedGetAttr(c_model, device_ids=[local_rank])
        c_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(c_model)
    # WANBD, TRAINING, SAVE
    run = init_wandb(cfg, global_rank)
    run_id = parse_wandb_run_id(run)
    c_model.eval()
    print(f"The length of the test_dataset is {len(validation_dataset)}.")
    print(f"The lenght of test_dataloader is {len(validation_dataloader)}.")
    print(f"The length of the train_dataset is {len(train_dataset)}.")
    print(f"The lenght of train_dataloader is {len(train_dataloader)}.")
    _ = validate(
        cfg=cfg,
        model=c_model,
        dataloader=validation_dataloader,
        epoch=0,
        device=device,
    )
    c_model.train()
    train(
        cfg=cfg,
        model=c_model,
        train_dataloader=train_dataloader,
        test_dataloader=validation_dataloader,
        mixup_fn=mixup_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsifier=sparsifier,
        criterion=criterion,
        epochs=cfg.training.num_epochs,
        device=device,
        run_id=run_id,
    )
    if run is not None:  # Global rank == 0
        run.finish()


def _get_device(cfg: DictConfig, local_rank: int) -> torch.device:
    if not cfg.training.use_cuda:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.training.use_cuda set to True but "
            "torch.cuda.is_available() is False!"
        )
    return torch.device(f"cuda:{local_rank}")


def _parse_dist_envs() -> Tuple[int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, global_rank, world_size


def _get_sampler(train_dataset: Dataset, test_dataset: Dataset):
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_sampler, test_sampler


def _get_mixup_fn(cfg: DictConfig):
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=cfg.data.num_classes,
    )
    return mixup_fn


def _get_transforms_v1(
    cfg: DictConfig,
) -> dict[str, transforms.Compose]:
    train_tsfm = create_transform(
        input_size=cfg.data.transforms._IMAGE_WIDTH,
        is_training=True,
        color_jitter=cfg.data.transforms.color_jitter,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="const",
        re_count=1,
    )
    test_tsfm = transforms.Compose(
        [
            transforms.Resize(
                cfg.data.transforms._RESIZE_X,
                interpolation=3,
            ),
            transforms.CenterCrop(cfg.data.transforms._IMAGE_WIDTH),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.data.transforms._MEAN_RGB,
                std=cfg.data.transforms._STDDEV_RGB,
            ),
        ]
    )
    return {"train": train_tsfm, "test": test_tsfm}


def _get_transforms(cfg: DictConfig) -> dict[str, transforms.Compose]:
    scale = (0.08, 1.0)
    interpolation = "bicubic"
    if cfg.data.transforms.random_resized_crop_and_interpolation:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                cfg.data.transforms._IMAGE_WIDTH,
                scale=scale,
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        primary_tfl = [
            transforms.Resize(cfg.data.transforms._RESIZE_X, interpolation=3),
            transforms.RandomCrop(
                cfg.data.transforms._IMAGE_WIDTH,
                padding=4,
                padding_mode="reflect",
            ),
            transforms.RandomHorizontalFlip(),
        ]

    secondary_tfl = [
        transforms.RandomChoice(
            [GrayScale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]
        )
    ]

    if (
        cfg.data.transforms.color_jitter is not None
        and not cfg.data.transforms.color_jitter == 0
    ):
        secondary_tfl.append(
            transforms.ColorJitter(
                cfg.data.transforms.color_jitter,
                cfg.data.transforms.color_jitter,
                cfg.data.transforms.color_jitter,
            )
        )
    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(cfg.data.transforms._MEAN_RGB),
            std=torch.tensor(cfg.data.transforms._STDDEV_RGB),
        ),
    ]
    train_tsfm = transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
    test_tsfm = transforms.Compose(
        [
            transforms.Resize(
                cfg.data.transforms._RESIZE_X,
                interpolation=3,
            ),
            transforms.CenterCrop(cfg.data.transforms._IMAGE_WIDTH),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.data.transforms._MEAN_RGB,
                std=cfg.data.transforms._STDDEV_RGB,
            ),
        ]
    )
    return {"train": train_tsfm, "test": test_tsfm}


def _get_datasets(
    cfg: DictConfig, transforms: Optional[dict[str, transforms.Compose]]
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    assert "train" in transforms and "test" in transforms
    if cfg.data.name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=cfg.data.directory,
            train=True,
            transform=transforms["train"],
            download=False,
        )
        test_dataset = datasets.CIFAR100(
            root=cfg.data.directory,
            train=False,
            transform=transforms["test"],
            download=False,
        )
    elif cfg.data.name == "aircrafts":
        train_dataset = datasets.FGVCAircraft(
            root=cfg.data.directory,
            split="train",
            transform=transforms["train"],
            download=False,
        )
        test_dataset = datasets.FGVCAircraft(
            root=cfg.data.directory,
            split="test",
            transform=transforms["test"],
            download=False,
        )
    elif cfg.data.name == "flowers":
        train_dataset = datasets.Flowers102(
            root=cfg.data.directory,
            split="train",
            transform=transforms["train"],
            download=False,
        )
        val_dataset = datasets.Flowers102(
            root=cfg.data.directory,
            split="val",
            transform=transforms["test"],
            download=False,
        )
        # combine train and val datasets
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset]
        )
        test_dataset = datasets.Flowers102(
            root=cfg.data.directory,
            split="test",
            transform=transforms["test"],
            download=False,
        )
    elif cfg.data.name == "pets":
        train_dataset = datasets.OxfordIIITPet(
            root=cfg.data.directory,
            split="trainval",
            transform=transforms["train"],
            download=False,
        )
        test_dataset = datasets.OxfordIIITPet(
            root=cfg.data.directory,
            split="test",
            transform=transforms["test"],
            download=False,
        )
    elif cfg.data.name == "inaturalist19":
        # Load the full dataset
        dataset = datasets.ImageFolder(root=cfg.data.directory)
        # In original DeiT paper, they use 1.2% of the trainval split for testing
        train_size = int((1 - 0.011195) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, val_size]
        )
        # train_dataset.transform=transforms["train"]
        # test_dataset.transform=transforms["test"]
        train_dataset = TransformDataset(
            train_dataset, transform=transforms["train"]
        )
        test_dataset = TransformDataset(
            test_dataset, transform=transforms["test"]
        )
    elif cfg.data.name == "fake_imagenet":
        train_dataset = FakeImageNetDataset()
        test_dataset = FakeImageNetDataset()
    else:
        raise ValueError(f"Dataset {cfg.data.name} not supported.")
    return train_dataset, test_dataset


def _get_dataloaders(
    cfg: DictConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_sampler: torch.utils.data.Sampler,
    test_sampler: torch.utils.data.Sampler,
) -> Tuple[DataLoader, DataLoader]:
    world_size = get_world_size()
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, seed=cfg.training.seed
        )
        test_sampler = DistributedSampler(test_dataset, seed=cfg.training.seed)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        drop_last=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def _init_compressed_model(cfg: DictConfig) -> nn.Module:
    if not hasattr(models, cfg.model.name):
        raise ValueError(f"Model {cfg.model.name} not supported.")
    model = models.create_model(
        cfg.model.name, pretrained=False, num_classes=cfg.data.num_classes
    )
    comp_model = SMModel(cfg.model.kwargs, model)
    return comp_model


def _load_checkpoint_v2(model: SMModel, checkpoint_model: Dict):
    model_state_dict = model.state_dict()
    pretrained_state_dict = checkpoint_model
    pretrained_state_dict = {
        k: v
        for k, v in pretrained_state_dict.items()
        if k in model_state_dict and model_state_dict[k].shape == v.shape
    }
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def _load_checkpoint(model: SMModel, checkpoint_model: Dict):
    state_dict = model.state_dict()
    for k in [
        "model.head.weight",
        "model.head.bias",
        "model.head_dist.weight",
        "model.head_dist.bias",
    ]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model["model.pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.model.patch_embed.num_patches
    num_extra_tokens = model.model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(
        -1, orig_size, orig_size, embedding_size
    ).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model["model.pos_embed"] = new_pos_embed
    model.load_state_dict(checkpoint_model, strict=False)


def _get_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    if not hasattr(torch.optim, cfg.optimizer.name):
        raise ValueError(f"Optimizer {cfg.optimizer.name} not supported.")
    optimizer_class = getattr(torch.optim, cfg.optimizer.name)
    optimizer = optimizer_class(model.parameters(), **cfg.optimizer.kwargs)
    return optimizer


def _get_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    if not hasattr(torch.optim.lr_scheduler, cfg.optimizer.scheduler.name):
        raise ValueError(
            f"Scheduler {cfg.optimizer.scheduler.name} not supported."
        )
    cfg.optimizer.scheduler.kwargs.T_max = total_steps
    scheduler_class = getattr(
        torch.optim.lr_scheduler, cfg.optimizer.scheduler.name
    )
    scheduler = scheduler_class(optimizer, **cfg.optimizer.scheduler.kwargs)
    return scheduler


def _compute_t_end(cfg: DictConfig, total_steps: int):
    t_end = int(cfg.sparsifier.scheduler.t_end_coeff * total_steps)
    return t_end


def _get_sparsifier(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    t_end: int,
    total_step: int,
):
    sparsity = cfg.sparsifier.sparsity
    pruning_ratio = cfg.sparsifier.pruning_ratio

    t_accel = total_step * 0.25
    delta_t = cfg.sparsifier.scheduler.delta_t

    initial_sparsity = cfg.sparsifier.initial_sparsity
    final_sparsity = cfg.sparsifier.final_sparsity
    accelerated_sparsity = cfg.sparsifier.accelerated_sparsity

    sparsifier_name = cfg.sparsifier.name.lower()

    if sparsifier_name == "dense" or cfg.sparsifier.name is None:
        sparsifier = None

    elif sparsifier_name == "static":
        sparsifier = StaticMagnitudeSparsifier(
            optimizer=optimizer,
            distribution=UniformDistribution(),
            sparsity=sparsity,
            random_mask_init=False,
            global_pruning=cfg.sparsifier.global_pruning,
        )
    elif sparsifier_name == "gradual":
        sparsifier = GMP(
            scheduler=AcceleratedCubicScheduler(
                t_end=t_end,
                delta_t=delta_t,
                t_accel=t_accel,
                initial_sparsity=initial_sparsity,
                accelerated_sparsity=accelerated_sparsity,
                final_sparsity=final_sparsity,
            ),
            distribution=UniformDistribution(),
            optimizer=optimizer,
            random_mask_init=False,
            global_pruning=cfg.sparsifier.global_pruning,
        )
    elif sparsifier_name == "rigl":
        sparsifier = RigL(
            scheduler=CosineDecayScheduler(
                quantity=pruning_ratio, t_end=t_end, delta_t=delta_t
            ),
            distribution=UniformDistribution(),
            optimizer=optimizer,
            sparsity=sparsity,
            init_method=None,
            grown_weights_init=cfg.sparsifier.grown_weights_init,
            random_mask_init=False,
            global_pruning=cfg.sparsifier.global_pruning,
        )
    else:
        raise ValueError(f"Sparsifier {cfg.sparsifier.name} not supported.")
    return sparsifier


def train(
    cfg: DictConfig,
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    mixup_fn: Mixup,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    sparsifier: DSTMixin,
    criterion: Callable,
    epochs: int,
    device: torch.device,
    run_id: str,
):
    world_size = get_world_size()
    step = 0
    best_accuracy = 0
    for epoch in range(epochs):
        for _, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            original_targets = (
                targets.argmax(dim=1) if len(targets.shape) > 1 else targets
            )
            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss *= 2
            loss.backward()
            accuracy = (logits.argmax(1) == original_targets).float().mean()
            if sparsifier is not None:
                sparsifier.step()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Sync b/w distributed nodes
            if world_size > 1:
                # Sum reduction since we want per instance loss across world
                dist.all_reduce(loss, dist.ReduceOp.SUM, async_op=False)
                dist.all_reduce(accuracy, dist.ReduceOp.AVG, async_op=False)

            wandb_finetune_log(
                step,
                loss.item() / (len(targets) * world_size),  # per instance loss
                accuracy.item() * 100,
                logits,
                cfg.wandb.log_images,
                inputs,
                targets,
                logits.argmax(1),
            )
            if int(os.environ.get("RANK", 0)) == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Step: {step}, Train Loss: {loss.item()/(len(targets)*world_size)}"
                )
            step += 1
        if (epoch + 1) % cfg.training.validation_interval == 0:
            model.eval()
            val_accuracy = validate(
                cfg,
                model,
                test_dataloader,
                (epoch + 1),
                device,
            )
            if val_accuracy > best_accuracy:
                save_name = os.path.join(
                    cfg.paths.models, f"compressed_model_{run_id}.pt"
                )
                torch.save(model.state_dict(), save_name)
            model.train()


def validate(
    cfg: DictConfig,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
):
    world_size = get_world_size()
    model.eval()
    loss = 0
    correct = 0
    top_5_correct = 0
    top_5_accuracy = None
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss += criterion(logits, targets)
            _, preds = torch.max(logits, 1)
            correct += preds.eq(targets.view_as(preds)).sum()

            if cfg.data.record_top_5:
                _, top_5_indices = torch.topk(logits, 5, dim=1, largest=True)
                top_5_preds = (
                    targets.reshape(-1, 1).expand_as(top_5_indices)
                    == top_5_indices
                ).any(dim=1)
                top_5_correct += top_5_preds.sum()

        # Sync b/w distributed nodes
        if world_size > 1:
            # Sum reduction since we want per instance metrics across world
            dist.all_reduce(loss, dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(correct, dist.ReduceOp.SUM, async_op=False)
            if cfg.data.record_top_5:
                dist.all_reduce(
                    top_5_correct, dist.ReduceOp.SUM, async_op=False
                )

        loss /= len(dataloader.dataset)  # avg. per instance loss
        accuracy = (correct / len(dataloader.dataset)) * 100
        top_5_accuracy = (top_5_correct / len(dataloader.dataset)) * 100
    wand_finetune_val_log(
        epoch,
        loss.item(),
        accuracy.item(),
        logits,
        cfg.wandb.log_images,
        top_5_accuracy.item(),
        inputs,
        targets,
        preds,
    )
    if int(os.environ.get("RANK", 0)) == 0:
        print(
            f"Epoch: {epoch}, Validation Loss: {loss}, Accuracy: {accuracy}, Top 5 Accuracy: {top_5_accuracy}"
        )
    return accuracy


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    wandb.login()
    main()
