# TODO - attribution to DOJO for inspiration

import math
import re
import os
import time
import pathlib
import collections
from functools import reduce
from typing import Optional, Tuple, List, Dict, Callable
import dotenv

import wandb
import torch
import torch.utils
import torch.nn as nn
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torchvision import datasets, transforms
from timm import models
import numpy as np
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from sparsimony.distributions.base import (
    UniformDistribution,
    UniformNMDistribution,  # will warn if shapes are not supported by kernels
)
from sparsimony.schedulers.base import (
    CosineDecayScheduler,
    AcceleratedCubicScheduler,
    AlwaysTrueScheduler,
)
from sparsimony.dst.base import DSTMixin
from sparsimony.dst.static import (
    StaticMagnitudeSparsifier,
)
from sparsimony.dst.gmp import GMP, SGMP
from sparsimony.pruners.ste import SRSTESparsifier
from sparsimony.dst.rigl import RigL
from sparsimony.dst.srigl import NMSRigL

from sd2s.lrm import SMLayer, SMModel
from sd2s.utils import DDPWrappedGetAttr, get_world_size
from data.stratified_sampler import (
    StratifiedSampler,
    DistributedStratifiedSampler,
)
from data.custom_hf_dataset import FakeImageNetDataset
from SETTINGS import CONFIG_DIR
from training.wandb_utils import (
    init_wandb,
    parse_wandb_run_id,
    wandb_training_log,
    wandb_validation_log,
)


@hydra.main(
    config_path=CONFIG_DIR, config_name="swin_base_config", version_base="1.3"
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
    if cfg.training.hooks.num_batches == -1:
        cfg.training.hooks.num_batches = int(
            cfg.training.hooks.optimal_steps / cfg.training.num_epochs
        )
    train_dataset, validation_dataset = _get_datasets(cfg, _get_transforms(cfg))
    train_dataloader, validation_dataloader = _get_dataloaders(
        cfg,
        train_dataset,
        validation_dataset,
    )

    # MODEL
    original_model = _get_model(cfg, device)
    original_model.eval()
    _set_model_parameters_gradient_irrelevant(original_model)
    _define_blocks(cfg, original_model)
    c_model = _get_compressed_model(cfg, original_model)

    # OPTIMIZER, SCHEDULER, SPARSIFIER, T_END for SPARSIFIER, CRITERION
    optimizer = _get_optimizer(cfg, c_model)
    scheduler = _get_scheduler(cfg, optimizer)
    t_end = _compute_t_end(cfg)
    sparsifier = _get_sparsifier(cfg, optimizer, t_end)
    criterion = _get_criterion(cfg)

    if sparsifier is not None:
        sparse_config = [
            {"tensor_fqn": f"{fqn}.V"}
            for fqn, module in c_model.named_modules()
            if isinstance(module, SMLayer)
        ]
        sparsifier.prepare(c_model, sparse_config)

    # INIT DDP
    if world_size > 1:
        c_model = DDPWrappedGetAttr(c_model, device_ids=[local_rank])
        c_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(c_model)

    # WANDB LOGS RUN INIT
    run = init_wandb(cfg, global_rank)
    run_id = parse_wandb_run_id(run)

    # GET ACTIVATIONS FOR DISTILLATION / EFFICIENT TRAINING
    original_model_activations = _get_k_activations(
        cfg,
        original_model,
        train_dataloader,
        device,
    )
    step = 0
    train_validate(
        cfg=cfg,
        model=c_model,
        original_model_activations=original_model_activations,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsifier=sparsifier,
        criterion=criterion,
        validation_dataloader=validation_dataloader,
        validation_dataset_size=len(validation_dataset),
        step=step,
        device=device,
    )
    if run is not None:
        # Only finish and save in global rank 0
        run.finish()
        save_name = os.path.join(
            cfg.paths.models, f"compressed_model_{run_id}.pt"
        )
        torch.save(c_model.state_dict(), save_name)


def _define_blocks(cfg: DictConfig, model: nn.Module) -> List[nn.Module]:
    pattern = rf"{re.escape(cfg.model.kwargs.block_fqn_prefix)}\.\d+$"
    blocks = []
    for fqn, module in model.named_modules():
        if re.search(pattern, fqn):
            blocks.append(module)
    model.defined_blocks = blocks


def _set_model_parameters_gradient_irrelevant(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


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


def _get_transforms(cfg: DictConfig) -> dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomChoice(
                [
                    transforms.Resize(cfg.data.transforms._RESIZE_X),
                    transforms.Resize(cfg.data.transforms._RESIZE_Y),
                ]
            ),
            transforms.RandomCrop(
                size=[
                    cfg.data.transforms._IMAGE_WIDTH,
                    cfg.data.transforms._IMAGE_HEIGHT,
                ]
            ),
            transforms.RandomHorizontalFlip(
                p=cfg.data.transforms._HORIZONTAL_FLIP_P
            ),
            transforms.ToTensor(),  # Applies min/max scaling
            transforms.Normalize(
                mean=cfg.data.transforms._MEAN_RGB,
                std=cfg.data.transforms._STDDEV_RGB,
            ),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.transforms._RESIZE_X),
            transforms.CenterCrop(
                size=[
                    cfg.data.transforms._IMAGE_WIDTH,
                    cfg.data.transforms._IMAGE_HEIGHT,
                ]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.data.transforms._MEAN_RGB,
                std=cfg.data.transforms._STDDEV_RGB,
            ),
        ]
    )
    return {"train": train_transform, "val": validation_transform}


def _get_datasets(
    cfg: DictConfig, transforms: Optional[dict[str, transforms.Compose]]
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    assert "train" in transforms and "val" in transforms
    if cfg.data.name == "imagenet":
        train_dataset = datasets.ImageFolder(
            os.path.join(cfg.data.directory, "train"),
            transform=transforms["train"],
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(cfg.data.directory, "val"),
            transform=transforms["val"],
        )
    elif cfg.data.name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=cfg.data.directory,
            train=True,
            transform=transforms["train"],
            download=False,
        )
        val_dataset = datasets.CIFAR100(
            root=cfg.data.directory,
            train=False,
            transform=transforms["val"],
            download=False,
        )
    elif cfg.data.name == "wikisql":
        raise NotImplementedError(f"WikiSQL not implemented yet")
    elif cfg.data.name == "fake_imagenet":
        train_dataset = FakeImageNetDataset()
        val_dataset = train_dataset
    else:
        raise ValueError(f"Dataset {cfg.data.name} not supported.")
    return train_dataset, val_dataset


def _get_dataloaders(
    cfg: DictConfig,
    train_dataset: torch.utils.data.Dataset,
    validation_dataset: torch.utils.data.Dataset,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dist.is_initialized():
        sampler = DistributedStratifiedSampler(
            train_dataset,
            cfg.data.batch_size,
            cfg.training.hooks.num_batches,
            cfg.data.num_classes,
            seed=cfg.training.seed,
        )
    else:
        sampler = StratifiedSampler(
            train_dataset,
            cfg.data.batch_size,
            cfg.training.hooks.num_batches,
            cfg.data.num_classes,
        )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
    )
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset=validation_dataset, seed=cfg.training.seed, drop_last=True
        )
    else:
        sampler = None
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        drop_last=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return train_dataloader, validation_dataloader


def _get_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    if not hasattr(models, cfg.model.name):
        raise ValueError(f"Model {cfg.model.name} not supported.")
    model_path = pathlib.Path(cfg.model.path)
    if not model_path.is_file():
        model = models.create_model(cfg.model.name, pretrained=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    model = models.create_model(cfg.model.name, pretrained=False)
    model.load_state_dict(torch.load(cfg.model.path))
    model.to(device)
    return model


def _get_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    if not hasattr(torch.optim, cfg.optimizer.name):
        raise ValueError(f"Optimizer {cfg.optimizer.name} not supported.")
    optimizer_class = getattr(torch.optim, cfg.optimizer.name)
    optimizer = optimizer_class(model.parameters(), **cfg.optimizer.kwargs)
    return optimizer


def _get_criterion(cfg: DictConfig) -> Callable:
    if cfg.training.task == "ImageClassification":
        criterion = nn.functional.cross_entropy
    elif cfg.training.task == "NLG":
        # Perplexity etc. to be added!
        ...
    else:
        raise ValueError(f"Task {cfg.training.task} not supported.")
    return criterion


def _get_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    if not hasattr(torch.optim.lr_scheduler, cfg.optimizer.scheduler.name):
        raise ValueError(
            f"Scheduler {cfg.optimizer.scheduler.name} not supported."
        )
    cfg.optimizer.scheduler.kwargs.T_max = (
        cfg.training.num_epochs * cfg.training.hooks.num_batches
    )
    scheduler_class = getattr(
        torch.optim.lr_scheduler, cfg.optimizer.scheduler.name
    )
    scheduler = scheduler_class(optimizer, **cfg.optimizer.scheduler.kwargs)
    return scheduler


def _compute_t_end(
    cfg: DictConfig,
):
    t_end = int(
        cfg.sparsifier.scheduler.t_end_coeff
        * cfg.training.num_epochs
        * cfg.training.hooks.num_batches
    )
    return t_end


def _get_sparsifier(
    cfg: DictConfig, optimizer: torch.optim.Optimizer, t_end: int
):
    sparsity = cfg.sparsifier.sparsity
    pruning_ratio = cfg.sparsifier.pruning_ratio

    t_accel = cfg.sparsifier.t_accel
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
    elif sparsifier_name == "srigl":
        m = 4
        n = math.floor((1 - sparsity) * m)
        sparsifier = NMSRigL(
            scheduler=CosineDecayScheduler(
                quantity=pruning_ratio,
                t_end=t_end,
                delta_t=delta_t,
            ),
            # set strict=True to skip bad layers
            distribution=UniformNMDistribution(n=n, m=m),
            optimizer=optimizer,
            random_mask_init=False,  # weights are pretrained, pick top magnitudes
            init_method=None,  # critical to avoid overwriting decomposition init
            global_pruning=False,  # n/a for N:M sparsity
            n=n,
            m=m,
            sparsity=sparsity,
        )
    elif sparsifier_name == "sgmp":
        m = 4
        n = math.floor((1 - sparsity) * m)
        sparsifier = SGMP(
            scheduler=AcceleratedCubicScheduler(
                t_end=t_end,
                delta_t=delta_t,
                t_accel=t_accel,
                initial_sparsity=initial_sparsity,
                accelerated_sparsity=accelerated_sparsity,
                final_sparsity=final_sparsity,
            ),
            # set strict=True to skip bad layers
            distribution=UniformNMDistribution(n=n, m=m),
            optimizer=optimizer,
            random_mask_init=False,  # weights are pretrained, pick top magnitudes
            init_method=None,  # critical to avoid overwriting decomposition init
            global_pruning=False,  # n/a for N:M sparsity
            n=n,
            m=m,
        )
    elif sparsifier_name == "srste":
        m = 4
        n = math.floor((1 - sparsity) * m)
        sparsifier = SRSTESparsifier(
            scheduler=AlwaysTrueScheduler(),
            distribution=UniformNMDistribution(n=n, m=m),
            n=n,
            m=m,
            decay=2e-4,
        )
    elif sparsifier_name == "ste":
        m = 4
        n = math.floor((1 - sparsity) * m)
        sparsifier = SRSTESparsifier(
            scheduler=AlwaysTrueScheduler(),
            distribution=UniformNMDistribution(n=n, m=m),
            n=n,
            m=m,
            decay=None,
        )
    else:
        raise ValueError(f"Sparsifier {cfg.sparsifier.name} not supported.")
    return sparsifier


def _get_compressed_model(
    cfg: DictConfig,
    model: nn.Module,
) -> nn.Module:
    c_model = SMModel(cfg.model.kwargs, model)
    return c_model


def _get_activation(name: str, activations: Dict[str, torch.Tensor]):
    def hook(model, input, output):
        activations[name] = output

    return hook


def _register_hook(
    cfg: DictConfig,
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    block_idx: int,
):
    if block_idx == -1:
        module = reduce(getattr, cfg.model.kwargs.emb_fqn.split("."), model)
        activation_func = _get_activation(cfg.model.kwargs.emb_fqn, activations)
    else:
        module = model.defined_blocks[block_idx]
        activation_func = _get_activation(f"block_{block_idx}", activations)

    return module.register_forward_hook(activation_func)


def _get_k_activations(
    cfg: DictConfig,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    world_size = get_world_size()
    handle_list = []
    activations = {}
    num_blocks = len(model.defined_blocks)
    # -1 for the embedding layer
    for i in range(-1, num_blocks):
        handle = _register_hook(cfg, model, activations, i)
        handle_list.append(handle)
    results_dict = collections.defaultdict(list)

    for ctr, (images, _) in enumerate(dataloader):
        images = images.to(device)
        if ctr == math.ceil(cfg.training.hooks.num_batches / world_size):
            # Should never be reached since
            # len(dataloader) <= cfg.training.hooks.num_batches
            break
        with torch.no_grad():
            _ = model(images)
        for key in activations:
            results_dict[key].append(activations[key].detach().cpu())
    for key in results_dict:
        results_dict[key] = torch.cat(results_dict[key])
    for handle in handle_list:
        handle.remove()

    return results_dict


def train_validate(
    cfg: DictConfig,
    model: SMModel,
    original_model_activations: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    sparsifier: Optional[DSTMixin],
    criterion: Callable,
    validation_dataloader: torch.utils.data.DataLoader,
    validation_dataset_size: int,
    step: int,
    device: torch.device,
):
    world_size = get_world_size()
    num_blocks = len(model.blocks)
    per_block_avg_error = collections.defaultdict(float)

    n_samples = original_model_activations["block_0"].shape[0]
    n_steps = n_samples // cfg.data.batch_size

    for epoch in range(cfg.training.num_epochs):
        permutation = np.random.permutation(n_samples)
        if epoch % cfg.training.validation_interval == 0:
            validation_start = time.time()
            validate(
                cfg=cfg,
                model=model,
                dataloader=validation_dataloader,
                dataset_size=validation_dataset_size,
                criterion=criterion,
                step=step,
                device=device,
            )
            wandb.log({"val_sec": time.time() - validation_start})
        epoch_start = time.time()
        sec_per_epoch = 0
        avg_block_error = 0
        for i in range(n_steps):
            step += 1
            error = 0
            step_start = time.time()
            avg_sec_per_step = 0
            start_id = i * cfg.data.batch_size
            end_id = (i + 1) * cfg.data.batch_size
            c_slice = permutation[start_id:end_id]
            for block_idx in range(num_blocks):
                if block_idx == 0:
                    x_original = original_model_activations[
                        cfg.model.kwargs.emb_fqn
                    ][c_slice]
                else:
                    x_original = original_model_activations[
                        f"block_{block_idx-1}"
                    ][c_slice]
                x_original = x_original.to(device)

                if cfg.model.kwargs.normalization.in_place:
                    if (
                        block_idx
                        in cfg.model.kwargs.normalization.apply_before_idx
                    ):
                        x_original = model.normalization_layers[block_idx](
                            x_original
                        )
                y_compressed = model.blocks[block_idx](x_original)
                y_original = original_model_activations[f"block_{block_idx}"][
                    c_slice
                ]
                y_original = y_original.to(device)
                temp_err = torch.nn.functional.mse_loss(
                    y_compressed, y_original
                )
                per_block_avg_error[f"block_{block_idx}"] += temp_err
                error += temp_err

            error.backward()
            current_sparsity = 0
            per_layer_sparsity = None
            if sparsifier is not None:
                sparsifier.step()
                current_sparsity = sparsifier.sparsity
                if (
                    not cfg.sparsifier.name == "srste"
                    and not cfg.sparsifier.name == "ste"
                    and sparsifier.global_pruning
                ):
                    per_layer_sparsity = sparsifier.get_layerwise_sparsity()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            avg_block_error += error / num_blocks
            avg_sec_per_step += time.time() - step_start

        parameter_count = model.model_param_count(sparsifier)

        # Sync b/w distributed nodes
        if world_size > 1:
            for _, v in per_block_avg_error.items():
                dist.all_reduce(v, dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(avg_block_error, dist.ReduceOp.SUM, async_op=False)

        # divide all values by n_steps
        per_block_avg_error = {
            k: v / n_steps for k, v in per_block_avg_error.items() if v != 0
        }
        sec_per_epoch = time.time() - epoch_start
        avg_sec_per_step /= n_steps
        avg_block_error /= n_steps
        log_kwargs = {
            "step": step,
            "epoch": epoch,
            "avg_sec_per_step": avg_sec_per_step,
            "sec_per_epoch": sec_per_epoch,
            "overall_block_avg_error": avg_block_error,
            "per_block_avg_error": per_block_avg_error,
            "lr": optimizer.param_groups[0]["lr"],
            "sparsity": current_sparsity,
            "parameter_count": parameter_count,
            "per_layer_sparsity": per_layer_sparsity,
        }
        wandb_training_log(**log_kwargs)
        print(
            f"Training Epoch: {epoch}, Avg. Blockwise Error: {avg_block_error}"
        )
    validation_now = time.time()
    validate(
        cfg=cfg,
        model=model,
        dataloader=validation_dataloader,
        dataset_size=validation_dataset_size,
        criterion=criterion,
        step=step,
        device=device,
    )
    wandb.log({"val_sec": time.time() - validation_now})
    return


def validate(
    cfg: DictConfig,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset_size: int,
    criterion: Callable,
    step: int,
    device: torch.device,
):
    world_size = get_world_size()
    model.eval()
    loss = 0
    correct = 0
    top_5_correct = 0
    top_5_accuracy = None
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss += criterion(
                logits, targets, label_smoothing=cfg.optimizer.label_smoothing
            )
            _, preds = torch.max(logits, 1)
            correct += preds.eq(targets.view_as(preds)).sum()

            if cfg.data.record_top_5:
                _, top_5_indices = torch.topk(logits, 5, dim=1, largest=True)
                top_5_preds = (
                    targets.reshape(-1, 1).expand_as(top_5_indices)
                    == top_5_indices
                ).any(dim=1)
                top_5_correct += top_5_preds.sum()

        if world_size > 1:
            dist.all_reduce(loss, dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(correct, dist.ReduceOp.SUM, async_op=False)
            if cfg.data.record_top_5:
                dist.all_reduce(
                    top_5_correct, dist.ReduceOp.SUM, async_op=False
                )

        loss = loss.item() / dataset_size
        accuracy = (correct.item() / dataset_size) * 100
        top_5_accuracy = (top_5_correct.item() / dataset_size) * 100

    wandb_validation_log(
        step,
        loss,
        accuracy,
        logits,
        cfg.wandb.log_images,
        top_5_accuracy,
        inputs,
        targets,
        preds,
    )
    print(
        f"Validation Loss: {loss}, Accuracy: {accuracy}, Top 5 Accuracy: {top_5_accuracy}"
    )


if __name__ == "__main__":
    # Override env vars using .env BEFORE initializing hydra config!
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    wandb.login()
    main()
