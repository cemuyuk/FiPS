# TODO - attribution to DOJO for inspiration

import math
import re
import os
import time
import collections
from functools import reduce
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import dotenv

import wandb
import torch
import torch.nn as nn
import torch.distributed as dist


import numpy as np
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from sparsimony.distributions.base import (
    UniformDistribution,
    UniformNMDistribution,
)
from sparsimony.schedulers.base import (
    AlwaysTrueScheduler,
    CosineDecayScheduler,
    AcceleratedCubicScheduler,
)
from sparsimony.dst.base import DSTMixin
from sparsimony.dst.static import (
    StaticMagnitudeSparsifier,
)
from sparsimony.pruners.ste import SRSTESparsifier
from sparsimony.dst.gmp import GMP, SGMP
from sparsimony.dst.rigl import RigL
from sparsimony.dst.srigl import NMSRigL

from sd2s.lrm import SMLayer, SMModel
from sd2s.utils import DDPWrappedGetAttr, get_world_size

from SETTINGS import CONFIG_DIR
from training.wandb_utils import (
    init_wandb,
    parse_wandb_run_id,
    wandb_llm_training_log,
    wandb_llm_validation_log,
)


@hydra.main(
    config_path=CONFIG_DIR, 
    config_name="gemma_2_2b_config", 
    version_base="1.3",
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

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    train_dataset, _ = _get_datasets(
        cfg,
        tokenizer,
    )
    wikitext_test = _get_wikitext_test()

    # MODEL
    original_model = _get_model(cfg, device)
    original_model = original_model.to(device)
    original_model.eval()

    def mlp_filter_fn(args):
        # print(args[0])
        last_key = args[0].split(".")[-1]
        if "mlp" == last_key:
            return True
        else:
            return False

    # each elements is <key, module>
    mlp_named_modules = filter(mlp_filter_fn, original_model.named_modules())

    original_model_activations = _get_k_activations(
        cfg=cfg,
        model=original_model,
        tokenizer=tokenizer,
        mlp_named_modules=mlp_named_modules,
        dataset=train_dataset,
        n_batch=cfg.training.hooks.num_batches,
        device=device,
    )

    _set_gradient_false(original_model)
    _define_blocks(cfg, original_model)
    c_model = _get_compressed_model(cfg, original_model)
    c_model.to(device)

    del original_model
    torch.cuda.empty_cache()
    # OPTIMIZER, SCHEDULER, SPARSIFIER, T_END for SPARSIFIER, CRITERION
    optimizer = _get_optimizer(cfg, c_model)
    scheduler = _get_scheduler(cfg, optimizer)
    t_end = _compute_t_end(cfg)
    sparsifier = _get_sparsifier(cfg, optimizer, t_end)

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

    # EVAL PRIOR TRAINING
    step = 0
    # START EFFICIENT TRAINING
    train_validate(
        cfg=cfg,
        c_model=c_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsifier=sparsifier,
        original_model_activations=original_model_activations,
        val_dataset=wikitext_test,
        step=step,
        device=device,
    )
    validation_now = time.time()
    c_model.eval()
    optimizer.zero_grad(set_to_none=True)
    sparsifier.squash_mask()
    del optimizer, scheduler, sparsifier, original_model_activations
    torch.cuda.empty_cache()
    perp = validate(
        cfg=cfg,
        tokenizer=tokenizer,
        model=c_model,
        dataset=wikitext_test,
        step=step,
        device=device,
    )
    wandb.log({"val_sec": time.time() - validation_now})
    print(f"Training End - Mean Perplexity: {perp['mean_perplexity']:.4f}.")

    if run is not None:
        # Only finish and save in global rank 0
        run.finish()
    # Synchronize (especially for multi-GPU training)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    save_name = os.path.join(
        "home/cem.uyuk/SparseDecompositions2Share/training/models/gemma2_2b/",
        f"{cfg.model.save_name}_{run_id}_pb50.pt",
    )
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    print(f"Current working directory is: {os.getcwd()}")
    torch.save(c_model.state_dict(), save_name)

def _define_blocks(cfg: DictConfig, model: nn.Module) -> List[nn.Module]:
    pattern = rf"{re.escape(cfg.model.kwargs.block_fqn_prefix)}\.\d+$"
    blocks = []
    for fqn, module in model.named_modules():
        if re.search(pattern, fqn):
            blocks.append(module)
    model.defined_blocks = blocks


def _set_gradient_false(model: nn.Module) -> None:
    for fqn, param in model.named_parameters():
        if "pre_feedforward_layernorm" in fqn or "post_feedforward_layernorm" in fqn:
            continue
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


def _get_wikitext_test():
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")


def _get_datasets(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if cfg.data.name == "redpajama":
        train_dataset = load_dataset(
            "DKYoon/SlimPajama-6B",
            split="train",
        )
        validation_dataset = load_dataset(
            "DKYoon/SlimPajama-6B",
            split="validation",
        )
    elif cfg.data.name == "wikitext":
        train_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train"
        )
        validation_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="validation"
        )

    else:
        raise ValueError(f"Dataset {cfg.data.name} not supported.")
    return train_dataset, validation_dataset


def _get_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    # TODO - update with assertions etc.
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    # model.to(device)
    return model


def _get_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    if not hasattr(torch.optim, cfg.optimizer.name):
        raise ValueError(f"Optimizer {cfg.optimizer.name} not supported.")
    optimizer_class = getattr(torch.optim, cfg.optimizer.name)
    optimizer = optimizer_class(model.parameters(), **cfg.optimizer.kwargs)
    return optimizer


def _get_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    if not hasattr(torch.optim.lr_scheduler, cfg.optimizer.scheduler.name):
        raise ValueError(
            f"Scheduler {cfg.optimizer.scheduler.name} not supported."
        )
    cfg.optimizer.scheduler.kwargs.T_max = (
        cfg.training.num_epochs
        * cfg.training.hooks.num_batches
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
            global_buffers_cpu_offload=True,
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
            low_mem_mode=True,
            global_buffers_cpu_offload=True,
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
            global_buffers_cpu_offload=True,
            # low_mem_mode=True,
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
            global_pruning=False,  # n/a for N:M sparsity
            n=n,
            m=m,
        )
    elif sparsifier_name == "srste":
        m = 4
        n = math.floor((1 - sparsity) * m)
        sparsifier = SRSTESparsifier(
            sparsifier=AlwaysTrueScheduler(),
            # set strict=True to skip bad layers
            distribution=UniformNMDistribution(n=n, m=m),
            optimizer=optimizer,
            random_mask_init=False,  # weights are pretrained, pick top magnitudes
            init_method=None,  # critical to avoid overwriting decomposition init
            global_pruning=False,  # n/a for N:M sparsity
            n=n,
            m=m,
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


def _register_hook(
    module_key: str,
    module: nn.Module,
    activations: Dict[str, torch.Tensor],
):
    def hook(model, input, output):
        x = input if isinstance(input, torch.Tensor) else input[0]
        y = output if isinstance(output, torch.Tensor) else output[0]
        activations[module_key] = [x, y]

    return module.register_forward_hook(hook)


def _get_k_activations(
    cfg: DictConfig,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    mlp_named_modules: List[Tuple[str, nn.Module]],
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    n_batch: int,
) -> Dict[str, torch.Tensor]:
    handle_list = []
    activations = {}

    # -1 for the embedding layer
    for key, mlp_module in mlp_named_modules:
        handle = _register_hook(key, mlp_module, activations)
        handle_list.append(handle)

    results_dict = collections.defaultdict(list)
    # num_in_tokens = model.model.config.max_position_embeddings
    num_in_tokens = cfg.data.batch_size
    encodings = tokenizer(
        "\n\n".join(dataset["text"][:10000]), return_tensors="pt"
    )
    seq_len = encodings.input_ids.size(1)

    for ctr, begin_loc in enumerate(range(0, seq_len, num_in_tokens)):
        # for ctr, begin_loc in enumerate(range(0, len(dataset["text"]), num_in_tokens)):
        end_loc = min(begin_loc + num_in_tokens, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        if ctr == n_batch:
            break

        with torch.no_grad():
            _ = model(input_ids)

        for key, (x, y) in activations.items():
            x, y = x.detach().cpu(), y.detach().cpu()
            results_dict[key].append((x, y))

    for key in results_dict:
        inputs = [x for x, _ in results_dict[key]]
        outputs = [y for _, y in results_dict[key]]
        mlp_dim = inputs[0].shape[-1]
        results_dict[key] = (
            torch.cat(inputs, dim=0).view(-1, mlp_dim),
            torch.cat(outputs, dim=0).view(-1, mlp_dim),
        )

    for handle in handle_list:
        handle.remove()

    return results_dict


def train_validate(
    cfg: DictConfig,
    c_model: SMModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    sparsifier: Optional[DSTMixin],
    original_model_activations: Dict[str, torch.Tensor],
    val_dataset: torch.utils.data.DataLoader,
    step: int,
    device: torch.device,
):
    print("Starting training...")
    world_size = get_world_size()
    num_blocks = len(c_model.blocks)
    per_mlp_avg_error = collections.defaultdict(float)

    # Assumes all activations have the same batch size. Find shape from the 1st in/out tuple.
    n_samples = next(iter(original_model_activations.values()))[0].shape[0]
    n_steps = n_samples // cfg.data.batch_size
    # validation_now = time.time()
    # c_model.eval()
    # perp = validate(
    #     cfg=cfg,
    #     tokenizer=tokenizer,
    #     model=c_model,
    #     dataset=val_dataset,
    #     step=step,
    #     device=device,
    # )
    # wandb.log({"val_sec": time.time() - validation_now})
    # print(f"Training Start - Mean Perplexity: {perp['mean_perplexity']:.4f}.")
    c_model.train()
    for epoch in range(cfg.training.num_epochs):
        permutation = np.random.permutation(n_samples)
        # if epoch % cfg.training.validation_interval == 0:
        #     validation_start =  time.time()
        #     c_model.eval()
        #     perp = validate(
        #         cfg=cfg,
        #         tokenizer=tokenizer,
        #         model=c_model,
        #         dataset=val_dataset,
        #         step=step,
        #         device=device,
        #     )
        #     print(
        #         f"Epoch: {epoch}, Mean Perplexity: {perp['mean_perplexity']:.4f}"
        #     )
        #     wandb.log({"val_sec": time.time() - validation_start})
        # c_model.train()
        epoch_start = time.time()
        sec_per_epoch = 0
        avg_mlp_error = 0
        epoch_start = time.time()
        for i in tqdm(range(n_steps)):
            step += 1
            error = 0
            avg_sec_per_step = 0
            step_start = time.time()
            start_id = i * cfg.data.batch_size
            end_id = (i + 1) * cfg.data.batch_size
            c_slice = permutation[start_id:end_id]
            for mlp_key, (x_orig, y_orig) in original_model_activations.items():
                x_orig = x_orig[c_slice].to(device)
                y_orig = y_orig[c_slice].to(device)
                compressed_mlp = c_model.get_submodule("model." + mlp_key).to(
                    device
                )
                y_compressed = compressed_mlp(x_orig)
                temp_err = torch.nn.functional.mse_loss(y_compressed, y_orig)
                mlp_idx = mlp_key.split(".")[2]
                per_mlp_avg_error[f"mlp_{mlp_idx}"] += temp_err.item()
                error += temp_err

            error.backward()
            if sparsifier is not None:
                sparsifier.step()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            avg_mlp_error += error / num_blocks
            avg_sec_per_step += time.time() - step_start

        # Sync b/w distributed nodes
        if world_size > 1:
            for _, v in per_mlp_avg_error.items():
                dist.all_reduce(v, dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(avg_mlp_error, dist.ReduceOp.SUM, async_op=False)

        avg_sec_per_step /= n_steps
        sec_per_epoch = time.time() - epoch_start
        per_mlp_avg_error = {
            k: v / n_steps for k, v in per_mlp_avg_error.items() if v != 0
        }
        overall_mlp_avg_error = avg_mlp_error.item() / n_steps
        current_sparsity = 0
        per_layer_sparsity = None
        if sparsifier is not None:
            current_sparsity = sparsifier.sparsity
            if sparsifier.global_pruning:
                per_layer_sparsity = sparsifier.get_layerwise_sparsity()
        parameter_count = c_model.model_param_count()

        log_kwargs = {
            "step": step,
            "epoch": epoch,
            "avg_sec_per_step": avg_sec_per_step,
            "sec_per_epoch": sec_per_epoch,
            "overall_mlp_avg_error": overall_mlp_avg_error,
            "per_mlp_avg_error": per_mlp_avg_error,
            "lr": optimizer.param_groups[0]["lr"],
            "sparsity": current_sparsity,
            "parameter_count": parameter_count,
            "per_layer_sparsity": per_layer_sparsity,
        }
        wandb_llm_training_log(**log_kwargs)
        print(f"Training Epoch: {epoch}, Avg. Blockwise Error: {avg_mlp_error}")
    return


def validate(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    step: int,
    device: torch.device,
):
    batch_size = cfg.data.batch_size
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        max_length=batch_size,
        truncation=True if batch_size else False,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]
    seq_len = encodings.input_ids.size(1)

    ppls = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, seq_len, batch_size)):
        end_index = min(start_index + batch_size, seq_len)
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask_batch
                ).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        wandb_llm_validation_log(
            step=step,
            perplexity=np.mean(ppls),
        )
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    # Override env vars using .env BEFORE initializing hydra config!
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    wandb.login()
    main()
