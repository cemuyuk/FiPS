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

from sparsimony.distributions.base import UniformDistribution
from sparsimony.dst.static import StaticMagnitudeSparsifier

from sd2s.lrm import SMLayer, SMModel
from sd2s.utils import DDPWrappedGetAttr

from SETTINGS import CONFIG_DIR


@hydra.main(
    config_path=CONFIG_DIR,
    config_name="gemma_2_2b_config_local",
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
    wikitext_test = _get_wikitext_test()

    # MODEL
    original_model = _get_model(cfg, device)
    original_model = original_model.to(device)
    original_model.eval()

    # perp = validate(
    #     cfg=cfg,
    #     tokenizer=tokenizer,
    #     model=original_model,
    #     dataset=wikitext_test,
    #     device=device,
    # )

    # print(f"Original Model. Mean Perplexity: {perp['mean_perplexity']:.4f}")
    # result_text = (
    #     f"Original Model. Mean Perplexity: {perp['mean_perplexity']:.4f}\n"
    # )

    _set_gradient_false(original_model)
    _define_blocks(cfg, original_model)
    c_model = _get_compressed_model(cfg, original_model)
    del original_model
    torch.cuda.empty_cache()

    checkpoint_model = torch.load(cfg.model.checkpoint_path, weights_only=True)
    # for key in list(checkpoint_model.keys()):
    #     if "model." in key:
    #         checkpoint_model[key.replace("model.", "")] = checkpoint_model.pop(
    #             key
    #         )
    c_model = _load_state_dict(c_model, checkpoint_model)
    c_model.to(device)

    for fqn, parameter in c_model.named_parameters():
        parameter.requires_grad = False
    # optimizer = torch.optim.Adam(c_model.parameters(), lr=1e-5)
    # sparsifier = _get_sparsifier(cfg, optimizer)

    # if sparsifier is not None:
    #     sparse_config = [
    #         {"tensor_fqn": f"{fqn}.V"}
    #         for fqn, module in c_model.named_modules()
    #         if isinstance(module, SMLayer)
    #     ]
    #     sparsifier.prepare(c_model, sparse_config)
    # sparsifier.squash_mask()

    # INIT DDP
    if world_size > 1:
        c_model = DDPWrappedGetAttr(c_model, device_ids=[local_rank])
        c_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(c_model)

    total_param_count = 0
    total_zero_param_count = 0
    for fqn, module in c_model.named_modules():
        if isinstance(module, SMLayer):
            total_param_count = module.V.numel()
            total_zero_param_count = module.V.data.eq(0).sum().item()
            print(
                f"Sparsity for {fqn}: {total_zero_param_count/total_param_count:.4f}."
            )

    print(f"Total Param Count: {total_param_count}.")
    print(f"Total Zero Param Count: {total_zero_param_count}.")
    print(f"Sparsity: {total_zero_param_count/total_param_count:.4f}.")

    c_model.eval()
    perp = validate(
        cfg=cfg,
        tokenizer=tokenizer,
        model=c_model,
        dataset=wikitext_test,
        device=device,
    )
    print(f"Post Training. Mean Perplexity: {perp['mean_perplexity']:.4f}")
    result_text = (
        f"Post Training. Mean Perplexity: {perp['mean_perplexity']:.4f}\n"
    )

    with open(
        f"./training/models/gemma2_2b/results_vcjw5ey4_pb50_norm.txt", "w"
    ) as f:
        f.write(result_text)


def _load_state_dict(model: nn.Module, checkpoint_model: dict) -> nn.Module:
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in checkpoint_model.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    return model


def _define_blocks(cfg: DictConfig, model: nn.Module) -> List[nn.Module]:
    pattern = rf"{re.escape(cfg.model.kwargs.block_fqn_prefix)}\.\d+$"
    blocks = []
    for fqn, module in model.named_modules():
        if re.search(pattern, fqn):
            blocks.append(module)
    model.defined_blocks = blocks


def _set_gradient_false(model: nn.Module) -> None:
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


def _get_sparsifier(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
):
    sparsity = cfg.sparsifier.sparsity
    sparsifier = StaticMagnitudeSparsifier(
        optimizer=optimizer,
        distribution=UniformDistribution(),
        sparsity=sparsity,
        random_mask_init=False,
        global_pruning=False,
        global_buffers_cpu_offload=True,
    )

    return sparsifier


def _get_compressed_model(
    cfg: DictConfig,
    model: nn.Module,
) -> nn.Module:
    c_model = SMModel(cfg.model.kwargs, model)
    return c_model


def validate(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
):
    batch_size = 4096
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

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    # Override env vars using .env BEFORE initializing hydra config!
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    wandb.login()
    main()
