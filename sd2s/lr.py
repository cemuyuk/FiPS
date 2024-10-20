import re
import copy

from functools import reduce

import torch
import torch.nn as nn

from omegaconf import DictConfig

from sd2s.utils import svd_decompose


class LRLayer(nn.Module):
    """Low Rank Masked Layer"""

    def __init__(
        self,
        U: nn.Parameter,
        V: nn.Parameter,
        in_features: int,
        out_features: int,
        bias: torch.tensor,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=True)

        self.U = nn.Parameter(U, requires_grad=True)
        self.V = nn.Parameter(V, requires_grad=True)

    def forward(self, x: torch.tensor):
        weight = self.U @ self.V
        return torch.nn.functional.linear(x, weight, self.bias)


class LRModel(nn.Module):
    """Low Rank Model"""

    def __init__(self, cfg: DictConfig, model: nn.Module):
        super().__init__()

        self.model = copy.deepcopy(model)

        self.cfg = cfg

        self._define_blocks(cfg)
        self._define_target_layers(cfg)
        self._init_layers(cfg.rank)

    def forward(self, x: torch.tensor):
        return self.model(x)

    def _define_blocks(self, cfg: DictConfig):
        pattern = rf"{re.escape(cfg.block_fqn_prefix)}\.\d+$"

        self.blocks = []
        for fqn, module in self.named_modules():
            if re.search(pattern, fqn):
                self.blocks.append(module)
        return

    def _define_target_layers(self, cfg: DictConfig):
        self.target_layers = [
            (fqn, module)
            for fqn, module in self.model.named_modules()
            if any(suffix in fqn for suffix in cfg.layer_fqn_suffixes)
        ]
        return

    def _init_layers(self, rank):
        for fqn, layer in self.target_layers:
            weight_tensor = layer.weight.data
            U, V = svd_decompose(weight_tensor, rank)
            setattr(
                reduce(getattr, fqn.split(".")[:-1], self.model),
                fqn.split(".")[-1],
                LRLayer(
                    U, V, layer.in_features, layer.out_features, layer.bias
                ),
            )
        return

    def model_param_count(self):
        return sum(p.numel() for p in self.parameters())
