import re
import copy
import collections
from typing import List, Tuple, Optional
from functools import reduce
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured

from omegaconf import DictConfig

from sd2s.utils import svd_decompose, compute_sparse_r

from sparsimony.utils import share_parametrizations, get_mask
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier


class SMLayer(ABC, nn.Module):
    """Shared Low Rank Masked Layer"""

    def __init__(
        self,
        U: nn.Parameter,
        V: nn.Parameter,
        in_features: int,
        out_features: int,
        bias: nn.Parameter,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.U = U
        self.V = V
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

    @abstractmethod
    def forward(self, x: torch.tensor): ...

    def set_custom_forward(self):
        self.forward = self.compress_forward

    def set_two_four_forward(self):
        self.forward = self.forward_two_four


class SharedMaskedFC1(SMLayer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.tensor):
        return torch.nn.functional.linear(x, (self.U @ self.V).T, self.bias)

    def fast_forward(self, x: torch.tensor):
        return torch.nn.functional.linear(x @ self.U, self.V.T, self.bias)

    def compress_forward(self, x: torch.tensor):
        return torch.nn.functional.linear(x, (self.V @ self.U), self.bias)

    def forward_two_four(self, x: torch.tensor):
        # Original implementation
        # return torch.nn.functional.linear(x, (self.V @ self.U), self.bias)

        # batch size 1 only option that is marginally faster than below
        # return torch.nn.functional.linear(
        #     x.squeeze() @ self.U, self.V, self.bias
        # ).unsqueeze(dim=0)

        # only transpose V for below
        return torch.nn.functional.linear(
            x.view(-1, x.shape[-1]) @ self.U, self.V, self.bias
        ).view((x.shape[0], x.shape[1], -1))


class SharedMaskedFC2(SMLayer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.tensor):
        return torch.nn.functional.linear(x, self.U @ self.V, self.bias)

    def fast_forward(self, x: torch.tensor):
        return torch.nn.functional.linear(x @ self.V.T, self.U, self.bias)

    def compress_forward(self, x: torch.tensor):
        return torch.nn.functional.linear(x, (self.V @ self.U).T, self.bias)

    def forward_two_four(self, x: torch.tensor):
        # transpose u to use below
        # return torch.nn.functional.linear(x, (self.V @ self.U).T, self.bias)

        # No transpose for below
        # bs 1 only
        # return torch.nn.functional.linear(
        #     torch.mm(self.V, x.squeeze().T).T, self.U, self.bias
        # ).unsqueeze(dim=0)

        return torch.nn.functional.linear(
            torch.mm(self.V, x.view(-1, x.shape[-1]).T).T, self.U, self.bias
        ).view((x.shape[0]), x.shape[1], -1)


class SMModel(nn.Module):
    """Shared Low Rank Masked Model"""

    def __init__(self, cfg: DictConfig, model: nn.Module):
        super().__init__()

        self.model = copy.deepcopy(model)

        self.cfg = cfg
        self._define_blocks(cfg)
        self._define_target_layers(cfg)
        self._validate_block_cfg()
        self._get_normalizer_layers(cfg)

        self.v_count = 1
        self.base_count = 1

        for indices in self.cfg.block_group_indices.values():
            self._initialize_slrml_layers(indices)

    def forward(self, x: torch.tensor, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def _define_target_layers(self, cfg: DictConfig):
        self.target_layers = [
            fqn
            for fqn, _ in self.model.named_modules()
            if any(suffix in fqn for suffix in cfg.layer_fqn_suffixes)
        ]
        return

    def _define_blocks(self, cfg: DictConfig):
        pattern = rf"{re.escape(cfg.block_fqn_prefix)}\.\d+$"

        self.blocks = []
        for fqn, module in self.named_modules():
            if re.search(pattern, fqn):
                self.blocks.append(module)
        return

    def _get_normalizer_layers(self, cfg: DictConfig):
        """
        In case there are normalization between some blocks,
        assign them to a class attr.
        """
        normalizer_layers = collections.defaultdict(nn.Module)
        if cfg.normalization.in_place:
            normalizer_idx = 0
            for fqn, module in self.named_modules():
                if fqn.endswith(cfg.normalization.fqn):
                    key = cfg.normalization.apply_before_idx[normalizer_idx]
                    normalizer_layers[key] = module
                    normalizer_idx += 1

        self.normalization_layers = normalizer_layers
        return

    def _validate_block_cfg(self):
        indices_set = set()
        for indices in self.cfg.block_group_indices.values():
            indices_set.update(set(indices))
        num_mlp_layers = len(self.target_layers)

        if (
            len(indices_set) * 3 != num_mlp_layers
            and len(indices_set) * 2 != num_mlp_layers
        ):
            raise ValueError(
                "Bad block configuration. "
                "Check block groups and ranks in the config."
            )
        return

    def _initialize_slrml_layers(self, indices: List[int]):
        """
        Concatenate MLPs in the given block_indices
        and perform SVD decomposition.
        """
        # Expand block indices to FC layer indices
        num_ffs = len(self.cfg.layer_fqn_suffixes)
        if num_ffs == 2:
            expanded_indices = [2 * i for i in indices] + [
                2 * i + 1 for i in indices
            ]
        elif num_ffs == 3:
            expanded_indices = (
                [3 * i for i in indices]
                + [3 * i + 1 for i in indices]
                + [3 * i + 2 for i in indices]
            )
        else:
            raise ValueError("Only 2 or 3 FFS are supported.")
        expanded_indices.sort()

        layer_names = [self.target_layers[i] for i in expanded_indices]
        layers = [
            reduce(getattr, name.split("."), self.model) for name in layer_names
        ]

        group_dim = layers[0].in_features
        rank = compute_sparse_r(
            n_blocks=len(indices),
            d=group_dim,
            p_budget=self.cfg.p_budget,
            sparsity=self.cfg.sparsity,
            num_ffs=num_ffs,
            rounding=self.cfg.structured_sparsity,
        )

        U, V = self._decompose_blocks(layers, rank)
        U = U.contiguous()
        U = nn.Parameter(U, requires_grad=self.cfg.base_requires_grad)
        base_name = f"SharedParamU{self.base_count}"
        self.model.register_parameter(base_name, U)
        self.base_count += 1

        c_slice = 0
        for layer_name, layer in zip(layer_names, layers):
            # taking mlp1 for the basis for slice computations
            if c_slice == 0:
                out_features = layer.out_features
            slice_start = out_features * c_slice
            slice_end = out_features * (c_slice + 1)
            V_t = V[:, slice_start:slice_end]
            V_t_C = V_t.contiguous()
            v_param = nn.Parameter(V_t_C, requires_grad=True)

            v_name = f"SliceParamV{self.v_count}"
            self.model.register_parameter(v_name, v_param)
            self.v_count += 1
            c_slice += 1

            if layer.in_features < layer.out_features:
                new_layer = SharedMaskedFC1(
                    U=U,
                    V=v_param,
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=layer.bias,
                )
            else:
                new_layer = SharedMaskedFC2(
                    U=U,
                    V=v_param,
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=layer.bias,
                )
            setattr(
                reduce(getattr, layer_name.split(".")[:-1], self.model),
                layer_name.split(".")[-1],
                new_layer,
            )
        return

    def _decompose_blocks(
        self, layers: List[nn.Linear], rank: int
    ) -> Tuple[torch.tensor, torch.tensor]:

        weights = []
        for layer in layers:
            if layer.weight.shape[0] > layer.weight.shape[1]:
                weights.append(layer.weight.T)
                continue

            weights.append(layer.weight)

        concat = torch.concat(
            weights,
            dim=self.cfg.concat_axis,
        )
        U, V = svd_decompose(concat, rank)
        return U, V

    def mlp_param_count(self) -> int:
        count = 0
        for fqn, param in self.model.named_parameters():
            if "SliceParamV" in fqn:
                continue
            if "SharedParamU" in fqn:
                count += param.numel()
        return count

    def model_param_count(
        self, sparsifier: Optional[BaseSparsifier] = None
    ) -> int:
        count = 0
        if sparsifier is None:
            for fqn, param in self.model.named_parameters():
                count += param.numel()
            return count
        for config in sparsifier.groups:
            mask = get_mask(config["module"], config["tensor_name"])
            n_zeros = torch.sum(mask == 0).item()
            count += mask.numel() - n_zeros
        for fqn, param in self.model.named_parameters():
            if "SliceParamV" in fqn:
                continue
            count += param.numel()
        return count

    def sync_parametrizations(self, tensor_name: str = "U"):
        if self.cfg.sparsity is None or self.cfg.sparsity == 1:
            return

        for _, (indices, _) in self.cfg.block_groups_indices_ranks.items():
            primary_base_idx = indices[0]
            block = self.blocks[primary_base_idx]
            primary_mod = reduce(
                getattr, self.cfg.layer_fqn_suffixes[0].split("."), block
            )
            for replica_idx in indices:
                for fqn_suffix in self.cfg.layer_fqn_suffixe:
                    block = self.blocks[replica_idx]
                    replica_mod = reduce(getattr, fqn_suffix.split("."), block)

                    share_parametrizations(
                        primary_mod, replica_mod, tensor_name=tensor_name
                    )
        return

    @torch.no_grad()
    def compress_sparse_weights(self):
        for _, module in self.named_modules():
            if (
                type(module) is SharedMaskedFC1
                or type(module) is SharedMaskedFC2
            ):
                module.V = nn.Parameter(
                    (module.V.T.type(torch.float16).to_sparse_csr()),
                    requires_grad=False,
                )
                module.U = nn.Parameter(module.U.T, requires_grad=False)
                module.set_custom_forward()

    @torch.no_grad()
    def set_two_four(self):
        for _, module in self.named_modules():
            if (
                type(module) is SharedMaskedFC1
                or type(module) is SharedMaskedFC2
            ):
                if isinstance(module, SharedMaskedFC1):
                    two_four = (
                        module.V.view(module.V.shape).detach().T.contiguous()
                    )
                else:
                    two_four = (
                        module.V.view(module.V.shape).detach().contiguous()
                    )

                orig_shape = two_four.shape
                two_four = two_four.view(-1, 4)
                two_four[:, 2:] = 0
                two_four = two_four.view(orig_shape)
                two_four = to_sparse_semi_structured(two_four)
                module.V = nn.Parameter(
                    two_four,
                    requires_grad=True,
                )
                module.set_two_four_forward()

    def set_fast_forward(self):
        for _, module in self.named_modules():
            if (
                type(module) is SharedMaskedFC1
                or type(module) is SharedMaskedFC2
            ):
                module.forward = module.fast_forward
