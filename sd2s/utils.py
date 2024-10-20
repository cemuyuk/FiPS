import os
from typing import Any, Dict, Optional
from unittest.result import failfast

import torch
from torch.nn.parallel import DistributedDataParallel


def svd_decompose(
    weight_tensor: torch.tensor, rank: int, tau: Optional[int] = 32
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Apply truncated SVD decomposition to the given weight tensor. Singular values are automatically multiplied in V.
    If the rank is higher than the rank of the weight tensor the function will pad the factor U with zeros.
    The factor V will be padded with TopK row vectors divided by tau.

    Args:
        weight_tensor (`torch.tensor`)
            The weight tensor to decompose.
        rank (`int`)
            The rank inut for truncated SVD.
        eta (`int`)
            The alpha value for dividing the TopK feature vectors of V.
    Returns:
        U (`torch.tensor`), V (`torch.tensor`)
            The decomposed matrices where V is multiplied with the singular values S.
    """
    U, S, V = torch.linalg.svd(weight_tensor, full_matrices=False)
    diag_S = torch.diag(S)
    V = diag_S @ V

    d = min(U.shape[0], U.shape[1])

    if rank > d:
        k = rank - d

        device = weight_tensor.device

        new_U_neurons = torch.zeros(U.shape[0], k, device=device)
        new_V_neurons = V[:k, :] / tau

        U = torch.cat([U, new_U_neurons], dim=1)
        V = torch.cat([V, new_V_neurons], dim=0)

        while V.shape[0] < rank:
            V = torch.cat([V, new_V_neurons], dim=0)[:rank, :]

        return U, V

    U = U[:, :rank]
    V = V[:rank, :]

    return U, V


def compute_sparse_r(n_blocks: int, d: int, p_budget: float, sparsity: float):
    """Returns rank of the model when initialized with n_blocks to be concatenated,
    model dimension d, parameter budget of FCs and sparsity.
    Args:
        n_blocks (int): Number of blocks to be concatenated.
        d (int): Model dimension.
        p_budget (float): Parameter budget of FCs.
        sparsity (float): Sparsity of the model.
    Returns:
        int: Rank of the compressed and sparse model.
    """
    n_params = 8 * n_blocks * (d**2) * p_budget
    f_coeff = 8 * d * n_blocks * (1 - sparsity) + d

    return int(n_params / f_coeff)


class DDPWrappedGetAttr(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_single_process_model_state_from_distributed_state(
    model_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Returns model state from a distributed training run in a format
    suitable for a single process model.

    In distributed training, `module.<param_name>` is appended to every
    parameter. If we wish to test/train this model further in a single
    process, we simply strip the `module` prefix to match keys expected in
    the model.

    Returns:
        Dict[str, torch.Tensor]: Model state ordered dict from distributed
            trained model for loading in single process.
    """
    return {".".join(k.split(".")[1:]): v for k, v in model_state.items()}


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))
