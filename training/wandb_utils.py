# TODO - attribution to DOJO

import os
import logging
from datetime import datetime
from typing import Callable, Any, Optional, Dict

import torch
import omegaconf

import wandb
import wandb.sdk
from wandb.sdk.wandb_run import Run


class WandbRunNameException(Exception):
    def __init__(self, message, name) -> None:
        super().__init__(f"Wandb run name of {name} is invalid! " f"{message}")


class WandbRunName:
    def __init__(self, name: str):
        self.name = name
        self._verify_name()

    def _verify_name(self):
        if " " in self.name:
            raise WandbRunNameException(
                message="No spaces allowed in name", name=self.name
            )
        if len(self.name) > 128:
            raise WandbRunNameException(
                message="Name must be <= 128 chars", name=self.name
            )


def _wandb_log_check(fn: Callable, log_to_wandb: bool = True) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        if log_to_wandb:
            return fn(*args, **kwargs)
        else:
            return None

    return wrapper


def _disable_wandb() -> None:
    wandb.log = _wandb_log_check(wandb.log, False)
    wandb.log_artifact = _wandb_log_check(wandb.log_artifact, False)
    wandb.watch = _wandb_log_check(wandb.watch, False)
    wandb.init = _wandb_log_check(wandb.init, False)
    wandb.Settings = _wandb_log_check(wandb.Settings, False)


# TODO - go over the method below
def init_wandb(
    cfg: omegaconf.DictConfig, global_rank: int
) -> wandb.sdk.wandb_run.Run | None:
    _logger = logging.getLogger(__name__)
    # We override logging functions now to avoid any calls
    if global_rank != 0:
        _disable_wandb()
        return None
    if not cfg.wandb.ENABLE:
        _disable_wandb()
        _logger.warning("No logging to WANDB! See cfg.wandb.ENABLE")
        return None

    now = datetime.now()
    formatted_time = now.strftime("%m/%d-%H/%M/%S")
    run_name = f"{cfg.wandb.name}-{formatted_time}"
    _ = WandbRunName(name=run_name)  # Verify name is OK
    resume = False
    id = None

    run = wandb.init(
        id=id,
        name=run_name,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg=cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method=cfg.wandb.start_method),
        dir=cfg.paths.logs,
        resume=resume,
    )
    return run


def parse_wandb_run_id(run: Optional[Run]) -> str:
    if run is None:
        return datetime.now().strftime("%h-%m-%d-%H-%M")
    else:
        return run.id


def wandb_finetune_log(
    step: int,
    loss: float,
    accuracy: float,
    logits: torch.Tensor,
    log_images: bool,
    inputs: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    pred: Optional[torch.Tensor],
):
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        return
    log_data = {
        "step": step,
        "train_loss": loss,
        "train_accuracy": accuracy,
        "train_logits": wandb.Histogram(logits.detach().cpu()),
    }
    if log_images:
        log_data.update(
            {
                "train_inputs": wandb.Image(inputs),
                "train_captions": wandb.Html(targets.cpu().numpy().__str__()),
                "train_predictions": wandb.Html(pred.cpu().numpy().__str__()),
            }
        )
    wandb.log(log_data)


def wand_finetune_val_log(
    epoch: int,
    loss: float,
    accuracy: float,
    logits: torch.Tensor,
    log_images: bool,
    top_5_accuracy: torch.Tensor,
    inputs: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    pred: Optional[torch.Tensor],
):
    # Check if this proc. is controller.
    # If not set, we are in single proc and should log
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        return
    log_data = {
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "logits": wandb.Histogram(logits.cpu()),
    }
    if top_5_accuracy is not None:
        log_data.update({"top_5_accuracy": top_5_accuracy})
    if log_images:
        log_data.update(
            {
                "val_inputs": wandb.Image(inputs),
                "val_captions": wandb.Html(targets.cpu().numpy().__str__()),
                "val_predictions": wandb.Html(pred.cpu().numpy().__str__()),
            }
        )
    wandb.log(log_data)


def wandb_training_log(
    step: int,
    epoch: int,
    avg_sec_per_step: float,
    sec_per_epoch: float,
    overall_block_avg_error: float,
    per_block_avg_error: Dict[str, float],
    lr: float,
    sparsity: float,
    parameter_count: Optional[int],
    per_layer_sparsity: Dict[str, float] = None,
):
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        return
    log_data = {
        "epoch": epoch,
        "avg_sec_per_step": avg_sec_per_step,
        "sec_per_epoch": sec_per_epoch,
        "avg_block_error": overall_block_avg_error,
        "lr": lr,
        "sparsity": sparsity,
        "parameter_count": parameter_count,
    }
    if per_layer_sparsity is not None:
        log_data.update({"per_layer_sparsity": per_layer_sparsity})
    log_data.update(per_block_avg_error)
    wandb.log(log_data, step=step)


def wandb_validation_log(
    step: int,
    loss: float,
    accuracy: float,
    logits: torch.Tensor,
    log_images: bool,
    top_5_accuracy: torch.Tensor,
    inputs: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    pred: Optional[torch.Tensor],
):
    # Check if this proc. is controller.
    # If not set, we are in single proc and should log
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        return
    log_data = {
        "loss": loss,
        "accuracy": accuracy,
        "val_logits": wandb.Histogram(logits.cpu()),
    }
    if top_5_accuracy is not None:
        log_data.update({"top_5_accuracy": top_5_accuracy})
    if log_images:
        log_data.update(
            {
                "val_inputs": wandb.Image(inputs),
                "val_captions": wandb.Html(targets.cpu().numpy().__str__()),
                "val_predictions": wandb.Html(pred.cpu().numpy().__str__()),
            }
        )
    wandb.log(log_data, step=step)
