import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Callable, Optional
from torch import Tensor
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    torch_command_ids_from_one_hot,
    torch_trajectory_distance,
    torch_valid_command_mask,
)
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
# from mmdet.models.losses import FocalLoss

def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def command_mode_mask(
    driving_command: Tensor,
    anchor_command_ids: Tensor,
    *,
    allow_unknown_all_modes: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return semantic command ids and the eligible ``[B, K]`` mode mask."""
    valid_commands = torch_valid_command_mask(driving_command)
    if torch.any(~valid_commands) and not allow_unknown_all_modes:
        raise ValueError("unknown driving_command has no command-conditioned Anchor group")
    if allow_unknown_all_modes:
        command_ids = (driving_command[:, :3] != 0).to(dtype=torch.int64).argmax(dim=1)
    else:
        command_ids = torch_command_ids_from_one_hot(driving_command)
    anchor_command_ids = anchor_command_ids.to(
        device=driving_command.device, dtype=torch.int64
    )
    if anchor_command_ids.ndim != 1:
        raise ValueError("anchor_command_ids must have shape [K]")
    if torch.any((anchor_command_ids < 0) | (anchor_command_ids > 2)):
        raise ValueError("anchor_command_ids must contain only ids 0--2")
    mask = command_ids[:, None] == anchor_command_ids[None]
    if allow_unknown_all_modes:
        mask = mask | (~valid_commands)[:, None]
    if torch.any(valid_commands & ~mask.any(dim=1)):
        raise ValueError("every batch command must have at least one eligible mode")
    return command_ids, mask


def command_masked_argmax(
    poses_cls: Tensor,
    driving_command: Tensor,
    anchor_command_ids: Tensor,
) -> Tensor:
    """Apply command eligibility before the unchanged maximum-logit selector."""
    if poses_cls.ndim != 2:
        raise ValueError("poses_cls must have shape [B, K]")
    _, mask = command_mode_mask(
        driving_command,
        anchor_command_ids,
        allow_unknown_all_modes=True,
    )
    if mask.shape != poses_cls.shape:
        raise ValueError("command mask and poses_cls must have identical shape")
    return poses_cls.masked_fill(~mask, -torch.inf).argmax(dim=-1)


def k_invariant_focal_loss(
    poses_cls: Tensor,
    winner_indices: Tensor,
    command_mask: Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Return positive focal loss plus per-sample mean valid-negative focal loss."""
    if poses_cls.ndim != 2 or command_mask.shape != poses_cls.shape:
        raise ValueError("poses_cls and command_mask must have shape [B, K]")
    if winner_indices.shape != (poses_cls.shape[0],):
        raise ValueError("winner_indices must have shape [B]")
    if torch.any((winner_indices < 0) | (winner_indices >= poses_cls.shape[1])):
        raise ValueError("winner_indices contain an out-of-range mode")
    if torch.any(~command_mask.gather(1, winner_indices[:, None]).squeeze(1)):
        raise ValueError("winner_indices must select command-valid modes")

    targets = torch.zeros_like(poses_cls)
    targets.scatter_(1, winner_indices[:, None], 1)
    loss_element = py_sigmoid_focal_loss(
        poses_cls,
        targets,
        weight=None,
        gamma=gamma,
        alpha=alpha,
        reduction="none",
        avg_factor=None,
    )
    positive_loss = loss_element.gather(1, winner_indices[:, None]).squeeze(1)
    negative_mask = command_mask & ~targets.bool()
    negative_count = negative_mask.sum(dim=1)
    negative_loss = (loss_element * negative_mask).sum(dim=1)
    negative_mean = torch.where(
        negative_count > 0,
        negative_loss / negative_count.clamp_min(1),
        torch.zeros_like(negative_loss),
    )
    return positive_loss.mean() + negative_mean.mean()


def current_prediction_responsibility(
    poses_reg: Tensor,
    target_trajectory: Tensor,
    driving_command: Tensor,
    anchor_command_ids: Tensor,
    delta_scales: Tensor,
    fde_scales: Tensor,
) -> Tensor:
    """Choose the detached current prediction with minimum command-local ``D_traj``."""
    command_ids, mask = command_mode_mask(driving_command, anchor_command_ids)
    with torch.no_grad():
        distances = torch_trajectory_distance(
            poses_reg.detach()[..., :2],
            target_trajectory[..., :2],
            command_ids,
            delta_scales,
            fde_scales,
        )
        distances = distances.masked_fill(~mask, torch.inf)
        return distances.argmin(dim=-1)


class LossComputer(nn.Module):
    def __init__(
        self,
        config: TransfuserConfig,
        anchor_command_ids: Tensor,
        delta_scales: Tensor,
        fde_scales: Tensor,
    ):
        super(LossComputer, self).__init__()
        self._config = config
        self.register_buffer("anchor_command_ids", anchor_command_ids.to(dtype=torch.int64))
        self.register_buffer("delta_scales", delta_scales.to(dtype=torch.float32))
        self.register_buffer("fde_scales", fde_scales.to(dtype=torch.float32))
        self.register_buffer(
            "winner_counts",
            torch.zeros(len(anchor_command_ids), dtype=torch.int64),
            persistent=False,
        )
        anchor_ids_cpu = anchor_command_ids.detach().to(device="cpu", dtype=torch.int64)
        self._command_anchor_indices = tuple(
            tuple(torch.flatnonzero(anchor_ids_cpu == command_id).tolist())
            for command_id in range(len(TrajectoryCommand))
        )
        # self.focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False)
        self.cls_loss_weight = config.trajectory_cls_weight
        self.reg_loss_weight = config.trajectory_reg_weight
    @torch.no_grad()
    def update_training_utilization(self, winner_indices: Tensor) -> None:
        """Accumulate command-local winner counts for training utilization logs."""
        self.winner_counts.add_(
            torch.bincount(winner_indices, minlength=len(self.anchor_command_ids)).to(
                device=self.winner_counts.device
            )
        )

    def training_utilization_metrics(
        self, winner_counts: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """Compute command-local utilization metrics from one global ``[K]`` histogram."""
        metrics: dict[str, Tensor] = {}
        if winner_counts is None:
            winner_counts = self.winner_counts
        if winner_counts.shape != self.winner_counts.shape:
            raise ValueError("winner_counts must have shape [K]")
        counts = winner_counts.to(device=self.winner_counts.device, dtype=torch.float32)
        for command_id, command in enumerate(TrajectoryCommand):
            anchor_indices = self._command_anchor_indices[command_id]
            if not anchor_indices:
                continue
            index_tensor = torch.tensor(
                anchor_indices, dtype=torch.int64, device=counts.device
            )
            command_counts = counts[index_tensor]
            frequencies = command_counts / command_counts.sum().clamp_min(1.0)
            active_mode_rate = (frequencies > 0.001).to(dtype=torch.float32).mean()
            positive_frequencies = frequencies[frequencies > 0]
            winner_entropy = -(
                positive_frequencies * positive_frequencies.log()
            ).sum()
            prefix = f"trajectory_utilization/{command.value}"
            metrics[f"{prefix}/active_mode_rate"] = active_mode_rate
            metrics[f"{prefix}/winner_entropy"] = winner_entropy
            for local_index, anchor_index in enumerate(anchor_indices):
                metrics[f"{prefix}/winner_mode_{anchor_index}_frequency"] = frequencies[
                    local_index
                ]
        return metrics

    @torch.no_grad()
    def synchronized_training_utilization_metrics(self) -> dict[str, Tensor]:
        """All-reduce the complete ``[K]`` histogram once, then compute global metrics."""
        global_counts = self.winner_counts.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_counts, op=torch.distributed.ReduceOp.SUM)
        return self.training_utilization_metrics(global_counts)

    @torch.no_grad()
    def reset_training_utilization(self) -> None:
        """Clear the local histogram after the epoch-level metrics are emitted."""
        self.winner_counts.zero_()

    def forward(
        self,
        poses_reg,
        poses_cls,
        targets,
        driving_command,
        *,
        track_utilization: bool = False,
    ):
        """
        pred_traj: (bs, K, 8, 3)
        pred_cls: (bs, K)
        driving_command: (bs, 4)
        targets['trajectory']: (bs, 8, 3)
        """
        valid_samples = torch_valid_command_mask(driving_command)
        if not torch.any(valid_samples):
            return (poses_reg.sum() + poses_cls.sum()) * 0.0

        poses_reg = poses_reg[valid_samples]
        poses_cls = poses_cls[valid_samples]
        target_traj = targets["trajectory"][valid_samples]
        driving_command = driving_command[valid_samples]
        _, _, ts, d = poses_reg.shape
        _, valid_mode_mask = command_mode_mask(
            driving_command, self.anchor_command_ids
        )
        mode_idx = current_prediction_responsibility(
            poses_reg,
            target_traj,
            driving_command,
            self.anchor_command_ids,
            self.delta_scales,
            self.fde_scales,
        )
        cls_target = mode_idx
        if track_utilization:
            self.update_training_utilization(cls_target)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        loss_cls = self.cls_loss_weight * k_invariant_focal_loss(
            poses_cls,
            cls_target,
            valid_mode_mask,
            gamma=2.0,
            alpha=0.25,
        )

        # Calculate regression loss
        reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)
        # import ipdb; ipdb.set_trace()
        # Combine classification and regression losses
        ret_loss = loss_cls + reg_loss
        return ret_loss
