import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from typing import Callable, Dict, Optional
from torch import Tensor
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
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


class LossComputer(nn.Module):
    def __init__(self,config: TransfuserConfig):
        self._config = config
        super(LossComputer, self).__init__()
        # self.focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False)
        self.cls_loss_weight = config.trajectory_cls_weight
        self.reg_loss_weight = config.trajectory_reg_weight
        self.energy_topk = 3
        self.energy_temperature = 0.5
        self.energy_start_epoch = 70
        self.energy_full_epoch = 85
        self.energy_gt_weight = 0.60
        self.energy_temporal_weight = 0.85
        self.energy_comfort_weight = 0.05
        self.energy_progress_weight = 0.10
        self.progress_path_len_threshold = 5.0
        self.progress_longitudinal_consistency = 0.70
        self.progress_lateral_ratio = 0.50
        self.progress_max_heading_change = 0.45
        self.progress_hard_negative_threshold = 0.05
        self.progress_hard_negative_cls_topk = 5
        self.progress_hard_negative_logit_weight = 0.10
        self.front_ttc_threshold = 2.0
        self.temporal_rank_weight_max = 0.30
        self.temporal_rank_margin = 0.10
        self.temporal_rank_energy_gap = 0.20
        self.temporal_rank_min_epoch = 50.0
        self.temporal_rank_ramp_epochs = 30.0
        self.temporal_rank_use_comfort = True
        self.temporal_aux_weight_max = 0.0

    @staticmethod
    def _as_schedule_tensor(value, device, dtype):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(float(value), device=device, dtype=dtype)

    @staticmethod
    def _standardize_topk(values: Tensor) -> Tensor:
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, unbiased=False, keepdim=True)
        return (values - mean) / (std + 1e-6)

    @staticmethod
    def _gather_topk(values: Tensor, topk_idx: Tensor) -> Tensor:
        return torch.gather(values, 1, topk_idx)

    @staticmethod
    def _batch_vector(value, device, dtype, batch_size):
        value = value.to(device=device)
        if value.dtype == torch.bool:
            value = value.bool()
        else:
            value = value.to(dtype=dtype)
        value = value.reshape(-1)
        if value.numel() == 1 and batch_size > 1:
            value = value.expand(batch_size)
        return value

    def _front_ttc_safe_gate(self, temporal_context: Dict[str, Tensor], device, dtype, batch_size) -> Tensor:
        if "front_ttc" not in temporal_context or "front_ttc_valid" not in temporal_context:
            return torch.ones(batch_size, device=device, dtype=torch.bool)

        front_ttc = self._batch_vector(temporal_context["front_ttc"], device, dtype, batch_size)
        front_ttc_valid = self._batch_vector(
            temporal_context["front_ttc_valid"],
            device,
            dtype,
            batch_size,
        ).bool()
        ttc_threshold = float(temporal_context.get("front_ttc_threshold", self.front_ttc_threshold))
        return (~front_ttc_valid) | (front_ttc > ttc_threshold)

    def _pdm_progress_cost(
        self,
        poses_reg: Tensor,
        target_traj: Tensor,
        temporal_context: Dict[str, Tensor],
    ):
        device = poses_reg.device
        dtype = poses_reg.dtype
        batch_size = poses_reg.shape[0]
        num_modes = poses_reg.shape[1]

        if poses_reg.shape[-2] < 2 or target_traj.shape[-2] < 2:
            zeros = torch.zeros(batch_size, num_modes, device=device, dtype=dtype)
            inactive = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return zeros, inactive

        target_xy = target_traj[..., :2]
        candidate_xy = poses_reg[..., :2]
        target_delta = target_xy[:, 1:] - target_xy[:, :-1]
        candidate_delta = candidate_xy[:, :, 1:] - candidate_xy[:, :, :-1]

        target_step = torch.linalg.norm(target_delta, dim=-1)
        target_path_len = target_step.sum(dim=-1)
        target_dir = target_delta / (target_step.unsqueeze(-1) + 1e-6)
        candidate_progress = (candidate_delta * target_dir.unsqueeze(1)).sum(dim=-1).clamp(min=0.0).sum(dim=-1)
        progress_ratio = candidate_progress / (target_path_len.unsqueeze(1) + 1e-6)
        progress_cost = torch.relu(1.0 - progress_ratio).clamp(max=1.0)

        path_len_threshold = float(
            temporal_context.get("progress_path_len_threshold", self.progress_path_len_threshold)
        )
        longitudinal_threshold = float(
            temporal_context.get("progress_longitudinal_consistency", self.progress_longitudinal_consistency)
        )
        lateral_threshold = float(temporal_context.get("progress_lateral_ratio", self.progress_lateral_ratio))
        heading_threshold = float(
            temporal_context.get("progress_max_heading_change", self.progress_max_heading_change)
        )

        target_displacement = target_xy[:, -1] - target_xy[:, 0]
        longitudinal_consistency = target_displacement[:, 0] / (target_path_len + 1e-6)
        lateral_ratio = target_displacement[:, 1].abs() / (target_path_len + 1e-6)
        if target_delta.shape[1] >= 2:
            heading = torch.atan2(target_delta[..., 1], target_delta[..., 0])
            heading_change = torch.atan2(
                torch.sin(heading[:, 1:] - heading[:, :-1]),
                torch.cos(heading[:, 1:] - heading[:, :-1]),
            ).abs()
            max_heading_change = heading_change.max(dim=-1).values
        else:
            max_heading_change = torch.zeros(batch_size, device=device, dtype=dtype)

        progress_active = (
            (target_path_len > path_len_threshold)
            & (longitudinal_consistency > longitudinal_threshold)
            & (lateral_ratio < lateral_threshold)
            & (max_heading_change < heading_threshold)
            & self._front_ttc_safe_gate(temporal_context, device, dtype, batch_size)
        )
        progress_cost = torch.where(progress_active.unsqueeze(1), progress_cost, torch.zeros_like(progress_cost))
        return progress_cost, progress_active

    def _energy_ramp(self, temporal_context: Dict[str, Tensor], device, dtype) -> Tensor:
        epoch = self._as_schedule_tensor(temporal_context.get("energy_training_epoch"), device, dtype)
        if epoch is None:
            return torch.zeros((), device=device, dtype=dtype)
        start_epoch = float(temporal_context.get("temporal_rank_min_epoch", self.temporal_rank_min_epoch))
        ramp_epochs = max(float(temporal_context.get("temporal_rank_ramp_epochs", self.temporal_rank_ramp_epochs)), 1.0)
        progress = ((epoch - start_epoch) / ramp_epochs).clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(progress * math.pi)

    def _energy_supervision(
        self,
        poses_reg: Tensor,
        poses_cls: Tensor,
        target_traj: Tensor,
        dist: Tensor,
        cls_target: Tensor,
        temporal_context: Optional[Dict[str, Tensor]],
    ) -> Optional[Dict[str, Tensor]]:
        if temporal_context is None:
            return None

        required_keys = ("energy_temporal_cost",)
        if any(key not in temporal_context for key in required_keys):
            return None

        device = poses_reg.device
        dtype = poses_reg.dtype
        ramp = self._energy_ramp(temporal_context, device, dtype)
        if ramp.item() <= 0.0:
            return None

        temporal_cost = temporal_context["energy_temporal_cost"].to(device=device, dtype=dtype)
        if temporal_cost.shape != dist.shape:
            return None
        if not torch.isfinite(temporal_cost).all():
            return None

        if dist.shape[1] < 2:
            return None
        topk = int(temporal_context.get("energy_topk", self.energy_topk))
        topk = min(max(2, topk), dist.shape[1])
        _, topk_idx = torch.topk(dist, k=topk, dim=-1, largest=False)

        gt_distance = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - poses_reg[..., :2], dim=-1).mean(dim=-1)
        topk_gt = self._gather_topk(gt_distance, topk_idx)
        topk_temporal = self._gather_topk(temporal_cost, topk_idx)
        use_comfort = bool(temporal_context.get("temporal_rank_use_comfort", self.temporal_rank_use_comfort))
        if use_comfort and "energy_comfort_cost" in temporal_context:
            comfort_cost = temporal_context["energy_comfort_cost"].to(device=device, dtype=dtype)
            if comfort_cost.shape != dist.shape or not torch.isfinite(comfort_cost).all():
                return None
            topk_comfort = self._gather_topk(comfort_cost, topk_idx)
        else:
            topk_comfort = torch.zeros_like(topk_temporal)

        gt_weight = float(temporal_context.get("energy_gt_weight", self.energy_gt_weight))
        temporal_weight = float(temporal_context.get("energy_temporal_weight", self.energy_temporal_weight))
        comfort_weight = float(temporal_context.get("energy_comfort_weight", self.energy_comfort_weight))
        progress_weight = float(temporal_context.get("energy_progress_weight", self.energy_progress_weight))

        progress_cost, progress_active = self._pdm_progress_cost(poses_reg, target_traj, temporal_context)
        topk_progress = self._gather_topk(progress_cost, topk_idx)

        if use_comfort:
            energy = (
                gt_weight * self._standardize_topk(topk_gt)
                + temporal_weight * self._standardize_topk(topk_temporal)
                + comfort_weight * self._standardize_topk(topk_comfort)
                + progress_weight * self._standardize_topk(topk_progress)
            )
        else:
            energy = (
                temporal_weight * self._standardize_topk(topk_temporal)
                + progress_weight * self._standardize_topk(topk_progress)
            )
        energy = energy.detach()

        good_pos = energy.argmin(dim=1, keepdim=True)
        fallback_bad_pos = energy.argmax(dim=1, keepdim=True)
        good_idx = torch.gather(topk_idx, 1, good_pos)
        fallback_bad_idx = torch.gather(topk_idx, 1, fallback_bad_pos)
        good_energy = torch.gather(energy, 1, good_pos).squeeze(1)
        fallback_bad_energy = torch.gather(energy, 1, fallback_bad_pos).squeeze(1)

        hard_threshold = float(
            temporal_context.get("progress_hard_negative_threshold", self.progress_hard_negative_threshold)
        )
        hard_cls_topk = min(
            max(1, int(temporal_context.get("progress_hard_negative_cls_topk", self.progress_hard_negative_cls_topk))),
            poses_cls.shape[1],
        )
        _, cls_topk_idx = torch.topk(poses_cls.detach(), k=hard_cls_topk, dim=1, largest=True)
        cls_topk_mask = torch.zeros_like(progress_cost, dtype=torch.bool)
        cls_topk_mask.scatter_(1, cls_topk_idx, True)
        mode_idx = torch.arange(progress_cost.shape[1], device=device).unsqueeze(0)
        hard_mask = (
            (progress_cost > hard_threshold)
            & cls_topk_mask
            & (mode_idx != good_idx)
            & torch.isfinite(progress_cost)
        )
        hard_negative_available = hard_mask.any(dim=1)

        logit_weight = float(
            temporal_context.get("progress_hard_negative_logit_weight", self.progress_hard_negative_logit_weight)
        )
        hard_score = progress_cost.detach() + logit_weight * self._standardize_topk(poses_cls.detach())
        hard_score = hard_score.masked_fill(~hard_mask, torch.finfo(dtype).min)
        hard_bad_idx = hard_score.argmax(dim=1, keepdim=True)

        good_progress = torch.gather(progress_cost, 1, good_idx).squeeze(1)
        hard_progress = torch.gather(progress_cost, 1, hard_bad_idx).squeeze(1)
        hard_progress_gap = hard_progress - good_progress
        use_hard_negative = hard_negative_available & (hard_progress_gap > hard_threshold)

        bad_idx = torch.where(use_hard_negative.unsqueeze(1), hard_bad_idx, fallback_bad_idx)
        fallback_energy_gap = fallback_bad_energy - good_energy
        energy_gap = torch.where(use_hard_negative, hard_progress_gap, fallback_energy_gap)

        min_gap = float(temporal_context.get("temporal_rank_energy_gap", self.temporal_rank_energy_gap))
        active = (energy_gap > min_gap) & torch.isfinite(energy_gap)
        if not active.any():
            zero = poses_cls.sum() * 0.0
            return {
                "temporal_rank_loss": zero,
                "temporal_rank_raw": zero.detach(),
                "temporal_rank_weight": zero.detach(),
                "temporal_rank_ramp": ramp.detach(),
                "temporal_rank_active_ratio": torch.zeros((), device=device, dtype=dtype),
                "temporal_rank_energy_gap": energy_gap.mean().detach(),
                "temporal_rank_logit_margin": torch.zeros((), device=device, dtype=dtype),
                "temporal_rank_good_progress_cost": zero.detach(),
                "temporal_rank_bad_progress_cost": zero.detach(),
                "temporal_rank_progress_active_ratio": progress_active.to(dtype=dtype).mean().detach(),
                "temporal_rank_hard_negative_ratio": torch.zeros((), device=device, dtype=dtype),
            }

        good_logit = torch.gather(poses_cls, 1, good_idx).squeeze(1)
        bad_logit = torch.gather(poses_cls, 1, bad_idx).squeeze(1)
        logit_margin = good_logit - bad_logit
        margin = float(temporal_context.get("temporal_rank_margin", self.temporal_rank_margin))
        rank_per_sample = F.softplus(margin - logit_margin)
        rank_raw = rank_per_sample[active].mean()
        rank_weight = ramp * float(temporal_context.get("temporal_rank_weight_max", self.temporal_rank_weight_max))
        rank_loss = rank_weight * rank_raw

        if not torch.isfinite(rank_loss):
            return None

        return {
            "temporal_rank_loss": rank_loss,
            "temporal_rank_raw": rank_raw.detach(),
            "temporal_rank_weight": rank_weight.detach(),
            "temporal_rank_ramp": ramp.detach(),
            "temporal_rank_active_ratio": active.to(dtype=dtype).mean().detach(),
            "temporal_rank_energy_gap": energy_gap[active].mean().detach(),
            "temporal_rank_logit_margin": logit_margin[active].mean().detach(),
            "temporal_rank_good_cost": torch.gather(temporal_cost, 1, good_idx).squeeze(1)[active].mean().detach(),
            "temporal_rank_bad_cost": torch.gather(temporal_cost, 1, bad_idx).squeeze(1)[active].mean().detach(),
            "temporal_rank_good_progress_cost": torch.gather(progress_cost, 1, good_idx).squeeze(1)[active].mean().detach(),
            "temporal_rank_bad_progress_cost": torch.gather(progress_cost, 1, bad_idx).squeeze(1)[active].mean().detach(),
            "temporal_rank_progress_active_ratio": progress_active.to(dtype=dtype).mean().detach(),
            "temporal_rank_hard_negative_ratio": use_hard_negative[active].to(dtype=dtype).mean().detach(),
        }

    def forward(self, poses_reg, poses_cls, targets, plan_anchor, temporal_context: Optional[Dict[str, Tensor]]=None):
        """
        pred_traj: (bs, 20, 8, 3)
        pred_cls: (bs, 20)
        plan_anchor: (bs,20, 8, 2)
        targets['trajectory']: (bs, 8, 3)
        """
        bs, num_mode, ts, d = poses_reg.shape
        target_traj = targets["trajectory"]
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        cls_target = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        mode_idx_flat = cls_target
        # import ipdb; ipdb.set_trace()
        # Calculate cls loss using focal loss
        target_classes_onehot = torch.zeros([bs, num_mode],
                                            dtype=poses_cls.dtype,
                                            layout=poses_cls.layout,
                                            device=poses_cls.device)
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)
        energy_context = temporal_context
        if temporal_context is not None:
            energy_context = dict(temporal_context)
            for key in ("front_ttc", "front_ttc_valid", "front_distance", "front_relative_speed"):
                if key in targets:
                    energy_context[key] = targets[key]

        energy_info = self._energy_supervision(
            poses_reg,
            poses_cls,
            target_traj,
            dist,
            mode_idx_flat,
            energy_context,
        )
        # Use py_sigmoid_focal_loss function for focal loss calculation
        loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls,
            target_classes_onehot,
            weight=None,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            avg_factor=None
        )

        # Calculate regression loss
        reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)
        # import ipdb; ipdb.set_trace()
        # Combine classification and regression losses
        ret_loss = loss_cls + reg_loss
        if energy_info is not None:
            ret_loss = ret_loss + energy_info["temporal_rank_loss"]
            logged_energy_info = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in energy_info.items()
            }
            logged_energy_info["trajectory_original_loss"] = (loss_cls + reg_loss).detach()
            return ret_loss, logged_energy_info
        return ret_loss
