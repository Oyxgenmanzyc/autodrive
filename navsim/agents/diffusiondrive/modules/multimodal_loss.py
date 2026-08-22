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
        self.energy_temporal_weight = 0.80
        self.energy_comfort_weight = 0.20
        self.temporal_rank_weight_max = 0.01
        self.temporal_rank_margin = 0.10
        self.temporal_rank_energy_gap = 0.20
        self.temporal_rank_min_epoch = 50.0
        self.temporal_rank_ramp_epochs = 30.0
        self.temporal_rank_use_comfort = True
        self.temporal_aux_weight_max = 0.005

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
        if use_comfort:
            energy = (
                gt_weight * self._standardize_topk(topk_gt)
                + temporal_weight * self._standardize_topk(topk_temporal)
                + comfort_weight * self._standardize_topk(topk_comfort)
            )
        else:
            energy = topk_temporal
        energy = energy.detach()

        good_pos = energy.argmin(dim=1, keepdim=True)
        bad_pos = energy.argmax(dim=1, keepdim=True)
        good_idx = torch.gather(topk_idx, 1, good_pos)
        bad_idx = torch.gather(topk_idx, 1, bad_pos)
        good_energy = torch.gather(energy, 1, good_pos).squeeze(1)
        bad_energy = torch.gather(energy, 1, bad_pos).squeeze(1)
        energy_gap = bad_energy - good_energy
        selected_temporal = torch.gather(temporal_cost.clamp(max=2.0), 1, cls_target.unsqueeze(1)).squeeze(1)
        aux_weight = ramp * float(temporal_context.get("temporal_aux_weight_max", self.temporal_aux_weight_max))
        aux_raw = selected_temporal.mean()
        aux_loss = aux_weight * aux_raw

        min_gap = float(temporal_context.get("temporal_rank_energy_gap", self.temporal_rank_energy_gap))
        active = (energy_gap > min_gap) & torch.isfinite(energy_gap)
        if not active.any():
            zero = poses_cls.sum() * 0.0
            total_loss = zero + aux_loss
            if not torch.isfinite(total_loss):
                return None
            return {
                "temporal_rank_loss": total_loss,
                "temporal_rank_raw": zero.detach(),
                "temporal_rank_weight": zero.detach(),
                "temporal_rank_ramp": ramp.detach(),
                "temporal_rank_active_ratio": torch.zeros((), device=device, dtype=dtype),
                "temporal_rank_energy_gap": energy_gap.mean().detach(),
                "temporal_rank_logit_margin": torch.zeros((), device=device, dtype=dtype),
                "temporal_aux_loss": aux_loss.detach(),
                "temporal_aux_raw": aux_raw.detach(),
                "temporal_aux_weight": aux_weight.detach(),
                "temporal_aux_selected_cost": selected_temporal.mean().detach(),
            }

        good_logit = torch.gather(poses_cls, 1, good_idx).squeeze(1)
        bad_logit = torch.gather(poses_cls, 1, bad_idx).squeeze(1)
        logit_margin = good_logit - bad_logit
        margin = float(temporal_context.get("temporal_rank_margin", self.temporal_rank_margin))
        rank_per_sample = F.softplus(margin - logit_margin)
        rank_raw = rank_per_sample[active].mean()
        rank_weight = ramp * float(temporal_context.get("temporal_rank_weight_max", self.temporal_rank_weight_max))
        rank_loss = rank_weight * rank_raw

        if not (torch.isfinite(rank_loss) and torch.isfinite(aux_loss)):
            return None

        return {
            "temporal_rank_loss": rank_loss + aux_loss,
            "temporal_rank_raw": rank_raw.detach(),
            "temporal_rank_weight": rank_weight.detach(),
            "temporal_rank_ramp": ramp.detach(),
            "temporal_rank_active_ratio": active.to(dtype=dtype).mean().detach(),
            "temporal_rank_energy_gap": energy_gap[active].mean().detach(),
            "temporal_rank_logit_margin": logit_margin[active].mean().detach(),
            "temporal_rank_good_cost": torch.gather(topk_temporal, 1, good_pos).squeeze(1)[active].mean().detach(),
            "temporal_rank_bad_cost": torch.gather(topk_temporal, 1, bad_pos).squeeze(1)[active].mean().detach(),
            "temporal_aux_loss": aux_loss.detach(),
            "temporal_aux_raw": aux_raw.detach(),
            "temporal_aux_weight": aux_weight.detach(),
            "temporal_aux_selected_cost": selected_temporal.mean().detach(),
        }

    def forward(self, poses_reg, poses_cls, targets, plan_anchor, temporal_context: Optional[Dict[str, Tensor]]=None):
        """
        pred_traj: (bs, K, 8, 3)
        pred_cls: (bs, K)
        plan_anchor: (bs, K, 8, 2)
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
        energy_info = self._energy_supervision(
            poses_reg,
            poses_cls,
            target_traj,
            dist,
            mode_idx_flat,
            temporal_context,
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
            original_loss = loss_cls + reg_loss
            connection_loss = energy_info["temporal_rank_loss"]
            ret_loss = original_loss + connection_loss
            logged_energy_info = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in energy_info.items()
            }
            # Keep these two terms differentiable so the training wrapper can
            # protect the original objective from conflicting temporal gradients.
            logged_energy_info["trajectory_original_loss"] = original_loss
            logged_energy_info["trajectory_connection_loss"] = connection_loss
            return ret_loss, logged_energy_info
        return ret_loss
