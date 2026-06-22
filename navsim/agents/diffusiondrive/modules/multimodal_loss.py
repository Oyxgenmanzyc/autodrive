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
        self.energy_target_gamma_max = 0.35
        self.energy_gt_weight = 0.60
        self.energy_temporal_weight = 0.30
        self.energy_comfort_weight = 0.10

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
    def _smooth_ramp(ramp: Tensor) -> Tensor:
        ramp = ramp.clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(ramp * math.pi)

    def _energy_ramp(self, temporal_context: Dict[str, Tensor], device, dtype) -> Tensor:
        ramp_override = temporal_context.get("energy_ramp_override")
        if ramp_override is not None:
            return self._smooth_ramp(self._as_schedule_tensor(ramp_override, device, dtype))

        epoch = self._as_schedule_tensor(temporal_context.get("energy_training_epoch"), device, dtype)
        if epoch is None:
            return torch.zeros((), device=device, dtype=dtype)
        start_epoch = float(temporal_context.get("energy_start_epoch", self.energy_start_epoch))
        full_epoch = float(temporal_context.get("energy_full_epoch", self.energy_full_epoch))
        linear_ramp = (epoch - start_epoch) / (full_epoch - start_epoch + 1e-6)
        return self._smooth_ramp(linear_ramp)

    def _energy_supervision(
        self,
        poses_reg: Tensor,
        target_traj: Tensor,
        dist: Tensor,
        cls_target: Tensor,
        temporal_context: Optional[Dict[str, Tensor]],
    ) -> Optional[Dict[str, Tensor]]:
        if temporal_context is None:
            return None

        required_keys = ("energy_temporal_cost", "energy_comfort_cost")
        if any(key not in temporal_context for key in required_keys):
            return None

        device = poses_reg.device
        dtype = poses_reg.dtype
        ramp = self._energy_ramp(temporal_context, device, dtype)
        if ramp.item() <= 0.0:
            return None

        temporal_cost = temporal_context["energy_temporal_cost"].to(device=device, dtype=dtype)
        comfort_cost = temporal_context["energy_comfort_cost"].to(device=device, dtype=dtype)
        if temporal_cost.shape != dist.shape or comfort_cost.shape != dist.shape:
            return None
        if not (torch.isfinite(temporal_cost).all() and torch.isfinite(comfort_cost).all()):
            return None

        topk = int(temporal_context.get("energy_topk", self.energy_topk))
        topk = max(1, min(topk, dist.shape[1]))
        _, topk_idx = torch.topk(dist, k=topk, dim=-1, largest=False)

        gt_distance = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - poses_reg[..., :2], dim=-1).mean(dim=-1)
        topk_gt = self._gather_topk(gt_distance, topk_idx)
        topk_temporal = self._gather_topk(temporal_cost, topk_idx)
        topk_comfort = self._gather_topk(comfort_cost, topk_idx)

        gt_weight = float(temporal_context.get("energy_gt_weight", self.energy_gt_weight))
        temporal_weight = float(temporal_context.get("energy_temporal_weight", self.energy_temporal_weight))
        comfort_weight = float(temporal_context.get("energy_comfort_weight", self.energy_comfort_weight))
        energy = (
            gt_weight * self._standardize_topk(topk_gt)
            + temporal_weight * self._standardize_topk(topk_temporal)
            + comfort_weight * self._standardize_topk(topk_comfort)
        )
        temperature = max(float(temporal_context.get("energy_temperature", self.energy_temperature)), 1e-3)
        topk_prob = torch.softmax(-energy / temperature, dim=1).detach()

        bs, num_mode = dist.shape
        energy_target = torch.zeros([bs, num_mode], dtype=dtype, device=device)
        energy_target.scatter_(1, topk_idx, topk_prob)
        one_hot_target = torch.zeros([bs, num_mode], dtype=dtype, device=device)
        one_hot_target.scatter_(1, cls_target.unsqueeze(1), 1)
        target_gamma = ramp * float(temporal_context.get("energy_target_gamma_max", self.energy_target_gamma_max))
        cls_soft_target = (1.0 - target_gamma) * one_hot_target + target_gamma * energy_target

        entropy = -(topk_prob * torch.log(topk_prob + 1e-6)).sum(dim=1).mean()
        return {
            "cls_soft_target": cls_soft_target,
            "energy_supervision_weight": target_gamma.detach(),
            "energy_target_gamma": target_gamma.detach(),
            "energy_soft_entropy": entropy.detach(),
            "energy_selected_cost": (topk_prob * energy).sum(dim=1).mean().detach(),
            "energy_weighted_temporal_cost": (topk_prob * topk_temporal).sum(dim=1).mean().detach(),
            "energy_weighted_comfort_cost": (topk_prob * topk_comfort).sum(dim=1).mean().detach(),
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
        energy_info = self._energy_supervision(
            poses_reg,
            target_traj,
            dist,
            mode_idx_flat,
            temporal_context,
        )
        cls_target_tensor = target_classes_onehot if energy_info is None else energy_info["cls_soft_target"]

        # Use py_sigmoid_focal_loss function for focal loss calculation
        loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls,
            cls_target_tensor,
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
            logged_energy_info = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in energy_info.items()
                if key != "cls_soft_target"
            }
            logged_energy_info["trajectory_original_loss"] = (loss_cls + reg_loss).detach()
            return ret_loss, logged_energy_info
        return ret_loss
