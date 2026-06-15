import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
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
        self.temporal_match_topk = 3
        self.temporal_match_alpha = 0.05
        self.temporal_aux_loss_weight = 0.02
        self.temporal_aux_start_weight = 0.45
        self.temporal_aux_path_weight = 0.30
        self.temporal_aux_velocity_weight = 0.25

    @staticmethod
    def _gather_mode(values: Tensor, mode_idx: Tensor) -> Tensor:
        return torch.gather(values, 1, mode_idx[:, None]).squeeze(1)

    def _select_temporal_aware_mode(self, dist: Tensor, temporal_context: Optional[Dict[str, Tensor]]) -> Tensor:
        if temporal_context is None or "temporal_match_cost" not in temporal_context:
            return torch.argmin(dist, dim=-1)

        temporal_match_cost = temporal_context["temporal_match_cost"].to(device=dist.device, dtype=dist.dtype)
        if temporal_match_cost.shape != dist.shape or not torch.isfinite(temporal_match_cost).all():
            return torch.argmin(dist, dim=-1)

        topk = int(temporal_context.get("temporal_match_topk", self.temporal_match_topk))
        topk = max(1, min(topk, dist.shape[1]))
        _, topk_idx = torch.topk(dist, k=topk, dim=-1, largest=False)
        topk_dist = torch.gather(dist, 1, topk_idx)
        topk_temporal = torch.gather(temporal_match_cost, 1, topk_idx)
        alpha = float(temporal_context.get("temporal_match_alpha", self.temporal_match_alpha))
        selected_in_topk = torch.argmin(topk_dist + alpha * topk_temporal, dim=-1)
        return torch.gather(topk_idx, 1, selected_in_topk[:, None]).squeeze(1)

    def _temporal_auxiliary_loss(
        self,
        mode_idx: Tensor,
        temporal_context: Optional[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        if temporal_context is None:
            return {}

        required_keys = ("pred_start_cost", "pred_path_cost", "pred_velocity_cost")
        if any(key not in temporal_context for key in required_keys):
            return {}

        pred_start_cost = temporal_context["pred_start_cost"]
        pred_path_cost = temporal_context["pred_path_cost"]
        pred_velocity_cost = temporal_context["pred_velocity_cost"]
        if pred_start_cost.shape != pred_path_cost.shape or pred_start_cost.shape != pred_velocity_cost.shape:
            return {}
        if mode_idx.shape[0] != pred_start_cost.shape[0]:
            return {}
        if not (
            torch.isfinite(pred_start_cost).all()
            and torch.isfinite(pred_path_cost).all()
            and torch.isfinite(pred_velocity_cost).all()
        ):
            return {}

        start = self._gather_mode(pred_start_cost, mode_idx).clamp(max=2.0)
        path = self._gather_mode(pred_path_cost, mode_idx).clamp(max=2.0)
        velocity = self._gather_mode(pred_velocity_cost, mode_idx).clamp(max=2.0)
        start_weight = float(temporal_context.get("temporal_aux_start_weight", self.temporal_aux_start_weight))
        path_weight = float(temporal_context.get("temporal_aux_path_weight", self.temporal_aux_path_weight))
        velocity_weight = float(temporal_context.get("temporal_aux_velocity_weight", self.temporal_aux_velocity_weight))
        aux_loss_weight = float(temporal_context.get("temporal_aux_loss_weight", self.temporal_aux_loss_weight))
        temporal_aux_raw = (
            start_weight * start
            + path_weight * path
            + velocity_weight * velocity
        ).mean()
        actual_temporal_aux = aux_loss_weight * temporal_aux_raw
        outer_trajectory_weight = max(float(getattr(self._config, "trajectory_weight", 1.0)), 1e-6)
        return {
            "trajectory_scaled_temporal_aux_loss": actual_temporal_aux / outer_trajectory_weight,
            "temporal_aux_loss": actual_temporal_aux.detach(),
            "temporal_aux_raw": temporal_aux_raw.detach(),
            "temporal_selected_start_cost": start.mean().detach(),
            "temporal_selected_path_cost": path.mean().detach(),
            "temporal_selected_velocity_cost": velocity.mean().detach(),
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
        mode_idx = self._select_temporal_aware_mode(dist, temporal_context)
        cls_target = mode_idx
        gather_mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(poses_reg, 1, gather_mode_idx).squeeze(1)
        # import ipdb; ipdb.set_trace()
        # Calculate cls loss using focal loss
        target_classes_onehot = torch.zeros([bs, num_mode],
                                            dtype=poses_cls.dtype,
                                            layout=poses_cls.layout,
                                            device=poses_cls.device)
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

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
        temporal_info = self._temporal_auxiliary_loss(mode_idx, temporal_context)
        if temporal_info:
            ret_loss = ret_loss + temporal_info["trajectory_scaled_temporal_aux_loss"]
            temporal_match_cost = temporal_context.get("temporal_match_cost")
            if temporal_match_cost is not None and temporal_match_cost.shape == dist.shape:
                temporal_info["temporal_match_cost"] = self._gather_mode(
                    temporal_match_cost.to(device=dist.device, dtype=dist.dtype),
                    mode_idx,
                ).mean().detach()
            temporal_info["temporal_selected_mode_idx"] = mode_idx.float().mean().detach()
            return ret_loss, temporal_info
        return ret_loss
