from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


class RiskModeRankingHead(nn.Module):
    """Predict an isolated risk-conditioned residual for each trajectory mode."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mode_norm = nn.LayerNorm(d_model)
        self.risk_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        mode_feature: torch.Tensor,
        history_risk_memory: Optional[torch.Tensor],
        history_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mode_feature = torch.nan_to_num(mode_feature.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if history_risk_memory is None:
            pooled_risk = torch.zeros_like(mode_feature[:, 0])
        else:
            memory = torch.nan_to_num(history_risk_memory.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            if history_valid is None:
                pooled_risk = memory.mean(dim=1)
            else:
                valid = history_valid.detach().float().unsqueeze(-1).clamp(0.0, 1.0)
                pooled_risk = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        risk_feature = self.risk_norm(pooled_risk)[:, None].expand_as(mode_feature)
        fused = torch.cat([self.mode_norm(mode_feature), risk_feature], dim=-1)
        return torch.nan_to_num(self.mlp(fused).squeeze(-1), nan=0.0, posinf=20.0, neginf=-20.0)


def _trajectory_statistics(
    poses: torch.Tensor,
    ego_v: torch.Tensor,
    dt: float,
    brake_threshold: float,
) -> Dict[str, torch.Tensor]:
    origin = torch.zeros_like(poses[..., :1, :2])
    displacements = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    step_distance = torch.linalg.norm(displacements, dim=-1)
    speed = step_distance / dt
    previous_speed = torch.cat(
        [ego_v[:, None, None].expand(-1, poses.shape[1], 1), speed[..., :-1]],
        dim=-1,
    )
    acceleration = (speed - previous_speed) / dt
    braking = acceleration <= brake_threshold
    sustained = braking[..., :-1] & braking[..., 1:]
    indices = torch.arange(sustained.shape[-1], device=poses.device).view(1, 1, -1)
    onset = torch.where(sustained, indices, poses.shape[-2]).min(dim=-1).values
    return {
        "brake_onset": onset,
        "progress": step_distance.sum(dim=-1),
        "speed": speed,
        "acceleration": acceleration,
    }


def _future_clearance(poses: torch.Tensor, future_front: torch.Tensor, config: Any) -> torch.Tensor:
    front_xy = future_front[:, None, :, :2]
    front_length = future_front[:, None, :, 3]
    front_width = future_front[:, None, :, 4]
    front_valid = future_front[:, None, :, 5] > 0.5
    relative = front_xy - poses[..., :2]
    heading = poses[..., 2]
    cos_heading, sin_heading = heading.cos(), heading.sin()
    longitudinal = relative[..., 0] * cos_heading + relative[..., 1] * sin_heading
    lateral = (-relative[..., 0] * sin_heading + relative[..., 1] * cos_heading).abs()
    overlap_limit = (
        0.5 * (float(_cfg(config, "risk_shadow_ego_width", 2.0)) + front_width)
        + float(_cfg(config, "risk_shadow_lateral_margin", 0.3))
    )
    overlap = front_valid & (lateral <= overlap_limit)
    clearance = longitudinal - 0.5 * front_length - float(_cfg(config, "risk_ego_front_offset", 2.0))
    clearance_cap = float(_cfg(config, "risk_shadow_clearance_cap", 40.0))
    return torch.where(overlap, clearance, torch.full_like(clearance, clearance_cap)).min(dim=-1).values


def mine_timing_pairs(
    poses: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Mine same-behavior trajectory pairs that differ in sustained brake onset."""

    poses = torch.nan_to_num(poses.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    context = targets["risk_pair_context"].to(poses.device).float()
    scene_active = targets["risk_pair_scene_active"].to(poses.device).float() > 0.5
    ego_v = context[:, 0].clamp(min=0.0)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    stats = _trajectory_statistics(
        poses,
        ego_v,
        dt,
        float(_cfg(config, "risk_rank_brake_accel_threshold", -0.5)),
    )

    lateral_distance = (poses[:, :, None, :, 1] - poses[:, None, :, :, 1]).abs().mean(dim=-1)
    heading_delta = poses[:, :, None, :, 2] - poses[:, None, :, :, 2]
    heading_distance = torch.atan2(heading_delta.sin(), heading_delta.cos()).abs().mean(dim=-1)
    same_behavior = (
        lateral_distance <= float(_cfg(config, "risk_rank_lateral_tolerance", 0.75))
    ) & (
        heading_distance <= float(_cfg(config, "risk_rank_heading_tolerance", 0.20))
    )

    onset_positive = stats["brake_onset"][:, :, None]
    onset_negative = stats["brake_onset"][:, None, :]
    timing_better = (
        onset_positive + int(_cfg(config, "risk_rank_min_onset_step_gain", 1))
        <= onset_negative
    )
    progress_positive = stats["progress"][:, :, None]
    progress_negative = stats["progress"][:, None, :]
    progress_ok = (
        progress_positive + float(_cfg(config, "risk_rank_progress_tolerance", 1.0))
        >= progress_negative
    )
    num_modes = poses.shape[1]
    non_self = ~torch.eye(num_modes, dtype=torch.bool, device=poses.device)[None]
    pair_mask = same_behavior & timing_better & progress_ok & non_self & scene_active[:, None, None]

    future_front = targets["risk_front_future"].to(poses.device).float()
    min_clearance = _future_clearance(poses, future_front, config)
    return {
        "pair_mask": pair_mask,
        "same_behavior": same_behavior & non_self,
        "brake_onset": stats["brake_onset"].float(),
        "progress": stats["progress"],
        "min_clearance": min_clearance,
        "scene_active": scene_active,
        "current_ttc": context[:, 1],
        "delayed_ttc": context[:, 2],
    }


def compute_risk_mode_ranking_loss(
    poses: torch.Tensor,
    base_logits: torch.Tensor,
    risk_delta: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Compute isolated pairwise ranking loss and scalar diagnostics."""

    if not all(
        key in targets
        for key in ("risk_pair_context", "risk_pair_scene_active", "risk_front_future")
    ):
        zero = risk_delta.sum() * 0.0
        return {"risk_mode_ranking_loss": zero}

    mined = mine_timing_pairs(poses, targets, config)
    pair_mask = mined["pair_mask"]
    rank_logits = torch.nan_to_num(base_logits.detach() + risk_delta, nan=0.0, posinf=20.0, neginf=-20.0)
    pair_margin = float(_cfg(config, "risk_rank_margin", 1.0))
    pair_difference = rank_logits[:, :, None] - rank_logits[:, None, :]
    pair_loss = _masked_mean(F.softplus(pair_margin - pair_difference), pair_mask)

    normal_scene = ~mined["scene_active"]
    normal_delta_loss = _masked_mean(risk_delta.square().mean(dim=-1), normal_scene)
    total_loss = (
        float(_cfg(config, "risk_rank_loss_weight", 0.2)) * pair_loss
        + float(_cfg(config, "risk_rank_normal_delta_weight", 0.01)) * normal_delta_loss
    )

    pair_count = pair_mask.sum(dim=(1, 2)).float()
    pair_scene = pair_count > 0
    base_logits_detached = base_logits.detach()
    base_difference = base_logits_detached[:, :, None] - base_logits_detached[:, None, :]
    base_rank = base_logits_detached.argsort(dim=-1, descending=True).argsort(dim=-1).float() + 1.0
    positive_rank = base_rank[:, :, None].expand_as(pair_difference)
    negative_rank = base_rank[:, None, :].expand_as(pair_difference)
    positive_onset = mined["brake_onset"][:, :, None].expand_as(pair_difference)
    negative_onset = mined["brake_onset"][:, None, :].expand_as(pair_difference)
    positive_clearance = mined["min_clearance"][:, :, None].expand_as(pair_difference)
    negative_clearance = mined["min_clearance"][:, None, :].expand_as(pair_difference)

    return {
        "risk_mode_ranking_loss": total_loss,
        "risk_rank_pair_loss": pair_loss.detach(),
        "risk_rank_normal_delta_loss": normal_delta_loss.detach(),
        "risk_rank_scene_rate": mined["scene_active"].float().mean().detach(),
        "risk_rank_pair_scene_rate": pair_scene.float().mean().detach(),
        "risk_rank_pair_count": pair_count.mean().detach(),
        "risk_rank_behavior_pair_count": mined["same_behavior"].sum(dim=(1, 2)).float().mean().detach(),
        "risk_rank_base_pair_accuracy": _masked_mean((base_difference > 0).float(), pair_mask).detach(),
        "risk_rank_adjusted_pair_accuracy": _masked_mean((pair_difference > 0).float(), pair_mask).detach(),
        "risk_rank_positive_brake_onset": _masked_mean(positive_onset, pair_mask).detach(),
        "risk_rank_negative_brake_onset": _masked_mean(negative_onset, pair_mask).detach(),
        "risk_rank_positive_original_rank": _masked_mean(positive_rank, pair_mask).detach(),
        "risk_rank_negative_original_rank": _masked_mean(negative_rank, pair_mask).detach(),
        "risk_rank_positive_min_clearance": _masked_mean(positive_clearance, pair_mask).detach(),
        "risk_rank_negative_min_clearance": _masked_mean(negative_clearance, pair_mask).detach(),
        "risk_rank_delta_abs_mean": risk_delta.detach().abs().mean(),
        "risk_rank_delta_abs_max": risk_delta.detach().abs().max(),
        "risk_rank_current_ttc": _masked_mean(mined["current_ttc"], mined["scene_active"]).detach(),
        "risk_rank_delayed_ttc": _masked_mean(mined["delayed_ttc"], mined["scene_active"]).detach(),
    }
