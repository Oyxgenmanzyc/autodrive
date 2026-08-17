from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _trajectory_progress(poses: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    return torch.linalg.norm(displacement, dim=-1).sum(dim=-1)


def _candidate_temporal_features(poses: torch.Tensor, dt: float) -> torch.Tensor:
    """Preserve the full speed/deceleration sequence for safety and timing heads."""

    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    step_distance = torch.linalg.norm(displacement, dim=-1)
    speed = step_distance / max(float(dt), 1e-3)
    previous_speed = torch.cat([speed[..., :1], speed[..., :-1]], dim=-1)
    acceleration = (speed - previous_speed) / max(float(dt), 1e-3)
    previous_acceleration = torch.cat(
        [acceleration[..., :1], acceleration[..., :-1]], dim=-1
    )
    jerk = (acceleration - previous_acceleration) / max(float(dt), 1e-3)
    braking = (-acceleration).clamp(min=0.0)

    summary = torch.stack(
        [
            step_distance.sum(dim=-1),
            braking.max(dim=-1).values,
            jerk.abs().max(dim=-1).values,
            poses[..., 1].abs().mean(dim=-1),
        ],
        dim=-1,
    )
    return torch.cat([speed, acceleration, summary], dim=-1)


class RiskModeRankingHead(nn.Module):
    """Predict candidate safety and brake-timing quality without changing generation."""

    def __init__(self, d_model: int, num_poses: int = 8):
        super().__init__()
        self.mode_norm = nn.LayerNorm(d_model)
        self.risk_norm = nn.LayerNorm(d_model)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2 * int(num_poses) + 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.shared = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
        )
        self.unsafe_head = nn.Linear(d_model, 1)
        self.timing_head = nn.Linear(d_model, 1)
        # Both heads initially preserve the original classifier selection.
        nn.init.zeros_(self.unsafe_head.weight)
        nn.init.zeros_(self.unsafe_head.bias)
        nn.init.zeros_(self.timing_head.weight)
        nn.init.zeros_(self.timing_head.bias)

    def forward(
        self,
        mode_feature: torch.Tensor,
        history_risk_memory: Optional[torch.Tensor],
        history_valid: Optional[torch.Tensor],
        poses: Optional[torch.Tensor] = None,
        dt: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        mode_feature = torch.nan_to_num(
            mode_feature.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )
        if history_risk_memory is None:
            pooled_risk = torch.zeros_like(mode_feature[:, 0])
        else:
            # Candidate losses train the risk memory, while mode features and poses
            # stay detached to protect the original trajectory generator.
            memory = torch.nan_to_num(
                history_risk_memory, nan=0.0, posinf=0.0, neginf=0.0
            )
            if history_valid is None:
                pooled_risk = memory.mean(dim=1)
            else:
                valid = history_valid.detach().float().unsqueeze(-1).clamp(0.0, 1.0)
                pooled_risk = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp(
                    min=1.0
                )
        risk_feature = self.risk_norm(pooled_risk)[:, None].expand_as(mode_feature)

        if poses is None:
            temporal_feature = torch.zeros_like(mode_feature)
        else:
            poses = torch.nan_to_num(
                poses.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            temporal_feature = self.temporal_encoder(
                _candidate_temporal_features(poses, dt)
            )

        fused = torch.cat(
            [self.mode_norm(mode_feature), risk_feature, temporal_feature], dim=-1
        )
        shared = self.shared(fused)
        return {
            "unsafe_logits": torch.nan_to_num(
                self.unsafe_head(shared).squeeze(-1),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            "timing_logits": torch.nan_to_num(
                self.timing_head(shared).squeeze(-1),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
        }


def _path_guard(
    base_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    poses: Optional[torch.Tensor],
    config: Optional[Any],
) -> Dict[str, torch.Tensor]:
    candidate_eligible = torch.ones_like(candidate_indices, dtype=torch.bool)
    zeros = torch.zeros_like(candidate_indices, dtype=base_logits.dtype)
    lateral_distance = zeros.clone()
    heading_distance = zeros.clone()
    progress_delta = zeros.clone()

    if not bool(_cfg(config, "risk_rank_use_inference_path_guard", True)) or poses is None:
        return {
            "eligible": candidate_eligible,
            "lateral_distance": lateral_distance,
            "heading_distance": heading_distance,
            "progress_delta": progress_delta,
        }

    poses = torch.nan_to_num(poses.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if poses.ndim != 4 or poses.shape[:2] != base_logits.shape:
        raise ValueError("poses must be [batch, modes, steps, state]")

    batch_size = poses.shape[0]
    batch_index = torch.arange(batch_size, device=poses.device)
    base_mode = base_logits.argmax(dim=-1)
    base_poses = poses[batch_index, base_mode]
    candidate_poses = torch.gather(
        poses,
        dim=1,
        index=candidate_indices[..., None, None].expand(
            -1, -1, poses.shape[-2], poses.shape[-1]
        ),
    )

    lateral_distance = (
        candidate_poses[..., 1] - base_poses[:, None, :, 1]
    ).abs().mean(dim=-1)
    heading_delta = candidate_poses[..., 2] - base_poses[:, None, :, 2]
    heading_distance = torch.atan2(
        heading_delta.sin(), heading_delta.cos()
    ).abs().mean(dim=-1)
    progress_delta = _trajectory_progress(candidate_poses) - _trajectory_progress(
        base_poses
    )[:, None]

    candidate_eligible = (
        lateral_distance
        <= float(_cfg(config, "risk_rank_inference_lateral_tolerance", 0.75))
    ) & (
        heading_distance
        <= float(_cfg(config, "risk_rank_inference_heading_tolerance", 0.20))
    )
    candidate_eligible |= candidate_indices == base_mode[:, None]
    return {
        "eligible": candidate_eligible,
        "lateral_distance": lateral_distance,
        "heading_distance": heading_distance,
        "progress_delta": progress_delta,
    }


def select_risk_ranked_mode(
    base_logits: torch.Tensor,
    unsafe_logits: torch.Tensor,
    topk: int,
    timing_logits: Optional[torch.Tensor] = None,
    poses: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """Preserve base unless an eligible mode has a learned relative safety gain."""

    num_modes = base_logits.shape[-1]
    candidate_count = min(max(int(topk), 1), num_modes)
    candidate_indices = base_logits.topk(candidate_count, dim=-1).indices
    candidate_base = torch.gather(base_logits, -1, candidate_indices)
    candidate_unsafe_logits = torch.gather(unsafe_logits, -1, candidate_indices)
    candidate_unsafe_probability = candidate_unsafe_logits.sigmoid()
    if timing_logits is None:
        candidate_timing = candidate_base
    else:
        candidate_timing = torch.gather(timing_logits, -1, candidate_indices)

    guard = _path_guard(base_logits, candidate_indices, poses, config)
    eligible = guard["eligible"]
    unsafe_threshold = float(
        _cfg(config, "risk_rank_unsafe_probability_threshold", 0.5)
    )
    safe_threshold = float(
        _cfg(config, "risk_rank_candidate_safe_probability_threshold", 0.35)
    )
    predicted_safe = (candidate_unsafe_probability < safe_threshold) & eligible
    base_mode = base_logits.argmax(dim=-1)
    base_position = (candidate_indices == base_mode[:, None]).float().argmax(
        dim=-1, keepdim=True
    )
    base_unsafe_probability = torch.gather(
        candidate_unsafe_probability, -1, base_position
    )
    unsafe_probability_gain = (
        base_unsafe_probability - candidate_unsafe_probability
    )
    base_predicted_unsafe = base_unsafe_probability >= unsafe_threshold
    qualified = predicted_safe & base_predicted_unsafe & (
        unsafe_probability_gain
        >= float(_cfg(config, "risk_rank_min_unsafe_probability_gain", 0.10))
    )
    has_qualified = qualified.any(dim=-1, keepdim=True)

    # Safety decides whether to intervene; timing decides among proven-safer modes.
    quality_score = candidate_timing + 1e-4 * candidate_base
    qualified_quality = quality_score.masked_fill(
        ~qualified, torch.finfo(quality_score.dtype).min
    )
    qualified_choice = qualified_quality.argmax(dim=-1, keepdim=True)
    choice = torch.where(has_qualified, qualified_choice, base_position)
    selected_mode = torch.gather(candidate_indices, -1, choice).squeeze(-1)

    base_rank = (
        base_logits.argsort(dim=-1, descending=True).argsort(dim=-1).float() + 1.0
    )
    selected_base_rank = torch.gather(
        base_rank, -1, selected_mode.unsqueeze(-1)
    ).squeeze(-1)

    def chosen(values: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, -1, choice).squeeze(-1)

    return {
        "adjusted_logits": base_logits,
        "base_mode": base_mode,
        "unconstrained_mode": selected_mode,
        "unconstrained_topk_mode": selected_mode,
        "selected_mode": selected_mode,
        "selected_base_rank": selected_base_rank,
        "guard_blocked": torch.zeros_like(selected_mode, dtype=torch.bool),
        "path_guard_blocked": torch.zeros_like(selected_mode, dtype=torch.bool),
        "eligible_candidate_count": eligible.sum(dim=-1),
        "predicted_safe_candidate_count": predicted_safe.sum(dim=-1),
        "qualified_candidate_count": qualified.sum(dim=-1),
        "safe_fallback_used": ~has_qualified.squeeze(-1),
        "base_predicted_unsafe": base_predicted_unsafe.squeeze(-1),
        "base_unsafe_probability": base_unsafe_probability.squeeze(-1),
        "selected_unsafe_probability": chosen(candidate_unsafe_probability),
        "selected_unsafe_probability_gain": chosen(unsafe_probability_gain),
        "selected_timing_logit": chosen(candidate_timing),
        "proposed_lateral_distance": chosen(guard["lateral_distance"]),
        "proposed_heading_distance": chosen(guard["heading_distance"]),
        "proposed_progress_delta": chosen(guard["progress_delta"]),
        "selected_lateral_distance": chosen(guard["lateral_distance"]),
        "selected_heading_distance": chosen(guard["heading_distance"]),
        "selected_progress_delta": chosen(guard["progress_delta"]),
    }


def _trajectory_statistics(
    poses: torch.Tensor,
    ego_v: torch.Tensor,
    dt: float,
    brake_threshold: float,
) -> Dict[str, torch.Tensor]:
    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    step_distance = torch.linalg.norm(displacement, dim=-1)
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
        "peak_decel": (-acceleration).clamp(min=0.0).max(dim=-1).values,
    }


def _future_longitudinal_metrics(
    poses: torch.Tensor,
    future_front: torch.Tensor,
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Candidate-front proxy used only to build train-time unsafe labels."""

    if future_front.shape[1] != poses.shape[-2]:
        raise ValueError("risk_front_future and poses must have the same time steps")

    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    ttc_cap = float(_cfg(config, "risk_ttc_max", 10.0))
    clearance_cap = float(_cfg(config, "risk_shadow_clearance_cap", 40.0))

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
    front_overlap = front_valid & (longitudinal > 0.0) & (lateral <= overlap_limit)
    raw_clearance = (
        longitudinal
        - 0.5 * front_length
        - float(_cfg(config, "risk_ego_front_offset", 2.0))
    )
    clearance = torch.where(
        front_overlap, raw_clearance, torch.full_like(raw_clearance, clearance_cap)
    )

    pair_valid = front_overlap[..., :-1] & front_overlap[..., 1:]
    closing_pair = ((clearance[..., :-1] - clearance[..., 1:]) / dt).clamp(min=0.0)
    closing_pair = torch.where(pair_valid, closing_pair, torch.zeros_like(closing_pair))
    closing_speed = torch.cat(
        [torch.zeros_like(closing_pair[..., :1]), closing_pair], dim=-1
    )
    positive_clearance = raw_clearance.clamp(min=1e-3)
    ttc = torch.full_like(raw_clearance, ttc_cap)
    valid_ttc = front_overlap & (closing_speed > 0.05) & (raw_clearance > 0.0)
    ttc = torch.where(
        valid_ttc,
        (positive_clearance / closing_speed.clamp(min=1e-3)).clamp(max=ttc_cap),
        ttc,
    )
    ttc = torch.where(
        front_overlap & (raw_clearance <= 0.0), torch.zeros_like(ttc), ttc
    )
    return {
        "min_clearance": clearance.min(dim=-1).values,
        "min_ttc": ttc.min(dim=-1).values,
        "front_overlap_rate": front_overlap.float().mean(dim=-1),
    }


def _future_agent_collision(
    poses: torch.Tensor,
    future_agents: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Return exact oriented-box overlap for every candidate and future step."""

    if future_agents.ndim != 4 or future_agents.shape[1] != poses.shape[-2]:
        raise ValueError(
            "risk_future_agents must be [batch, steps, agents, 6] and align with poses"
        )
    agents = torch.nan_to_num(
        future_agents.to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    ego_heading = poses[..., 2]
    center_offset = float(_cfg(config, "risk_rank_ego_rear_axle_to_center", 1.461))
    ego_center = poses[..., :2] + center_offset * torch.stack(
        [ego_heading.cos(), ego_heading.sin()], dim=-1
    )
    ego_center = ego_center[:, :, :, None, :]
    agent_center = agents[:, None, :, :, :2]
    delta = agent_center - ego_center

    ego_u = torch.stack([ego_heading.cos(), ego_heading.sin()], dim=-1)[
        :, :, :, None, :
    ]
    ego_v = torch.stack([-ego_u[..., 1], ego_u[..., 0]], dim=-1)
    agent_heading = agents[..., 2][:, None]
    agent_u = torch.stack([agent_heading.cos(), agent_heading.sin()], dim=-1)
    agent_v = torch.stack([-agent_u[..., 1], agent_u[..., 0]], dim=-1)

    margin = float(_cfg(config, "risk_rank_collision_margin", 0.0))
    ego_half_length = 0.5 * float(_cfg(config, "risk_rank_ego_length", 5.176)) + margin
    ego_half_width = 0.5 * float(_cfg(config, "risk_rank_ego_width", 2.297)) + margin
    agent_half_length = 0.5 * agents[..., 3][:, None] + margin
    agent_half_width = 0.5 * agents[..., 4][:, None] + margin

    def overlaps(axis: torch.Tensor) -> torch.Tensor:
        distance = (delta * axis).sum(dim=-1).abs()
        ego_radius = (
            ego_half_length * (ego_u * axis).sum(dim=-1).abs()
            + ego_half_width * (ego_v * axis).sum(dim=-1).abs()
        )
        agent_radius = (
            agent_half_length * (agent_u * axis).sum(dim=-1).abs()
            + agent_half_width * (agent_v * axis).sum(dim=-1).abs()
        )
        return distance <= ego_radius + agent_radius

    valid = agents[..., 5][:, None] > 0.5
    collision = valid
    for axis in (ego_u, ego_v, agent_u, agent_v):
        collision = collision & overlaps(axis)
    return collision


def mine_timing_pairs(
    poses: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Build one safety label and one GT quality value for each candidate."""

    poses = torch.nan_to_num(poses.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    context = targets["risk_pair_context"].to(poses.device).float()
    cached_scene_active = targets["risk_pair_scene_active"].to(poses.device).float() > 0.5
    ego_v = context[:, 0].clamp(min=0.0)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    stats = _trajectory_statistics(
        poses,
        ego_v,
        dt,
        float(_cfg(config, "risk_rank_brake_accel_threshold", -0.5)),
    )

    target_trajectory = targets["trajectory"].to(poses.device).float()[..., :3]
    gt_xy_error = torch.linalg.norm(
        poses[..., :2] - target_trajectory[:, None, :, :2], dim=-1
    ).mean(dim=-1)
    heading_delta = poses[..., 2] - target_trajectory[:, None, :, 2]
    gt_heading_error = torch.atan2(
        heading_delta.sin(), heading_delta.cos()
    ).abs().mean(dim=-1)
    gt_error = (
        float(_cfg(config, "risk_rank_gt_xy_weight", 1.0)) * gt_xy_error
        + float(_cfg(config, "risk_rank_gt_heading_weight", 0.5)) * gt_heading_error
    )

    gt_lateral_error = (
        poses[..., 1] - target_trajectory[:, None, :, 1]
    ).abs().mean(dim=-1)
    gt_path_compatible = (
        gt_lateral_error
        <= float(_cfg(config, "risk_rank_gt_lateral_tolerance", 1.50))
    ) & (
        gt_heading_error
        <= float(_cfg(config, "risk_rank_gt_heading_tolerance", 0.35))
    )

    future_front = targets["risk_front_future"].to(poses.device).float()
    longitudinal = _future_longitudinal_metrics(poses, future_front, config)
    front_continuous = future_front[..., 5].sum(dim=-1) >= 2.0
    collision_proxy = longitudinal["min_clearance"] <= float(
        _cfg(config, "risk_rank_collision_clearance_threshold", 0.0)
    )
    ttc_failure = longitudinal["min_ttc"] < float(
        _cfg(config, "risk_rank_unsafe_ttc_threshold", 2.0)
    )
    future_agent_collision = _future_agent_collision(
        poses, targets["risk_future_agents"], config
    ).any(dim=(-1, -2))
    unsafe_target = future_agent_collision | collision_proxy | ttc_failure
    # Future annotations are available in every cached scene, including empty scenes.
    unsafe_valid = torch.ones_like(unsafe_target, dtype=torch.bool)

    # Scene activation is broad, but no additional hand-written quality costs are used.
    gt_risk_active = unsafe_target.any(dim=-1) | (
        front_continuous
        & (
            (context[:, 1] <= float(_cfg(config, "risk_pair_ttc_warning_threshold", 4.0)))
            | (context[:, 2] <= float(_cfg(config, "risk_pair_ttc_warning_threshold", 4.0)))
        )
    )
    scene_active = cached_scene_active | gt_risk_active

    positive_unsafe = unsafe_target[:, :, None]
    negative_unsafe = unsafe_target[:, None, :]
    safe_over_unsafe = (~positive_unsafe) & negative_unsafe

    same_safety = positive_unsafe == negative_unsafe
    positive_gt_error = gt_error[:, :, None]
    negative_gt_error = gt_error[:, None, :]
    gt_better = positive_gt_error + float(
        _cfg(config, "risk_rank_min_gt_error_gain", 0.10)
    ) <= negative_gt_error

    progress_gain = stats["progress"][:, :, None] - stats["progress"][:, None, :]
    decel_gain = stats["peak_decel"][:, None, :] - stats["peak_decel"][:, :, None]
    min_progress_gain = float(_cfg(config, "risk_rank_timing_min_progress_gain", 1.0))
    min_decel_gain = float(_cfg(config, "risk_rank_timing_min_decel_gain", 0.5))
    max_decel_regression = float(
        _cfg(config, "risk_rank_timing_max_decel_regression", 0.5)
    )
    progress_better = (progress_gain >= min_progress_gain) & (
        decel_gain >= -max_decel_regression
    )
    similar_progress = progress_gain.abs() < min_progress_gain
    decel_better = similar_progress & (decel_gain >= min_decel_gain)
    timing_tie_break = (
        similar_progress & (decel_gain.abs() < min_decel_gain) & gt_better
    )
    timing_better = progress_better | decel_better | timing_tie_break

    lateral_distance = (
        poses[:, :, None, :, 1] - poses[:, None, :, :, 1]
    ).abs().mean(dim=-1)
    pair_heading_delta = poses[:, :, None, :, 2] - poses[:, None, :, :, 2]
    pair_heading_distance = torch.atan2(
        pair_heading_delta.sin(), pair_heading_delta.cos()
    ).abs().mean(dim=-1)
    same_behavior = (
        lateral_distance <= float(_cfg(config, "risk_rank_lateral_tolerance", 0.75))
    ) & (
        pair_heading_distance <= float(_cfg(config, "risk_rank_heading_tolerance", 0.20))
    )

    num_modes = poses.shape[1]
    non_self = ~torch.eye(num_modes, dtype=torch.bool, device=poses.device)[None]
    safety_pair_mask = (
        safe_over_unsafe
        & same_behavior
        & non_self
    )
    quality_pair_mask = (
        same_safety
        & (~positive_unsafe)
        & timing_better
        & gt_path_compatible[:, :, None]
        & gt_path_compatible[:, None, :]
        & non_self
        & scene_active[:, None, None]
    )

    return {
        "pair_mask": safety_pair_mask,
        "safety_pair_mask": safety_pair_mask,
        "quality_pair_mask": quality_pair_mask,
        "unsafe_target": unsafe_target.float(),
        "unsafe_valid": unsafe_valid,
        "collision_proxy": collision_proxy.float(),
        "future_agent_collision": future_agent_collision.float(),
        "ttc_failure": ttc_failure.float(),
        "gt_error": gt_error,
        "gt_path_compatible": gt_path_compatible,
        "min_clearance": longitudinal["min_clearance"],
        "min_ttc": longitudinal["min_ttc"],
        "brake_onset": stats["brake_onset"].float(),
        "progress": stats["progress"],
        "peak_decel": stats["peak_decel"],
        "scene_active": scene_active,
        "cached_scene_active": cached_scene_active,
        "gt_risk_active": gt_risk_active,
        "current_ttc": context[:, 1],
        "delayed_ttc": context[:, 2],
    }


def compute_risk_mode_ranking_loss(
    poses: torch.Tensor,
    base_logits: torch.Tensor,
    unsafe_logits: torch.Tensor,
    timing_logits: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Train PDM-aligned collision filtering and safe-candidate timing quality."""

    required = {
        "risk_pair_context",
        "risk_pair_scene_active",
        "risk_front_future",
        "risk_future_agents",
        "trajectory",
    }
    if not required.issubset(targets):
        zero = (unsafe_logits.sum() + timing_logits.sum()) * 0.0
        return {"risk_mode_ranking_loss": zero}

    mined = mine_timing_pairs(poses, targets, config)
    valid = mined["unsafe_valid"]
    unsafe_target = mined["unsafe_target"]
    pos_weight = torch.tensor(
        float(_cfg(config, "risk_rank_unsafe_pos_weight", 3.0)),
        device=unsafe_logits.device,
        dtype=unsafe_logits.dtype,
    )
    raw_bce = F.binary_cross_entropy_with_logits(
        unsafe_logits, unsafe_target, reduction="none", pos_weight=pos_weight
    )
    bce_loss = _masked_mean(raw_bce, valid)

    safety_pair_mask = mined["safety_pair_mask"]
    positive = unsafe_logits[:, :, None]
    negative = unsafe_logits[:, None, :]
    # Positive candidate is safe, so its unsafe logit must be lower.
    safety_pair_loss = _masked_mean(
        F.softplus(float(_cfg(config, "risk_rank_margin", 1.0)) + positive - negative),
        safety_pair_mask,
    )

    timing_pair_mask = mined["quality_pair_mask"]
    better_timing = timing_logits[:, :, None]
    worse_timing = timing_logits[:, None, :]
    timing_pair_loss = _masked_mean(
        F.softplus(
            float(_cfg(config, "risk_rank_margin", 1.0))
            - better_timing
            + worse_timing
        ),
        timing_pair_mask,
    )
    total_loss = float(_cfg(config, "risk_rank_loss_weight", 0.2)) * (
        bce_loss
        + float(_cfg(config, "risk_rank_pair_loss_weight", 0.5)) * safety_pair_loss
        + float(_cfg(config, "risk_rank_timing_loss_weight", 0.5))
        * timing_pair_loss
    )

    probability = unsafe_logits.sigmoid()
    prediction = probability >= float(
        _cfg(config, "risk_rank_unsafe_probability_threshold", 0.5)
    )
    correct = (prediction == (unsafe_target > 0.5)) & valid
    unsafe_count = (unsafe_target * valid.float()).sum()
    safe_count = ((1.0 - unsafe_target) * valid.float()).sum()
    true_positive = (prediction & (unsafe_target > 0.5) & valid).float().sum()
    true_negative = ((~prediction) & (unsafe_target <= 0.5) & valid).float().sum()

    base_difference = base_logits.detach()[:, :, None] - base_logits.detach()[:, None, :]
    timing_difference = timing_logits[:, :, None] - timing_logits[:, None, :]
    selection = select_risk_ranked_mode(
        base_logits.detach(),
        unsafe_logits,
        int(_cfg(config, "risk_rank_inference_topk", unsafe_logits.shape[-1])),
        timing_logits=timing_logits,
        poses=poses,
        config=config,
    )
    batch_index = torch.arange(poses.shape[0], device=poses.device)
    base_mode = selection["base_mode"]
    selected_mode = selection["selected_mode"]
    collision = mined["future_agent_collision"] > 0.5
    base_collision = collision[batch_index, base_mode]
    selected_collision = collision[batch_index, selected_mode]
    rescuable_collision = base_collision & (
        (~collision) & mined["gt_path_compatible"]
    ).any(dim=-1)
    rescued_collision = rescuable_collision & (~selected_collision)
    new_collision = (~base_collision) & selected_collision
    selection_changed = selected_mode != base_mode

    return {
        "risk_mode_ranking_loss": total_loss,
        "risk_rank_unsafe_bce_loss": bce_loss.detach(),
        "risk_rank_unsafe_pair_loss": safety_pair_loss.detach(),
        "risk_rank_timing_pair_loss": timing_pair_loss.detach(),
        "risk_rank_scene_rate": mined["scene_active"].float().mean().detach(),
        "risk_rank_pair_scene_rate": safety_pair_mask.any(dim=(1, 2)).float().mean().detach(),
        "risk_rank_pair_count": safety_pair_mask.sum(dim=(1, 2)).float().mean().detach(),
        "risk_rank_timing_pair_scene_rate": timing_pair_mask.any(dim=(1, 2)).float().mean().detach(),
        "risk_rank_unsafe_rate": _masked_mean(unsafe_target, valid).detach(),
        "risk_rank_unsafe_accuracy": _masked_mean(correct.float(), valid).detach(),
        "risk_rank_unsafe_recall": (true_positive / unsafe_count.clamp(min=1.0)).detach(),
        "risk_rank_safe_recall": (true_negative / safe_count.clamp(min=1.0)).detach(),
        "risk_rank_collision_proxy_rate": _masked_mean(
            mined["collision_proxy"], valid
        ).detach(),
        "risk_rank_future_agent_collision_rate": _masked_mean(
            mined["future_agent_collision"], valid
        ).detach(),
        "risk_rank_ttc_failure_rate": _masked_mean(mined["ttc_failure"], valid).detach(),
        "risk_rank_base_quality_pair_accuracy": _masked_mean(
            (base_difference > 0).float(), timing_pair_mask
        ).detach(),
        "risk_rank_timing_pair_accuracy": _masked_mean(
            (timing_difference > 0).float(), timing_pair_mask
        ).detach(),
        "risk_rank_gt_error_mean": _masked_mean(
            mined["gt_error"], mined["gt_path_compatible"]
        ).detach(),
        "risk_rank_min_ttc_mean": _masked_mean(mined["min_ttc"], valid).detach(),
        "risk_rank_min_clearance_mean": _masked_mean(
            mined["min_clearance"], valid
        ).detach(),
        "risk_rank_current_ttc": _masked_mean(
            mined["current_ttc"], mined["scene_active"]
        ).detach(),
        "risk_rank_delayed_ttc": _masked_mean(
            mined["delayed_ttc"], mined["scene_active"]
        ).detach(),
        # Exact epoch counters.
        "risk_rank_stat_scene_count": torch.tensor(
            float(poses.shape[0]), device=poses.device
        ),
        "risk_rank_stat_active_scene_count": mined["scene_active"].sum().float().detach(),
        "risk_rank_stat_pair_scene_count": safety_pair_mask.any(dim=(1, 2)).sum().float().detach(),
        "risk_rank_stat_pair_count": safety_pair_mask.sum().float().detach(),
        "risk_rank_stat_valid_mode_count": valid.sum().float().detach(),
        "risk_rank_stat_unsafe_mode_count": (unsafe_target * valid.float()).sum().detach(),
        "risk_rank_stat_correct_mode_count": correct.sum().float().detach(),
        "risk_rank_stat_unsafe_true_count": unsafe_count.detach(),
        "risk_rank_stat_unsafe_true_positive_count": true_positive.detach(),
        "risk_rank_stat_safe_true_count": safe_count.detach(),
        "risk_rank_stat_safe_true_negative_count": true_negative.detach(),
        "risk_rank_stat_collision_proxy_count": (
            mined["collision_proxy"] * valid.float()
        ).sum().detach(),
        "risk_rank_stat_future_agent_collision_count": (
            mined["future_agent_collision"] * valid.float()
        ).sum().detach(),
        "risk_rank_stat_ttc_failure_count": (
            mined["ttc_failure"] * valid.float()
        ).sum().detach(),
        "risk_rank_stat_quality_pair_count": timing_pair_mask.sum().float().detach(),
        "risk_rank_stat_base_quality_pair_correct_count": (
            (base_difference > 0).float() * timing_pair_mask.float()
        ).sum().detach(),
        "risk_rank_stat_timing_pair_correct_count": (
            (timing_difference > 0).float() * timing_pair_mask.float()
        ).sum().detach(),
        "risk_rank_stat_base_collision_scene_count": base_collision.sum().float().detach(),
        "risk_rank_stat_selected_collision_scene_count": selected_collision.sum().float().detach(),
        "risk_rank_stat_rescuable_collision_scene_count": rescuable_collision.sum().float().detach(),
        "risk_rank_stat_rescued_collision_scene_count": rescued_collision.sum().float().detach(),
        "risk_rank_stat_new_collision_scene_count": new_collision.sum().float().detach(),
        "risk_rank_stat_selection_change_count": selection_changed.sum().float().detach(),
    }
