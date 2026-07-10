from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _mode_value(values: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
    return values.gather(1, mode[:, None]).squeeze(1)


def _estimate_front_state(
    history_risk_tokens: torch.Tensor,
    agent_states: Optional[torch.Tensor],
    agent_labels: Optional[torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Fuse the LiDAR history proxy with the current predicted front vehicle."""

    tokens = torch.nan_to_num(history_risk_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    current = tokens[:, -1]
    previous = tokens[:, -2] if tokens.shape[1] > 1 else current
    lidar_valid = current[:, -1] > 0.5
    previous_valid = previous[:, -1] > 0.5
    lidar_gap = current[:, 0].clamp(min=0.0)
    ego_v = current[:, 2].clamp(min=0.0)

    max_lead_speed = float(_cfg(config, "risk_shadow_max_lead_speed", 40.0))
    current_lead_v = current[:, 3].clamp(min=0.0, max=max_lead_speed)
    previous_lead_v = previous[:, 3].clamp(min=0.0, max=max_lead_speed)
    lead_v = torch.where(previous_valid, 0.5 * (current_lead_v + previous_lead_v), current_lead_v)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    lead_a = torch.where(
        lidar_valid & previous_valid,
        (current_lead_v - previous_lead_v) / dt,
        torch.zeros_like(lead_v),
    ).clamp(
        min=float(_cfg(config, "risk_shadow_lead_accel_min", -4.0)),
        max=float(_cfg(config, "risk_shadow_lead_accel_max", 2.0)),
    )

    batch_size = tokens.shape[0]
    zeros = torch.zeros(batch_size, device=tokens.device, dtype=tokens.dtype)
    agent_valid = torch.zeros(batch_size, device=tokens.device, dtype=torch.bool)
    agent_gap = zeros.clone()
    agent_y = zeros.clone()
    agent_width = torch.full_like(zeros, float(_cfg(config, "risk_shadow_default_lead_width", 2.0)))
    agent_confidence = zeros.clone()

    if agent_states is not None and agent_labels is not None and agent_states.shape[1] > 0:
        raw_states = agent_states.float()
        raw_labels = agent_labels.float()
        finite_prediction = torch.isfinite(raw_states).all(dim=-1) & torch.isfinite(raw_labels)
        states = torch.nan_to_num(raw_states, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = torch.sigmoid(torch.nan_to_num(raw_labels, nan=-20.0, posinf=20.0, neginf=-20.0))
        x = states[..., 0]
        y = states[..., 1]
        length = states[..., 3].abs().clamp(min=0.5, max=12.0)
        width = states[..., 4].abs().clamp(min=0.5, max=4.0)
        gaps = x - 0.5 * length - float(_cfg(config, "risk_ego_front_offset", 2.0))
        candidate = (
            finite_prediction
            & (confidence >= float(_cfg(config, "risk_shadow_agent_confidence", 0.35)))
            & (x >= float(_cfg(config, "risk_front_x_min", 1.0)))
            & (x <= float(_cfg(config, "risk_front_x_max", 32.0)))
            & (gaps >= 0.0)
            & (
                y.abs()
                <= float(_cfg(config, "risk_front_y_abs", 1.8)) + 0.5 * width
            )
        )
        masked_gap = gaps.masked_fill(~candidate, torch.inf)
        agent_gap, agent_index = masked_gap.min(dim=1)
        agent_valid = torch.isfinite(agent_gap)
        gather_index = agent_index[:, None]
        agent_gap = torch.where(agent_valid, agent_gap, zeros)
        agent_y = torch.where(agent_valid, y.gather(1, gather_index).squeeze(1), zeros)
        agent_width = torch.where(
            agent_valid,
            width.gather(1, gather_index).squeeze(1),
            agent_width,
        )
        agent_confidence = torch.where(
            agent_valid,
            confidence.gather(1, gather_index).squeeze(1),
            zeros,
        )

    both_valid = lidar_valid & agent_valid
    gap_agreement = (lidar_gap - agent_gap).abs()
    agree = both_valid & (
        gap_agreement <= float(_cfg(config, "risk_shadow_gap_agreement", 3.0))
    )
    front_valid = lidar_valid | agent_valid
    gap = torch.where(
        agree,
        0.5 * (lidar_gap + agent_gap),
        torch.where(lidar_valid, lidar_gap, agent_gap),
    )

    # 0:none, 1:LiDAR only, 2:agent only, 3:fused, 4:disagreement.
    source = torch.zeros_like(gap)
    source = torch.where(lidar_valid & ~agent_valid, torch.ones_like(source), source)
    source = torch.where(agent_valid & ~lidar_valid, torch.full_like(source, 2.0), source)
    source = torch.where(agree, torch.full_like(source, 3.0), source)
    source = torch.where(both_valid & ~agree, torch.full_like(source, 4.0), source)

    reliability = torch.zeros_like(gap)
    reliability = torch.where(lidar_valid & ~agent_valid, torch.full_like(gap, 0.30), reliability)
    reliability = torch.where(agent_valid & ~lidar_valid, 0.40 * agent_confidence, reliability)
    reliability = torch.where(agree, 0.50 + 0.50 * agent_confidence, reliability)
    reliability = torch.where(both_valid & ~agree, 0.15 * agent_confidence, reliability)

    use_agent_geometry = agree | (agent_valid & ~lidar_valid)
    lead_y = torch.where(use_agent_geometry, agent_y, zeros)
    lead_width = torch.where(
        use_agent_geometry,
        agent_width,
        torch.full_like(gap, float(_cfg(config, "risk_shadow_default_lead_width", 2.0))),
    )
    lead_v = torch.where(lidar_valid, lead_v, ego_v)
    lead_a = torch.where(lidar_valid, lead_a, torch.zeros_like(lead_a))

    return {
        "valid": front_valid,
        "reliability": reliability,
        "source": source,
        "gap": gap,
        "lead_y": lead_y,
        "lead_width": lead_width,
        "lead_v": lead_v,
        "lead_a": lead_a,
        "ego_v": ego_v,
        "ego_a": current[:, 4],
        "ttc": current[:, 6],
        "drac": current[:, 7].clamp(min=0.0),
        "agent_confidence": agent_confidence,
        "gap_agreement": torch.where(both_valid, gap_agreement, torch.zeros_like(gap_agreement)),
    }


def _trajectory_dynamics(
    poses_reg: torch.Tensor,
    ego_v: torch.Tensor,
    ego_a: torch.Tensor,
    dt: float,
    config: Any,
) -> Dict[str, torch.Tensor]:
    xy = poses_reg[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    segment = torch.cat([xy[..., :1, :] - origin, xy[..., 1:, :] - xy[..., :-1, :]], dim=-2)
    speed = torch.linalg.vector_norm(segment, dim=-1) / dt
    previous_speed = torch.cat(
        [ego_v[:, None, None].expand(-1, speed.shape[1], 1), speed[..., :-1]],
        dim=-1,
    )
    acceleration = (speed - previous_speed) / dt
    previous_acceleration = torch.cat(
        [ego_a[:, None, None].expand(-1, acceleration.shape[1], 1), acceleration[..., :-1]],
        dim=-1,
    )
    jerk = (acceleration - previous_acceleration) / dt

    brake_mask = acceleration <= float(_cfg(config, "risk_shadow_brake_accel", -0.5))
    if acceleration.shape[-1] > 1:
        sustained = brake_mask[..., :-1] & brake_mask[..., 1:]
        indices = torch.arange(sustained.shape[-1], device=poses_reg.device).view(1, 1, -1)
        first_index = torch.where(sustained, indices, sustained.shape[-1]).min(dim=-1).values
        has_sustained_brake = first_index < sustained.shape[-1]
        brake_onset = (first_index.float() + 1.0) * dt
    else:
        has_sustained_brake = torch.zeros_like(acceleration[..., 0], dtype=torch.bool)
        brake_onset = torch.zeros_like(acceleration[..., 0])
    no_brake_time = float(poses_reg.shape[-2] + 1) * dt
    brake_onset = torch.where(has_sustained_brake, brake_onset, torch.full_like(brake_onset, no_brake_time))

    return {
        "acceleration": acceleration,
        "brake_onset": brake_onset,
        "max_abs_jerk": jerk.abs().max(dim=-1).values,
        "min_accel": acceleration.min(dim=-1).values,
    }


def _sample_drivable_probability(
    poses_reg: torch.Tensor,
    bev_semantic_map: Optional[torch.Tensor],
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_modes = poses_reg.shape[:2]
    if bev_semantic_map is None:
        zeros = torch.zeros(batch_size, num_modes, device=poses_reg.device, dtype=poses_reg.dtype)
        return zeros, torch.zeros(batch_size, device=poses_reg.device, dtype=torch.bool)

    probabilities = torch.softmax(
        torch.nan_to_num(bev_semantic_map.float(), nan=0.0, posinf=20.0, neginf=-20.0),
        dim=1,
    )
    class_indices = [index for index in (1, 3, 5, 6) if index < probabilities.shape[1]]
    drivable = probabilities[:, class_indices].sum(dim=1, keepdim=True).clamp(max=1.0)
    height, width = drivable.shape[-2:]
    pixel_size = max(float(_cfg(config, "bev_pixel_size", 0.25)), 1e-3)
    row = poses_reg[..., 0] / pixel_size
    column = poses_reg[..., 1] / pixel_size + width / 2.0
    grid_x = 2.0 * column / max(width - 1, 1) - 1.0
    grid_y = 2.0 * row / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    sampled = F.grid_sample(drivable, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[:, 0].mean(dim=-1), torch.ones(batch_size, device=poses_reg.device, dtype=torch.bool)


def evaluate_risk_shadow(
    poses_reg: torch.Tensor,
    poses_cls: torch.Tensor,
    history_risk_tokens: torch.Tensor,
    agent_states: Optional[torch.Tensor],
    agent_labels: Optional[torch.Tensor],
    bev_semantic_map: Optional[torch.Tensor],
    config: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Score modes without changing the trajectory unless soft re-ranking is enabled."""

    poses = torch.nan_to_num(poses_reg.float(), nan=0.0, posinf=0.0, neginf=0.0)
    logits = torch.nan_to_num(poses_cls.float(), nan=-20.0, posinf=20.0, neginf=-20.0)
    front = _estimate_front_state(history_risk_tokens, agent_states, agent_labels, config)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    dynamics = _trajectory_dynamics(poses, front["ego_v"], front["ego_a"], dt, config)

    lead_velocity = front["lead_v"]
    lead_displacement = torch.zeros_like(lead_velocity)
    future_gap = []
    for _ in range(poses.shape[-2]):
        next_velocity = (lead_velocity + front["lead_a"] * dt).clamp(
            min=0.0,
            max=float(_cfg(config, "risk_shadow_max_lead_speed", 40.0)),
        )
        lead_displacement = lead_displacement + 0.5 * (lead_velocity + next_velocity) * dt
        future_gap.append(front["gap"] + lead_displacement)
        lead_velocity = next_velocity
    future_gap = torch.stack(future_gap, dim=-1)[:, None, :]

    lateral_limit = (
        0.5
        * (
            float(_cfg(config, "risk_shadow_ego_width", 2.0))
            + front["lead_width"][:, None, None]
        )
        + float(_cfg(config, "risk_shadow_lateral_margin", 0.3))
    )
    lateral_overlap = (poses[..., 1] - front["lead_y"][:, None, None]).abs() <= lateral_limit
    dynamic_clearance_steps = future_gap - poses[..., 0]
    clearance_cap = float(_cfg(config, "risk_shadow_clearance_cap", 40.0))
    dynamic_clearance = torch.where(
        lateral_overlap,
        dynamic_clearance_steps,
        torch.full_like(dynamic_clearance_steps, clearance_cap),
    ).min(dim=-1).values
    static_clearance = front["gap"][:, None] - poses[..., 0].max(dim=-1).values

    warning_gap = float(_cfg(config, "risk_shadow_warning_gap", 3.0))
    clearance_cost = ((warning_gap - dynamic_clearance) / max(warning_gap, 1e-3)).clamp(0.0, 1.0)
    crossing = lateral_overlap & (dynamic_clearance_steps < warning_gap)
    indices = torch.arange(poses.shape[-2], device=poses.device).view(1, 1, -1)
    first_crossing_index = torch.where(crossing, indices, poses.shape[-2]).min(dim=-1).values
    has_crossing = first_crossing_index < poses.shape[-2]
    crossing_time = (first_crossing_index.float() + 1.0) * dt
    preparation_time = float(_cfg(config, "risk_shadow_brake_preparation_time", 1.0))
    desired_brake_onset = (crossing_time - preparation_time).clamp(min=dt)
    timing_cost = torch.where(
        has_crossing,
        ((dynamics["brake_onset"] - desired_brake_onset) / max(preparation_time, dt)).clamp(0.0, 1.0),
        torch.zeros_like(dynamic_clearance),
    )

    drivable_probability, map_valid = _sample_drivable_probability(poses, bev_semantic_map, config)
    map_cost = torch.where(map_valid[:, None], 1.0 - drivable_probability, torch.zeros_like(drivable_probability))
    jerk_cost = (
        (dynamics["max_abs_jerk"] - float(_cfg(config, "risk_shadow_jerk_free", 4.0)))
        / max(float(_cfg(config, "risk_shadow_jerk_scale", 4.0)), 1e-3)
    ).clamp(0.0, 1.0)
    decel_cost = (
        (-dynamics["min_accel"] - float(_cfg(config, "risk_shadow_decel_free", 4.0)))
        / max(float(_cfg(config, "risk_shadow_decel_scale", 4.0)), 1e-3)
    ).clamp(0.0, 1.0)
    comfort_cost = torch.maximum(jerk_cost, decel_cost)

    total_cost = (
        float(_cfg(config, "risk_shadow_clearance_weight", 0.45)) * clearance_cost
        + float(_cfg(config, "risk_shadow_timing_weight", 0.25)) * timing_cost
        + float(_cfg(config, "risk_shadow_map_weight", 0.20)) * map_cost
        + float(_cfg(config, "risk_shadow_comfort_weight", 0.10)) * comfort_cost
    ).clamp(0.0, 1.0)

    raw_mode = logits.argmax(dim=-1)
    reliable = front["valid"] & (
        front["reliability"] >= float(_cfg(config, "risk_shadow_min_reliability", 0.5))
    )
    max_logit = logits.max(dim=-1, keepdim=True).values
    eligible = logits >= max_logit - float(_cfg(config, "risk_shadow_cls_margin", 1.0))
    adjusted_logits = logits - (
        float(_cfg(config, "risk_shadow_logit_penalty", 1.0))
        * front["reliability"][:, None]
        * total_cost
    )
    adjusted_logits = adjusted_logits.masked_fill(~eligible, -torch.inf)
    proposed_mode = torch.where(reliable, adjusted_logits.argmax(dim=-1), raw_mode)

    diagnostics = {
        "risk_shadow_active": torch.ones_like(front["gap"]),
        "risk_shadow_reliable": reliable.float(),
        "risk_front_source": front["source"],
        "risk_front_gap": front["gap"],
        "risk_front_lead_v": front["lead_v"],
        "risk_front_lead_a": front["lead_a"],
        "risk_front_agent_confidence": front["agent_confidence"],
        "risk_front_gap_agreement": front["gap_agreement"],
        "risk_shadow_base_mode": raw_mode.float(),
        "risk_shadow_proposed_mode": proposed_mode.float(),
        "risk_shadow_changed": (proposed_mode != raw_mode).float(),
    }
    for prefix, mode in (("base", raw_mode), ("proposed", proposed_mode)):
        diagnostics[f"risk_{prefix}_cls"] = _mode_value(logits, mode)
        diagnostics[f"risk_{prefix}_static_clearance"] = _mode_value(static_clearance, mode)
        diagnostics[f"risk_{prefix}_dynamic_clearance"] = _mode_value(dynamic_clearance, mode)
        diagnostics[f"risk_{prefix}_brake_onset"] = _mode_value(dynamics["brake_onset"], mode)
        diagnostics[f"risk_{prefix}_max_abs_jerk"] = _mode_value(dynamics["max_abs_jerk"], mode)
        diagnostics[f"risk_{prefix}_drivable_probability"] = _mode_value(drivable_probability, mode)
        diagnostics[f"risk_{prefix}_clearance_cost"] = _mode_value(clearance_cost, mode)
        diagnostics[f"risk_{prefix}_timing_cost"] = _mode_value(timing_cost, mode)
        diagnostics[f"risk_{prefix}_map_cost"] = _mode_value(map_cost, mode)
        diagnostics[f"risk_{prefix}_comfort_cost"] = _mode_value(comfort_cost, mode)
        diagnostics[f"risk_{prefix}_total_cost"] = _mode_value(total_cost, mode)

    return proposed_mode, diagnostics
