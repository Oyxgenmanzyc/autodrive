from typing import Any, Dict

import torch
import torch.nn.functional as F


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.to(values.dtype)).sum()


def _trajectory_dynamics(poses: torch.Tensor, ego_v: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
    """Compute finite speed, acceleration, and jerk from trajectory positions."""

    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    # A stationary valid trajectory has zero displacement. Evaluate this branch
    # in FP32 and smooth the zero-vector norm while retaining zero speed and a
    # zero displacement gradient at that point.
    distance_sq = displacement.float().square().sum(dim=-1)
    epsilon_root = distance_sq.new_tensor(1e-6)
    speed = (torch.sqrt(distance_sq + epsilon_root.square()) - epsilon_root).clamp_min(0.0) / dt
    initial_speed = ego_v
    while initial_speed.ndim < speed.ndim:
        initial_speed = initial_speed.unsqueeze(-1)
    previous_speed = torch.cat([initial_speed, speed[..., :-1]], dim=-1)
    acceleration = (speed - previous_speed) / dt
    jerk = torch.diff(acceleration, dim=-1) / dt
    return {"speed": speed, "acceleration": acceleration, "jerk": jerk}


def _sustained_brake_onset(acceleration: torch.Tensor, threshold: float) -> torch.Tensor:
    braking = acceleration <= threshold
    sustained = braking[..., :-1] & braking[..., 1:]
    indices = torch.arange(sustained.shape[-1], device=acceleration.device)
    return torch.where(sustained, indices, acceleration.shape[-1]).min(dim=-1).values


def _brake_timing_observations(
    poses: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Return per-scene timing values before any batch or epoch reduction."""

    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    threshold = float(_cfg(config, "brake_timing_accel_threshold", -0.5))
    temperature = max(float(_cfg(config, "brake_timing_temperature", 0.35)), 1e-3)
    ego_v = context[:, 0].clamp(min=0.0)
    pred_dynamics = _trajectory_dynamics(poses, ego_v, dt)
    gt_dynamics = _trajectory_dynamics(target, ego_v, dt)
    pred_acceleration = pred_dynamics["acceleration"]
    gt_acceleration = gt_dynamics["acceleration"]

    pred_brake = torch.sigmoid((threshold - pred_acceleration) / temperature)
    gt_brake = torch.sigmoid((threshold - gt_acceleration) / temperature)
    profile_loss = F.smooth_l1_loss(pred_brake, gt_brake, reduction="none").mean(dim=-1)
    onset_loss = F.smooth_l1_loss(
        torch.cummax(pred_brake, dim=-1).values,
        torch.cummax(gt_brake, dim=-1).values,
        reduction="none",
    ).mean(dim=-1)
    profile_weight = float(_cfg(config, "brake_timing_profile_weight", 0.25))
    per_scene_loss = profile_weight * profile_loss + (1.0 - profile_weight) * onset_loss

    current_ttc, delayed_ttc, t1, continuous_steps = context[:, 1], context[:, 2], context[:, 3], context[:, 4]
    preparation_time = float(_cfg(config, "brake_timing_preparation_time", 1.0))
    pre_risk = (continuous_steps >= 2.0) & (torch.minimum(current_ttc, delayed_ttc) <= t1 + preparation_time)
    gt_onset = _sustained_brake_onset(gt_acceleration, threshold)
    pred_onset = _sustained_brake_onset(pred_acceleration, threshold)
    gt_has_sustained_brake = gt_onset < gt_acceleration.shape[-1]
    active = pre_risk & gt_has_sustained_brake
    onset_error = (pred_onset - gt_onset).float() * dt
    pred_oscillation = (
        (pred_acceleration[:, 1:] * pred_acceleration[:, :-1] < 0.0)
        & (pred_acceleration[:, 1:].abs() > 0.2)
        & (pred_acceleration[:, :-1].abs() > 0.2)
    ).float().mean(dim=-1)

    return {
        "pre_risk": pre_risk,
        "gt_has_sustained_brake": gt_has_sustained_brake,
        "active": active,
        "raw_loss": per_scene_loss,
        "gt_onset_s": gt_onset.float() * dt,
        "pred_onset_s": pred_onset.float() * dt,
        "onset_error_s": onset_error,
        "onset_abs_error_s": onset_error.abs(),
        "late_rate": (onset_error > 0.25).float(),
        "early_rate": (onset_error < -0.25).float(),
        "speed_mae": (pred_dynamics["speed"] - gt_dynamics["speed"]).abs().mean(dim=-1),
        "pred_min_accel": pred_acceleration.min(dim=-1).values,
        "gt_min_accel": gt_acceleration.min(dim=-1).values,
        "pred_jerk_abs": pred_dynamics["jerk"].abs().mean(dim=-1),
        "gt_jerk_abs": gt_dynamics["jerk"].abs().mean(dim=-1),
        "pred_accel_oscillation": pred_oscillation,
        "current_ttc": current_ttc,
        "delayed_ttc": delayed_ttc,
        "continuous_front_steps": continuous_steps,
    }


def _trajectory_progress(poses: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    distance_sq = displacement.float().square().sum(dim=-1)
    epsilon_root = distance_sq.new_tensor(1e-6)
    step_distance = torch.sqrt(distance_sq + epsilon_root.square()) - epsilon_root
    return step_distance.clamp_min(0.0).cumsum(dim=-1)


def _path_tangent(poses: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    norm = torch.linalg.vector_norm(displacement.float(), dim=-1, keepdim=True)
    fallback = torch.zeros_like(displacement)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-3, displacement / norm.clamp_min(1e-3), fallback)


def _time_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def _all_mode_risk_corridor_observations(
    poses: torch.Tensor,
    gt_target: torch.Tensor,
    context: torch.Tensor,
    front_boxes: torch.Tensor,
    front_mask: torch.Tensor,
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Constrain every front-conflicting mode between early- and late-braking bounds."""

    if poses.ndim == 3:
        poses = poses[:, None]
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    ego_v = context[:, 0].clamp(min=0.0)
    dynamics = _trajectory_dynamics(poses, ego_v, dt)
    speed = dynamics["speed"]
    acceleration = dynamics["acceleration"]
    jerk = dynamics["jerk"]
    gt_acceleration = _trajectory_dynamics(gt_target, ego_v, dt)["acceleration"]

    tangent = _path_tangent(poses.detach())
    relative = front_boxes[:, None, :, :2] - poses[..., :2]
    longitudinal = (relative * tangent).sum(dim=-1)
    lateral = torch.abs(relative[..., 0] * tangent[..., 1] - relative[..., 1] * tangent[..., 0])
    front_length = front_boxes[:, None, :, 2].clamp_min(0.5)
    front_width = front_boxes[:, None, :, 3].clamp_min(0.5)
    ego_width = float(_cfg(config, "risk_shadow_ego_width", 2.0))
    lateral_margin = float(_cfg(config, "risk_shadow_lateral_margin", 0.3))
    ego_front_offset = float(_cfg(config, "risk_ego_front_offset", 2.0))
    overlap_limit = 0.5 * (ego_width + front_width) + lateral_margin

    valid_front = front_mask[:, None, :] > 0.5
    relevant = (
        valid_front
        & (lateral.detach() <= overlap_limit)
        & (longitudinal.detach() > -(0.5 * front_length + ego_front_offset))
    )
    min_gap = float(_cfg(config, "transport_min_gap", 1.5))
    max_gap = float(_cfg(config, "transport_max_gap", 8.0))
    time_headway = float(_cfg(config, "transport_time_headway", 0.75))
    desired_gap = (time_headway * speed.detach()).clamp(min=min_gap, max=max_gap)
    clearance = longitudinal - 0.5 * front_length - ego_front_offset - desired_gap

    preparation_time = float(_cfg(config, "risk_corridor_preparation_time", 1.0))
    preparation_distance = (speed.detach() * preparation_time).clamp(min=0.5, max=max_gap)
    threat = relevant & (clearance.detach() <= preparation_distance)
    min_mode_steps = int(_cfg(config, "risk_corridor_min_mode_steps", 2))
    reliable_scene = front_mask.sum(dim=-1) >= min_mode_steps
    mode_active = (
        reliable_scene[:, None]
        & (relevant.sum(dim=-1) >= min_mode_steps)
        & threat.any(dim=-1)
    )

    safety_excess = (-clearance).clamp_min(0.0)
    safety_loss = _time_masked_mean(
        F.smooth_l1_loss(safety_excess, torch.zeros_like(safety_excess), reduction="none"),
        relevant,
    )

    previous_clearance = torch.cat([clearance[..., :1], clearance[..., :-1]], dim=-1)
    closing_speed = ((previous_clearance - clearance).detach() / dt).clamp_min(0.0)
    max_decel = float(_cfg(config, "risk_corridor_max_decel", 4.0))
    required_decel = (
        closing_speed.square() / (2.0 * clearance.detach().clamp_min(0.5))
    ).clamp(max=max_decel)
    proximity = (
        (preparation_distance - clearance.detach()) / preparation_distance.clamp_min(0.5)
    ).clamp(0.0, 1.0)
    late_excess = (acceleration + required_decel).clamp_min(0.0) * proximity
    late_loss = _time_masked_mean(
        F.smooth_l1_loss(late_excess, torch.zeros_like(late_excess), reduction="none"),
        threat,
    )

    threat_seen = threat.to(torch.int32).cumsum(dim=-1) > 0
    early_threshold = float(_cfg(config, "risk_corridor_early_accel_threshold", -0.5))
    gt_braking = gt_acceleration <= early_threshold
    early_mask = relevant & ~threat_seen & ~gt_braking[:, None, :]
    early_excess = (early_threshold - acceleration).clamp_min(0.0)
    early_loss = _time_masked_mean(
        F.smooth_l1_loss(early_excess, torch.zeros_like(early_excess), reduction="none"),
        early_mask,
    )

    decel_excess = (-max_decel - acceleration).clamp_min(0.0)
    decel_loss = _time_masked_mean(
        F.smooth_l1_loss(decel_excess, torch.zeros_like(decel_excess), reduction="none"),
        relevant,
    )
    jerk_free = float(_cfg(config, "risk_corridor_jerk_free", 4.0))
    jerk_excess = (jerk.abs() - jerk_free).clamp_min(0.0)
    jerk_loss = _time_masked_mean(
        F.smooth_l1_loss(jerk_excess, torch.zeros_like(jerk_excess), reduction="none"),
        relevant[..., 1:],
    )
    raw_loss = (
        float(_cfg(config, "risk_corridor_safety_weight", 2.0)) * safety_loss
        + float(_cfg(config, "risk_corridor_late_weight", 0.5)) * late_loss
        + float(_cfg(config, "risk_corridor_early_weight", 0.5)) * early_loss
        + float(_cfg(config, "risk_corridor_decel_weight", 0.05)) * decel_loss
        + float(_cfg(config, "risk_corridor_jerk_weight", 0.05)) * jerk_loss
    )

    threshold = float(_cfg(config, "brake_timing_accel_threshold", -0.5))
    target_onset = torch.where(
        threat,
        torch.arange(threat.shape[-1], device=poses.device),
        threat.shape[-1],
    ).min(dim=-1).values
    pred_onset = _sustained_brake_onset(acceleration, threshold)
    onset_error = (pred_onset - target_onset).float() * dt
    pre_risk_scene = (
        (context[:, 4] >= 2.0)
        & (
            torch.minimum(context[:, 1], context[:, 2])
            <= context[:, 3] + float(_cfg(config, "brake_timing_preparation_time", 1.0))
        )
    )
    pre_risk = pre_risk_scene[:, None].expand_as(mode_active)
    oscillation = (
        (acceleration[..., 1:] * acceleration[..., :-1] < 0.0)
        & (acceleration[..., 1:].abs() > 0.2)
        & (acceleration[..., :-1].abs() > 0.2)
    ).float().mean(dim=-1)
    min_clearance = torch.where(
        relevant,
        clearance,
        torch.full_like(clearance, 1e3),
    ).min(dim=-1).values
    min_clearance = torch.where(relevant.any(dim=-1), min_clearance, torch.zeros_like(min_clearance))
    safety_violation = torch.where(relevant, safety_excess, torch.zeros_like(safety_excess)).max(dim=-1).values
    required_decel_metric = torch.where(threat, required_decel, torch.zeros_like(required_decel)).max(dim=-1).values
    early_excess_metric = torch.where(early_mask, early_excess, torch.zeros_like(early_excess)).max(dim=-1).values
    late_excess_metric = torch.where(threat, late_excess, torch.zeros_like(late_excess)).max(dim=-1).values

    return {
        "pre_risk": pre_risk,
        "gt_has_sustained_brake": threat.any(dim=-1),
        "active": mode_active,
        "raw_loss": raw_loss,
        "gt_onset_s": target_onset.float() * dt,
        "pred_onset_s": pred_onset.float() * dt,
        "onset_error_s": onset_error,
        "onset_abs_error_s": onset_error.abs(),
        "late_rate": (onset_error > 0.25).float(),
        "early_rate": (onset_error < -0.25).float(),
        "pred_min_accel": acceleration.min(dim=-1).values,
        "gt_min_accel": -required_decel.max(dim=-1).values,
        "pred_jerk_abs": jerk.abs().mean(dim=-1),
        "pred_accel_oscillation": oscillation,
        "corridor_relevant_steps": relevant.sum(dim=-1).float(),
        "corridor_min_clearance": min_clearance,
        "corridor_safety_violation": safety_violation,
        "corridor_required_decel": required_decel_metric,
        "corridor_early_brake_excess": early_excess_metric,
        "corridor_late_brake_excess": late_excess_metric,
        "corridor_safety_loss": safety_loss,
        "corridor_early_loss": early_loss,
        "corridor_late_loss": late_loss,
    }


def _temporal_transport_observations(
    poses: torch.Tensor,
    gt_target: torch.Tensor,
    transport_target: torch.Tensor,
    context: torch.Tensor,
    upper_s: torch.Tensor,
    constraint_mask: torch.Tensor,
    transport_valid: torch.Tensor,
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Measure the GT-matched mode against the endpoint-conditioned ST target."""

    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    threshold = float(_cfg(config, "brake_timing_accel_threshold", -0.5))
    ego_v = context[:, 0].clamp(min=0.0)
    pred_dynamics = _trajectory_dynamics(poses, ego_v, dt)
    teacher_dynamics = _trajectory_dynamics(transport_target, ego_v, dt)
    gt_dynamics = _trajectory_dynamics(gt_target, ego_v, dt)
    pred_progress = _trajectory_progress(poses)
    teacher_progress = _trajectory_progress(transport_target)
    gt_progress = _trajectory_progress(gt_target)
    constraint_mask = constraint_mask.float().clamp(0.0, 1.0)
    constraint_denom = constraint_mask.sum(dim=-1).clamp(min=1.0)

    progress_loss = F.smooth_l1_loss(pred_progress, teacher_progress, reduction="none").mean(dim=-1)
    terminal_loss = F.smooth_l1_loss(pred_progress[:, -1], teacher_progress[:, -1], reduction="none")
    safety_excess = (pred_progress - upper_s).clamp_min(0.0)
    safety_loss = (safety_excess.square() * constraint_mask).sum(dim=-1) / constraint_denom
    acceleration_loss = F.smooth_l1_loss(
        pred_dynamics["acceleration"], teacher_dynamics["acceleration"], reduction="none"
    ).mean(dim=-1)
    jerk_loss = F.smooth_l1_loss(pred_dynamics["jerk"], teacher_dynamics["jerk"], reduction="none").mean(dim=-1)
    raw_loss = (
        float(_cfg(config, "transport_progress_weight", 1.0)) * progress_loss
        + float(_cfg(config, "transport_terminal_weight", 1.0)) * terminal_loss
        + float(_cfg(config, "transport_safety_weight", 2.0)) * safety_loss
        + float(_cfg(config, "transport_acceleration_weight", 0.10)) * acceleration_loss
        + float(_cfg(config, "transport_jerk_weight", 0.05)) * jerk_loss
    )

    current_ttc, delayed_ttc, t1, continuous_steps = context[:, 1], context[:, 2], context[:, 3], context[:, 4]
    preparation_time = float(_cfg(config, "brake_timing_preparation_time", 1.0))
    pre_risk = (continuous_steps >= 2.0) & (torch.minimum(current_ttc, delayed_ttc) <= t1 + preparation_time)
    teacher_onset = _sustained_brake_onset(teacher_dynamics["acceleration"], threshold)
    pred_onset = _sustained_brake_onset(pred_dynamics["acceleration"], threshold)
    teacher_has_sustained_brake = teacher_onset < teacher_dynamics["acceleration"].shape[-1]
    active = transport_valid.reshape(-1) > 0.5
    onset_error = (pred_onset - teacher_onset).float() * dt
    pred_oscillation = (
        (pred_dynamics["acceleration"][:, 1:] * pred_dynamics["acceleration"][:, :-1] < 0.0)
        & (pred_dynamics["acceleration"][:, 1:].abs() > 0.2)
        & (pred_dynamics["acceleration"][:, :-1].abs() > 0.2)
    ).float().mean(dim=-1)

    return {
        "pre_risk": pre_risk,
        "gt_has_sustained_brake": teacher_has_sustained_brake,
        "active": active,
        "raw_loss": raw_loss,
        "gt_onset_s": teacher_onset.float() * dt,
        "pred_onset_s": pred_onset.float() * dt,
        "onset_error_s": onset_error,
        "onset_abs_error_s": onset_error.abs(),
        "late_rate": (onset_error > 0.25).float(),
        "early_rate": (onset_error < -0.25).float(),
        "speed_mae": (pred_dynamics["speed"] - teacher_dynamics["speed"]).abs().mean(dim=-1),
        "pred_min_accel": pred_dynamics["acceleration"].min(dim=-1).values,
        "gt_min_accel": teacher_dynamics["acceleration"].min(dim=-1).values,
        "pred_jerk_abs": pred_dynamics["jerk"].abs().mean(dim=-1),
        "gt_jerk_abs": teacher_dynamics["jerk"].abs().mean(dim=-1),
        "pred_accel_oscillation": pred_oscillation,
        "current_ttc": current_ttc,
        "delayed_ttc": delayed_ttc,
        "continuous_front_steps": continuous_steps,
        "transport_progress_mae": (pred_progress - teacher_progress).abs().mean(dim=-1),
        "transport_terminal_error": (pred_progress[:, -1] - teacher_progress[:, -1]).abs(),
        "transport_safety_violation": safety_excess.max(dim=-1).values,
        "transport_constraint_steps": constraint_mask.sum(dim=-1),
        "transport_teacher_shift": (teacher_progress - gt_progress).abs().max(dim=-1).values,
    }


def _summarize_observations(prefix: str, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Expose both live batch means and sufficient statistics for exact epoch metrics."""

    active = values["active"]
    pre_risk = values["pre_risk"]
    gt_has_sustained_brake = values["gt_has_sustained_brake"]
    if active.ndim > 1:
        scene_active = active.any(dim=-1)
        scene_pre_risk = pre_risk.any(dim=-1)
        scene_gt_brake = gt_has_sustained_brake.any(dim=-1)
        scene_count = float(active.shape[0])
    else:
        scene_active = active
        scene_pre_risk = pre_risk
        scene_gt_brake = gt_has_sustained_brake
        scene_count = float(active.numel())
    result = {
        f"{prefix}_scene_count": active.new_tensor(scene_count, dtype=torch.float32),
        f"{prefix}_pre_risk_count": scene_pre_risk.float().sum().detach(),
        f"{prefix}_gt_brake_count": scene_gt_brake.float().sum().detach(),
        f"{prefix}_active_count": scene_active.float().sum().detach(),
        f"{prefix}_pre_risk_rate": scene_pre_risk.float().mean().detach(),
        f"{prefix}_gt_brake_rate": scene_gt_brake.float().mean().detach(),
        f"{prefix}_active_rate": scene_active.float().mean().detach(),
    }
    if active.ndim > 1:
        result.update(
            {
                f"{prefix}_mode_count": active.new_tensor(float(active.numel()), dtype=torch.float32),
                f"{prefix}_active_mode_count": active.float().sum().detach(),
                f"{prefix}_active_mode_rate": active.float().mean().detach(),
            }
        )
    for name, per_scene_value in values.items():
        if name in {"pre_risk", "gt_has_sustained_brake", "active"}:
            continue
        result[f"{prefix}_{name}"] = _masked_mean(per_scene_value, active).detach()
        result[f"{prefix}_{name}_sum"] = _masked_sum(per_scene_value, active).detach()
    return result


def compute_brake_timing_loss(
    poses: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    plan_anchor: torch.Tensor,
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Supervise longitudinal timing with the configured matched- or all-mode objective."""

    required = ("trajectory", "brake_timing_context")
    if not all(key in targets for key in required):
        zero = poses.sum() * 0.0
        return {"brake_timing_loss": zero}

    poses = torch.nan_to_num(poses.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(targets["trajectory"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    context = torch.nan_to_num(targets["brake_timing_context"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    all_mode_keys = (
        "temporal_transport_front_boxes",
        "temporal_transport_front_mask",
    )
    if getattr(config, "use_all_mode_risk_corridor", False) and all(
        key in targets for key in all_mode_keys
    ):
        observations = _all_mode_risk_corridor_observations(
            poses,
            target,
            context,
            torch.nan_to_num(
                targets["temporal_transport_front_boxes"].to(poses.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_front_mask"].to(poses.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            config,
        )
    else:
        anchors = plan_anchor.to(poses.device).float()
        anchor_distance = torch.linalg.norm(target[:, None, :, :2] - anchors, dim=-1).mean(dim=-1)
        matched_mode = anchor_distance.argmin(dim=-1)
        gather_index = matched_mode[:, None, None, None].expand(
            -1, 1, poses.shape[-2], poses.shape[-1]
        )
        matched_poses = torch.gather(poses, 1, gather_index).squeeze(1)
        observations = _matched_mode_observations(
            matched_poses,
            target,
            context,
            targets,
            config,
        )

    raw_loss = _masked_mean(observations["raw_loss"], observations["active"])
    total_loss = float(_cfg(config, "brake_timing_loss_weight", 0.1)) * raw_loss
    result = _summarize_observations("brake_timing", observations)
    result.update(
        {
            "brake_timing_loss": total_loss,
            "brake_timing_loss_sum": (
                float(_cfg(config, "brake_timing_loss_weight", 0.1))
                * _masked_sum(observations["raw_loss"], observations["active"])
            ).detach(),
        }
    )
    return result


def _matched_mode_observations(
    matched_poses: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Retain the prior matched-mode objectives for reproducible ablations."""

    transport_keys = (
        "temporal_transport_target",
        "temporal_transport_upper_s",
        "temporal_transport_constraint_mask",
        "temporal_transport_valid",
    )
    if all(key in targets for key in transport_keys):
        device = matched_poses.device
        observations = _temporal_transport_observations(
            matched_poses,
            target,
            torch.nan_to_num(
                targets["temporal_transport_target"].to(device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            context,
            torch.nan_to_num(
                targets["temporal_transport_upper_s"].to(device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_constraint_mask"].to(device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_valid"].to(device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            config,
        )
        return observations
    return _brake_timing_observations(matched_poses, target, context, config)


def compute_brake_timing_diagnostics(
    trajectory: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    config: Any,
    prefix: str = "brake_timing_selected",
) -> Dict[str, torch.Tensor]:
    """Measure timing on the trajectory actually selected at validation/inference time."""

    if "trajectory" not in targets or "brake_timing_context" not in targets:
        return {}
    trajectory = torch.nan_to_num(trajectory.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(
        targets["trajectory"].to(trajectory.device).float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    context = torch.nan_to_num(
        targets["brake_timing_context"].to(trajectory.device).float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    all_mode_keys = (
        "temporal_transport_front_boxes",
        "temporal_transport_front_mask",
    )
    if getattr(config, "use_all_mode_risk_corridor", False) and all(
        key in targets for key in all_mode_keys
    ):
        observations = _all_mode_risk_corridor_observations(
            trajectory,
            target,
            context,
            torch.nan_to_num(
                targets["temporal_transport_front_boxes"].to(trajectory.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_front_mask"].to(trajectory.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            config,
        )
        return _summarize_observations(prefix, observations)

    transport_keys = (
        "temporal_transport_target",
        "temporal_transport_upper_s",
        "temporal_transport_constraint_mask",
        "temporal_transport_valid",
    )
    if all(key in targets for key in transport_keys):
        observations = _temporal_transport_observations(
            trajectory,
            target,
            torch.nan_to_num(
                targets["temporal_transport_target"].to(trajectory.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            context,
            torch.nan_to_num(
                targets["temporal_transport_upper_s"].to(trajectory.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_constraint_mask"].to(trajectory.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_valid"].to(trajectory.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            config,
        )
    else:
        observations = _brake_timing_observations(trajectory, target, context, config)
    return _summarize_observations(prefix, observations)
