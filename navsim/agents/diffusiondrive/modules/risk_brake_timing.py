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
    previous_speed = torch.cat([ego_v[:, None], speed[..., :-1]], dim=-1)
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
    result = {
        f"{prefix}_scene_count": active.new_tensor(float(active.numel()), dtype=torch.float32),
        f"{prefix}_pre_risk_count": pre_risk.float().sum().detach(),
        f"{prefix}_gt_brake_count": gt_has_sustained_brake.float().sum().detach(),
        f"{prefix}_active_count": active.float().sum().detach(),
        f"{prefix}_pre_risk_rate": pre_risk.float().mean().detach(),
        f"{prefix}_gt_brake_rate": gt_has_sustained_brake.float().mean().detach(),
        f"{prefix}_active_rate": active.float().mean().detach(),
    }
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
    """Supervise the GT-matched mode's longitudinal timing in pre-risk scenes."""

    required = ("trajectory", "brake_timing_context")
    if not all(key in targets for key in required):
        zero = poses.sum() * 0.0
        return {"brake_timing_loss": zero}

    poses = torch.nan_to_num(poses.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(targets["trajectory"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    context = torch.nan_to_num(targets["brake_timing_context"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    anchors = plan_anchor.to(poses.device).float()
    anchor_distance = torch.linalg.norm(target[:, None, :, :2] - anchors, dim=-1).mean(dim=-1)
    matched_mode = anchor_distance.argmin(dim=-1)
    gather_index = matched_mode[:, None, None, None].expand(-1, 1, poses.shape[-2], poses.shape[-1])
    matched_poses = torch.gather(poses, 1, gather_index).squeeze(1)
    transport_keys = (
        "temporal_transport_target",
        "temporal_transport_upper_s",
        "temporal_transport_constraint_mask",
        "temporal_transport_valid",
    )
    if all(key in targets for key in transport_keys):
        observations = _temporal_transport_observations(
            matched_poses,
            target,
            torch.nan_to_num(
                targets["temporal_transport_target"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0
            ),
            context,
            torch.nan_to_num(
                targets["temporal_transport_upper_s"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0
            ),
            torch.nan_to_num(
                targets["temporal_transport_constraint_mask"].to(poses.device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.nan_to_num(
                targets["temporal_transport_valid"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0
            ),
            config,
        )
    else:
        observations = _brake_timing_observations(matched_poses, target, context, config)
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
