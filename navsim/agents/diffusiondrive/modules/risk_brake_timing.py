from typing import Any, Dict

import torch
import torch.nn.functional as F


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _trajectory_dynamics(poses: torch.Tensor, ego_v: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
    """Compute finite speed, acceleration, and jerk from trajectory positions."""

    origin = torch.zeros_like(poses[..., :1, :2])
    displacement = torch.diff(torch.cat([origin, poses[..., :2]], dim=-2), dim=-2)
    speed = torch.linalg.norm(displacement, dim=-1) / dt
    previous_speed = torch.cat([ego_v[:, None], speed[..., :-1]], dim=-1)
    acceleration = (speed - previous_speed) / dt
    jerk = torch.diff(acceleration, dim=-1) / dt
    return {"speed": speed, "acceleration": acceleration, "jerk": jerk}


def _sustained_brake_onset(acceleration: torch.Tensor, threshold: float) -> torch.Tensor:
    braking = acceleration <= threshold
    sustained = braking[..., :-1] & braking[..., 1:]
    indices = torch.arange(sustained.shape[-1], device=acceleration.device)
    return torch.where(sustained, indices, acceleration.shape[-1]).min(dim=-1).values


def compute_brake_timing_loss(
    poses: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    plan_anchor: torch.Tensor,
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Supervise the GT-matched mode's soft brake onset in pre-risk following scenes."""

    required = ("trajectory", "brake_timing_context")
    if not all(key in targets for key in required):
        zero = poses.sum() * 0.0
        return {"brake_timing_loss": zero}

    poses = torch.nan_to_num(poses.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(targets["trajectory"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    context = torch.nan_to_num(targets["brake_timing_context"].to(poses.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    anchors = plan_anchor.to(poses.device).float()
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    threshold = float(_cfg(config, "brake_timing_accel_threshold", -0.5))
    temperature = max(float(_cfg(config, "brake_timing_temperature", 0.35)), 1e-3)

    anchor_distance = torch.linalg.norm(target[:, None, :, :2] - anchors, dim=-1).mean(dim=-1)
    matched_mode = anchor_distance.argmin(dim=-1)
    gather_index = matched_mode[:, None, None, None].expand(-1, 1, poses.shape[-2], poses.shape[-1])
    matched_poses = torch.gather(poses, 1, gather_index).squeeze(1)

    ego_v = context[:, 0].clamp(min=0.0)
    pred_dynamics = _trajectory_dynamics(matched_poses, ego_v, dt)
    gt_dynamics = _trajectory_dynamics(target, ego_v, dt)
    pred_acceleration = pred_dynamics["acceleration"]
    gt_acceleration = gt_dynamics["acceleration"]

    pred_brake = torch.sigmoid((threshold - pred_acceleration) / temperature)
    gt_brake = torch.sigmoid((threshold - gt_acceleration) / temperature)
    pred_onset_curve = torch.cummax(pred_brake, dim=-1).values
    gt_onset_curve = torch.cummax(gt_brake, dim=-1).values

    profile_loss = F.smooth_l1_loss(pred_brake, gt_brake, reduction="none").mean(dim=-1)
    onset_loss = F.smooth_l1_loss(pred_onset_curve, gt_onset_curve, reduction="none").mean(dim=-1)
    profile_weight = float(_cfg(config, "brake_timing_profile_weight", 0.25))
    per_scene_loss = profile_weight * profile_loss + (1.0 - profile_weight) * onset_loss

    current_ttc, delayed_ttc, t1, continuous_steps = context[:, 1], context[:, 2], context[:, 3], context[:, 4]
    preparation_time = float(_cfg(config, "brake_timing_preparation_time", 1.0))
    pre_risk = (continuous_steps >= 2.0) & (torch.minimum(current_ttc, delayed_ttc) <= t1 + preparation_time)
    gt_onset = _sustained_brake_onset(gt_acceleration, threshold)
    pred_onset = _sustained_brake_onset(pred_acceleration, threshold)
    gt_has_sustained_brake = gt_onset < gt_acceleration.shape[-1]
    active = pre_risk & gt_has_sustained_brake

    raw_loss = _masked_mean(per_scene_loss, active)
    total_loss = float(_cfg(config, "brake_timing_loss_weight", 0.1)) * raw_loss
    onset_error = (pred_onset - gt_onset).float() * dt
    speed_mae = (pred_dynamics["speed"] - gt_dynamics["speed"]).abs().mean(dim=-1)
    pred_jerk = pred_dynamics["jerk"].abs().mean(dim=-1)
    gt_jerk = gt_dynamics["jerk"].abs().mean(dim=-1)
    pred_oscillation = (
        (pred_acceleration[:, 1:] * pred_acceleration[:, :-1] < 0.0)
        & (pred_acceleration[:, 1:].abs() > 0.2)
        & (pred_acceleration[:, :-1].abs() > 0.2)
    ).float().mean(dim=-1)

    return {
        "brake_timing_loss": total_loss,
        "brake_timing_raw_loss": raw_loss.detach(),
        "brake_timing_pre_risk_rate": pre_risk.float().mean().detach(),
        "brake_timing_gt_brake_rate": gt_has_sustained_brake.float().mean().detach(),
        "brake_timing_active_rate": active.float().mean().detach(),
        "brake_timing_gt_onset_s": _masked_mean(gt_onset.float() * dt, active).detach(),
        "brake_timing_pred_onset_s": _masked_mean(pred_onset.float() * dt, active).detach(),
        "brake_timing_onset_error_s": _masked_mean(onset_error, active).detach(),
        "brake_timing_onset_abs_error_s": _masked_mean(onset_error.abs(), active).detach(),
        "brake_timing_late_rate": _masked_mean((onset_error > 0.25).float(), active).detach(),
        "brake_timing_early_rate": _masked_mean((onset_error < -0.25).float(), active).detach(),
        "brake_timing_speed_mae": _masked_mean(speed_mae, active).detach(),
        "brake_timing_pred_min_accel": _masked_mean(pred_acceleration.min(dim=-1).values, active).detach(),
        "brake_timing_gt_min_accel": _masked_mean(gt_acceleration.min(dim=-1).values, active).detach(),
        "brake_timing_pred_jerk_abs": _masked_mean(pred_jerk, active).detach(),
        "brake_timing_gt_jerk_abs": _masked_mean(gt_jerk, active).detach(),
        "brake_timing_pred_accel_oscillation": _masked_mean(pred_oscillation, active).detach(),
        "brake_timing_current_ttc": _masked_mean(current_ttc, active).detach(),
        "brake_timing_delayed_ttc": _masked_mean(delayed_ttc, active).detach(),
        "brake_timing_continuous_front_steps": _masked_mean(continuous_steps, active).detach(),
    }
