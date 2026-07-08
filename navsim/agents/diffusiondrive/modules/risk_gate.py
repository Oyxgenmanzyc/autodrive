from typing import Any

import torch


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def select_risk_gated_mode(
    poses_reg: torch.Tensor,
    poses_cls: torch.Tensor,
    history_risk_tokens: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Select a mode using risk safety sets before confidence ranking."""

    raw_best = poses_cls.argmax(dim=-1)
    if history_risk_tokens is None:
        return raw_best

    current_token = history_risk_tokens[:, -1]
    valid = current_token[:, -1] > 0.5
    if valid.sum().item() == 0:
        return raw_best

    gap = current_token[:, 0].clamp(min=0.0)
    ttc = current_token[:, 6]
    drac = current_token[:, 7].clamp(min=0.0)
    dt = float(_cfg(config, "risk_history_dt", 0.5))
    min_gap = float(_cfg(config, "risk_gate_min_gap", 1.0))
    cls_margin = float(_cfg(config, "risk_gate_cls_margin", 2.0))

    x_steps = poses_reg[..., 0]
    start_x = torch.zeros_like(x_steps[..., :1])
    x_with_start = torch.cat([start_x, x_steps], dim=-1)
    speed = (x_with_start[..., 1:] - x_with_start[..., :-1]) / max(dt, 1e-3)
    start_speed = speed[..., :1]
    speed_with_start = torch.cat([start_speed, speed], dim=-1)
    acceleration = (speed_with_start[..., 1:] - speed_with_start[..., :-1]) / max(dt, 1e-3)

    min_clearance = gap[:, None] - x_steps.max(dim=-1).values
    collision_like = min_clearance < min_gap
    min_accel = acceleration.min(dim=-1).values
    required_accel = -drac[:, None]

    in_t1 = (ttc[:, None] <= 2.5) | (drac[:, None] >= 2.0)
    in_t2 = (ttc[:, None] <= 1.5) | (drac[:, None] >= 4.0)
    has_response = min_accel <= torch.minimum(required_accel * 0.5, torch.full_like(required_accel, -0.5))
    has_strong_response = min_accel <= torch.minimum(required_accel * 0.8, torch.full_like(required_accel, -1.0))

    safe = (~collision_like) & ((~in_t1) | has_response) & (drac[:, None] < 4.0)
    warning = (~collision_like) & ((~in_t2) | has_strong_response)
    dangerous = ~(safe | warning)

    selected = raw_best.clone()
    for batch_idx in range(poses_reg.shape[0]):
        if not valid[batch_idx]:
            continue

        safe_idx = torch.where(safe[batch_idx])[0]
        if safe_idx.numel() > 0:
            selected[batch_idx] = safe_idx[poses_cls[batch_idx, safe_idx].argmax()]
            continue

        warning_idx = torch.where(warning[batch_idx])[0]
        if warning_idx.numel() > 0:
            max_cls = poses_cls[batch_idx].max()
            eligible = warning_idx[poses_cls[batch_idx, warning_idx] >= max_cls - cls_margin]
            if eligible.numel() == 0:
                eligible = warning_idx
            risk_value = collision_like[batch_idx, eligible].float() * 100.0 + drac[batch_idx]
            selected[batch_idx] = eligible[risk_value.argmin()]
            continue

        danger_idx = torch.where(dangerous[batch_idx])[0]
        if danger_idx.numel() > 0:
            risk_value = collision_like[batch_idx, danger_idx].float() * 100.0 + min_clearance[batch_idx, danger_idx].neg()
            selected[batch_idx] = danger_idx[risk_value.argmin()]

    return selected
