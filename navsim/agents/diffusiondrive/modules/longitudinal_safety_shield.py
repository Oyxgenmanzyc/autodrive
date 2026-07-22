"""Inference-only longitudinal safety shield.

The shield never changes the lateral path or candidate mode. It only moves the
selected trajectory backward along its own polyline when an inference-observable,
high-confidence front-vehicle estimate predicts insufficient longitudinal
clearance. The raw trajectory is retained as an exact counterfactual.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from navsim.agents.diffusiondrive.modules.risk_shadow import (
    _delayed_time_risk,
    _estimate_front_state,
)


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _has_consecutive_true(
    mask: torch.Tensor,
    required_steps: int,
) -> torch.Tensor:
    """Return whether every sample has a persistent unsafe interval."""

    required_steps = max(int(required_steps), 1)

    if required_steps == 1:
        return mask.any(dim=-1)

    if mask.shape[-1] < required_steps:
        return torch.zeros(
            mask.shape[0],
            dtype=torch.bool,
            device=mask.device,
        )

    windows = mask.unfold(
        dimension=-1,
        size=required_steps,
        step=1,
    )

    return windows.all(dim=-1).any(dim=-1)


def _polyline_arclength(
    xy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate segment length and cumulative path length."""

    origin = torch.zeros_like(xy[:, :1])
    points = torch.cat([origin, xy], dim=1)

    segment = points[:, 1:] - points[:, :-1]
    segment_length = torch.linalg.vector_norm(segment, dim=-1)

    return segment_length, segment_length.cumsum(dim=-1)


def _interpolate_heading(
    heading_start: torch.Tensor,
    heading_end: torch.Tensor,
    ratio: torch.Tensor,
) -> torch.Tensor:
    """Interpolate heading through the shortest angular distance."""

    delta = torch.atan2(
        (heading_end - heading_start).sin(),
        (heading_end - heading_start).cos(),
    )

    return heading_start + ratio * delta


def _sample_along_polyline(
    trajectory: torch.Tensor,
    target_distance: torch.Tensor,
) -> torch.Tensor:
    """Move poses backward along the original polyline without changing its path."""

    batch_size, num_poses, _ = trajectory.shape

    xy = trajectory[..., :2]
    heading = trajectory[..., 2]

    _, cumulative_distance = _polyline_arclength(xy)

    output = trajectory.clone()

    for batch_index in range(batch_size):
        origin_xy = torch.zeros_like(xy[batch_index : batch_index + 1, :1]).squeeze(0)

        points = torch.cat(
            [
                origin_xy,
                xy[batch_index],
            ],
            dim=0,
        )

        headings = torch.cat(
            [
                heading.new_zeros(1),
                heading[batch_index],
            ],
            dim=0,
        )

        cumulative = torch.cat(
            [
                cumulative_distance.new_zeros(1),
                cumulative_distance[batch_index],
            ],
            dim=0,
        )

        query = target_distance[batch_index].clamp(
            min=0.0,
            max=cumulative[-1],
        )

        upper = torch.searchsorted(
            cumulative,
            query,
            right=False,
        ).clamp(
            min=1,
            max=num_poses,
        )

        lower = upper - 1

        lower_distance = cumulative[lower]
        upper_distance = cumulative[upper]

        ratio = (
            (query - lower_distance)
            / (upper_distance - lower_distance).clamp(min=1e-6)
        ).clamp(
            min=0.0,
            max=1.0,
        )

        output[batch_index, :, :2] = (
            points[lower]
            + ratio[:, None] * (points[upper] - points[lower])
        )

        output[batch_index, :, 2] = _interpolate_heading(
            headings[lower],
            headings[upper],
            ratio,
        )

    return output


def _constant_deceleration_distance(
    ego_velocity: torch.Tensor,
    deceleration: torch.Tensor,
    delay: torch.Tensor,
    num_poses: int,
    dt: float,
) -> torch.Tensor:
    """Generate a smooth constant-deceleration longitudinal distance profile."""

    batch_size = ego_velocity.shape[0]

    speed = ego_velocity.clamp_min(0.0)
    distance = torch.zeros_like(speed)

    distance_sequence = []

    for step in range(num_poses):
        step_start_time = step * dt

        braking_active = step_start_time >= delay

        next_speed = torch.where(
            braking_active,
            (speed - deceleration * dt).clamp_min(0.0),
            speed,
        )

        distance = distance + 0.5 * (speed + next_speed) * dt
        distance_sequence.append(distance)

        speed = next_speed

    return torch.stack(
        distance_sequence,
        dim=-1,
    ).reshape(
        batch_size,
        num_poses,
    )


@torch.no_grad()
def apply_longitudinal_safety_shield(
    trajectory: torch.Tensor,
    history_risk_tokens: Optional[torch.Tensor],
    agent_states: Optional[torch.Tensor],
    agent_labels: Optional[torch.Tensor],
    config: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply a path-preserving longitudinal safety correction."""

    raw_trajectory = torch.nan_to_num(
        trajectory.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    batch_size, num_poses, _ = raw_trajectory.shape

    zeros = raw_trajectory.new_zeros(batch_size)

    diagnostics: Dict[str, torch.Tensor] = {
        "shield_active": zeros,
        "shield_trigger": zeros,
        "shield_reliable": zeros,
        "shield_emergency": zeros,
        "shield_progress_before": zeros,
        "shield_progress_after": zeros,
        "shield_progress_reduction": zeros,
        "shield_min_clearance_before": zeros,
        "shield_min_clearance_after": zeros,
        "shield_clearance_gain": zeros,
        "shield_deceleration": zeros,
    }

    if history_risk_tokens is None:
        return raw_trajectory, diagnostics

    front_state = _estimate_front_state(
        history_risk_tokens,
        agent_states,
        agent_labels,
        config,
    )

    time_risk = _delayed_time_risk(
        front_state,
        config,
    )

    dt = max(
        float(_cfg(config, "risk_history_dt", 0.5)),
        1e-3,
    )

    # source:
    # 2 = detector only
    # 3 = LiDAR and detector agree
    fused_front = front_state["source"] == 3.0

    high_confidence_detector = (
        (front_state["source"] == 2.0)
        & (
            front_state["agent_confidence"]
            >= float(
                _cfg(
                    config,
                    "shield_agent_only_confidence",
                    0.75,
                )
            )
        )
    )

    reliable_front = front_state["valid"] & (
        (
            fused_front
            & (
                front_state["reliability"]
                >= float(
                    _cfg(
                        config,
                        "shield_min_reliability",
                        0.50,
                    )
                )
            )
        )
        | high_confidence_detector
    )

    # Predict the longitudinal position of the front vehicle.
    lead_velocity = front_state["lead_v"].clamp_min(0.0)
    lead_displacement = torch.zeros_like(lead_velocity)

    future_front_gap = []

    for _ in range(num_poses):
        next_lead_velocity = (
            lead_velocity
            + front_state["lead_a"] * dt
        ).clamp(
            min=0.0,
            max=float(
                _cfg(
                    config,
                    "risk_shadow_max_lead_speed",
                    40.0,
                )
            ),
        )

        lead_displacement = (
            lead_displacement
            + 0.5
            * (lead_velocity + next_lead_velocity)
            * dt
        )

        future_front_gap.append(
            front_state["gap"] + lead_displacement
        )

        lead_velocity = next_lead_velocity

    future_front_gap = torch.stack(
        future_front_gap,
        dim=-1,
    )

    # Only handle longitudinally overlapping trajectories.
    lateral_limit = (
        0.5
        * (
            float(
                _cfg(
                    config,
                    "risk_shadow_ego_width",
                    2.0,
                )
            )
            + front_state["lead_width"][:, None]
        )
        + float(
            _cfg(
                config,
                "risk_shadow_lateral_margin",
                0.3,
            )
        )
    )

    lateral_overlap = (
        raw_trajectory[..., 1]
        - front_state["lead_y"][:, None]
    ).abs() <= lateral_limit

    clearance_cap = float(
        _cfg(
            config,
            "risk_shadow_clearance_cap",
            40.0,
        )
    )

    clearance_before_steps = (
        future_front_gap
        - raw_trajectory[..., 0]
    )

    minimum_clearance_before = torch.where(
        lateral_overlap,
        clearance_before_steps,
        torch.full_like(
            clearance_before_steps,
            clearance_cap,
        ),
    ).min(
        dim=-1,
    ).values

    speed_based_gap = (
        float(
            _cfg(
                config,
                "shield_time_headway",
                0.75,
            )
        )
        * front_state["ego_v"]
    )

    safe_gap = torch.maximum(
        raw_trajectory.new_full(
            (batch_size,),
            float(
                _cfg(
                    config,
                    "shield_min_gap",
                    2.5,
                )
            ),
        ),
        speed_based_gap,
    ).clamp(
        max=float(
            _cfg(
                config,
                "shield_max_gap",
                8.0,
            )
        )
    )

    minimum_unsafe_steps = max(
        int(
            _cfg(
                config,
                "shield_min_consecutive_unsafe_steps",
                2,
            )
        ),
        1,
    )

    unsafe_before = (
        lateral_overlap
        & (
            clearance_before_steps
            < safe_gap[:, None]
        )
    )

    persistent_unsafe_before = _has_consecutive_true(
        unsafe_before,
        minimum_unsafe_steps,
    )

    hard_clearance = float(
        _cfg(
            config,
            "shield_hard_clearance",
            1.5,
        )
    )

    clearance_trigger_margin = float(
        _cfg(
            config,
            "shield_clearance_trigger_margin",
            0.25,
        )
    )

    # 只放宽进入候选修正阶段的阈值：
    # 默认由1.50m放宽到1.75m。
    near_hard_clearance = (
        minimum_clearance_before
        < (
            hard_clearance
            + clearance_trigger_margin
        )
    )

    risk_signal = (
        time_risk["trigger"]
        | time_risk["emergency"]
        | near_hard_clearance
    )

    # 即使是emergency，也必须确认原轨迹存在持续危险。
    trigger = (
        reliable_front
        & lateral_overlap.any(dim=-1)
        & persistent_unsafe_before
        & risk_signal
    )

    relative_velocity = (
        front_state["ego_v"]
        - front_state["lead_v"]
    ).clamp_min(0.0)

    available_gap = (
        front_state["gap"]
        - safe_gap
    ).clamp_min(0.5)

    kinematic_deceleration = (
        relative_velocity.square()
        / (2.0 * available_gap)
    )

    required_deceleration = torch.maximum(
        kinematic_deceleration,
        time_risk["delayed_drac"],
    )

    required_deceleration = (
        required_deceleration
        + float(
            _cfg(
                config,
                "shield_decel_margin",
                0.35,
            )
        )
    ).clamp(
        min=float(
            _cfg(
                config,
                "shield_min_decel",
                0.8,
            )
        ),
        max=float(
            _cfg(
                config,
                "shield_max_decel",
                3.5,
            )
        ),
    )

    brake_delay = torch.where(
        time_risk["emergency"],
        torch.zeros_like(required_deceleration),
        torch.full_like(
            required_deceleration,
            float(
                _cfg(
                    config,
                    "shield_brake_delay",
                    0.5,
                )
            ),
        ),
    )

    _, raw_path_distance = _polyline_arclength(
        raw_trajectory[..., :2]
    )

    braking_path_distance = _constant_deceleration_distance(
        front_state["ego_v"],
        required_deceleration,
        brake_delay,
        num_poses,
        dt,
    )

    target_path_distance = torch.minimum(
        raw_path_distance,
        braking_path_distance,
    )

    target_path_distance = torch.cummax(
        target_path_distance,
        dim=-1,
    ).values

    candidate_trajectory = _sample_along_polyline(
        raw_trajectory,
        target_path_distance,
    )

    clearance_after_steps = (
        future_front_gap
        - candidate_trajectory[..., 0]
    )

    candidate_lateral_overlap = (
        candidate_trajectory[..., 1]
        - front_state["lead_y"][:, None]
    ).abs() <= lateral_limit

    minimum_clearance_after = torch.where(
        candidate_lateral_overlap,
        clearance_after_steps,
        torch.full_like(
            clearance_after_steps,
            clearance_cap,
        ),
    ).min(
        dim=-1,
    ).values

    after_safe_gap_ratio = float(
        _cfg(
            config,
            "shield_after_safe_gap_ratio",
            0.75,
        )
    )

    required_clearance_after = torch.maximum(
        raw_trajectory.new_full(
            (batch_size,),
            hard_clearance,
        ),
        safe_gap * after_safe_gap_ratio,
    )

    unsafe_after = (
        candidate_lateral_overlap
        & (
            clearance_after_steps
            < required_clearance_after[:, None]
        )
    )

    persistent_unsafe_after = _has_consecutive_true(
        unsafe_after,
        minimum_unsafe_steps,
    )

    # 修正后不得继续存在连续危险，并且最小间距需要达到要求。
    safety_resolved = (
        (~persistent_unsafe_after)
        & (
            minimum_clearance_after
            >= required_clearance_after
        )
    )

    clearance_gain = (
        minimum_clearance_after
        - minimum_clearance_before
    )

    progress_before = raw_path_distance[:, -1]

    _, candidate_path_distance = _polyline_arclength(
        candidate_trajectory[..., :2]
    )

    progress_after = candidate_path_distance[:, -1]

    progress_reduction = (
        progress_before
        - progress_after
    ).clamp_min(0.0)

    normal_progress_limit = float(
        _cfg(
            config,
            "shield_accept_progress_reduction",
            4.0,
        )
    )

    absolute_progress_limit = float(
        _cfg(
            config,
            "shield_absolute_max_progress_reduction",
            5.0,
        )
    )

    absolute_progress_ok = (
        progress_reduction
        <= absolute_progress_limit
    )

    normal_progress_ok = (
        time_risk["emergency"]
        | (
            progress_reduction
            <= normal_progress_limit
        )
    )

    progress_limit_ok = (
        absolute_progress_ok
        & normal_progress_ok
    )

    accept = (
        trigger
        & safety_resolved
        & (
            clearance_gain
            >= float(
                _cfg(
                    config,
                    "shield_min_clearance_gain",
                    0.75,
                )
            )
        )
        & progress_limit_ok
    )

    shielded_trajectory = torch.where(
        accept[:, None, None],
        candidate_trajectory,
        raw_trajectory,
    )

    diagnostics.update(
        {
            "shield_active": accept.float(),
            "shield_trigger": trigger.float(),
            "shield_reliable": reliable_front.float(),
            "shield_emergency": time_risk["emergency"].float(),
            "shield_front_source": front_state["source"],
            "shield_front_gap": front_state["gap"],
            "shield_front_ttc": time_risk["ttc"],
            "shield_front_delayed_ttc": time_risk["delayed_ttc"],
            "shield_front_drac": time_risk["drac"],
            "shield_front_delayed_drac": time_risk["delayed_drac"],
            "shield_progress_before": progress_before,
            "shield_progress_after": torch.where(
                accept,
                progress_after,
                progress_before,
            ),
            "shield_progress_reduction": torch.where(
                accept,
                progress_reduction,
                zeros,
            ),
            "shield_min_clearance_before": minimum_clearance_before,
            "shield_min_clearance_after": torch.where(
                accept,
                minimum_clearance_after,
                minimum_clearance_before,
            ),
            "shield_clearance_gain": torch.where(
                accept,
                clearance_gain,
                zeros,
            ),
            "shield_deceleration": torch.where(
                accept,
                required_deceleration,
                zeros,
            ),
        }
    )

    return shielded_trajectory, diagnostics
