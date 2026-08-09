from typing import Any, Dict, List, Optional

import numpy as np

from navsim.common.enums import LidarIndex


RISK_TOKEN_DIM = 10
RISK_TOKEN_FIELDS = (
    "gap",
    "signed_rel_v",
    "closing_speed",
    "ego_v",
    "lead_v",
    "ego_a",
    "thw",
    "ttc",
    "drac",
    "valid",
)

RISK_LABEL_NAMES = (
    "ttc_worsening",
    "drac_worsening",
    "urgency",
)

RISK_COLLISION_OBJECT_NAMES = frozenset(
    {"vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "generic_object"}
)


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def compute_longitudinal_risk(
    gap: float,
    rel_v: float,
    ego_v: float,
    eps: float = 1e-3,
    ttc_max: float = 10.0,
    drac_max: float = 6.0,
) -> Dict[str, float]:
    """Compute THW/TTC/DRAC for a longitudinal following proxy."""

    gap = max(float(np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)), eps)
    rel_v = float(np.nan_to_num(rel_v, nan=0.0, posinf=0.0, neginf=0.0))
    ego_v = max(float(np.nan_to_num(ego_v, nan=0.0, posinf=0.0, neginf=0.0)), 0.0)

    thw = min(gap / max(ego_v, eps), ttc_max)
    if rel_v > 0.0:
        ttc = min(gap / max(rel_v, eps), ttc_max)
        drac = min((rel_v * rel_v) / (2.0 * gap), drac_max)
    else:
        ttc = ttc_max
        drac = 0.0

    return {
        "thw": float(thw),
        "ttc": float(ttc),
        "drac": float(drac),
        "a_required": -float(drac),
    }


def tm_brake_thresholds(speed: float) -> Dict[str, float]:
    """Zhu-style braking time thresholds for longitudinal collision avoidance."""

    speed = float(max(speed, 0.0))
    if speed <= 10.0:
        return {"t1": 2.0, "t2": 1.0}
    if speed < 25.0:
        return {"t1": 2.78 - 0.078 * speed, "t2": 1.42 - 0.042 * speed}
    return {"t1": 0.83, "t2": 0.37}



def classify_risk_labels(
    ttc: float,
    drac: float,
    ttc_slope: float,
    drac_slope: float,
    valid: float,
    trend_valid: float,
    config: Any,
) -> Dict[str, int]:
    """Create auxiliary labels from quantities observable in the history window.

    Slopes are fitted over all consecutive valid history frames and are measured
    per history frame.  The function deliberately does not use TM.
    """

    current_valid = valid > 0.5
    trend_is_valid = current_valid and trend_valid > 0.5

    ttc_moderate = float(_cfg(config, "risk_ttc_slope_moderate", -0.35))
    ttc_severe = float(_cfg(config, "risk_ttc_slope_severe", -0.90))
    drac_moderate = float(_cfg(config, "risk_drac_slope_moderate", 0.25))
    drac_severe = float(_cfg(config, "risk_drac_slope_severe", 0.75))

    if trend_is_valid and ttc_slope <= ttc_severe:
        ttc_worsening = 2
    elif trend_is_valid and ttc_slope <= ttc_moderate:
        ttc_worsening = 1
    else:
        ttc_worsening = 0

    if trend_is_valid and drac_slope >= drac_severe:
        drac_worsening = 2
    elif trend_is_valid and drac_slope >= drac_moderate:
        drac_worsening = 1
    else:
        drac_worsening = 0

    # Simple enlarged TTC zones.  DRAC is kept as an independent physical cue.
    if not current_valid:
        urgency = 0
    elif (
        ttc <= float(_cfg(config, "risk_urgency_ttc_emergency", 1.5))
        or drac >= float(_cfg(config, "risk_urgency_drac_emergency", 5.0))
    ):
        urgency = 3
    elif (
        ttc <= float(_cfg(config, "risk_urgency_ttc_critical", 2.5))
        or drac >= float(_cfg(config, "risk_urgency_drac_critical", 3.0))
    ):
        urgency = 2
    elif (
        ttc <= float(_cfg(config, "risk_urgency_ttc_warning", 4.0))
        or drac >= float(_cfg(config, "risk_urgency_drac_warning", 1.5))
    ):
        urgency = 1
    else:
        urgency = 0

    return {
        "ttc_worsening": int(ttc_worsening),
        "drac_worsening": int(drac_worsening),
        "urgency": int(urgency),
        "current_valid": int(current_valid),
        "trend_valid": int(trend_is_valid),
    }



def estimate_front_gap_from_lidar(lidar_pc: np.ndarray, config: Any) -> Dict[str, float]:
    """Estimate front gap from raw LiDAR points in a narrow forward corridor."""

    if lidar_pc is None or lidar_pc.size == 0:
        return {"gap": 0.0, "valid": 0.0}

    position_slice = slice(0, 3)
    points = lidar_pc[position_slice].T
    if points.size == 0:
        return {"gap": 0.0, "valid": 0.0}
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.size == 0:
        return {"gap": 0.0, "valid": 0.0}

    x_min = float(_cfg(config, "risk_front_x_min", 1.0))
    x_max = float(_cfg(config, "risk_front_x_max", _cfg(config, "lidar_max_x", 32.0)))
    y_abs = float(_cfg(config, "risk_front_y_abs", 1.8))
    z_min = float(_cfg(config, "risk_lidar_min_z", _cfg(config, "lidar_split_height", 0.2)))
    z_max = float(_cfg(config, "risk_lidar_max_z", _cfg(config, "max_height_lidar", 100.0)))

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (np.abs(points[:, 1]) <= y_abs)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    corridor_x = points[mask, 0]
    min_points = int(_cfg(config, "risk_lidar_min_points", 3))
    if corridor_x.shape[0] < min_points:
        return {"gap": 0.0, "valid": 0.0}

    percentile = float(_cfg(config, "risk_lidar_gap_percentile", 10.0))
    front_distance = float(np.percentile(corridor_x, percentile))
    ego_front_offset = float(_cfg(config, "risk_ego_front_offset", 2.0))
    gap = max(front_distance - ego_front_offset, 0.0)
    return {"gap": gap, "valid": 1.0}



def _fit_valid_slope(values: List[float], valid: List[float], min_points: int = 3) -> Dict[str, float]:
    """Fit a least-squares slope over the latest consecutive valid suffix."""

    suffix_values: List[float] = []
    for value, is_valid in zip(reversed(values), reversed(valid)):
        if is_valid <= 0.5:
            break
        suffix_values.append(float(value))
    suffix_values.reverse()
    if len(suffix_values) < min_points:
        return {"slope": 0.0, "valid": 0.0, "count": float(len(suffix_values))}

    y = np.asarray(suffix_values, dtype=np.float64)
    x = np.arange(y.shape[0], dtype=np.float64)
    x = x - x.mean()
    denominator = float(np.square(x).sum())
    slope = 0.0 if denominator <= 1e-12 else float((x * (y - y.mean())).sum() / denominator)
    return {"slope": slope, "valid": 1.0, "count": float(y.shape[0])}


def build_history_risk_tokens(agent_input: Any, config: Any) -> np.ndarray:
    """Build raw history tokens; trend supervision is computed from multi-frame slopes."""

    history_frames = int(_cfg(config, "risk_history_num_frames", 4))
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    ttc_max = float(_cfg(config, "risk_ttc_max", 10.0))
    drac_max = float(_cfg(config, "risk_drac_max", 6.0))

    num_available = min(history_frames, len(agent_input.ego_statuses), len(agent_input.lidars))
    start = max(0, len(agent_input.ego_statuses) - num_available)

    entries: List[Dict[str, float]] = []
    for idx in range(start, start + num_available):
        ego_status = agent_input.ego_statuses[idx]
        gap_info = estimate_front_gap_from_lidar(agent_input.lidars[idx].lidar_pc, config)
        ego_velocity = np.asarray(ego_status.ego_velocity, dtype=np.float32)
        ego_acceleration = np.asarray(ego_status.ego_acceleration, dtype=np.float32)
        entries.append(
            {
                "gap": float(gap_info["gap"]),
                "valid": float(gap_info["valid"]),
                "ego_v": max(float(np.nan_to_num(ego_velocity[0], nan=0.0)), 0.0),
                "ego_a": float(np.nan_to_num(ego_acceleration[0], nan=0.0)),
            }
        )

    pad_count = history_frames - len(entries)
    if pad_count > 0:
        entries = [
            {"gap": 0.0, "valid": 0.0, "ego_v": 0.0, "ego_a": 0.0}
        ] * pad_count + entries

    tokens = np.zeros((history_frames, RISK_TOKEN_DIM), dtype=np.float32)
    for idx, entry in enumerate(entries):
        previous = entries[idx - 1] if idx > 0 else None
        valid = entry["valid"] > 0.5
        previous_valid = previous is not None and previous["valid"] > 0.5
        signed_rel_v = (
            (previous["gap"] - entry["gap"]) / dt
            if valid and previous_valid
            else 0.0
        )
        closing_speed = max(signed_rel_v, 0.0)
        if valid:
            metrics = compute_longitudinal_risk(
                entry["gap"],
                closing_speed,
                entry["ego_v"],
                ttc_max=ttc_max,
                drac_max=drac_max,
            )
            lead_v = max(entry["ego_v"] - signed_rel_v, 0.0)
        else:
            metrics = {"thw": 0.0, "ttc": ttc_max, "drac": 0.0}
            lead_v = 0.0

        tokens[idx] = np.asarray(
            [
                entry["gap"],
                signed_rel_v,
                closing_speed,
                entry["ego_v"],
                lead_v,
                entry["ego_a"],
                metrics["thw"],
                metrics["ttc"],
                metrics["drac"],
                entry["valid"],
            ],
            dtype=np.float32,
        )

    return np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)



def _select_history_front_track(scene: Any, config: Any) -> Optional[str]:
    current_idx = scene.scene_metadata.num_history_frames - 1
    annotations = scene.frames[current_idx].annotations
    front = _select_front_vehicle(annotations, config, preferred_track_token=None)
    return front["track_token"] if front is not None else None


def _select_front_vehicle(annotations: Any, config: Any, preferred_track_token: Optional[str]) -> Optional[Dict[str, Any]]:
    y_abs = float(_cfg(config, "risk_front_y_abs", 1.8))
    x_min = float(_cfg(config, "risk_front_x_min", 1.0))
    x_max = float(_cfg(config, "risk_front_x_max", 32.0))
    ego_front_offset = float(_cfg(config, "risk_ego_front_offset", 2.0))

    candidates: List[Dict[str, float]] = []
    for idx, (box, name) in enumerate(zip(annotations.boxes, annotations.names)):
        if name != "vehicle":
            continue
        box_x = float(np.nan_to_num(box[0], nan=0.0, posinf=0.0, neginf=0.0))
        box_y = float(np.nan_to_num(box[1], nan=0.0, posinf=0.0, neginf=0.0))
        box_length = float(np.nan_to_num(box[3], nan=0.0, posinf=0.0, neginf=0.0))
        box_width = float(np.nan_to_num(box[4], nan=0.0, posinf=0.0, neginf=0.0))
        box_heading = float(np.nan_to_num(box[6], nan=0.0, posinf=0.0, neginf=0.0))
        if box_x < x_min or box_x > x_max or abs(box_y) > y_abs:
            continue
        track_token = annotations.track_tokens[idx] if idx < len(annotations.track_tokens) else ""
        velocity = annotations.velocity_3d[idx] if idx < len(annotations.velocity_3d) else np.zeros(3, dtype=np.float32)
        gap = max(box_x - box_length / 2.0 - ego_front_offset, 0.0)
        lead_v = float(np.nan_to_num(velocity[0], nan=0.0, posinf=0.0, neginf=0.0))
        candidates.append(
            {
                "gap": gap,
                "lead_v": lead_v,
                "track_token": track_token,
                "preferred": float(track_token == preferred_track_token),
                "index": idx,
                "x": box_x,
                "y": box_y,
                "heading": box_heading,
                "length": box_length,
                "width": box_width,
            }
        )

    if len(candidates) == 0:
        return None
    if preferred_track_token:
        preferred = [candidate for candidate in candidates if candidate["preferred"] > 0.5]
        if len(preferred) > 0:
            return min(preferred, key=lambda item: item["gap"])
    return min(candidates, key=lambda item: item["gap"])


def _box_in_origin_frame(box: np.ndarray, frame_ego_pose: np.ndarray, origin_ego_pose: np.ndarray) -> np.ndarray:
    """Transform a frame-local box into the current ego coordinate frame."""

    frame_x, frame_y, frame_heading = [float(value) for value in frame_ego_pose]
    origin_x, origin_y, origin_heading = [float(value) for value in origin_ego_pose]
    local_x = float(np.nan_to_num(box[0], nan=0.0, posinf=0.0, neginf=0.0))
    local_y = float(np.nan_to_num(box[1], nan=0.0, posinf=0.0, neginf=0.0))
    local_heading = float(np.nan_to_num(box[6], nan=0.0, posinf=0.0, neginf=0.0))

    frame_cos, frame_sin = np.cos(frame_heading), np.sin(frame_heading)
    global_x = frame_x + frame_cos * local_x - frame_sin * local_y
    global_y = frame_y + frame_sin * local_x + frame_cos * local_y

    delta_x, delta_y = global_x - origin_x, global_y - origin_y
    origin_cos, origin_sin = np.cos(origin_heading), np.sin(origin_heading)
    origin_local_x = origin_cos * delta_x + origin_sin * delta_y
    origin_local_y = -origin_sin * delta_x + origin_cos * delta_y
    heading = np.arctan2(
        np.sin(frame_heading + local_heading - origin_heading),
        np.cos(frame_heading + local_heading - origin_heading),
    )
    return np.array(
        [origin_local_x, origin_local_y, heading, float(box[3]), float(box[4]), 1.0],
        dtype=np.float32,
    )



def _simple_ttc_warning_threshold(config: Any) -> float:
    """Single enlarged TTC threshold used only to activate ranking scenes."""

    return float(_cfg(config, "risk_pair_ttc_warning_threshold", 4.0))


def build_gt_future_agent_targets(scene: Any, config: Any) -> np.ndarray:
    """Track nearby collision-relevant objects over the candidate horizon.

    The returned boxes are training-only GT in the current ego frame with fields
    ``x, y, heading, length, width, valid``.  Fixed padding keeps the cache schema
    deterministic and permits vectorized candidate/agent overlap tests.
    """

    num_poses = int(config.trajectory_sampling.num_poses)
    max_agents = int(_cfg(config, "risk_rank_max_future_agents", 30))
    future_agents = np.zeros((num_poses, max_agents, 6), dtype=np.float32)
    current_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_idx]
    annotations = current_frame.annotations

    tracks = []
    for box, name, track_token in zip(
        annotations.boxes, annotations.names, annotations.track_tokens
    ):
        if name not in RISK_COLLISION_OBJECT_NAMES or not track_token:
            continue
        x = float(np.nan_to_num(box[0], nan=0.0, posinf=0.0, neginf=0.0))
        y = float(np.nan_to_num(box[1], nan=0.0, posinf=0.0, neginf=0.0))
        if not (
            float(_cfg(config, "lidar_min_x", -32.0)) <= x <= float(_cfg(config, "lidar_max_x", 32.0))
            and float(_cfg(config, "lidar_min_y", -32.0)) <= y <= float(_cfg(config, "lidar_max_y", 32.0))
        ):
            continue
        tracks.append((x * x + y * y, str(track_token)))
    track_tokens = [token for _, token in sorted(tracks)[:max_agents]]
    origin_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float64)

    for step in range(num_poses):
        frame_idx = current_idx + step + 1
        if frame_idx >= len(scene.frames):
            break
        frame = scene.frames[frame_idx]
        token_to_index = {
            str(token): idx for idx, token in enumerate(frame.annotations.track_tokens)
        }
        for agent_idx, token in enumerate(track_tokens):
            box_idx = token_to_index.get(token)
            if box_idx is None:
                continue
            future_agents[step, agent_idx] = _box_in_origin_frame(
                frame.annotations.boxes[box_idx],
                np.asarray(frame.ego_status.ego_pose, dtype=np.float64),
                origin_pose,
            )
    return future_agents



def build_gt_future_front_targets(scene: Any, config: Any) -> Dict[str, np.ndarray]:
    """Build training-only continuous future-front targets for mode pair mining."""

    num_poses = int(config.trajectory_sampling.num_poses)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    ttc_max = float(_cfg(config, "risk_ttc_max", 10.0))
    future_front = np.zeros((num_poses, 6), dtype=np.float32)
    context = np.zeros(5, dtype=np.float32)

    current_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_idx]
    front = _select_front_vehicle(current_frame.annotations, config, preferred_track_token=None)
    if front is None or not front["track_token"]:
        return {
            "risk_front_future": future_front,
            "risk_pair_context": context,
            "risk_pair_scene_active": np.array(0.0, dtype=np.float32),
        }

    origin_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float64)
    continuous_steps = 0
    for step in range(num_poses):
        frame_idx = current_idx + step + 1
        if frame_idx >= len(scene.frames):
            break
        frame = scene.frames[frame_idx]
        try:
            track_idx = frame.annotations.track_tokens.index(front["track_token"])
        except ValueError:
            break
        future_front[step] = _box_in_origin_frame(
            frame.annotations.boxes[track_idx],
            np.asarray(frame.ego_status.ego_pose, dtype=np.float64),
            origin_pose,
        )
        continuous_steps += 1

    ego_v = max(
        float(np.nan_to_num(current_frame.ego_status.ego_velocity[0], nan=0.0, posinf=0.0, neginf=0.0)),
        0.0,
    )
    rel_v = max(ego_v - float(front["lead_v"]), 0.0)
    current_ttc = min(float(front["gap"]) / max(rel_v, 1e-3), ttc_max) if rel_v > 0.1 else ttc_max
    delayed_ttc = ttc_max
    if continuous_steps > 0:
        gt_ego = scene.get_future_trajectory(num_trajectory_frames=num_poses).poses
        delayed_gap = max(
            float(future_front[0, 0])
            - 0.5 * float(future_front[0, 3])
            - float(gt_ego[0, 0])
            - float(_cfg(config, "risk_ego_front_offset", 2.0)),
            1e-3,
        )
        delayed_rel_v = max((float(front["gap"]) - delayed_gap) / dt, 0.0)
        if delayed_rel_v > 0.1:
            delayed_ttc = min(delayed_gap / delayed_rel_v, ttc_max)

    t1 = _simple_ttc_warning_threshold(config)
    risk_active = float(
        continuous_steps >= 2
        and (current_ttc <= t1 or (current_ttc > t1 and delayed_ttc <= t1))
    )
    context[:] = [ego_v, current_ttc, delayed_ttc, t1, float(continuous_steps)]
    return {
        "risk_front_future": future_front,
        "risk_pair_context": context,
        "risk_pair_scene_active": np.array(risk_active, dtype=np.float32),
    }



def build_gt_history_risk_targets(scene: Any, config: Any) -> Dict[str, np.ndarray]:
    """Build split history-trend labels and future GT braking supervision."""

    history_frames = int(_cfg(config, "risk_history_num_frames", 4))
    ttc_max = float(_cfg(config, "risk_ttc_max", 10.0))
    drac_max = float(_cfg(config, "risk_drac_max", 6.0))
    preferred_track_token = _select_history_front_track(scene, config)

    start = max(0, scene.scene_metadata.num_history_frames - history_frames)
    entries: List[Dict[str, float]] = []
    for frame_idx in range(start, scene.scene_metadata.num_history_frames):
        frame = scene.frames[frame_idx]
        front = _select_front_vehicle(frame.annotations, config, preferred_track_token)
        # Trend labels must describe one continuous target, not whichever object is closest.
        if (
            front is not None
            and preferred_track_token
            and front.get("track_token", "") != preferred_track_token
        ):
            front = None
        ego_v = max(float(np.nan_to_num(frame.ego_status.ego_velocity[0], nan=0.0)), 0.0)
        if front is None:
            entries.append({"ttc": ttc_max, "drac": 0.0, "ego_v": ego_v, "valid": 0.0})
            continue
        signed_rel_v = ego_v - float(front["lead_v"])
        metrics = compute_longitudinal_risk(
            front["gap"],
            max(signed_rel_v, 0.0),
            ego_v,
            ttc_max=ttc_max,
            drac_max=drac_max,
        )
        entries.append(
            {
                "ttc": metrics["ttc"],
                "drac": metrics["drac"],
                "ego_v": ego_v,
                "valid": 1.0,
            }
        )

    pad_count = history_frames - len(entries)
    if pad_count > 0:
        entries = [
            {"ttc": ttc_max, "drac": 0.0, "ego_v": 0.0, "valid": 0.0}
        ] * pad_count + entries

    ttc_fit = _fit_valid_slope(
        [entry["ttc"] for entry in entries],
        [entry["valid"] for entry in entries],
        min_points=int(_cfg(config, "risk_trend_min_valid_frames", 3)),
    )
    drac_fit = _fit_valid_slope(
        [entry["drac"] for entry in entries],
        [entry["valid"] for entry in entries],
        min_points=int(_cfg(config, "risk_trend_min_valid_frames", 3)),
    )
    current = entries[-1]
    labels = classify_risk_labels(
        ttc=current["ttc"],
        drac=current["drac"],
        ttc_slope=ttc_fit["slope"],
        drac_slope=drac_fit["slope"],
        valid=current["valid"],
        trend_valid=min(ttc_fit["valid"], drac_fit["valid"]),
        config=config,
    )

    label_valid = np.asarray(
        [
            labels["trend_valid"],
            labels["trend_valid"],
            labels["current_valid"],
        ],
        dtype=np.float32,
    )
    raw = np.asarray(
        [
            ttc_fit["slope"],
            drac_fit["slope"],
            current["ttc"],
            current["drac"],
        ],
        dtype=np.float32,
    )

    return {
        "risk_aux_labels": np.asarray(
            [
                labels["ttc_worsening"],
                labels["drac_worsening"],
                labels["urgency"],
            ],
            dtype=np.int64,
        ),
        "risk_aux_label_valid": label_valid,
        "risk_aux_raw": raw,
    }
