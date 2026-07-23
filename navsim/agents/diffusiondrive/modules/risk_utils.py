from typing import Any, Dict, List, Optional

import numpy as np

from navsim.common.enums import LidarIndex


RISK_TOKEN_DIM = 12
RISK_TOKEN_FIELDS = (
    "gap",
    "rel_v",
    "ego_v",
    "lead_v",
    "ego_a",
    "thw",
    "ttc",
    "drac",
    "delta_thw",
    "delta_ttc",
    "delta_drac",
    "valid",
)

RISK_LABEL_NAMES = ("risk_trend", "urgency", "brake_need")


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
    delta_ttc: float,
    delta_drac: float,
    ego_v: float,
    valid: float,
    step_margin: float = 0.5,
    ttc_max: float = 10.0,
) -> Dict[str, int]:
    """Create discrete auxiliary labels from explicit risk metrics."""

    if valid <= 0.5:
        return {"risk_trend": 0, "urgency": 0, "brake_need": 0, "valid": 0}

    if delta_ttc <= -1.0 or delta_drac >= 1.0:
        risk_trend = 2
    elif delta_ttc < -0.25 or delta_drac > 0.25:
        risk_trend = 1
    else:
        risk_trend = 0

    thresholds = tm_brake_thresholds(ego_v)
    if drac >= 6.0:
        urgency = 3
    elif ttc <= thresholds["t2"] + step_margin or drac >= 4.0:
        urgency = 2
    elif ttc <= thresholds["t1"] + step_margin or drac >= 2.0 or ttc < ttc_max:
        urgency = 1
    else:
        urgency = 0

    if drac < 1.0:
        brake_need = 0
    elif drac < 2.0:
        brake_need = 1
    elif drac < 4.0:
        brake_need = 2
    elif drac < 6.0:
        brake_need = 3
    else:
        brake_need = 4

    return {
        "risk_trend": int(risk_trend),
        "urgency": int(urgency),
        "brake_need": int(brake_need),
        "valid": 1,
    }


def estimate_front_gap_from_lidar(lidar_pc: np.ndarray, config: Any) -> Dict[str, float]:
    """Estimate front gap from raw LiDAR points in a narrow forward corridor."""

    if lidar_pc is None or lidar_pc.size == 0:
        return {"gap": 0.0, "valid": 0.0}

    points = lidar_pc[LidarIndex.POSITION].T
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


def build_history_risk_tokens(agent_input: Any, config: Any) -> np.ndarray:
    """Build [K, F] risk tokens from history LiDAR and ego states available at inference."""

    history_frames = int(_cfg(config, "risk_history_num_frames", 4))
    dt = float(_cfg(config, "risk_history_dt", 0.5))
    ttc_max = float(_cfg(config, "risk_ttc_max", 10.0))
    drac_max = float(_cfg(config, "risk_drac_max", 6.0))

    num_available = min(history_frames, len(agent_input.ego_statuses), len(agent_input.lidars))
    start = max(0, len(agent_input.ego_statuses) - num_available)

    entries: List[Dict[str, float]] = []
    for idx in range(start, start + num_available):
        ego_status = agent_input.ego_statuses[idx]
        lidar = agent_input.lidars[idx]
        gap_info = estimate_front_gap_from_lidar(lidar.lidar_pc, config)
        ego_velocity = np.asarray(ego_status.ego_velocity, dtype=np.float32)
        ego_acceleration = np.asarray(ego_status.ego_acceleration, dtype=np.float32)
        ego_v = max(float(np.nan_to_num(ego_velocity[0], nan=0.0, posinf=0.0, neginf=0.0)), 0.0)
        ego_a = float(np.nan_to_num(ego_acceleration[0], nan=0.0, posinf=0.0, neginf=0.0))
        entries.append(
            {
                "gap": float(gap_info["gap"]),
                "valid": float(gap_info["valid"]),
                "ego_v": ego_v,
                "ego_a": ego_a,
            }
        )

    pad_count = history_frames - len(entries)
    if pad_count > 0:
        entries = [{"gap": 0.0, "valid": 0.0, "ego_v": 0.0, "ego_a": 0.0}] * pad_count + entries

    tokens = np.zeros((history_frames, RISK_TOKEN_DIM), dtype=np.float32)
    prev_metrics = None
    for idx, entry in enumerate(entries):
        prev_entry = entries[idx - 1] if idx > 0 else None
        valid = entry["valid"] > 0.5
        prev_valid = prev_entry is not None and prev_entry["valid"] > 0.5
        if valid and prev_valid:
            rel_v = max((prev_entry["gap"] - entry["gap"]) / max(dt, 1e-3), 0.0)
        else:
            rel_v = 0.0

        if valid:
            metrics = compute_longitudinal_risk(
                entry["gap"],
                rel_v,
                entry["ego_v"],
                ttc_max=ttc_max,
                drac_max=drac_max,
            )
            lead_v = entry["ego_v"] - rel_v
        else:
            metrics = {"thw": 0.0, "ttc": ttc_max, "drac": 0.0}
            lead_v = 0.0

        if valid and prev_metrics is not None:
            delta_thw = metrics["thw"] - prev_metrics["thw"]
            delta_ttc = metrics["ttc"] - prev_metrics["ttc"]
            delta_drac = metrics["drac"] - prev_metrics["drac"]
        else:
            delta_thw = 0.0
            delta_ttc = 0.0
            delta_drac = 0.0

        tokens[idx] = np.array(
            [
                entry["gap"],
                rel_v,
                entry["ego_v"],
                lead_v,
                entry["ego_a"],
                metrics["thw"],
                metrics["ttc"],
                metrics["drac"],
                delta_thw,
                delta_ttc,
                delta_drac,
                entry["valid"],
            ],
            dtype=np.float32,
        )
        prev_metrics = metrics if valid else None

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
    frame_cos, frame_sin = np.cos(frame_heading), np.sin(frame_heading)
    global_x = frame_x + frame_cos * local_x - frame_sin * local_y
    global_y = frame_y + frame_sin * local_x + frame_cos * local_y
    delta_x, delta_y = global_x - origin_x, global_y - origin_y
    origin_cos, origin_sin = np.cos(origin_heading), np.sin(origin_heading)
    return np.array(
        [
            origin_cos * delta_x + origin_sin * delta_y,
            -origin_sin * delta_x + origin_cos * delta_y,
            float(box[3]),
            float(box[4]),
        ],
        dtype=np.float32,
    )


def _speed_dependent_t1(ego_v: float) -> float:
    if ego_v <= 10.0:
        return 2.0
    if ego_v < 25.0:
        return 2.78 - 0.078 * ego_v
    return 0.83


def build_gt_brake_timing_context(scene: Any, config: Any) -> np.ndarray:
    """Build training-only TTC context for brake-timing supervision."""

    num_poses = int(config.trajectory_sampling.num_poses)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    ttc_max = float(_cfg(config, "risk_ttc_max", 10.0))
    context = np.zeros(5, dtype=np.float32)
    current_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_idx]
    front = _select_front_vehicle(current_frame.annotations, config, preferred_track_token=None)
    if front is None or not front["track_token"]:
        return context

    origin_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float64)
    first_future_front = None
    continuous_steps = 0
    x_min = float(_cfg(config, "risk_front_x_min", 1.0))
    x_max = float(_cfg(config, "risk_front_x_max", 32.0))
    y_abs = float(_cfg(config, "risk_front_y_abs", 1.8))
    for step in range(num_poses):
        frame_idx = current_idx + step + 1
        if frame_idx >= len(scene.frames):
            break
        frame = scene.frames[frame_idx]
        try:
            track_idx = frame.annotations.track_tokens.index(front["track_token"])
        except ValueError:
            break
        tracked_box = frame.annotations.boxes[track_idx]
        tracked_name = frame.annotations.names[track_idx]
        if (
            tracked_name != "vehicle"
            or float(tracked_box[0]) < x_min
            or float(tracked_box[0]) > x_max
            or abs(float(tracked_box[1])) > y_abs
        ):
            break
        transformed = _box_in_origin_frame(
            tracked_box,
            np.asarray(frame.ego_status.ego_pose, dtype=np.float64),
            origin_pose,
        )
        if first_future_front is None:
            first_future_front = transformed
        continuous_steps += 1

    ego_v = max(
        float(np.nan_to_num(current_frame.ego_status.ego_velocity[0], nan=0.0, posinf=0.0, neginf=0.0)),
        0.0,
    )
    rel_v = max(ego_v - float(front["lead_v"]), 0.0)
    current_ttc = min(float(front["gap"]) / max(rel_v, 1e-3), ttc_max) if rel_v > 0.1 else ttc_max
    delayed_ttc = ttc_max
    if first_future_front is not None:
        gt_ego = scene.get_future_trajectory(num_trajectory_frames=num_poses).poses
        delayed_gap = max(
            float(first_future_front[0])
            - 0.5 * float(first_future_front[2])
            - float(gt_ego[0, 0])
            - float(_cfg(config, "risk_ego_front_offset", 2.0)),
            1e-3,
        )
        delayed_rel_v = max((float(front["gap"]) - delayed_gap) / dt, 0.0)
        if delayed_rel_v > 0.1:
            delayed_ttc = min(delayed_gap / delayed_rel_v, ttc_max)

    context[:] = [ego_v, current_ttc, delayed_ttc, _speed_dependent_t1(ego_v), float(continuous_steps)]
    return context


def _cumulative_path_progress(path_xy: np.ndarray) -> np.ndarray:
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), path_xy.astype(np.float32)], axis=0)
    return np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)).astype(np.float32)


def _project_to_path(point_xy: np.ndarray, path_xy: np.ndarray, progress: np.ndarray) -> tuple[float, float]:
    """Project a point onto the fixed GT path and return longitudinal/lateral coordinates."""

    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), path_xy.astype(np.float32)], axis=0)
    starts, ends = points[:-1], points[1:]
    segments = ends - starts
    length_sq = np.maximum((segments * segments).sum(axis=1), 1e-6)
    raw_ratio = ((point_xy[None] - starts) * segments).sum(axis=1) / length_sq
    ratio = np.clip(raw_ratio, 0.0, 1.0)
    projections = starts + ratio[:, None] * segments
    index = int(np.argmin(((projections - point_xy[None]) ** 2).sum(axis=1)))
    if index == len(segments) - 1 and raw_ratio[index] > 1.0:
        ratio[index] = raw_ratio[index]
        projections[index] = starts[index] + ratio[index] * segments[index]
    segment_length = float(np.sqrt(length_sq[index]))
    return float(progress[index] + ratio[index] * segment_length), float(
        np.linalg.norm(projections[index] - point_xy)
    )


def _minimum_jerk_progress(
    terminal_progress: float,
    initial_speed: float,
    terminal_speed: float,
    num_poses: int,
    dt: float,
) -> np.ndarray:
    """Create a minimum-jerk longitudinal profile with fixed start/end state."""

    horizon = num_poses * dt
    system = np.array(
        [
            [horizon**3, horizon**4, horizon**5],
            [3.0 * horizon**2, 4.0 * horizon**3, 5.0 * horizon**4],
            [6.0 * horizon, 12.0 * horizon**2, 20.0 * horizon**3],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [terminal_progress - initial_speed * horizon, terminal_speed - initial_speed, 0.0],
        dtype=np.float64,
    )
    try:
        c3, c4, c5 = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.full(num_poses, np.nan, dtype=np.float32)
    time = np.arange(1, num_poses + 1, dtype=np.float64) * dt
    progress = initial_speed * time + c3 * time**3 + c4 * time**4 + c5 * time**5
    # A feasible stop near the horizon can produce a sub-centimeter polynomial
    # overshoot in finite precision. Progress must remain forward-only because
    # the target reuses the original path geometry rather than reversing on it.
    return np.maximum.accumulate(progress).astype(np.float32)


def _sample_path_by_progress(path: np.ndarray, progress: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Sample the original GT geometry at a new monotonic progress schedule."""

    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), path[:, :2].astype(np.float32)], axis=0)
    headings = np.concatenate([np.zeros(1, dtype=np.float32), path[:, 2].astype(np.float32)])
    path_progress = np.concatenate([np.zeros(1, dtype=np.float32), progress.astype(np.float32)])
    sampled = np.zeros_like(path, dtype=np.float32)
    for index, value in enumerate(query):
        upper = int(np.clip(np.searchsorted(path_progress, value, side="left"), 1, len(path_progress) - 1))
        lower = upper - 1
        denominator = max(float(path_progress[upper] - path_progress[lower]), 1e-6)
        ratio = float(np.clip((value - path_progress[lower]) / denominator, 0.0, 1.0))
        sampled[index, :2] = points[lower] + ratio * (points[upper] - points[lower])
        heading_delta = np.arctan2(
            np.sin(headings[upper] - headings[lower]),
            np.cos(headings[upper] - headings[lower]),
        )
        sampled[index, 2] = headings[lower] + ratio * heading_delta
    return sampled


def _transport_profile_is_feasible(
    progress: np.ndarray,
    upper_bounds: np.ndarray,
    initial_speed: float,
    config: Any,
    dt: float,
) -> bool:
    if not np.isfinite(progress).all() or np.any(np.diff(progress) < -1e-4):
        return False
    speed = np.diff(np.concatenate([np.zeros(1, dtype=np.float32), progress])) / dt
    acceleration = np.diff(np.concatenate([[initial_speed], speed])) / dt
    jerk = np.diff(acceleration) / dt
    return bool(
        np.all(progress <= upper_bounds + 1e-3)
        and np.all(speed >= -1e-4)
        and np.all(speed <= float(_cfg(config, "transport_max_speed", 30.0)))
        and np.all(acceleration >= -float(_cfg(config, "transport_max_decel", 4.0)))
        and np.all(acceleration <= float(_cfg(config, "transport_max_accel", 3.0)))
        and np.all(np.abs(jerk) <= float(_cfg(config, "transport_max_jerk", 6.0)))
    )


def build_gt_temporal_transport_target(scene: Any, config: Any) -> Dict[str, np.ndarray]:
    """Build a GT-only, endpoint-conditioned longitudinal transport target.

    The target preserves the GT polyline geometry and optimizes only its progress
    schedule. Future tracked boxes supply an ST upper bound during training; they
    are never exposed through model features at inference.
    """

    num_poses = int(config.trajectory_sampling.num_poses)
    dt = max(float(_cfg(config, "risk_history_dt", 0.5)), 1e-3)
    trajectory = np.asarray(
        scene.get_future_trajectory(num_trajectory_frames=num_poses).poses,
        dtype=np.float32,
    )
    gt_progress = _cumulative_path_progress(trajectory[:, :2])
    unconstrained_bound = max(float(gt_progress[-1]), 1.0) + 100.0
    result = {
        "target": trajectory.copy(),
        "upper_s": np.full(num_poses, unconstrained_bound, dtype=np.float32),
        "constraint_mask": np.zeros(num_poses, dtype=np.float32),
        "valid": np.zeros(1, dtype=np.float32),
        "front_boxes": np.zeros((num_poses, 4), dtype=np.float32),
        "front_mask": np.zeros(num_poses, dtype=np.float32),
    }

    current_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_idx]
    front = _select_front_vehicle(current_frame.annotations, config, preferred_track_token=None)
    if front is None or not front["track_token"]:
        return result

    origin_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float64)
    initial_speed = max(float(np.nan_to_num(current_frame.ego_status.ego_velocity[0])), 0.0)
    gt_speed = np.diff(np.concatenate([[0.0], gt_progress])) / dt
    terminal_speed = max(float(gt_speed[-1]), 0.0)
    upper_s = result["upper_s"]
    constraint_mask = result["constraint_mask"]
    continuous_steps = 0
    ego_width = float(_cfg(config, "risk_shadow_ego_width", 2.0))
    lateral_margin = float(_cfg(config, "risk_shadow_lateral_margin", 0.3))
    min_gap = float(_cfg(config, "transport_min_gap", 1.5))
    time_headway = float(_cfg(config, "transport_time_headway", 0.75))
    max_gap = float(_cfg(config, "transport_max_gap", 8.0))
    ego_front_offset = float(_cfg(config, "risk_ego_front_offset", 2.0))

    for step in range(num_poses):
        frame_idx = current_idx + step + 1
        if frame_idx >= len(scene.frames):
            break
        frame = scene.frames[frame_idx]
        try:
            track_idx = frame.annotations.track_tokens.index(front["track_token"])
        except ValueError:
            break
        if frame.annotations.names[track_idx] != "vehicle":
            break
        transformed = _box_in_origin_frame(
            frame.annotations.boxes[track_idx],
            np.asarray(frame.ego_status.ego_pose, dtype=np.float64),
            origin_pose,
        )
        result["front_boxes"][step] = transformed[:4]
        result["front_mask"][step] = 1.0
        continuous_steps += 1
        front_s, lateral_distance = _project_to_path(transformed[:2], trajectory[:, :2], gt_progress)
        overlap_limit = 0.5 * (ego_width + max(float(transformed[3]), 0.5)) + lateral_margin
        if lateral_distance > overlap_limit:
            continue
        desired_gap = min(max(min_gap, time_headway * max(float(gt_speed[step]), 0.0)), max_gap)
        upper_s[step] = max(front_s - 0.5 * max(float(transformed[2]), 0.5) - ego_front_offset - desired_gap, 0.0)
        constraint_mask[step] = 1.0

    if getattr(config, "use_all_mode_risk_corridor", False):
        return result

    context = build_gt_brake_timing_context(scene, config)
    pre_risk = bool(
        continuous_steps >= int(_cfg(config, "transport_min_front_steps", 2))
        and min(float(context[1]), float(context[2]))
        <= float(context[3]) + float(_cfg(config, "brake_timing_preparation_time", 1.0))
    )
    if not pre_risk or constraint_mask.sum() <= 0.0:
        return result

    constrained_indices = np.flatnonzero(constraint_mask > 0.5)
    if len(constrained_indices) >= 2:
        last_index, previous_index = constrained_indices[-1], constrained_indices[-2]
        boundary_speed = max(
            float((upper_s[last_index] - upper_s[previous_index]) / ((last_index - previous_index) * dt)),
            0.0,
        )
        terminal_speed = min(terminal_speed, boundary_speed)

    best_progress = None
    for terminal_progress in np.linspace(float(gt_progress[-1]), 0.0, num=33):
        candidate = _minimum_jerk_progress(
            terminal_progress,
            initial_speed,
            terminal_speed,
            num_poses,
            dt,
        )
        if _transport_profile_is_feasible(candidate, upper_s, initial_speed, config, dt):
            best_progress = candidate
            break
    if best_progress is None:
        return result

    max_shift = float(np.max(np.abs(best_progress - gt_progress)))
    if max_shift < float(_cfg(config, "transport_min_progress_shift", 0.10)):
        return result

    result["target"] = _sample_path_by_progress(trajectory, gt_progress, best_progress)
    result["upper_s"] = upper_s.astype(np.float32)
    result["constraint_mask"] = constraint_mask.astype(np.float32)
    result["valid"][0] = 1.0
    return result


def build_gt_history_risk_targets(scene: Any, config: Any) -> Dict[str, np.ndarray]:
    """Build auxiliary risk labels from training-only GT boxes and velocities."""

    history_frames = int(_cfg(config, "risk_history_num_frames", 4))
    ttc_max = float(_cfg(config, "risk_ttc_max", 10.0))
    drac_max = float(_cfg(config, "risk_drac_max", 6.0))
    preferred_track_token = _select_history_front_track(scene, config)

    start = max(0, scene.scene_metadata.num_history_frames - history_frames)
    entries: List[Dict[str, float]] = []
    for frame_idx in range(start, scene.scene_metadata.num_history_frames):
        frame = scene.frames[frame_idx]
        front = _select_front_vehicle(frame.annotations, config, preferred_track_token)
        ego_v = max(float(np.nan_to_num(frame.ego_status.ego_velocity[0], nan=0.0, posinf=0.0, neginf=0.0)), 0.0)
        ego_a = float(np.nan_to_num(frame.ego_status.ego_acceleration[0], nan=0.0, posinf=0.0, neginf=0.0))
        if front is None:
            entries.append({"gap": 0.0, "rel_v": 0.0, "ego_v": ego_v, "ego_a": ego_a, "valid": 0.0})
            continue
        rel_v = max(ego_v - front["lead_v"], 0.0)
        entries.append(
            {
                "gap": front["gap"],
                "rel_v": rel_v,
                "ego_v": ego_v,
                "ego_a": ego_a,
                "valid": 1.0,
            }
        )

    pad_count = history_frames - len(entries)
    if pad_count > 0:
        entries = [{"gap": 0.0, "rel_v": 0.0, "ego_v": 0.0, "ego_a": 0.0, "valid": 0.0}] * pad_count + entries

    prev_metrics = None
    current_metrics = None
    current_entry = entries[-1]
    for entry in entries:
        if entry["valid"] > 0.5:
            metrics = compute_longitudinal_risk(
                entry["gap"],
                entry["rel_v"],
                entry["ego_v"],
                ttc_max=ttc_max,
                drac_max=drac_max,
            )
        else:
            metrics = {"thw": 0.0, "ttc": ttc_max, "drac": 0.0}
        current_metrics = metrics
        if entry is current_entry:
            break
        prev_metrics = metrics if entry["valid"] > 0.5 else None

    if prev_metrics is None or current_entry["valid"] <= 0.5:
        delta_ttc = 0.0
        delta_drac = 0.0
    else:
        delta_ttc = current_metrics["ttc"] - prev_metrics["ttc"]
        delta_drac = current_metrics["drac"] - prev_metrics["drac"]

    labels = classify_risk_labels(
        ttc=current_metrics["ttc"],
        drac=current_metrics["drac"],
        delta_ttc=delta_ttc,
        delta_drac=delta_drac,
        ego_v=current_entry["ego_v"],
        valid=current_entry["valid"],
        ttc_max=ttc_max,
    )
    return {
        "risk_aux_labels": np.array(
            [labels["risk_trend"], labels["urgency"], labels["brake_need"]],
            dtype=np.int64,
        ),
        "risk_aux_valid": np.array(labels["valid"], dtype=np.float32),
    }
