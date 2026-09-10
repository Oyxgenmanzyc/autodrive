from typing import Any, Dict, List, Optional

import numpy as np


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


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
