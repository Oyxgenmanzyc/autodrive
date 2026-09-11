# Source: autodrive 233457e, history-only risk proxy helpers.
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


