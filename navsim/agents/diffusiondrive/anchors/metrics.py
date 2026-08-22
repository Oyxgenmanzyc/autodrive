"""Pure NumPy trajectory distance and coverage metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def _validate_trajectory_array(name: str, values: Array) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [N, T, 2], got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def pairwise_ade(trajectories: Array, anchors: Array) -> np.ndarray:
    """Return mean point-wise L2 distance with shape ``[N, K]``."""
    trajectories = _validate_trajectory_array("trajectories", trajectories)
    anchors = _validate_trajectory_array("anchors", anchors)
    if trajectories.shape[1:] != anchors.shape[1:]:
        raise ValueError("trajectories and anchors must have identical [T, 2] shape")
    distance = np.linalg.norm(trajectories[:, None] - anchors[None], axis=-1)
    return distance.mean(axis=-1)


def pairwise_max_point_distance(trajectories: Array, anchors: Array) -> np.ndarray:
    """Return maximum point-wise L2 distance with shape ``[N, K]``."""
    trajectories = _validate_trajectory_array("trajectories", trajectories)
    anchors = _validate_trajectory_array("anchors", anchors)
    if trajectories.shape[1:] != anchors.shape[1:]:
        raise ValueError("trajectories and anchors must have identical [T, 2] shape")
    distance = np.linalg.norm(trajectories[:, None] - anchors[None], axis=-1)
    return distance.max(axis=-1)


def nearest_anchor_assignment(
    trajectories: Array,
    anchors: Array,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each trajectory by ADE without materializing the full NAVSIM matrix."""
    trajectories = _validate_trajectory_array("trajectories", trajectories)
    anchors = _validate_trajectory_array("anchors", anchors)
    if trajectories.shape[1:] != anchors.shape[1:]:
        raise ValueError("trajectories and anchors must have identical [T, 2] shape")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    assignments = np.empty(len(trajectories), dtype=np.int64)
    nearest_ade = np.empty(len(trajectories), dtype=np.float64)
    for start in range(0, len(trajectories), batch_size):
        end = min(start + batch_size, len(trajectories))
        distances = pairwise_ade(trajectories[start:end], anchors)
        assignments[start:end] = distances.argmin(axis=1)
        nearest_ade[start:end] = distances[np.arange(end - start), assignments[start:end]]
    return assignments, nearest_ade


def nearest_max_point_distance(
    trajectories: Array,
    anchors: Array,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return ``min_anchor max_t ||trajectory_t-anchor_t||`` for each sample."""
    trajectories = _validate_trajectory_array("trajectories", trajectories)
    anchors = _validate_trajectory_array("anchors", anchors)
    nearest = np.empty(len(trajectories), dtype=np.float64)
    for start in range(0, len(trajectories), batch_size):
        end = min(start + batch_size, len(trajectories))
        nearest[start:end] = pairwise_max_point_distance(trajectories[start:end], anchors).min(axis=1)
    return nearest


def coverage_ratio(
    trajectories: Array,
    anchors: Array,
    epsilon: float,
    batch_size: int = 4096,
) -> float:
    """Fraction of trajectories covered under the strict maximum-point metric."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float((nearest_max_point_distance(trajectories, anchors, batch_size) <= epsilon).mean())


def cluster_variance(cluster: Array) -> float:
    """Mean temporal squared spread around the cluster trajectory mean."""
    cluster = _validate_trajectory_array("cluster", cluster)
    center = cluster.mean(axis=0, keepdims=True)
    squared_distance = np.square(cluster - center).sum(axis=-1)
    return float(squared_distance.mean())


def cluster_statistics(
    trajectories: Array,
    anchors: Array,
    assignments: npt.NDArray[np.integer],
    coverage_epsilon: float,
) -> list[Dict[str, float]]:
    """Compute support, assigned-tail error, spread, and uncovered count per anchor."""
    trajectories = _validate_trajectory_array("trajectories", trajectories)
    anchors = _validate_trajectory_array("anchors", anchors)
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.shape != (len(trajectories),):
        raise ValueError("assignments must have shape [N]")

    stats: list[Dict[str, float]] = []
    for anchor_idx, anchor in enumerate(anchors):
        samples = trajectories[assignments == anchor_idx]
        if len(samples) == 0:
            stats.append(
                {
                    "anchor_index": int(anchor_idx),
                    "support": 0,
                    "coverage_error_p95": 0.0,
                    "variance": 0.0,
                    "uncovered_count": 0,
                }
            )
            continue
        max_error = np.linalg.norm(samples - anchor[None], axis=-1).max(axis=-1)
        stats.append(
            {
                "anchor_index": int(anchor_idx),
                "support": int(len(samples)),
                "coverage_error_p95": float(np.quantile(max_error, 0.95)),
                "variance": cluster_variance(samples),
                "uncovered_count": int((max_error > coverage_epsilon).sum()),
            }
        )
    return stats


def evaluate_anchor_bank(
    trajectories: Array,
    anchors: Array,
    coverage_epsilons: Iterable[float] = (0.5, 1.0, 1.5, 2.0),
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Return the paper-facing coverage and nearest-error summary."""
    _, nearest_ade = nearest_anchor_assignment(trajectories, anchors, batch_size)
    nearest_max = nearest_max_point_distance(trajectories, anchors, batch_size)
    result: Dict[str, float] = {
        "anchor_count": int(len(anchors)),
        "mean_nearest_ADE": float(nearest_ade.mean()),
        "p95_nearest_ADE": float(np.quantile(nearest_ade, 0.95)),
        "p95_max_error": float(np.quantile(nearest_max, 0.95)),
    }
    for epsilon in coverage_epsilons:
        result[f"coverage@{epsilon:g}m"] = float((nearest_max <= epsilon).mean())
    return result
