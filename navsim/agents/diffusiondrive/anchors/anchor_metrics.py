"""Audit metrics for a command-conditioned adaptive Anchor Bank."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from navsim.agents.diffusiondrive.anchors.kmedoids_splitter import (
    pairwise_trajectory_distance,
)
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


COMMAND_ORDER = tuple(TrajectoryCommand)
DIVERSITY_THRESHOLDS = (0.10, 0.15, 0.20, 0.25)


def command_anchor_coverage_metrics(
    trajectories: npt.ArrayLike,
    anchors: npt.ArrayLike,
    assignments: npt.ArrayLike,
    nearest_distances: npt.ArrayLike,
) -> dict[str, float]:
    """Summarize formal nearest distance and legacy geometric coverage diagnostics."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    nearest_distances = np.asarray(nearest_distances, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError("trajectories must have shape [N, 8, 2]")
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError("anchors must have shape [K, 8, 2]")
    if assignments.shape != (len(trajectories),) or nearest_distances.shape != (
        len(trajectories),
    ):
        raise ValueError("assignments and nearest_distances must have shape [N]")
    if np.any((assignments < 0) | (assignments >= len(anchors))):
        raise ValueError("assignments contain an out-of-range anchor index")
    if not np.isfinite(trajectories).all() or not np.isfinite(nearest_distances).all():
        raise ValueError("coverage inputs must contain finite values")

    point_errors = np.linalg.norm(trajectories - anchors[assignments], axis=-1)
    ade = point_errors.mean(axis=-1)
    fde = point_errors[:, -1]
    max_point_error = point_errors.max(axis=-1)
    return {
        "mean_nearest_d_traj": float(nearest_distances.mean()),
        "p50_nearest_d_traj": float(np.quantile(nearest_distances, 0.50)),
        "p95_nearest_d_traj": float(np.quantile(nearest_distances, 0.95)),
        "mean_nearest_ADE": float(ade.mean()),
        "mean_nearest_FDE": float(fde.mean()),
        "p95_max_point_error": float(np.quantile(max_point_error, 0.95)),
        "coverage@0.5m": float((max_point_error <= 0.5).mean()),
        "coverage@1m": float((max_point_error <= 1.0).mean()),
        "coverage@1.5m": float((max_point_error <= 1.5).mean()),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "histogram_edges": [],
            "histogram_counts": [],
            **{f"ratio_d_below_{threshold:.2f}": 0.0 for threshold in DIVERSITY_THRESHOLDS},
        }
    bin_count = min(20, max(5, int(np.ceil(np.sqrt(len(values))))))
    histogram_counts, histogram_edges = np.histogram(values, bins=bin_count)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "histogram_edges": histogram_edges,
        "histogram_counts": histogram_counts,
        **{
            f"ratio_d_below_{threshold:.2f}": float((values < threshold).mean())
            for threshold in DIVERSITY_THRESHOLDS
        },
    }


def command_anchor_diversity_metrics(
    anchors: npt.ArrayLike,
    command_ids: npt.ArrayLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Measure same-command nearest-neighbor and pairwise ``D_traj`` diversity."""
    anchors = np.asarray(anchors, dtype=np.float32)
    command_ids = np.asarray(command_ids, dtype=np.int64)
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError("anchors must have shape [K, 8, 2]")
    if command_ids.shape != (len(anchors),):
        raise ValueError("command_ids must have shape [K]")
    if np.any((command_ids < 0) | (command_ids >= len(COMMAND_ORDER))):
        raise ValueError("command_ids must contain only ids 0--2")

    all_nearest_neighbors = []
    all_pairwise = []
    by_command: dict[str, Any] = {}
    for command_id, command in enumerate(COMMAND_ORDER):
        command_anchors = anchors[command_ids == command_id]
        if len(command_anchors) < 2:
            by_command[command.value] = {
                "anchor_count": int(len(command_anchors)),
                "mean_nearest_neighbor_d_traj": None,
                "median_nearest_neighbor_d_traj": None,
                "pairwise_d_traj_distribution": _distribution_summary(
                    np.asarray([], dtype=np.float32)
                ),
            }
            continue
        distances = pairwise_trajectory_distance(
            command_anchors,
            command_anchors,
            command,
            scales,
            batch_size=batch_size,
        )
        upper_triangle = distances[np.triu_indices(len(command_anchors), k=1)]
        np.fill_diagonal(distances, np.inf)
        nearest_neighbors = distances.min(axis=1)
        all_nearest_neighbors.append(nearest_neighbors)
        all_pairwise.append(upper_triangle)
        by_command[command.value] = {
            "anchor_count": int(len(command_anchors)),
            "mean_nearest_neighbor_d_traj": float(nearest_neighbors.mean()),
            "median_nearest_neighbor_d_traj": float(np.median(nearest_neighbors)),
            "pairwise_d_traj_distribution": _distribution_summary(upper_triangle),
        }

    nearest_neighbors = (
        np.concatenate(all_nearest_neighbors)
        if all_nearest_neighbors
        else np.asarray([], dtype=np.float32)
    )
    pairwise_values = (
        np.concatenate(all_pairwise) if all_pairwise else np.asarray([], dtype=np.float32)
    )
    return {
        "mean_nearest_neighbor_d_traj": (
            float(nearest_neighbors.mean()) if len(nearest_neighbors) else None
        ),
        "median_nearest_neighbor_d_traj": (
            float(np.median(nearest_neighbors)) if len(nearest_neighbors) else None
        ),
        "pairwise_d_traj_distribution": _distribution_summary(pairwise_values),
        "by_command": by_command,
    }
