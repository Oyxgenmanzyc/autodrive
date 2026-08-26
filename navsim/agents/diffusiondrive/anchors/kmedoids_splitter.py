"""Trajectory-native K-Medoids split for one command-conditioned parent cluster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import kmedoids
import numpy as np
import numpy.typing as npt

from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    CommandLike,
    TrajectoryCommand,
    TrajectoryDistanceScale,
    trajectory_distance,
)


@dataclass(frozen=True)
class KMedoidsSplitResult:
    """Real-trajectory medoids and assignments for the complete input cluster."""

    medoids: np.ndarray
    assignments: np.ndarray
    support: np.ndarray
    sampled_indices: np.ndarray
    medoid_parent_indices: np.ndarray


def pairwise_trajectory_distance(
    trajectories: npt.ArrayLike,
    anchors: npt.ArrayLike,
    command: CommandLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    batch_size: int,
) -> np.ndarray:
    """Compute a bounded-memory ``[N, K]`` matrix using the formal ``D_traj``."""
    trajectories = np.asarray(trajectories)
    anchors = np.asarray(anchors)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError(f"trajectories must have shape [N, 8, 2], got {trajectories.shape}")
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError(f"anchors must have shape [K, 8, 2], got {anchors.shape}")
    if len(trajectories) == 0 or len(anchors) == 0:
        raise ValueError("trajectories and anchors must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    distances = np.empty((len(trajectories), len(anchors)), dtype=np.float32)
    for start in range(0, len(trajectories), batch_size):
        end = min(start + batch_size, len(trajectories))
        distances[start:end] = trajectory_distance(
            trajectories[start:end, None],
            anchors[None],
            command,
            scales,
        )
    return distances


def split_parent_cluster(
    trajectories: npt.ArrayLike,
    command: CommandLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    *,
    max_medoid_samples: int = 4096,
    distance_batch_size: int = 128,
    random_seed: int = 0,
) -> KMedoidsSplitResult:
    """Split one parent with ``K-Medoids(k=2)`` and fully reassign all parent GT."""
    return select_k_medoids(
        trajectories,
        command,
        scales,
        num_medoids=2,
        max_medoid_samples=max_medoid_samples,
        distance_batch_size=distance_batch_size,
        random_seed=random_seed,
    )


def select_k_medoids(
    trajectories: npt.ArrayLike,
    command: CommandLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    *,
    num_medoids: int,
    max_medoid_samples: int,
    distance_batch_size: int = 128,
    random_seed: int = 0,
) -> KMedoidsSplitResult:
    """Select real GT medoids on a fixed sample, then fully reassign the input."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError(f"trajectories must have shape [N, 8, 2], got {trajectories.shape}")
    if num_medoids < 1:
        raise ValueError("num_medoids must be positive")
    if len(trajectories) < num_medoids:
        raise ValueError("trajectories must contain at least num_medoids samples")
    if not np.isfinite(trajectories).all():
        raise ValueError("trajectories contain non-finite values")
    if max_medoid_samples < num_medoids:
        raise ValueError("max_medoid_samples cannot be smaller than num_medoids")

    rng = np.random.RandomState(random_seed)
    if len(trajectories) > max_medoid_samples:
        sampled_indices = np.sort(
            rng.choice(len(trajectories), max_medoid_samples, replace=False)
        )
    else:
        sampled_indices = np.arange(len(trajectories), dtype=np.int64)
    sampled = trajectories[sampled_indices]

    sample_distances = pairwise_trajectory_distance(
        sampled,
        sampled,
        command,
        scales,
        distance_batch_size,
    )
    result = kmedoids.fasterpam(
        sample_distances,
        num_medoids,
        init="random",
        random_state=random_seed,
    )
    local_medoid_indices = np.asarray(result.medoids, dtype=np.int64)
    if local_medoid_indices.shape != (num_medoids,) or len(
        np.unique(local_medoid_indices)
    ) != num_medoids:
        raise RuntimeError("K-Medoids did not return the requested distinct medoids")
    if np.any(local_medoid_indices < 0) or np.any(local_medoid_indices >= len(sampled)):
        raise RuntimeError("K-Medoids returned an out-of-range medoid index")

    medoid_parent_indices = sampled_indices[local_medoid_indices]
    order = np.argsort(medoid_parent_indices)
    medoid_parent_indices = medoid_parent_indices[order]
    medoids = trajectories[medoid_parent_indices]
    full_distances = pairwise_trajectory_distance(
        trajectories,
        medoids,
        command,
        scales,
        distance_batch_size,
    )
    assignments = full_distances.argmin(axis=1).astype(np.int64)
    support = np.bincount(assignments, minlength=num_medoids).astype(np.int64)
    return KMedoidsSplitResult(
        medoids=medoids,
        assignments=assignments,
        support=support,
        sampled_indices=sampled_indices,
        medoid_parent_indices=medoid_parent_indices,
    )
