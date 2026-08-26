"""Command-local anchor deduplication using the formal trajectory distance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt

from navsim.agents.diffusiondrive.anchors.command_anchor_bank import (
    COMMAND_ORDER,
    CommandAnchorBankResult,
    command_conditioned_assignment,
)
from navsim.agents.diffusiondrive.anchors.kmedoids_splitter import (
    pairwise_trajectory_distance,
)
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


@dataclass
class AnchorDedupResult:
    """Deduplicated bank and the mandatory post-dedup GT reassignment."""

    anchors: np.ndarray
    command_ids: np.ndarray
    support: np.ndarray
    parent_ids: np.ndarray
    root_ids: np.ndarray
    node_ids: np.ndarray
    assignments: np.ndarray
    nearest_distances: np.ndarray
    local_mean_distances: np.ndarray
    source_indices: np.ndarray
    removed_count: int


def deduplicate_command_anchor_bank(
    trajectories: npt.ArrayLike,
    trajectory_commands: npt.ArrayLike,
    bank: CommandAnchorBankResult,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    *,
    threshold: float = 0.15,
    assignment_batch_size: int = 4096,
) -> AnchorDedupResult:
    """Greedily retain higher-support, lower-local-error anchors within each command."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    anchors = np.asarray(bank.anchors, dtype=np.float32)
    command_ids = np.asarray(bank.command_ids, dtype=np.int64)
    support = np.asarray(bank.support, dtype=np.int64)
    if command_ids.shape != (len(anchors),) or support.shape != (len(anchors),):
        raise ValueError("bank command_ids and support must have shape [K]")
    local_mean = np.asarray(
        [float(stat["mean_d_traj"]) for stat in bank.cluster_stats], dtype=np.float32
    )
    if local_mean.shape != (len(anchors),):
        raise ValueError("bank cluster_stats must contain one entry per anchor")

    kept_indices: list[int] = []
    for command_id, command in enumerate(COMMAND_ORDER):
        group_indices = np.flatnonzero(command_ids == command_id)
        if len(group_indices) == 0:
            continue
        distances = pairwise_trajectory_distance(
            anchors[group_indices],
            anchors[group_indices],
            command,
            scales,
            batch_size=assignment_batch_size,
        )
        priority = sorted(
            range(len(group_indices)),
            key=lambda local_index: (
                -int(support[group_indices[local_index]]),
                float(local_mean[group_indices[local_index]]),
                int(group_indices[local_index]),
            ),
        )
        kept_local: list[int] = []
        for local_index in priority:
            if kept_local and np.any(distances[local_index, kept_local] < threshold):
                continue
            kept_local.append(local_index)
        kept_indices.extend(int(group_indices[index]) for index in kept_local)

    source_indices = np.asarray(sorted(kept_indices), dtype=np.int64)
    if len(source_indices) == 0:
        raise RuntimeError("deduplication produced an empty anchor bank")
    deduplicated_anchors = anchors[source_indices]
    deduplicated_commands = command_ids[source_indices]
    assignments, nearest_distances = command_conditioned_assignment(
        trajectories,
        trajectory_commands,
        deduplicated_anchors,
        deduplicated_commands,
        scales,
        batch_size=assignment_batch_size,
    )
    reassigned_support = np.bincount(
        assignments, minlength=len(deduplicated_anchors)
    ).astype(np.int64)
    reassigned_local_mean = np.zeros(len(deduplicated_anchors), dtype=np.float32)
    for anchor_index in range(len(deduplicated_anchors)):
        assigned_distances = nearest_distances[assignments == anchor_index]
        if len(assigned_distances):
            reassigned_local_mean[anchor_index] = float(assigned_distances.mean())

    return AnchorDedupResult(
        anchors=deduplicated_anchors,
        command_ids=deduplicated_commands,
        support=reassigned_support,
        parent_ids=bank.parent_ids[source_indices],
        root_ids=bank.root_ids[source_indices],
        node_ids=bank.node_ids[source_indices],
        assignments=assignments,
        nearest_distances=nearest_distances,
        local_mean_distances=reassigned_local_mean,
        source_indices=source_indices,
        removed_count=int(len(anchors) - len(source_indices)),
    )
