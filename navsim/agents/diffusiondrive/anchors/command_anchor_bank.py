"""Command-conditioned, worst-first expansion of a flat DiffusionDrive anchor bank."""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import numpy.typing as npt

from navsim.agents.diffusiondrive.anchors.calibration import allocate_command_anchors
from navsim.agents.diffusiondrive.anchors.kmedoids_splitter import (
    pairwise_trajectory_distance,
    select_k_medoids,
    split_parent_cluster,
)
from navsim.agents.diffusiondrive.anchors.metrics import cluster_variance
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


COMMAND_ORDER = tuple(TrajectoryCommand)


@dataclass(frozen=True)
class CommandAnchorBankConfig:
    """Fixed offline parameters for sections 11--17 of the implementation plan."""

    tau_split: float = 1.0
    child_min_ratio: float = 0.03
    max_split_medoid_samples: int = 4096
    max_total_anchors: int = 72
    distance_batch_size: int = 128
    assignment_batch_size: int = 4096
    diagnostics_coverage_epsilon: float = 1.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau_split) or self.tau_split <= 0:
            raise ValueError("tau_split must be finite and positive")
        if not 0 < self.child_min_ratio < 0.5:
            raise ValueError("child_min_ratio must be in (0, 0.5)")
        if self.max_split_medoid_samples < 2:
            raise ValueError("max_split_medoid_samples must be at least two")
        if self.max_total_anchors < 1:
            raise ValueError("max_total_anchors must be positive")
        if self.distance_batch_size < 1 or self.assignment_batch_size < 1:
            raise ValueError("distance batch sizes must be positive")
        if self.diagnostics_coverage_epsilon <= 0:
            raise ValueError("diagnostics_coverage_epsilon must be positive")


@dataclass(frozen=True)
class InitialCommandAnchorBankConfig:
    """Parameters for regenerating the initial Base Bank directly from command GT."""

    total_anchors: int = 20
    min_per_command: int = 1
    max_medoid_samples: int = 4096
    distance_batch_size: int = 128
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.min_per_command < 1:
            raise ValueError("min_per_command must be positive")
        if self.total_anchors < len(COMMAND_ORDER) * self.min_per_command:
            raise ValueError("total_anchors cannot satisfy min_per_command")
        if self.max_medoid_samples < 1 or self.distance_batch_size < 1:
            raise ValueError("sample and distance batch sizes must be positive")


@dataclass
class CommandAnchorBankResult:
    """Expanded flat bank with command, support, and split-tree metadata."""

    anchors: np.ndarray
    command_ids: np.ndarray
    support: np.ndarray
    parent_ids: np.ndarray
    root_ids: np.ndarray
    node_ids: np.ndarray
    assignments: np.ndarray
    nearest_distances: np.ndarray
    cluster_stats: list[Dict[str, Any]]
    history: list[Dict[str, Any]]
    stop_reason: str


@dataclass
class InitialCommandAnchorBankResult:
    """Initial real-GT medoids with labels inherited from their source GT groups."""

    anchors: np.ndarray
    command_ids: np.ndarray
    support: np.ndarray
    source_indices: np.ndarray
    command_counts: np.ndarray
    anchor_counts: np.ndarray
    sampled_counts: np.ndarray


def _normalize_commands(commands: npt.ArrayLike, expected_length: int, name: str) -> np.ndarray:
    commands = np.asarray(commands)
    if commands.shape != (expected_length,):
        raise ValueError(f"{name} must have shape [{expected_length}], got {commands.shape}")
    normalized = np.empty(expected_length, dtype=np.int64)
    for index, value in enumerate(commands):
        if isinstance(value, (int, np.integer)):
            command_id = int(value)
            if command_id < 0 or command_id >= len(COMMAND_ORDER):
                raise ValueError(f"unsupported {name} value at index {index}: {value}")
            normalized[index] = command_id
            continue
        try:
            normalized[index] = COMMAND_ORDER.index(TrajectoryCommand(str(value)))
        except ValueError as error:
            raise ValueError(f"unsupported {name} value at index {index}: {value}") from error
    return normalized


def command_conditioned_assignment(
    trajectories: npt.ArrayLike,
    trajectory_command_ids: npt.ArrayLike,
    anchors: npt.ArrayLike,
    anchor_command_ids: npt.ArrayLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign GT only to anchors carrying the same semantic command metadata."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError(f"trajectories must have shape [N, 8, 2], got {trajectories.shape}")
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError(f"anchors must have shape [K, 8, 2], got {anchors.shape}")
    if len(trajectories) == 0 or len(anchors) == 0:
        raise ValueError("trajectories and anchors must not be empty")
    if not np.isfinite(trajectories).all() or not np.isfinite(anchors).all():
        raise ValueError("trajectories and anchors must contain finite values")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    trajectory_command_ids = _normalize_commands(
        trajectory_command_ids, len(trajectories), "trajectory_command_ids"
    )
    anchor_command_ids = _normalize_commands(
        anchor_command_ids, len(anchors), "anchor_command_ids"
    )

    assignments = np.empty(len(trajectories), dtype=np.int64)
    nearest_distances = np.empty(len(trajectories), dtype=np.float32)
    for command_id, command in enumerate(COMMAND_ORDER):
        sample_indices = np.flatnonzero(trajectory_command_ids == command_id)
        if len(sample_indices) == 0:
            continue
        anchor_indices = np.flatnonzero(anchor_command_ids == command_id)
        if len(anchor_indices) == 0:
            raise ValueError(f"initial bank has no anchor for command: {command.value}")
        for start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[start : start + batch_size]
            distances = pairwise_trajectory_distance(
                trajectories[batch_indices],
                anchors[anchor_indices],
                command,
                scales,
                batch_size,
            )
            local_assignments = distances.argmin(axis=1)
            assignments[batch_indices] = anchor_indices[local_assignments]
            nearest_distances[batch_indices] = distances[
                np.arange(len(batch_indices)), local_assignments
            ]
    return assignments, nearest_distances


def build_initial_command_anchor_bank(
    trajectories: npt.ArrayLike,
    trajectory_commands: npt.ArrayLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    config: InitialCommandAnchorBankConfig = InitialCommandAnchorBankConfig(),
) -> InitialCommandAnchorBankResult:
    """Regenerate the initial bank inside each GT command group using formal ``D_traj``."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError(f"trajectories must have shape [N, 8, 2], got {trajectories.shape}")
    if len(trajectories) == 0 or not np.isfinite(trajectories).all():
        raise ValueError("trajectories must contain finite samples")
    trajectory_command_ids = _normalize_commands(
        trajectory_commands, len(trajectories), "trajectory_commands"
    )
    command_counts = {
        command: int(np.sum(trajectory_command_ids == command_id))
        for command_id, command in enumerate(COMMAND_ORDER)
    }
    allocations = allocate_command_anchors(
        config.total_anchors,
        command_counts,
        config.min_per_command,
    )

    all_anchors = []
    all_command_ids = []
    all_support = []
    all_source_indices = []
    sampled_counts = []
    for command_id, command in enumerate(COMMAND_ORDER):
        command_indices = np.flatnonzero(trajectory_command_ids == command_id)
        selection = select_k_medoids(
            trajectories[command_indices],
            command,
            scales,
            num_medoids=allocations[command],
            max_medoid_samples=config.max_medoid_samples,
            distance_batch_size=config.distance_batch_size,
            random_seed=config.random_seed + command_id,
        )
        all_anchors.append(selection.medoids)
        all_command_ids.append(
            np.full(allocations[command], command_id, dtype=np.int64)
        )
        all_support.append(selection.support)
        all_source_indices.append(command_indices[selection.medoid_parent_indices])
        sampled_counts.append(len(selection.sampled_indices))

    return InitialCommandAnchorBankResult(
        anchors=np.concatenate(all_anchors).astype(np.float32),
        command_ids=np.concatenate(all_command_ids),
        support=np.concatenate(all_support),
        source_indices=np.concatenate(all_source_indices),
        command_counts=np.asarray(
            [command_counts[command] for command in COMMAND_ORDER], dtype=np.int64
        ),
        anchor_counts=np.asarray(
            [allocations[command] for command in COMMAND_ORDER], dtype=np.int64
        ),
        sampled_counts=np.asarray(sampled_counts, dtype=np.int64),
    )


def _membership_is_unchanged(
    blocked_memberships: dict[int, np.ndarray],
    node_id: int,
    current_membership: np.ndarray,
) -> bool:
    """Keep a rejection blocked only while its exact GT membership is unchanged."""
    rejected_membership = blocked_memberships.get(node_id)
    if rejected_membership is None:
        return False
    if np.array_equal(rejected_membership, current_membership):
        return True
    del blocked_memberships[node_id]
    return False


def _cluster_statistics(
    trajectories: np.ndarray,
    trajectory_command_ids: np.ndarray,
    anchors: np.ndarray,
    anchor_command_ids: np.ndarray,
    node_ids: np.ndarray,
    assignments: np.ndarray,
    nearest_distances: np.ndarray,
    coverage_epsilon: float,
) -> list[Dict[str, Any]]:
    stats: list[Dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors):
        sample_indices = np.flatnonzero(assignments == anchor_index)
        command_id = int(anchor_command_ids[anchor_index])
        base: Dict[str, Any] = {
            "anchor_index": int(anchor_index),
            "node_id": int(node_ids[anchor_index]),
            "command_id": command_id,
            "command": COMMAND_ORDER[command_id].value,
            "support": int(len(sample_indices)),
        }
        if len(sample_indices) == 0:
            stats.append(
                {
                    **base,
                    "p95_d_traj": 0.0,
                    "mean_d_traj": 0.0,
                    "coverage_error_p95": 0.0,
                    "variance": 0.0,
                    "uncovered_count": 0,
                }
            )
            continue
        samples = trajectories[sample_indices]
        if not np.all(trajectory_command_ids[sample_indices] == command_id):
            raise RuntimeError("command mask invariant was violated")
        d_traj = nearest_distances[sample_indices]
        max_point_error = np.linalg.norm(samples - anchor[None], axis=-1).max(axis=-1)
        stats.append(
            {
                **base,
                "p95_d_traj": float(np.quantile(d_traj, 0.95)),
                "mean_d_traj": float(d_traj.mean()),
                "coverage_error_p95": float(np.quantile(max_point_error, 0.95)),
                "variance": cluster_variance(samples),
                "uncovered_count": int((max_point_error > coverage_epsilon).sum()),
            }
        )
    return stats


def _history_entry(
    trajectories: np.ndarray,
    anchors: np.ndarray,
    assignments: np.ndarray,
    nearest_distances: np.ndarray,
    split_event: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    assigned_anchors = anchors[assignments]
    point_error = np.linalg.norm(trajectories - assigned_anchors, axis=-1)
    ade = point_error.mean(axis=-1)
    fde = point_error[:, -1]
    max_point_error = point_error.max(axis=-1)
    return {
        "anchor_count": int(len(anchors)),
        "mean_nearest_d_traj": float(nearest_distances.mean()),
        "p50_nearest_d_traj": float(np.quantile(nearest_distances, 0.50)),
        "p95_nearest_d_traj": float(np.quantile(nearest_distances, 0.95)),
        "mean_nearest_ADE": float(ade.mean()),
        "mean_nearest_FDE": float(fde.mean()),
        "p95_max_error": float(np.quantile(max_point_error, 0.95)),
        "coverage@0.5m": float((max_point_error <= 0.5).mean()),
        "coverage@1m": float((max_point_error <= 1.0).mean()),
        "coverage@1.5m": float((max_point_error <= 1.5).mean()),
        "split_event": split_event,
    }


def build_command_anchor_bank(
    trajectories: npt.ArrayLike,
    trajectory_commands: npt.ArrayLike,
    initial_anchors: npt.ArrayLike,
    initial_anchor_commands: npt.ArrayLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
    config: CommandAnchorBankConfig = CommandAnchorBankConfig(),
) -> CommandAnchorBankResult:
    """Recursively split the highest-P95 command cluster with formal ``D_traj``."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    anchors = np.asarray(initial_anchors, dtype=np.float32).copy()
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError(f"trajectories must have shape [N, 8, 2], got {trajectories.shape}")
    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError(f"initial_anchors must have shape [K, 8, 2], got {anchors.shape}")
    if len(trajectories) == 0 or len(anchors) == 0:
        raise ValueError("trajectories and initial_anchors must not be empty")
    if not np.isfinite(trajectories).all() or not np.isfinite(anchors).all():
        raise ValueError("trajectories and initial_anchors must contain finite values")
    if config.max_total_anchors < len(anchors):
        raise ValueError("max_total_anchors cannot be smaller than the initial bank")

    trajectory_command_ids = _normalize_commands(
        trajectory_commands, len(trajectories), "trajectory_commands"
    )
    anchor_command_ids = _normalize_commands(
        initial_anchor_commands, len(anchors), "initial_anchor_commands"
    )
    root_ids = np.arange(len(anchors), dtype=np.int64)
    node_ids = np.arange(len(anchors), dtype=np.int64)
    parent_ids = np.full(len(anchors), -1, dtype=np.int64)
    next_node_id = len(anchors)
    blocked_memberships: dict[int, np.ndarray] = {}
    history: list[Dict[str, Any]] = []
    stop_reason = "no_cluster_above_tau_split"

    while True:
        assignments, nearest_distances = command_conditioned_assignment(
            trajectories,
            trajectory_command_ids,
            anchors,
            anchor_command_ids,
            scales,
            batch_size=config.assignment_batch_size,
        )
        stats = _cluster_statistics(
            trajectories,
            trajectory_command_ids,
            anchors,
            anchor_command_ids,
            node_ids,
            assignments,
            nearest_distances,
            config.diagnostics_coverage_epsilon,
        )
        if not history:
            history.append(
                _history_entry(trajectories, anchors, assignments, nearest_distances, None)
            )

        queue = []
        for stat in stats:
            if stat["p95_d_traj"] <= config.tau_split:
                continue
            anchor_index = int(stat["anchor_index"])
            node_id = int(stat["node_id"])
            membership = np.flatnonzero(assignments == anchor_index)
            if _membership_is_unchanged(blocked_memberships, node_id, membership):
                continue
            queue.append((-float(stat["p95_d_traj"]), node_id, anchor_index))
        heapq.heapify(queue)
        if not queue:
            if any(stat["p95_d_traj"] > config.tau_split for stat in stats):
                stop_reason = "no_acceptable_split"
            break
        if len(anchors) >= config.max_total_anchors:
            stop_reason = "max_total_anchors"
            break

        negative_p95, split_node_id, split_index = heapq.heappop(queue)
        command_id = int(anchor_command_ids[split_index])
        command = COMMAND_ORDER[command_id]
        parent_sample_indices = np.flatnonzero(assignments == split_index)
        event: Dict[str, Any] = {
            "parent_node_id": split_node_id,
            "command_id": command_id,
            "command": command.value,
            "parent_support": int(len(parent_sample_indices)),
            "parent_p95_d_traj": float(-negative_p95),
            "accepted": False,
        }

        if len(parent_sample_indices) < 2:
            event["reject_reason"] = "parent_has_fewer_than_two_samples"
            blocked_memberships[split_node_id] = parent_sample_indices.copy()
            history.append(
                _history_entry(trajectories, anchors, assignments, nearest_distances, event)
            )
            continue

        split = split_parent_cluster(
            trajectories[parent_sample_indices],
            command,
            scales,
            max_medoid_samples=config.max_split_medoid_samples,
            distance_batch_size=config.distance_batch_size,
            random_seed=config.random_seed + split_node_id,
        )
        command_sample_count = int(np.sum(trajectory_command_ids == command_id))
        required_support = config.child_min_ratio * command_sample_count
        event.update(
            {
                "child_support": split.support.tolist(),
                "required_child_support": float(required_support),
                "sampled_for_kmedoids": int(len(split.sampled_indices)),
                "medoid_parent_indices": parent_sample_indices[
                    split.medoid_parent_indices
                ].tolist(),
            }
        )
        if np.any(split.support < required_support):
            event["reject_reason"] = "child_support_below_command_ratio"
            blocked_memberships[split_node_id] = parent_sample_indices.copy()
            history.append(
                _history_entry(trajectories, anchors, assignments, nearest_distances, event)
            )
            continue

        keep = np.arange(len(anchors)) != split_index
        parent_root_id = int(root_ids[split_index])
        anchors = np.concatenate([anchors[keep], split.medoids.astype(np.float32)], axis=0)
        anchor_command_ids = np.concatenate(
            [anchor_command_ids[keep], np.full(2, command_id, dtype=np.int64)]
        )
        root_ids = np.concatenate(
            [root_ids[keep], np.full(2, parent_root_id, dtype=np.int64)]
        )
        node_ids = np.concatenate(
            [node_ids[keep], np.asarray([next_node_id, next_node_id + 1], dtype=np.int64)]
        )
        parent_ids = np.concatenate(
            [parent_ids[keep], np.full(2, split_node_id, dtype=np.int64)]
        )
        next_node_id += 2
        event["accepted"] = True

        updated_assignments, updated_distances = command_conditioned_assignment(
            trajectories,
            trajectory_command_ids,
            anchors,
            anchor_command_ids,
            scales,
            batch_size=config.assignment_batch_size,
        )
        history.append(
            _history_entry(
                trajectories,
                anchors,
                updated_assignments,
                updated_distances,
                event,
            )
        )

    support = np.bincount(assignments, minlength=len(anchors)).astype(np.int64)
    return CommandAnchorBankResult(
        anchors=anchors,
        command_ids=anchor_command_ids,
        support=support,
        parent_ids=parent_ids,
        root_ids=root_ids,
        node_ids=node_ids,
        assignments=assignments,
        nearest_distances=nearest_distances,
        cluster_stats=stats,
        history=history,
        stop_reason=stop_reason,
    )


def config_to_dict(config: CommandAnchorBankConfig) -> Dict[str, Any]:
    return asdict(config)
