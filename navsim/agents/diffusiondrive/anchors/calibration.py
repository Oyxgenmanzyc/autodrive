"""Offline calibration bank and fixed scales for normalized trajectory distance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import kmedoids
import numpy as np
import numpy.typing as npt

from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    CommandLike,
    TrajectoryCommand,
    TrajectoryDistanceScale,
    raw_delta_distance,
    raw_fde_distance,
)


COMMAND_ORDER = (
    TrajectoryCommand.LEFT,
    TrajectoryCommand.STRAIGHT,
    TrajectoryCommand.RIGHT,
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Parameters for the one-time command-conditioned calibration bank."""

    total_anchors: int = 60
    min_per_command: int = 10
    max_kmedoids_samples: int = 10_000
    distance_batch_size: int = 128
    assignment_batch_size: int = 4096
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.total_anchors < len(COMMAND_ORDER) * self.min_per_command:
            raise ValueError("total_anchors cannot satisfy min_per_command for all commands")
        if self.min_per_command < 1:
            raise ValueError("min_per_command must be positive")
        if self.max_kmedoids_samples < 1:
            raise ValueError("max_kmedoids_samples must be positive")
        if self.distance_batch_size < 1 or self.assignment_batch_size < 1:
            raise ValueError("distance batch sizes must be positive")


@dataclass
class CalibrationResult:
    """Calibration medoids, fixed scales, and audit metadata."""

    anchors: np.ndarray
    command_ids: np.ndarray
    source_indices: np.ndarray
    support: np.ndarray
    command_counts: np.ndarray
    anchor_counts: np.ndarray
    sampled_counts: np.ndarray
    scales: Dict[TrajectoryCommand, TrajectoryDistanceScale]

    def scale_for(self, command: CommandLike) -> TrajectoryDistanceScale:
        return self.scales[TrajectoryCommand(command)]


def allocate_command_anchors(
    total_k: int,
    command_counts: Mapping[CommandLike, int],
    min_per_command: int,
) -> Dict[TrajectoryCommand, int]:
    """Allocate an exact total using proportional quotas, lower bounds, and largest remainder."""
    if min_per_command < 1:
        raise ValueError("min_per_command must be positive")
    if total_k < len(COMMAND_ORDER) * min_per_command:
        raise ValueError("total_k cannot satisfy min_per_command for all commands")

    counts = {command: int(command_counts.get(command, command_counts.get(command.value, 0))) for command in COMMAND_ORDER}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("every command must contain at least one trajectory")
    if total_k > sum(counts.values()):
        raise ValueError("total_k cannot exceed the total trajectory count")

    fixed: Dict[TrajectoryCommand, int] = {}
    free = list(COMMAND_ORDER)
    remaining = total_k
    while True:
        free_count = sum(counts[command] for command in free)
        quotas = {command: remaining * counts[command] / free_count for command in free}
        below_minimum = [command for command in free if quotas[command] < min_per_command]
        if not below_minimum:
            break
        for command in below_minimum:
            fixed[command] = min_per_command
            free.remove(command)
            remaining -= min_per_command

    floors = {command: int(np.floor(quotas[command])) for command in free}
    unassigned = remaining - sum(floors.values())
    remainder_order = sorted(
        free,
        key=lambda command: (-(quotas[command] - floors[command]), COMMAND_ORDER.index(command)),
    )
    for command in remainder_order[:unassigned]:
        floors[command] += 1

    allocation = {**fixed, **floors}
    allocation = {command: allocation[command] for command in COMMAND_ORDER}
    if any(allocation[command] > counts[command] for command in COMMAND_ORDER):
        raise ValueError("a command has fewer trajectories than its allocated anchors")
    return allocation


def pairwise_raw_delta_distance(
    trajectories: npt.ArrayLike,
    anchors: npt.ArrayLike,
    command: CommandLike,
    batch_size: int,
) -> np.ndarray:
    """Compute a bounded-memory ``[N, K]`` raw Delta Distance matrix."""
    trajectories = np.asarray(trajectories)
    anchors = np.asarray(anchors)
    if trajectories.ndim != 3 or anchors.ndim != 3:
        raise ValueError("trajectories and anchors must have shape [N/K, 8, 2]")
    if trajectories.shape[1:] != (8, 2) or anchors.shape[1:] != (8, 2):
        raise ValueError("trajectories and anchors must have shape [N/K, 8, 2]")
    if len(trajectories) == 0 or len(anchors) == 0:
        raise ValueError("trajectories and anchors must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    distances = np.empty((len(trajectories), len(anchors)), dtype=np.float32)
    for start in range(0, len(trajectories), batch_size):
        end = min(start + batch_size, len(trajectories))
        distances[start:end] = raw_delta_distance(
            trajectories[start:end, None],
            anchors[None],
            command,
        )
    return distances


def calibrate_command_scales(
    trajectories: npt.ArrayLike,
    anchors: npt.ArrayLike,
    command: CommandLike,
    batch_size: int,
) -> Tuple[TrajectoryDistanceScale, np.ndarray]:
    """Compute P95 Delta/FDE scales using the same Delta-nearest medoid per GT."""
    trajectories = np.asarray(trajectories)
    anchors = np.asarray(anchors)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if trajectories.ndim != 3 or anchors.ndim != 3:
        raise ValueError("trajectories and anchors must have shape [N/K, 8, 2]")
    if trajectories.shape[1:] != (8, 2) or anchors.shape[1:] != (8, 2):
        raise ValueError("trajectories and anchors must have shape [N/K, 8, 2]")
    if len(trajectories) == 0 or len(anchors) == 0:
        raise ValueError("trajectories and anchors must not be empty")
    delta_errors = np.empty(len(trajectories), dtype=np.float32)
    fde_errors = np.empty(len(trajectories), dtype=np.float32)
    assignments = np.empty(len(trajectories), dtype=np.int64)

    for start in range(0, len(trajectories), batch_size):
        end = min(start + batch_size, len(trajectories))
        distances = pairwise_raw_delta_distance(
            trajectories[start:end],
            anchors,
            command,
            batch_size=batch_size,
        )
        local_assignments = distances.argmin(axis=1)
        assignments[start:end] = local_assignments
        delta_errors[start:end] = distances[np.arange(end - start), local_assignments]
        fde_errors[start:end] = raw_fde_distance(
            trajectories[start:end],
            anchors[local_assignments],
            command,
        )

    scale = TrajectoryDistanceScale(
        delta_scale=float(np.quantile(delta_errors, 0.95)),
        fde_scale=float(np.quantile(fde_errors, 0.95)),
    )
    support = np.bincount(assignments, minlength=len(anchors)).astype(np.int64)
    return scale, support


def _normalize_commands(commands: npt.ArrayLike, expected_length: int) -> np.ndarray:
    commands = np.asarray(commands)
    if commands.shape != (expected_length,):
        raise ValueError(f"commands must have shape [{expected_length}], got {commands.shape}")
    normalized = np.empty(expected_length, dtype="<U8")
    for index, command in enumerate(commands):
        try:
            normalized[index] = TrajectoryCommand(str(command)).value
        except ValueError as error:
            raise ValueError(f"unsupported command at index {index}: {command}") from error
    return normalized


def build_calibration_bank(
    trajectories: npt.ArrayLike,
    commands: npt.ArrayLike,
    config: CalibrationConfig = CalibrationConfig(),
) -> CalibrationResult:
    """Build command-specific raw-Delta K-Medoids and fixed P95 normalization scales."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 2):
        raise ValueError(f"trajectories must have shape [N, 8, 2], got {trajectories.shape}")
    if len(trajectories) == 0 or not np.isfinite(trajectories).all():
        raise ValueError("trajectories must contain finite samples")
    commands = _normalize_commands(commands, len(trajectories))

    command_counts = {command: int(np.sum(commands == command.value)) for command in COMMAND_ORDER}
    allocations = allocate_command_anchors(
        config.total_anchors,
        command_counts,
        config.min_per_command,
    )

    all_anchors = []
    all_command_ids = []
    all_source_indices = []
    all_support = []
    sampled_counts = []
    scales: Dict[TrajectoryCommand, TrajectoryDistanceScale] = {}

    for command_id, command in enumerate(COMMAND_ORDER):
        command_indices = np.flatnonzero(commands == command.value)
        rng = np.random.RandomState(config.random_seed + command_id)
        if len(command_indices) > config.max_kmedoids_samples:
            sampled_indices = np.sort(
                rng.choice(command_indices, config.max_kmedoids_samples, replace=False)
            )
        else:
            sampled_indices = command_indices
        sampled_trajectories = trajectories[sampled_indices]
        sampled_counts.append(len(sampled_indices))
        if allocations[command] > len(sampled_trajectories):
            raise ValueError(
                f"{command.value} has fewer sampled trajectories than allocated anchors"
            )

        distance_matrix = pairwise_raw_delta_distance(
            sampled_trajectories,
            sampled_trajectories,
            command,
            batch_size=config.distance_batch_size,
        )
        kmedoids_result = kmedoids.fasterpam(
            distance_matrix,
            allocations[command],
            init="random",
            random_state=config.random_seed + command_id,
        )
        local_medoid_indices = np.sort(np.asarray(kmedoids_result.medoids, dtype=np.int64))
        if len(local_medoid_indices) != allocations[command]:
            raise RuntimeError("K-Medoids returned an unexpected number of medoids")
        if len(np.unique(local_medoid_indices)) != len(local_medoid_indices):
            raise RuntimeError("K-Medoids returned duplicate medoid indices")
        if np.any(local_medoid_indices < 0) or np.any(local_medoid_indices >= len(sampled_indices)):
            raise RuntimeError("K-Medoids returned an out-of-range medoid index")
        source_indices = sampled_indices[local_medoid_indices]
        anchors = trajectories[source_indices]

        scale, support = calibrate_command_scales(
            trajectories[command_indices],
            anchors,
            command,
            batch_size=config.assignment_batch_size,
        )
        scales[command] = scale
        all_anchors.append(anchors)
        all_command_ids.append(np.full(len(anchors), command_id, dtype=np.int64))
        all_source_indices.append(source_indices)
        all_support.append(support)

    return CalibrationResult(
        anchors=np.concatenate(all_anchors).astype(np.float32),
        command_ids=np.concatenate(all_command_ids),
        source_indices=np.concatenate(all_source_indices),
        support=np.concatenate(all_support),
        command_counts=np.asarray([command_counts[command] for command in COMMAND_ORDER]),
        anchor_counts=np.asarray([allocations[command] for command in COMMAND_ORDER]),
        sampled_counts=np.asarray(sampled_counts),
        scales=scales,
    )


def save_calibration_artifacts(
    output_dir: Path,
    result: CalibrationResult,
    config: CalibrationConfig,
    stem: str = "trajectory_distance_calibration",
) -> Dict[str, Path]:
    """Save calibration medoids to NPZ and the six fixed scales to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        anchors=result.anchors,
        command_ids=result.command_ids,
        source_indices=result.source_indices,
        support=result.support,
        command_counts=result.command_counts,
        anchor_counts=result.anchor_counts,
        sampled_counts=result.sampled_counts,
        delta_scales=np.asarray([result.scales[command].delta_scale for command in COMMAND_ORDER]),
        fde_scales=np.asarray([result.scales[command].fde_scale for command in COMMAND_ORDER]),
    )
    report: Dict[str, Any] = {
        "version": 1,
        "config": asdict(config),
        "command_order": [command.value for command in COMMAND_ORDER],
        "calibration_anchor_count": int(len(result.anchors)),
        "npz_path": npz_path.name,
        "commands": {},
    }
    for command_id, command in enumerate(COMMAND_ORDER):
        command_support = result.support[result.command_ids == command_id]
        report["commands"][command.value] = {
            "trajectory_count": int(result.command_counts[command_id]),
            "anchor_count": int(result.anchor_counts[command_id]),
            "sampled_for_kmedoids": int(result.sampled_counts[command_id]),
            "delta_scale": result.scales[command].delta_scale,
            "fde_scale": result.scales[command].fde_scale,
            "support": command_support.tolist(),
        }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return {"npz": npz_path, "json": json_path}


def load_calibration_scales(path: Path) -> Dict[TrajectoryCommand, TrajectoryDistanceScale]:
    """Load immutable command-specific scales from a calibration JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as file:
        report = json.load(file)
    return {
        command: TrajectoryDistanceScale(
            delta_scale=float(report["commands"][command.value]["delta_scale"]),
            fde_scale=float(report["commands"][command.value]["fde_scale"]),
        )
        for command in COMMAND_ORDER
    }
