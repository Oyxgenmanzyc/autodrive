"""Command-conditioned trajectory distance for adaptive anchor construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch


NUM_TIMESTEPS = 8
FDE_WEIGHT = 0.2
NORMALIZATION_EPSILON = 1e-6


class TrajectoryCommand(str, Enum):
    """Semantic command labels used by the command-conditioned anchor bank."""

    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"


CommandLike = Union[TrajectoryCommand, str]
NAVSIM_ONE_HOT_COMMANDS: Tuple[Optional[TrajectoryCommand], ...] = (
    TrajectoryCommand.LEFT,
    TrajectoryCommand.STRAIGHT,
    TrajectoryCommand.RIGHT,
    None,
)


@dataclass(frozen=True)
class TrajectoryDistanceScale:
    """Fixed P95 normalization scales for one semantic command."""

    delta_scale: float
    fde_scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta_scale) or self.delta_scale <= 0:
            raise ValueError("delta_scale must be finite and positive")
        if not np.isfinite(self.fde_scale) or self.fde_scale <= 0:
            raise ValueError("fde_scale must be finite and positive")


def navsim_command_from_one_hot(driving_command: npt.ArrayLike) -> Optional[TrajectoryCommand]:
    """Map NAVSIM ``[left, straight, right, unknown]`` one-hot data to a semantic command."""
    driving_command = np.asarray(driving_command)
    if driving_command.shape != (len(NAVSIM_ONE_HOT_COMMANDS),):
        raise ValueError(
            f"driving_command must have shape [{len(NAVSIM_ONE_HOT_COMMANDS)}], "
            f"got {driving_command.shape}"
        )
    active = np.flatnonzero(driving_command)
    if len(active) != 1:
        raise ValueError("driving_command must be one-hot")
    return NAVSIM_ONE_HOT_COMMANDS[int(active[0])]


def command_xy_weights(command: CommandLike) -> Tuple[float, float]:
    """Return ``(longitudinal_x, lateral_y)`` weights for one command."""
    try:
        command = TrajectoryCommand(command)
    except ValueError as error:
        valid = ", ".join(item.value for item in TrajectoryCommand)
        raise ValueError(f"command must be one of: {valid}") from error

    if command is TrajectoryCommand.STRAIGHT:
        return 0.6, 0.4
    return 0.4, 0.6


def _validate_trajectory(name: str, trajectory: npt.ArrayLike) -> np.ndarray:
    trajectory = np.asarray(trajectory)
    if trajectory.ndim < 2 or trajectory.shape[-2:] != (NUM_TIMESTEPS, 2):
        raise ValueError(
            f"{name} must end with shape [{NUM_TIMESTEPS}, 2], got {trajectory.shape}"
        )
    if not np.issubdtype(trajectory.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} contains non-finite values")
    return trajectory


def _validate_trajectory_pair(
    trajectory_a: npt.ArrayLike,
    trajectory_b: npt.ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    trajectory_a = _validate_trajectory("trajectory_a", trajectory_a)
    trajectory_b = _validate_trajectory("trajectory_b", trajectory_b)
    try:
        np.broadcast_shapes(trajectory_a.shape[:-2], trajectory_b.shape[:-2])
    except ValueError as error:
        raise ValueError("trajectory leading dimensions must be broadcast-compatible") from error
    return trajectory_a, trajectory_b


def trajectory_deltas(trajectory: npt.ArrayLike) -> np.ndarray:
    """Return all eight deltas after prepending the ego origin ``p0=(0, 0)``."""
    trajectory = _validate_trajectory("trajectory", trajectory)
    origin = np.zeros_like(trajectory[..., :1, :])
    return np.diff(np.concatenate([origin, trajectory], axis=-2), axis=-2)


def raw_delta_distance(
    trajectory_a: npt.ArrayLike,
    trajectory_b: npt.ArrayLike,
    command: CommandLike,
) -> np.ndarray:
    """Return mean command-weighted L2 distance between the eight trajectory deltas."""
    trajectory_a, trajectory_b = _validate_trajectory_pair(trajectory_a, trajectory_b)
    weight_x, weight_y = command_xy_weights(command)
    delta_difference = trajectory_deltas(trajectory_a) - trajectory_deltas(trajectory_b)
    squared_distance = (
        weight_x * np.square(delta_difference[..., 0])
        + weight_y * np.square(delta_difference[..., 1])
    )
    return np.sqrt(squared_distance).mean(axis=-1)


def raw_fde_distance(
    trajectory_a: npt.ArrayLike,
    trajectory_b: npt.ArrayLike,
    command: CommandLike,
) -> np.ndarray:
    """Return command-weighted L2 distance between the final trajectory points."""
    trajectory_a, trajectory_b = _validate_trajectory_pair(trajectory_a, trajectory_b)
    weight_x, weight_y = command_xy_weights(command)
    endpoint_difference = trajectory_a[..., -1, :] - trajectory_b[..., -1, :]
    squared_distance = (
        weight_x * np.square(endpoint_difference[..., 0])
        + weight_y * np.square(endpoint_difference[..., 1])
    )
    return np.sqrt(squared_distance)


def normalized_trajectory_distance(
    trajectory_a: npt.ArrayLike,
    trajectory_b: npt.ArrayLike,
    command: CommandLike,
    delta_scale: float,
    fde_scale: float,
) -> np.ndarray:
    """Return normalized ``D_traj`` with the fixed weak FDE contribution."""
    if not np.isfinite(delta_scale) or delta_scale <= 0:
        raise ValueError("delta_scale must be finite and positive")
    if not np.isfinite(fde_scale) or fde_scale <= 0:
        raise ValueError("fde_scale must be finite and positive")

    delta_distance = raw_delta_distance(trajectory_a, trajectory_b, command)
    fde_distance = raw_fde_distance(trajectory_a, trajectory_b, command)
    return (
        delta_distance / (delta_scale + NORMALIZATION_EPSILON)
        + FDE_WEIGHT * fde_distance / (fde_scale + NORMALIZATION_EPSILON)
    )


def trajectory_distance(
    trajectory_a: npt.ArrayLike,
    trajectory_b: npt.ArrayLike,
    command: CommandLike,
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
) -> np.ndarray:
    """Return formal ``D_traj`` using the fixed offline scale for ``command``."""
    command = TrajectoryCommand(command)
    try:
        scale = scales[command]
    except KeyError as error:
        raise ValueError(f"missing calibration scale for command: {command.value}") from error
    return normalized_trajectory_distance(
        trajectory_a,
        trajectory_b,
        command,
        delta_scale=scale.delta_scale,
        fde_scale=scale.fde_scale,
    )


def torch_valid_command_mask(driving_command: torch.Tensor) -> torch.Tensor:
    """Return samples carrying a supported left/straight/right one-hot command."""
    if driving_command.ndim != 2 or driving_command.shape[1] != len(NAVSIM_ONE_HOT_COMMANDS):
        raise ValueError(
            "driving_command must have shape "
            f"[B, {len(NAVSIM_ONE_HOT_COMMANDS)}], got {tuple(driving_command.shape)}"
        )
    active = driving_command != 0
    if torch.any(active.sum(dim=1) != 1):
        raise ValueError("driving_command rows must be one-hot")
    return ~active[:, -1]


def torch_command_ids_from_one_hot(driving_command: torch.Tensor) -> torch.Tensor:
    """Convert supported NAVSIM one-hot command batches to ids 0--2."""
    valid_command = torch_valid_command_mask(driving_command)
    if torch.any(~valid_command):
        raise ValueError("unknown driving_command has no command-conditioned Anchor group")
    active = driving_command != 0
    return active[:, :3].to(dtype=torch.int64).argmax(dim=1)


def torch_trajectory_distance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    command_ids: torch.Tensor,
    delta_scales: torch.Tensor,
    fde_scales: torch.Tensor,
) -> torch.Tensor:
    """Torch implementation of formal ``D_traj`` for ``[B, K, 8, 2]`` predictions."""
    if predictions.ndim != 4 or predictions.shape[-2:] != (NUM_TIMESTEPS, 2):
        raise ValueError(
            f"predictions must have shape [B, K, {NUM_TIMESTEPS}, 2], "
            f"got {tuple(predictions.shape)}"
        )
    if targets.shape != (predictions.shape[0], NUM_TIMESTEPS, 2):
        raise ValueError(
            f"targets must have shape [B, {NUM_TIMESTEPS}, 2], got {tuple(targets.shape)}"
        )
    if command_ids.shape != (predictions.shape[0],):
        raise ValueError(f"command_ids must have shape [B], got {tuple(command_ids.shape)}")
    if torch.any((command_ids < 0) | (command_ids >= len(TrajectoryCommand))):
        raise ValueError("command_ids must contain only left/straight/right ids 0--2")

    delta_scales = torch.as_tensor(
        delta_scales, dtype=predictions.dtype, device=predictions.device
    )
    fde_scales = torch.as_tensor(
        fde_scales, dtype=predictions.dtype, device=predictions.device
    )
    if delta_scales.shape != (len(TrajectoryCommand),) or fde_scales.shape != (
        len(TrajectoryCommand),
    ):
        raise ValueError("delta_scales and fde_scales must have shape [3]")
    if torch.any(~torch.isfinite(delta_scales)) or torch.any(delta_scales <= 0):
        raise ValueError("delta_scales must be finite and positive")
    if torch.any(~torch.isfinite(fde_scales)) or torch.any(fde_scales <= 0):
        raise ValueError("fde_scales must be finite and positive")

    targets = targets.to(dtype=predictions.dtype, device=predictions.device)
    command_ids = command_ids.to(device=predictions.device, dtype=torch.int64)
    weight_x = torch.where(
        command_ids == list(TrajectoryCommand).index(TrajectoryCommand.STRAIGHT),
        predictions.new_tensor(0.6),
        predictions.new_tensor(0.4),
    ).view(-1, 1, 1)
    weight_y = 1.0 - weight_x

    prediction_origin = torch.zeros_like(predictions[..., :1, :])
    target_origin = torch.zeros_like(targets[..., :1, :])
    prediction_deltas = torch.diff(
        torch.cat([prediction_origin, predictions], dim=-2), dim=-2
    )
    target_deltas = torch.diff(torch.cat([target_origin, targets], dim=-2), dim=-2)
    delta_difference = prediction_deltas - target_deltas[:, None]
    raw_delta = torch.sqrt(
        weight_x * delta_difference[..., 0].square()
        + weight_y * delta_difference[..., 1].square()
    ).mean(dim=-1)

    endpoint_difference = predictions[..., -1, :] - targets[:, None, -1, :]
    raw_fde = torch.sqrt(
        weight_x[..., 0] * endpoint_difference[..., 0].square()
        + weight_y[..., 0] * endpoint_difference[..., 1].square()
    )
    selected_delta_scales = delta_scales[command_ids].view(-1, 1)
    selected_fde_scales = fde_scales[command_ids].view(-1, 1)
    return raw_delta / (selected_delta_scales + NORMALIZATION_EPSILON) + FDE_WEIGHT * (
        raw_fde / (selected_fde_scales + NORMALIZATION_EPSILON)
    )
