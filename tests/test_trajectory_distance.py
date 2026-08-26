import numpy as np
import pytest
import torch

from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    FDE_WEIGHT,
    NORMALIZATION_EPSILON,
    TrajectoryCommand,
    TrajectoryDistanceScale,
    command_xy_weights,
    navsim_command_from_one_hot,
    normalized_trajectory_distance,
    raw_delta_distance,
    raw_fde_distance,
    trajectory_distance,
    trajectory_deltas,
    torch_command_ids_from_one_hot,
    torch_trajectory_distance,
)


def _linear_trajectory(step_x: float = 0.0, step_y: float = 0.0) -> np.ndarray:
    steps = np.arange(1, 9, dtype=np.float64)
    return np.stack([steps * step_x, steps * step_y], axis=-1)


def test_command_xy_weights_follow_driving_intent():
    assert command_xy_weights(TrajectoryCommand.STRAIGHT) == (0.6, 0.4)
    assert command_xy_weights(TrajectoryCommand.LEFT) == (0.4, 0.6)
    assert command_xy_weights(TrajectoryCommand.RIGHT) == (0.4, 0.6)


def test_navsim_one_hot_command_mapping_and_unknown_filter():
    assert navsim_command_from_one_hot([1, 0, 0, 0]) is TrajectoryCommand.LEFT
    assert navsim_command_from_one_hot([0, 1, 0, 0]) is TrajectoryCommand.STRAIGHT
    assert navsim_command_from_one_hot([0, 0, 1, 0]) is TrajectoryCommand.RIGHT
    assert navsim_command_from_one_hot([0, 0, 0, 1]) is None
    with pytest.raises(ValueError, match="one-hot"):
        navsim_command_from_one_hot([1, 1, 0, 0])


def test_trajectory_deltas_include_ego_origin_and_all_eight_steps():
    trajectory = _linear_trajectory(step_x=1.0, step_y=-0.5)

    deltas = trajectory_deltas(trajectory)

    assert deltas.shape == (8, 2)
    np.testing.assert_allclose(deltas, np.repeat([[1.0, -0.5]], 8, axis=0))


def test_raw_delta_distance_uses_command_weights_and_equal_timestep_weights():
    origin = np.zeros((8, 2), dtype=np.float64)
    longitudinal = _linear_trajectory(step_x=1.0)
    lateral = _linear_trajectory(step_y=1.0)
    first_delta_only = np.repeat([[1.0, 0.0]], 8, axis=0)
    last_delta_only = origin.copy()
    last_delta_only[-1, 0] = 1.0

    assert raw_delta_distance(origin, longitudinal, "straight") == pytest.approx(np.sqrt(0.6))
    assert raw_delta_distance(origin, lateral, "straight") == pytest.approx(np.sqrt(0.4))
    assert raw_delta_distance(origin, longitudinal, "left") == pytest.approx(np.sqrt(0.4))
    assert raw_delta_distance(origin, lateral, "right") == pytest.approx(np.sqrt(0.6))
    assert raw_delta_distance(origin, first_delta_only, "straight") == pytest.approx(
        raw_delta_distance(origin, last_delta_only, "straight")
    )


def test_raw_fde_distance_reuses_command_weights():
    trajectory_a = np.zeros((8, 2), dtype=np.float64)
    trajectory_b = np.zeros((8, 2), dtype=np.float64)
    trajectory_b[-1] = [3.0, 4.0]

    assert raw_fde_distance(trajectory_a, trajectory_b, "straight") == pytest.approx(
        np.sqrt(0.6 * 3.0**2 + 0.4 * 4.0**2)
    )
    assert raw_fde_distance(trajectory_a, trajectory_b, "left") == pytest.approx(
        np.sqrt(0.4 * 3.0**2 + 0.6 * 4.0**2)
    )


def test_normalized_trajectory_distance_uses_weak_fde_constraint():
    trajectory_a = np.zeros((8, 2), dtype=np.float64)
    trajectory_b = _linear_trajectory(step_x=1.0)
    delta_scale = np.sqrt(0.6) - NORMALIZATION_EPSILON
    fde_scale = 8.0 * np.sqrt(0.6) - NORMALIZATION_EPSILON

    distance = normalized_trajectory_distance(
        trajectory_a,
        trajectory_b,
        TrajectoryCommand.STRAIGHT,
        delta_scale=delta_scale,
        fde_scale=fde_scale,
    )

    assert distance == pytest.approx(1.0 + FDE_WEIGHT)


def test_formal_trajectory_distance_uses_fixed_command_scale():
    trajectory_a = np.zeros((8, 2), dtype=np.float64)
    trajectory_b = _linear_trajectory(step_x=1.0)
    scales = {
        TrajectoryCommand.STRAIGHT: TrajectoryDistanceScale(
            delta_scale=np.sqrt(0.6) - NORMALIZATION_EPSILON,
            fde_scale=8.0 * np.sqrt(0.6) - NORMALIZATION_EPSILON,
        )
    }

    distance = trajectory_distance(trajectory_a, trajectory_b, "straight", scales)

    assert distance == pytest.approx(1.0 + FDE_WEIGHT)


def test_distances_support_pairwise_broadcasting():
    trajectories = np.stack([_linear_trajectory(step_x=1.0), _linear_trajectory(step_y=1.0)])
    anchors = np.stack(
        [
            np.zeros((8, 2), dtype=np.float64),
            _linear_trajectory(step_x=1.0),
            _linear_trajectory(step_y=1.0),
        ]
    )

    distances = raw_delta_distance(trajectories[:, None], anchors[None], "straight")

    assert distances.shape == (2, 3)
    np.testing.assert_allclose(np.diag(distances[:, 1:]), 0.0)


@pytest.mark.parametrize("scale_name", ["delta_scale", "fde_scale"])
def test_normalized_trajectory_distance_rejects_non_positive_scales(scale_name):
    trajectory = np.zeros((8, 2), dtype=np.float64)
    scales = {"delta_scale": 1.0, "fde_scale": 1.0}
    scales[scale_name] = 0.0

    with pytest.raises(ValueError, match=scale_name):
        normalized_trajectory_distance(trajectory, trajectory, "straight", **scales)


def test_trajectory_distance_rejects_unknown_command_and_wrong_shape():
    trajectory = np.zeros((8, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="command"):
        raw_delta_distance(trajectory, trajectory, "unknown")
    with pytest.raises(ValueError, match=r"\[8, 2\]"):
        raw_delta_distance(trajectory[:-1], trajectory[:-1], "straight")


def test_torch_distance_matches_numpy_formal_distance_per_command():
    predictions = np.stack(
        [
            np.stack([_linear_trajectory(step_x=1.0), _linear_trajectory(step_y=0.5)]),
            np.stack([_linear_trajectory(step_x=0.5), _linear_trajectory(step_y=1.0)]),
        ]
    ).astype(np.float32)
    targets = np.zeros((2, 8, 2), dtype=np.float32)
    command_ids = torch.tensor([0, 1])
    delta_scales = torch.tensor([1.2, 1.3, 1.4])
    fde_scales = torch.tensor([2.1, 2.2, 2.3])

    actual = torch_trajectory_distance(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
        command_ids,
        delta_scales,
        fde_scales,
    ).numpy()
    expected = np.empty((2, 2), dtype=np.float32)
    for batch_index, command in enumerate(("left", "straight")):
        for mode_index in range(2):
            expected[batch_index, mode_index] = normalized_trajectory_distance(
                predictions[batch_index, mode_index],
                targets[batch_index],
                command,
                delta_scale=float(delta_scales[batch_index]),
                fde_scale=float(fde_scales[batch_index]),
            )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_torch_command_mapping_rejects_unknown():
    command = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="unknown"):
        torch_command_ids_from_one_hot(command)
