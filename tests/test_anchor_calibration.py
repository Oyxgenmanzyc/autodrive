import json

import numpy as np
import pytest

from navsim.agents.diffusiondrive.anchors import calibration
from navsim.agents.diffusiondrive.anchors.calibration import (
    COMMAND_ORDER,
    CalibrationConfig,
    CalibrationResult,
    allocate_command_anchors,
    build_calibration_bank,
    calibrate_command_scales,
    load_calibration_scales,
    save_calibration_artifacts,
)
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


def _trajectory_from_deltas(deltas):
    return np.cumsum(np.asarray(deltas, dtype=np.float32), axis=0)


def _linear_trajectory(step_x: float, step_y: float = 0.0) -> np.ndarray:
    return _trajectory_from_deltas(np.repeat([[step_x, step_y]], 8, axis=0))


def test_allocate_command_anchors_applies_minimum_and_largest_remainder():
    allocation = allocate_command_anchors(
        total_k=60,
        command_counts={"left": 15, "straight": 70, "right": 15},
        min_per_command=10,
    )

    assert allocation == {
        TrajectoryCommand.LEFT: 10,
        TrajectoryCommand.STRAIGHT: 40,
        TrajectoryCommand.RIGHT: 10,
    }
    assert sum(allocation.values()) == 60


def test_calibration_scales_use_same_delta_nearest_medoid_for_fde():
    ground_truth = _linear_trajectory(1.0)[None]
    delta_nearest = _linear_trajectory(1.1)
    endpoint_nearest = _trajectory_from_deltas(
        [[2.0, 0.0], [0.0, 0.0]] + [[1.0, 0.0]] * 6
    )
    anchors = np.stack([delta_nearest, endpoint_nearest])

    scale, support = calibrate_command_scales(
        ground_truth,
        anchors,
        TrajectoryCommand.STRAIGHT,
        batch_size=8,
    )

    assert scale.delta_scale == pytest.approx(np.sqrt(0.6) * 0.1)
    assert scale.fde_scale == pytest.approx(np.sqrt(0.6) * 0.8)
    np.testing.assert_array_equal(support, [1, 0])


def test_build_calibration_bank_keeps_real_sample_medoids(monkeypatch):
    trajectories = []
    commands = []
    for command_id, command in enumerate(COMMAND_ORDER):
        for offset in range(3):
            trajectories.append(_linear_trajectory(1.0 + command_id + offset * 0.1, offset * 0.05))
            commands.append(command.value)
    trajectories = np.stack(trajectories)

    class FakeKMedoidsResult:
        medoids = np.asarray([1], dtype=np.int64)

    monkeypatch.setattr(calibration.kmedoids, "fasterpam", lambda *args, **kwargs: FakeKMedoidsResult())
    config = CalibrationConfig(
        total_anchors=3,
        min_per_command=1,
        max_kmedoids_samples=3,
        distance_batch_size=2,
        assignment_batch_size=2,
    )

    result = build_calibration_bank(trajectories, np.asarray(commands), config)

    np.testing.assert_array_equal(result.source_indices, [1, 4, 7])
    np.testing.assert_allclose(result.anchors, trajectories[result.source_indices])
    np.testing.assert_array_equal(result.command_ids, [0, 1, 2])
    assert sum(result.support) == len(trajectories)


def test_save_and_load_calibration_artifacts(tmp_path):
    scales = {
        command: TrajectoryDistanceScale(delta_scale=1.0 + index, fde_scale=2.0 + index)
        for index, command in enumerate(COMMAND_ORDER)
    }
    result = CalibrationResult(
        anchors=np.zeros((3, 8, 2), dtype=np.float32),
        command_ids=np.arange(3, dtype=np.int64),
        source_indices=np.arange(3, dtype=np.int64),
        support=np.asarray([10, 20, 30], dtype=np.int64),
        command_counts=np.asarray([10, 20, 30], dtype=np.int64),
        anchor_counts=np.ones(3, dtype=np.int64),
        sampled_counts=np.asarray([10, 20, 30], dtype=np.int64),
        scales=scales,
    )
    config = CalibrationConfig(total_anchors=3, min_per_command=1)

    paths = save_calibration_artifacts(tmp_path, result, config)
    loaded_scales = load_calibration_scales(paths["json"])

    assert paths["npz"].is_file()
    assert paths["json"].is_file()
    assert loaded_scales == scales
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["calibration_anchor_count"] == 3
    assert report["command_order"] == ["left", "straight", "right"]
