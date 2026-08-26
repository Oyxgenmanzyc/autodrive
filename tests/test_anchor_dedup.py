import numpy as np

from navsim.agents.diffusiondrive.anchors.anchor_dedup import (
    deduplicate_command_anchor_bank,
)
from navsim.agents.diffusiondrive.anchors.command_anchor_bank import CommandAnchorBankResult
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


def _constant_trajectories(offsets):
    offsets = np.asarray(offsets, dtype=np.float32)
    return np.repeat(offsets[:, None, :], 8, axis=1)


def _scales():
    return {
        command: TrajectoryDistanceScale(delta_scale=1.0, fde_scale=1.0)
        for command in TrajectoryCommand
    }


def test_dedup_is_command_local_and_uses_support_then_local_mean_priority():
    anchors = _constant_trajectories(
        [[0.0, 0.0], [0.01, 0.0], [5.0, 0.0], [5.01, 0.0], [0.01, 0.0]]
    )
    bank = CommandAnchorBankResult(
        anchors=anchors,
        command_ids=np.asarray([0, 0, 0, 0, 1]),
        support=np.asarray([5, 10, 10, 10, 20]),
        parent_ids=np.full(5, -1, dtype=np.int64),
        root_ids=np.arange(5, dtype=np.int64),
        node_ids=np.arange(5, dtype=np.int64),
        assignments=np.zeros(6, dtype=np.int64),
        nearest_distances=np.zeros(6, dtype=np.float32),
        cluster_stats=[
            {"mean_d_traj": value}
            for value in (0.1, 0.2, 0.5, 0.1, 0.1)
        ],
        history=[],
        stop_reason="test",
    )
    trajectories = _constant_trajectories(
        [[0.01, 0.0], [0.02, 0.0], [5.0, 0.0], [5.01, 0.0], [0.01, 0.0], [0.02, 0.0]]
    )
    commands = np.asarray(["left"] * 4 + ["straight"] * 2)

    result = deduplicate_command_anchor_bank(
        trajectories,
        commands,
        bank,
        _scales(),
        threshold=0.15,
    )

    np.testing.assert_array_equal(result.source_indices, [1, 3, 4])
    np.testing.assert_array_equal(result.command_ids, [0, 0, 1])
    assert result.removed_count == 2
    assert result.support.sum() == len(trajectories)
    assert result.support[result.command_ids == 0].sum() == 4
    assert result.support[result.command_ids == 1].sum() == 2
