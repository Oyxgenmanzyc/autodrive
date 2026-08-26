import numpy as np

from navsim.agents.diffusiondrive.anchors.anchor_metrics import (
    command_anchor_coverage_metrics,
    command_anchor_diversity_metrics,
)
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


def test_coverage_metrics_include_formal_distance_and_legacy_diagnostics():
    anchors = _constant_trajectories([[0.0, 0.0], [1.0, 0.0]])
    trajectories = anchors.copy()

    metrics = command_anchor_coverage_metrics(
        trajectories,
        anchors,
        assignments=np.asarray([0, 1]),
        nearest_distances=np.zeros(2, dtype=np.float32),
    )

    assert metrics["mean_nearest_d_traj"] == 0.0
    assert metrics["p50_nearest_d_traj"] == 0.0
    assert metrics["p95_nearest_d_traj"] == 0.0
    assert metrics["mean_nearest_ADE"] == 0.0
    assert metrics["mean_nearest_FDE"] == 0.0
    assert metrics["p95_max_point_error"] == 0.0
    assert metrics["coverage@0.5m"] == 1.0
    assert metrics["coverage@1m"] == 1.0
    assert metrics["coverage@1.5m"] == 1.0


def test_diversity_metrics_compare_only_anchors_from_the_same_command():
    anchors = _constant_trajectories(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    )
    metrics = command_anchor_diversity_metrics(
        anchors,
        command_ids=np.asarray([0, 0, 1, 1]),
        scales=_scales(),
    )

    distribution = metrics["pairwise_d_traj_distribution"]
    assert distribution["count"] == 2
    assert distribution["min"] > 0
    assert len(distribution["histogram_edges"]) == 6
    assert len(distribution["histogram_counts"]) == 5
    assert metrics["by_command"]["right"]["anchor_count"] == 0
    assert metrics["by_command"]["right"]["mean_nearest_neighbor_d_traj"] is None
