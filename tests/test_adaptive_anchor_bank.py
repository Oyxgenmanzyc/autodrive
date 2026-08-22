import numpy as np

from navsim.agents.diffusiondrive.anchors.base_expander import (
    AnchorBuilderConfig,
    expand_base_anchors,
)
from navsim.agents.diffusiondrive.anchors.composer import compose_anchor_bank
from navsim.agents.diffusiondrive.anchors.metrics import (
    coverage_ratio,
    nearest_anchor_assignment,
    pairwise_ade,
    pairwise_max_point_distance,
)
from navsim.agents.diffusiondrive.anchors.residual_codebook import (
    learn_residual_codebooks,
    local_residual_to_xy,
    xy_to_local_residual,
)


def _constant_trajectories(offsets, steps=8):
    offsets = np.asarray(offsets, dtype=np.float32)
    return np.repeat(offsets[:, None, :], steps, axis=1)


def test_distance_assignment_and_coverage_metrics():
    anchors = _constant_trajectories([[0.0, 0.0], [2.0, 0.0]], steps=2)
    trajectories = _constant_trajectories([[0.0, 0.0], [1.5, 0.0]], steps=2)

    ade = pairwise_ade(trajectories, anchors)
    max_distance = pairwise_max_point_distance(trajectories, anchors)
    assignments, nearest_ade = nearest_anchor_assignment(trajectories, anchors)

    np.testing.assert_allclose(ade, [[0.0, 2.0], [1.5, 0.5]])
    np.testing.assert_allclose(max_distance, ade)
    np.testing.assert_array_equal(assignments, [0, 1])
    np.testing.assert_allclose(nearest_ade, [0.0, 0.5])
    assert coverage_ratio(trajectories, anchors, epsilon=0.25) == 0.5


def test_bimodal_cluster_splits_and_reduces_distortion():
    rng = np.random.RandomState(0)
    left = _constant_trajectories(np.column_stack([np.full(40, -2.0), rng.normal(0, 0.02, 40)]))
    right = _constant_trajectories(np.column_stack([np.full(40, 2.0), rng.normal(0, 0.02, 40)]))
    trajectories = np.concatenate([left, right])
    initial = _constant_trajectories([[0.0, 0.0]])
    config = AnchorBuilderConfig(
        coverage_target=1.0,
        coverage_error_threshold=0.5,
        variance_threshold=0.1,
        min_split_samples=10,
        max_base_anchors=2,
        residual_modes=1,
    )

    result = expand_base_anchors(trajectories, initial, config)

    assert len(result.base_anchors) == 2
    assert result.history[-1]["mean_nearest_ADE"] < result.history[0]["mean_nearest_ADE"]


def test_tight_unimodal_cluster_does_not_split():
    rng = np.random.RandomState(1)
    offsets = rng.normal(0, 0.01, size=(100, 2))
    trajectories = _constant_trajectories(offsets)
    initial = _constant_trajectories([[0.0, 0.0]])
    config = AnchorBuilderConfig(
        coverage_target=1.0,
        coverage_epsilon=0.001,
        coverage_error_threshold=0.001,
        variance_threshold=0.1,
        min_split_samples=10,
        max_base_anchors=3,
        residual_modes=1,
    )

    result = expand_base_anchors(trajectories, initial, config)

    assert len(result.base_anchors) == 1
    assert result.stop_reason == "no_split_candidate"


def test_descendants_keep_original_root_after_repeated_splits():
    trajectories = _constant_trajectories(
        [[-3.0, 0.0]] * 20 + [[0.0, 0.0]] * 20 + [[3.0, 0.0]] * 20
    )
    initial = _constant_trajectories([[0.0, 0.0]])
    config = AnchorBuilderConfig(
        coverage_target=1.0,
        coverage_epsilon=0.1,
        coverage_error_threshold=0.1,
        variance_threshold=0.01,
        min_split_samples=5,
        max_base_anchors=3,
        residual_modes=1,
    )

    result = expand_base_anchors(trajectories, initial, config)

    assert len(result.base_anchors) == 3
    np.testing.assert_array_equal(result.root_ids, np.zeros(3, dtype=np.int64))
    assert np.all(result.parent_ids >= 0)


def test_tangent_normal_residual_round_trip():
    anchor = np.stack([np.arange(1, 9), np.arange(1, 9) * 0.5], axis=-1)[None].astype(np.float32)
    trajectory = anchor + np.array([[[0.2, -0.3]]], dtype=np.float32)

    local = xy_to_local_residual(trajectory, anchor)
    reconstructed = local_residual_to_xy(anchor, local)

    np.testing.assert_allclose(reconstructed, trajectory, atol=1e-6)


def test_zero_residual_preserves_every_base_anchor():
    trajectories = _constant_trajectories([[0.0, 0.0]] * 10 + [[2.0, 0.0]] * 10)
    base_anchors = _constant_trajectories([[0.0, 0.0], [2.0, 0.0]])
    assignments, _ = nearest_anchor_assignment(trajectories, base_anchors)
    config = AnchorBuilderConfig(
        max_base_anchors=2,
        residual_modes=1,
        residual_min_support=100,
    )
    residuals = learn_residual_codebooks(
        trajectories,
        base_anchors,
        np.array([0, 1]),
        assignments,
        config,
    )

    composition = compose_anchor_bank(base_anchors, np.array([0, 1]), residuals, config)

    assert len(composition.anchors) == 2
    for base_anchor in base_anchors:
        assert np.any(np.all(np.isclose(composition.anchors, base_anchor), axis=(1, 2)))
