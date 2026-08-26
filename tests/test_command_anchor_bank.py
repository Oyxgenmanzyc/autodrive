import numpy as np

from navsim.agents.diffusiondrive.anchors.command_anchor_bank import (
    CommandAnchorBankConfig,
    InitialCommandAnchorBankConfig,
    _membership_is_unchanged,
    build_command_anchor_bank,
    build_initial_command_anchor_bank,
    command_conditioned_assignment,
)
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


def _constant_trajectories(offsets):
    offsets = np.asarray(offsets, dtype=np.float32)
    return np.repeat(offsets[:, None, :], 8, axis=1)


def _scales(value=0.1):
    return {
        command: TrajectoryDistanceScale(delta_scale=value, fde_scale=value)
        for command in TrajectoryCommand
    }


def test_assignment_applies_command_mask_before_distance_argmin():
    trajectories = _constant_trajectories([[0.0, 0.0]])
    anchors = _constant_trajectories([[10.0, 0.0], [0.0, 0.0]])

    assignments, _ = command_conditioned_assignment(
        trajectories,
        np.asarray(["left"]),
        anchors,
        np.asarray(["left", "straight"]),
        _scales(),
    )

    np.testing.assert_array_equal(assignments, [0])


def test_initial_bank_inherits_command_from_source_gt_group():
    trajectories = _constant_trajectories(
        [[-4.0, 1.0], [-2.0, 1.0], [-1.0, 1.0], [-0.5, 1.0]]
        + [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        + [[-4.0, -1.0], [-2.0, -1.0], [-1.0, -1.0], [-0.5, -1.0]]
    )
    commands = np.asarray(["left"] * 4 + ["straight"] * 4 + ["right"] * 4)

    result = build_initial_command_anchor_bank(
        trajectories,
        commands,
        _scales(),
        InitialCommandAnchorBankConfig(total_anchors=6, max_medoid_samples=4),
    )

    assert len(result.anchors) == 6
    np.testing.assert_array_equal(result.anchor_counts, [2, 2, 2])
    np.testing.assert_array_equal(result.command_ids, [0, 0, 1, 1, 2, 2])
    source_commands = commands[result.source_indices]
    np.testing.assert_array_equal(
        source_commands,
        ["left", "left", "straight", "straight", "right", "right"],
    )
    np.testing.assert_array_equal(
        [result.support[result.command_ids == command_id].sum() for command_id in range(3)],
        result.command_counts,
    )


def test_rejected_node_is_unblocked_only_after_global_membership_changes():
    blocked = {7: np.asarray([1, 3, 5], dtype=np.int64)}

    assert _membership_is_unchanged(blocked, 7, np.asarray([1, 3, 5])) is True
    assert _membership_is_unchanged(blocked, 7, np.asarray([1, 4, 5])) is False
    assert 7 not in blocked


def test_worst_first_split_uses_real_medoids_and_preserves_metadata():
    left = _constant_trajectories([[-3.0, 0.0]] * 10 + [[3.0, 0.0]] * 10)
    straight = _constant_trajectories([[0.0, 0.0]] * 20)
    trajectories = np.concatenate([left, straight])
    commands = np.asarray(["left"] * 20 + ["straight"] * 20)
    initial = _constant_trajectories([[0.0, 0.0], [0.0, 0.0]])
    config = CommandAnchorBankConfig(tau_split=1.0, max_total_anchors=3)

    result = build_command_anchor_bank(
        trajectories,
        commands,
        initial,
        np.asarray(["left", "straight"]),
        _scales(),
        config,
    )

    assert len(result.anchors) == 3
    event = result.history[1]["split_event"]
    assert event["accepted"] is True
    assert event["command"] == "left"
    left_anchors = result.anchors[result.command_ids == 0]
    assert all(np.any(np.all(anchor == left, axis=(1, 2))) for anchor in left_anchors)
    np.testing.assert_array_equal(result.root_ids[result.command_ids == 0], [0, 0])
    assert np.all(result.parent_ids[result.command_ids == 0] == 0)
    np.testing.assert_array_equal(result.support, [20, 10, 10])


def test_large_parent_caps_medoid_sample_but_reports_full_child_support():
    trajectories = _constant_trajectories(
        [[-4.0, 0.0]] * 10 + [[4.0, 0.0]] * 10
    )
    result = build_command_anchor_bank(
        trajectories,
        np.asarray(["left"] * 20),
        _constant_trajectories([[0.0, 0.0]]),
        np.asarray(["left"]),
        _scales(),
        CommandAnchorBankConfig(
            tau_split=1.0,
            max_split_medoid_samples=6,
            max_total_anchors=2,
            random_seed=2,
        ),
    )

    event = result.history[1]["split_event"]
    assert event["sampled_for_kmedoids"] == 6
    assert sum(event["child_support"]) == 20


def test_split_is_rejected_when_one_child_is_below_three_percent_of_command():
    trajectories = _constant_trajectories([[1.0, 0.0]] * 98 + [[10.0, 0.0]] * 2)
    result = build_command_anchor_bank(
        trajectories,
        np.asarray(["right"] * 100),
        _constant_trajectories([[0.0, 0.0]]),
        np.asarray(["right"]),
        _scales(),
        CommandAnchorBankConfig(tau_split=1.0, max_total_anchors=3),
    )

    assert len(result.anchors) == 1
    event = result.history[-1]["split_event"]
    assert event["accepted"] is False
    assert event["reject_reason"] == "child_support_below_command_ratio"
    assert min(event["child_support"]) == 2
    assert len(result.history) == 2


def test_safety_cap_stops_before_an_eligible_split():
    trajectories = _constant_trajectories([[-3.0, 0.0]] * 10 + [[3.0, 0.0]] * 10)
    result = build_command_anchor_bank(
        trajectories,
        np.asarray(["straight"] * 20),
        _constant_trajectories([[0.0, 0.0]]),
        np.asarray(["straight"]),
        _scales(),
        CommandAnchorBankConfig(tau_split=1.0, max_total_anchors=1),
    )

    assert len(result.anchors) == 1
    assert result.stop_reason == "max_total_anchors"
