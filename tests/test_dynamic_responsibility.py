from types import SimpleNamespace

import torch

from navsim.agents.diffusiondrive.modules.multimodal_loss import (
    LossComputer,
    command_masked_argmax,
    current_prediction_responsibility,
    k_invariant_focal_loss,
)


def _constant_predictions(offsets):
    offsets = torch.tensor(offsets, dtype=torch.float32)
    xy = offsets[:, :, None, :].repeat(1, 1, 8, 1)
    heading = torch.zeros(*xy.shape[:-1], 1)
    return torch.cat([xy, heading], dim=-1)


def test_responsibility_uses_current_prediction_and_command_mask():
    predictions = _constant_predictions([[[2.0, 0.0], [0.5, 0.0], [0.0, 0.0]]])
    target = torch.zeros(1, 8, 3)
    driving_command = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    anchor_command_ids = torch.tensor([0, 0, 1])

    winner = current_prediction_responsibility(
        predictions,
        target,
        driving_command,
        anchor_command_ids,
        torch.ones(3),
        torch.ones(3),
    )

    assert winner.item() == 1
    assert winner.requires_grad is False


def test_original_l1_regression_is_applied_only_to_dynamic_winner():
    config = SimpleNamespace(trajectory_cls_weight=0.0, trajectory_reg_weight=1.0)
    loss_computer = LossComputer(
        config,
        anchor_command_ids=torch.tensor([0, 0]),
        delta_scales=torch.ones(3),
        fde_scales=torch.ones(3),
    )
    predictions = _constant_predictions([[[2.0, 0.0], [0.25, 0.0]]]).requires_grad_()
    logits = torch.zeros(1, 2, requires_grad=True)
    target = torch.zeros(1, 8, 3)
    driving_command = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    loss = loss_computer(
        predictions,
        logits,
        {"trajectory": target},
        driving_command,
    )
    expected = torch.nn.functional.l1_loss(predictions[:, 1], target)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert torch.count_nonzero(predictions.grad[:, 0]) == 0
    assert torch.count_nonzero(predictions.grad[:, 1]) > 0


def test_k_invariant_focal_reduction_ignores_mode_count_and_invalid_modes():
    short_logits = torch.tensor([[0.2, -0.4]])
    short_loss = k_invariant_focal_loss(
        short_logits,
        winner_indices=torch.tensor([0]),
        command_mask=torch.tensor([[True, True]]),
    )
    long_logits = torch.tensor([[0.2, -0.4, -0.4, -0.4, 100.0]], requires_grad=True)
    long_loss = k_invariant_focal_loss(
        long_logits,
        winner_indices=torch.tensor([0]),
        command_mask=torch.tensor([[True, True, True, True, False]]),
    )

    torch.testing.assert_close(short_loss, long_loss)
    long_loss.backward()
    assert long_logits.grad[0, 4] == 0


def test_k_invariant_focal_is_finite_with_one_valid_mode_and_no_negatives():
    logits = torch.tensor([[0.2]], requires_grad=True)

    loss = k_invariant_focal_loss(
        logits,
        winner_indices=torch.tensor([0]),
        command_mask=torch.tensor([[True]]),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_selector_applies_command_mask_and_keeps_unknown_all_mode_fallback():
    logits = torch.tensor(
        [
            [1.0, 2.0, 100.0],
            [1.0, 2.0, 100.0],
        ]
    )
    commands = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    anchor_command_ids = torch.tensor([0, 0, 1])

    selected = command_masked_argmax(logits, commands, anchor_command_ids)

    torch.testing.assert_close(selected, torch.tensor([1, 2]))


def test_training_utilization_tracks_histogram_active_rate_and_entropy():
    config = SimpleNamespace(trajectory_cls_weight=1.0, trajectory_reg_weight=1.0)
    loss_computer = LossComputer(
        config,
        anchor_command_ids=torch.tensor([0, 0, 0, 1]),
        delta_scales=torch.ones(3),
        fde_scales=torch.ones(3),
    )
    loss_computer.update_training_utilization(
        winner_indices=torch.tensor([0, 1, 1, 3]),
    )

    metrics = loss_computer.training_utilization_metrics()
    torch.testing.assert_close(
        metrics["trajectory_utilization/left/winner_mode_0_frequency"],
        torch.tensor(1.0 / 3.0),
    )
    torch.testing.assert_close(
        metrics["trajectory_utilization/left/winner_mode_1_frequency"],
        torch.tensor(2.0 / 3.0),
    )
    torch.testing.assert_close(
        metrics["trajectory_utilization/left/active_mode_rate"],
        torch.tensor(2.0 / 3.0),
    )
    assert metrics["trajectory_utilization/left/winner_entropy"] > 0
    torch.testing.assert_close(
        metrics["trajectory_utilization/straight/winner_entropy"],
        torch.tensor(0.0),
    )


def test_training_utilization_all_reduces_one_full_count_vector(monkeypatch):
    config = SimpleNamespace(trajectory_cls_weight=1.0, trajectory_reg_weight=1.0)
    loss_computer = LossComputer(
        config,
        anchor_command_ids=torch.tensor([0, 0, 0, 1]),
        delta_scales=torch.ones(3),
        fde_scales=torch.ones(3),
    )
    loss_computer.update_training_utilization(torch.tensor([0, 1, 1, 3]))
    reduced_shapes = []

    def _fake_all_reduce(counts, op):
        reduced_shapes.append(tuple(counts.shape))
        counts.add_(torch.tensor([2, 0, 0, 0]))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)

    metrics = loss_computer.synchronized_training_utilization_metrics()

    assert reduced_shapes == [(4,)]
    torch.testing.assert_close(
        metrics["trajectory_utilization/left/winner_mode_0_frequency"],
        torch.tensor(3.0 / 5.0),
    )
    torch.testing.assert_close(
        metrics["trajectory_utilization/left/winner_mode_1_frequency"],
        torch.tensor(2.0 / 5.0),
    )


def test_unknown_samples_are_excluded_from_all_trajectory_supervision():
    config = SimpleNamespace(trajectory_cls_weight=1.0, trajectory_reg_weight=1.0)
    loss_computer = LossComputer(
        config,
        anchor_command_ids=torch.tensor([0, 0]),
        delta_scales=torch.ones(3),
        fde_scales=torch.ones(3),
    )
    predictions = _constant_predictions(
        [
            [[2.0, 0.0], [0.25, 0.0]],
            [[10.0, 0.0], [20.0, 0.0]],
        ]
    ).requires_grad_()
    logits = torch.tensor([[0.0, 1.0], [10.0, -10.0]], requires_grad=True)
    targets = {"trajectory": torch.zeros(2, 8, 3)}
    commands = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    mixed_loss = loss_computer(predictions, logits, targets, commands)
    valid_only_loss = loss_computer(
        predictions[:1],
        logits[:1],
        {"trajectory": targets["trajectory"][:1]},
        commands[:1],
    )
    torch.testing.assert_close(mixed_loss, valid_only_loss)

    mixed_loss.backward()
    assert torch.count_nonzero(predictions.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_all_unknown_batch_returns_differentiable_zero_trajectory_loss():
    config = SimpleNamespace(trajectory_cls_weight=1.0, trajectory_reg_weight=1.0)
    loss_computer = LossComputer(
        config,
        anchor_command_ids=torch.tensor([0]),
        delta_scales=torch.ones(3),
        fde_scales=torch.ones(3),
    )
    predictions = _constant_predictions([[[1.0, 0.0]]]).requires_grad_()
    logits = torch.zeros(1, 1, requires_grad=True)
    targets = {"trajectory": torch.zeros(1, 8, 3)}
    commands = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    loss = loss_computer(predictions, logits, targets, commands)
    torch.testing.assert_close(loss, torch.tensor(0.0))

    loss.backward()
    assert torch.count_nonzero(predictions.grad) == 0
    assert torch.count_nonzero(logits.grad) == 0
