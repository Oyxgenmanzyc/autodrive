import pytest
import torch

from navsim.agents.diffusiondrive.modules.semantic_auxiliary import (
    HybridSpeedSemanticDecoder,
    future_speed_targets,
)


def test_future_speed_targets_build_four_ordered_classes():
    speeds = torch.tensor([0.25, 1.0, 3.0, 6.0])
    interval_length = 0.5
    steps = torch.arange(8, dtype=torch.float32)
    x = speeds[:, None] * interval_length * steps[None]
    trajectory = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)

    targets = future_speed_targets(
        trajectory,
        interval_length=interval_length,
        thresholds=(0.5, 2.0, 5.0),
    )

    assert targets.dtype == torch.long
    assert targets.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("num_modes", [20, 67])
def test_hybrid_speed_decoder_preserves_variable_mode_count(num_modes):
    decoder = HybridSpeedSemanticDecoder(
        feature_dim=16,
        num_heads=4,
        dim_feedforward=32,
        dropout=0.0,
    )
    trajectory_tokens = torch.randn(2, num_modes, 16, requires_grad=True)

    refined_tokens, speed_logits = decoder(trajectory_tokens)
    torch.nn.functional.cross_entropy(speed_logits, torch.tensor([0, 3])).backward()

    assert refined_tokens.shape == trajectory_tokens.shape
    assert speed_logits.shape == (2, 4)
    assert trajectory_tokens.grad is not None
    assert torch.count_nonzero(trajectory_tokens.grad) > 0


@pytest.mark.parametrize(
    "trajectory, interval_length, thresholds",
    [
        (torch.zeros(2, 1, 3), 0.5, (0.5, 2.0, 5.0)),
        (torch.zeros(2, 8, 3), 0.0, (0.5, 2.0, 5.0)),
        (torch.zeros(2, 8, 3), 0.5, (0.5, 0.5, 5.0)),
    ],
)
def test_future_speed_targets_reject_invalid_inputs(trajectory, interval_length, thresholds):
    with pytest.raises(ValueError):
        future_speed_targets(trajectory, interval_length, thresholds)
