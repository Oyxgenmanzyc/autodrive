import math

import pytest
import torch

from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR


def test_scheduler_initializes_on_pytorch_without_verbose_argument():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=3e-4)

    scheduler = WarmupCosLR(
        optimizer=optimizer,
        min_lr=1e-6,
        lr=3e-4,
        warmup_epochs=3,
        epochs=100,
        verbose=True,
    )

    assert scheduler.optimizer is optimizer
    assert all(math.isfinite(value) for value in scheduler.get_last_lr())
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(2e-4)
