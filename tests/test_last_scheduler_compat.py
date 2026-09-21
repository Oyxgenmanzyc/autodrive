"""Compatibility tests for the original DiffusionDrive LR scheduler."""

import math
import unittest

import torch

from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR


class WarmupCosLRCompatibilityTest(unittest.TestCase):
    def test_constructor_accepts_legacy_verbose_without_forwarding_it(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            min_lr=1e-6,
            lr=1e-4,
            warmup_epochs=3,
            epochs=100,
            verbose=True,
        )

        for epoch in (0, 1, 2, 3, 50, 100):
            scheduler.last_epoch = epoch
            values = scheduler.get_lr()
            self.assertEqual(len(values), 1)
            self.assertTrue(math.isfinite(values[0]))
            self.assertGreater(values[0], 0)

        scheduler.last_epoch = 2
        self.assertAlmostEqual(scheduler.get_lr()[0], 1e-4)
        scheduler.last_epoch = 100
        self.assertAlmostEqual(scheduler.get_lr()[0], 1e-6)


if __name__ == "__main__":
    unittest.main()
