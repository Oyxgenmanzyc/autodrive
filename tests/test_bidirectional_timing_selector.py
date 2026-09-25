import unittest

import numpy as np
import torch

from navsim.agents.diffusiondrive.timing_oracle.geometry import ACTIONS
from navsim.agents.diffusiondrive.timing_selector.metrics import calibrated_reports
from navsim.agents.diffusiondrive.timing_selector.model import (
    BidirectionalTimingSelector, scalar_targets, timing_ranking_loss,
)


class TestBidirectionalTimingSelector(unittest.TestCase):
    def _batch(self, size=2):
        return {
            "feature": torch.randn(size, 512),
            "subscores": torch.rand(size, 5),
            "pcs_score": torch.rand(size),
            "base_probability": torch.rand(size),
            "proposal": torch.randn(size, 8, 3),
        }

    def test_zero_initialization_preserves_identity_tie(self):
        model = BidirectionalTimingSelector()
        logits = model(self._batch())
        self.assertEqual(tuple(logits.shape), (2, len(ACTIONS)))
        torch.testing.assert_close(logits, torch.zeros_like(logits))
        self.assertTrue(torch.equal(logits.argmax(-1), torch.zeros(2, dtype=torch.long)))

    def test_unsafe_action_receives_worst_scalar_target(self):
        labels = torch.ones(1, len(ACTIONS), 5)
        scores = torch.full((1, len(ACTIONS)), 0.5)
        direction = torch.ones(1, len(ACTIONS))
        scores[:, 1] = 0.9
        labels[:, 1, 3] = 0.0
        targets = scalar_targets(labels, scores, direction)
        self.assertEqual(float(targets[0, 0]), 0.0)
        self.assertEqual(float(targets[0, 1]), -1.0)

    def test_single_ranking_loss_is_finite(self):
        logits = torch.zeros(2, len(ACTIONS), requires_grad=True)
        labels = torch.ones(2, len(ACTIONS), 5)
        scores = torch.zeros(2, len(ACTIONS))
        scores[:, 2] = 0.25
        direction = torch.ones(2, len(ACTIONS))
        loss, stats = timing_ranking_loss(logits, labels, scores, direction)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(int(stats["pairs"]), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_calibration_locks_policy_before_audit(self):
        count = 12
        records = [{"log_name": f"log-{index // 2}"} for index in range(count)]
        logits = np.zeros((count, len(ACTIONS)), np.float32)
        logits[:, -1] = np.linspace(0.0, 1.0, count)
        labels = np.ones((count, len(ACTIONS), 5), np.float32)
        scores = np.full((count, len(ACTIONS)), 0.5, np.float32)
        scores[:, -1] = 0.6
        direction = np.ones((count, len(ACTIONS)), np.float32)
        report = calibrated_reports({"logits": logits, "labels": labels,
                                     "scores": scores, "direction": direction}, records)
        self.assertIn("threshold", report["policy"])
        self.assertTrue(report["reports"]["calibration"]["safety_pass"])
        self.assertTrue(report["reports"]["audit"]["safety_pass"])


if __name__ == "__main__":
    unittest.main()
