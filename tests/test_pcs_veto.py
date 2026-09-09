"""Run on navhigh: python -m unittest discover -s tests -p 'test_pcs*.py'"""
import unittest

import torch

from navsim.agents.diffusiondrive.pcs.model import PDMCSHead
from navsim.agents.diffusiondrive.pcs.veto import (
    TripleRiskVeto, calibrate_thresholds, relative_risk_targets,
)


class RelativeRiskTests(unittest.TestCase):
    def test_targets_measure_only_degradation_from_base(self):
        labels = torch.ones(2, 3, 5)
        labels[0, 1, 0] = 0.0
        labels[0, 1, 1] = 0.0
        labels[0, 1, 3] = 0.5
        labels[1, 0, (0, 1, 3)] = torch.tensor([0.0, 0.0, 0.0])
        labels[1, 1, (0, 1, 3)] = torch.tensor([0.0, 1.0, 1.0])
        targets, _, _ = relative_risk_targets(
            labels, torch.tensor([1, 1]), torch.tensor([0, 0]),
        )
        torch.testing.assert_close(targets[0], torch.tensor([1.0, 1.0, 0.5]))
        torch.testing.assert_close(targets[1], torch.tensor([0.0, 0.0, 0.0]))

    def test_validation_calibration_can_preserve_safe_wins_and_veto_losses(self):
        risks = torch.tensor([
            [0.01, 0.01, 0.01], [0.90, 0.01, 0.01],
            [0.01, 0.01, 0.01], [0.01, 0.90, 0.01],
        ])
        pcs = torch.tensor([1.0, 0.0, 1.0, 0.0])
        base = torch.full((4,), 0.8)
        thresholds, result = calibrate_thresholds(risks, pcs, base)
        self.assertEqual(thresholds, [0.02, 0.02, 0.02])
        self.assertAlmostEqual(result["pdm"], 0.9, places=6)
        self.assertGreaterEqual(result["pdm"], float(pcs.mean()))
        self.assertEqual(result["new_zero_count"], 0)

    def test_calibration_never_counts_unchanged_decisions_as_vetoes(self):
        risks = torch.full((2, 3), 0.9)
        pcs = torch.tensor([0.7, 0.0])
        base = torch.tensor([0.7, 0.8])
        changed = torch.tensor([False, True])
        _, result = calibrate_thresholds(risks, pcs, base, changed)
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["vetoed_count"], 1)
        self.assertEqual(result["accepted_switch_count"], 0)


class TripleRiskVetoNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_any_factor_can_veto_and_only_risk_heads_receive_gradients(self):
        model = TripleRiskVeto(PDMCSHead(), thresholds=(0.0, 0.0, 0.0))
        context = {
            "proposals": torch.randn(1, 3, 8, 3, requires_grad=True),
            "base_logits": torch.tensor([[2.0, 0.0, 0.0]]),
            "bev": torch.randn(1, 256, 8, 8, requires_grad=True),
            "agents": torch.randn(1, 30, 256, requires_grad=True),
            "ego": torch.randn(1, 1, 256, requires_grad=True),
        }
        output = model(context, torch.tensor([1]), torch.tensor([0]))
        self.assertEqual(tuple(output["risks"].shape), (1, 3))
        self.assertTrue(bool(output["vetoed"][0]))
        self.assertEqual(int(output["final_mode"][0]), 0)

        labels = torch.ones(1, 3, 5)
        labels[0, 1, 1] = 0
        scores = torch.tensor([[1.0, 0.0, 1.0]])
        loss = model.loss(output, labels, scores)["loss"]
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.risk_heads.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.proposer.parameters()))
        for key in ("proposals", "bev", "agents", "ego"):
            self.assertIsNone(context[key].grad)


if __name__ == "__main__":
    unittest.main()
