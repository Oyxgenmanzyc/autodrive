"""Server tests: python -m unittest discover -s tests -p 'test_pcs*.py'."""
import io
import unittest
from unittest.mock import patch

import torch

from navsim.agents.diffusiondrive.pcs.model import PDMCSHead
from navsim.agents.diffusiondrive.pcs.veto import TripleRiskVeto
from navsim.agents.diffusiondrive.pcs.timing_gain import (
    SCHEMA, TimingGainSelector, timing_features, choose_action, calibrate_policy, load_timing_gain,
)


class TimingTests(unittest.TestCase):
    def test_stationary_is_finite_and_initial_acceleration_is_unknown(self):
        values = timing_features(torch.zeros(2, 8, 3)).reshape(2, 8, 11)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values[..., 2] == 0).all())
        self.assertTrue((values[:, 0, 10] == 0).all())
        self.assertTrue((values[:, 0, 5:8] == 0).all())

    def test_isolated_brake_spike_is_not_sustained(self):
        trajectories = torch.zeros(2, 8, 3)
        speeds = torch.tensor([[8., 6., 6., 6., 6., 6., 6., 6.],
                               [8., 6., 4., 4., 4., 4., 4., 4.]])
        trajectories[..., 0] = speeds.cumsum(-1) * .5
        values = timing_features(trajectories).reshape(2, 8, 11)
        self.assertLess(float(values[0, :, 6].max()), .25)
        self.assertGreater(float(values[1, :, 6].max()), .9)

    def test_late_and_early_braking_are_distinct(self):
        poses = torch.zeros(2, 8, 3)
        poses[..., 0] = torch.tensor([[10., 8., 6., 6., 6., 6., 6., 6.],
                                     [10., 10., 10., 10., 8., 6., 6., 6.]]).cumsum(-1)*.5
        values = timing_features(poses).reshape(2, 8, 11)
        self.assertGreater(float(values[0, 2, 7]), float(values[1, 2, 7]))


class UtilityTests(unittest.TestCase):
    def test_same_high_risk_can_accept_large_gain_but_reject_loss(self):
        gc = torch.tensor([[.8, .05], [.01, .5], [.01, .5]])
        risk = torch.full((3, 3), .8)
        policy = {"mode": "weighted", "threshold": 0., "loss_multiplier": 1., "risk_weights": [.1, .1, .04]}
        veto, _ = choose_action(gc, risk, risk, torch.tensor([True, True, False]), policy)
        self.assertEqual(veto.tolist(), [False, True, False])

    def test_calibration_preserves_reference_safety_fallback(self):
        risk = torch.tensor([[0., 1., 0.], [0., 0., 0.]])
        gc = torch.tensor([[.9, .0], [.1, .2]])  # intentionally wrong predictions
        pcs, base = torch.tensor([0., 1.]), torch.tensor([1., .5])
        labels = torch.ones(2, 5)
        plabels = labels.clone()
        plabels[0, 1] = 0
        policy, result, reference = calibrate_policy(gc, torch.zeros_like(risk), risk,
            torch.ones(2, dtype=torch.bool), pcs, base, plabels, labels, [.5, .5, .5])
        self.assertGreaterEqual(result["pdm"], reference["pdm"])
        self.assertEqual(result["new_zero_count"], 0)
        self.assertEqual(policy["mode"], "reference")

    def test_gain_targets_retain_real_positive_and_negative_magnitudes(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        gc = torch.tensor([[.2, 0.], [0., .7]], requires_grad=True)
        out = {"risk_logits": logits, "gain_cost": gc,
               "pcs_mode": torch.ones(2, dtype=torch.long), "base_mode": torch.zeros(2, dtype=torch.long)}
        labels = torch.ones(2, 2, 5)
        scores = torch.tensor([[.5, .7], [.7, 0.]])
        losses = TimingGainSelector.loss(out, labels, scores, torch.ones(3))
        self.assertLess(float(losses["gain_loss"]), 1e-12)


class NetworkTests(unittest.TestCase):
    def test_checkpoint_roundtrip_restores_policy_and_checks_identity(self):
        model = TimingGainSelector(TripleRiskVeto(PDMCSHead(), thresholds=(.5, .5, .5)))
        policy = {"mode": "weighted", "threshold": -.01, "risk_weights": [.1, .1, .04], "loss_multiplier": 1.5}
        metadata = {"schema": SCHEMA, "provenance": {"seed": 0}, "pcs_settings": {},
                    "reference_thresholds": [.5, .5, .5], "hidden_dim": 256,
                    "use_timing": True, "policy": policy}
        stream = io.BytesIO()
        torch.save({"state_dict": {"model."+k: v for k, v in model.state_dict().items()},
                    "timing_gain_metadata": metadata}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        with patch("navsim.agents.diffusiondrive.pcs.timing_gain.load_torch", return_value=checkpoint):
            restored, _ = load_timing_gain("unused", "cpu", {"seed": 0})
            self.assertEqual(restored.policy, policy)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            with self.assertRaises(ValueError):
                load_timing_gain("unused", "cpu", {"seed": 1})

    def test_initial_reference_identity_and_frozen_gradient_boundary(self):
        torch.set_num_threads(2)
        reference = TripleRiskVeto(PDMCSHead(), thresholds=(.5, .5, .5)).eval()
        model = TimingGainSelector(reference).eval()
        context = {"proposals": torch.randn(2, 3, 8, 3, requires_grad=True),
                   "base_logits": torch.tensor([[2., 0., 0.], [2., 0., 0.]]),
                   "bev": torch.randn(2, 256, 8, 8, requires_grad=True),
                   "agents": torch.randn(2, 30, 256, requires_grad=True),
                   "ego": torch.randn(2, 1, 256, requires_grad=True)}
        pcs, base = torch.ones(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
        old, new = reference(context, pcs, base), model(context, pcs, base)
        torch.testing.assert_close(old["risks"], new["reference_risks"])
        torch.testing.assert_close(old["risks"], new["risks"])
        torch.testing.assert_close(old["final_mode"], new["final_mode"])
        labels, scores = torch.ones(2, 3, 5), torch.tensor([[.8, .9, .8], [.8, .0, .8]])
        labels[1, 1, 3] = 0
        model.loss(new, labels, scores, torch.ones(3))["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.proposer.parameters()))
        self.assertTrue(all(p.grad is None for p in model.reference_heads.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.gain_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.risk_heads.parameters()))
        self.assertTrue(all(context[k].grad is None for k in ("proposals", "bev", "agents", "ego")))


if __name__ == "__main__":
    unittest.main()
