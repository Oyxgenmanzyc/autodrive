"""Run in navhigh: python -m unittest discover -s tests -p test_gated_refinement.py."""
import unittest
import numpy as np
import torch
from navsim.agents.diffusiondrive.gated_refinement.labels import (
    build_labels, match_negatives, PairedBatchSampler, calibration_mask,
)
from navsim.agents.diffusiondrive.gated_refinement.model import GatedRefiner, decide, loss_terms
from navsim.agents.diffusiondrive.gated_refinement.training import gate_metrics, calibrate
from navsim.agents.diffusiondrive.post_selection.geometry import brake_bank


class LabelTests(unittest.TestCase):
    def make(self, n=4):
        return torch.ones(n, 25, 5), torch.full((n, 25), .8), torch.ones(n, 25)

    def test_safe_gain_and_protected_failure(self):
        labels, scores, direction = self.make()
        scores[0, 1] = .9
        scores[1, 1] = .99
        labels[1, 1, 0] = 0
        scores[2, 1] = .99
        direction[2, 1] = 0
        scores[3, 1] += 1e-7
        out = build_labels(labels, scores, direction)
        self.assertEqual(out['need'].tolist(), [True, False, False, False])
        self.assertEqual(out['teacher'].tolist(), [1, 0, 0, 0])

    def test_rescue_still_requires_gain_and_no_safety_regression(self):
        labels, scores, direction = self.make(3)
        labels[:, 0, 3] = 0
        scores[0, 1] += .005
        scores[1, 1] -= .01
        labels[2, 1:, 0] = 0
        scores[2, 1:] += .1
        out = build_labels(labels, scores, direction)
        self.assertEqual(out['need'].tolist(), [True, False, False])
        self.assertEqual(out['reasons'][0, 2].item(), 1)
        self.assertTrue(out['unresolved'][1:].all())

    def test_multiple_good_actions_soft_target(self):
        labels, scores, direction = self.make(1)
        scores[0, 1:3] = .9
        out = build_labels(labels, scores, direction)
        torch.testing.assert_close(out['action_target'][0, :2], torch.tensor([.5, .5]))
        self.assertEqual(out['action_target'][0, 2:].sum().item(), 0.)

    def test_matching_excludes_same_log_and_positive(self):
        selected = torch.randn(10, 8, 3)
        need = torch.tensor([True, True]+[False]*8)
        records = [dict(log_name='same' if i < 4 else str(i)) for i in range(10)]
        pairs = match_negatives(selected, need, records)
        self.assertTrue((pairs[:2] >= 4).all())
        self.assertTrue((pairs[2:] == -1).all())

    def test_sampler_groups_cover_positives_and_change_each_epoch(self):
        need = torch.tensor([True]*8+[False]*8)
        pairs = torch.full((16, 4), -1, dtype=torch.long)
        pairs[:8] = torch.tensor([8, 9, 10, 11])
        samplers = [PairedBatchSampler(need, pairs, 8, rank, 2) for rank in range(2)]
        positives = []
        for sampler in samplers:
            batches = list(sampler)
            for batch in batches:
                group = torch.tensor(batch).reshape(-1, 4)
                self.assertTrue(need[group[:, 0]].all())
                self.assertFalse(need[group[:, 1:]].any())
                positives.extend(group[:, 0].tolist())
            sampler.set_epoch(1)
            self.assertNotEqual(batches, list(sampler))
        self.assertEqual(sorted(positives), list(range(8)))

    def test_validation_log_separation(self):
        records = [dict(log_name=str(i//3)) for i in range(18)]
        mask = calibration_mask(records)
        for i in range(0, 18, 3):
            self.assertTrue((mask[i:i+3] == mask[i]).all())
        self.assertEqual(int(mask.sum()), 9)

    def test_mining_cannot_sample_positive_as_negative(self):
        sampler = PairedBatchSampler(torch.tensor([True, False, False, False, False]),
                                     torch.tensor([[1, 2, 3, 4]]*5), 4)
        with self.assertRaises(ValueError):
            sampler.set_hard_negatives([0])
        sampler.set_hard_negatives([2, 3])
        self.assertEqual(len(list(sampler)), 1)


class DecisionTests(unittest.TestCase):
    def test_gate_can_choose_negative_scored_edit_without_identity_competition(self):
        out = dict(gain=torch.tensor([[0., -.2, -.1], [0., -.2, -.1]]),
                   unsafe_logits=torch.full((2, 3), -10.), gate_logits=torch.tensor([10., -10.]))
        self.assertEqual(decide(out).tolist(), [2, 0])
        out['unsafe_logits'][:] = 10
        self.assertEqual(decide(out).tolist(), [0, 0])

    def test_ap_handles_constant_scores(self):
        result = gate_metrics(torch.zeros(100), torch.arange(100) < 5)
        self.assertAlmostEqual(result['auprc'], .05)
        self.assertEqual(result['actual_fraction_at_5pct'], 1.)

    def test_calibration_rejects_harmful_and_accepts_safe(self):
        out = dict(gate_logits=torch.tensor([10., -10.]), gain=torch.tensor([[0., .5], [0., .5]]),
                   unsafe_logits=torch.full((2, 2), -10.))
        labels = torch.ones(2, 2, 5)
        direction = torch.ones(2, 2)
        policy, _ = calibrate(out, labels, torch.tensor([[.8, .7], [.8, .7]]), direction)
        self.assertFalse(policy['enabled'])
        policy, _ = calibrate(out, labels, torch.tensor([[.8, .9], [.8, .7]]), direction)
        self.assertTrue(policy['enabled'])
        self.assertEqual(decide(out, policy['gate_threshold'], policy['risk_limit']).tolist(), [1, 0])

    def test_joint_and_probe_gradients_and_shape(self):
        path = np.stack([np.arange(1, 9)*2, np.zeros(8), np.zeros(8)], -1).astype(np.float32)
        variants = torch.from_numpy(brake_bank(path))[None].repeat(4, 1, 1, 1)
        context = dict(bev=torch.randn(4, 256, 8, 8), agents=torch.randn(4, 3, 256), ego=torch.randn(4, 1, 256))
        labels, scores, direction = LabelTests().make(4)
        scores[0, 1] = .9
        target = build_labels(labels, scores, direction)
        batch = dict(**target, scores=scores)
        for stage in ('probe', 'joint'):
            model = GatedRefiner()
            if stage == 'probe':
                model.head.requires_grad_(False)
            output = model(context, variants)
            loss, _ = loss_terms(output, batch, stage)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.gate[-1].weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters() if p.requires_grad))
            if stage == 'joint':
                self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
