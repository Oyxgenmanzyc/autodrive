"""Run on navhigh: python -m unittest discover -s tests -p test_post_selection.py."""
import unittest
import numpy as np
import torch
from navsim.agents.diffusiondrive.post_selection.geometry import ACTIONS, brake_bank, safe_oracle
from navsim.agents.diffusiondrive.post_selection.model import PostSelectionRefiner, decide, refinement_loss


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.path = np.stack([np.arange(1, 9)*2., np.zeros(8), np.zeros(8)], -1).astype(np.float32)

    def test_identity_exact(self):
        out = brake_bank(self.path)
        self.assertEqual(out.shape, (25, 8, 3))
        np.testing.assert_array_equal(out[0], self.path)

    def test_no_reverse_or_extension(self):
        out = brake_bank(self.path)
        self.assertTrue((np.diff(out[..., 0], axis=1) >= -1e-6).all())
        self.assertTrue((out[..., 0] <= self.path[None, :, 0]+1e-6).all())
        self.assertTrue((out[1:, -1, 0] < self.path[-1, 0]).all())

    def test_curved_path_not_fixed_ego_y(self):
        self.path[:, 1] = .03*self.path[:, 0]**2
        self.path[:, 2] = np.arctan(.06*self.path[:, 0])
        out = brake_bank(self.path)
        knots = np.vstack([np.zeros((1, 2)), self.path[:, :2]])
        for point in out[..., :2].reshape(-1, 2):
            edge = np.diff(knots, axis=0)
            u = np.clip(((point-knots[:-1])*edge).sum(-1)/(edge*edge).sum(-1), 0, 1)
            dist = np.linalg.norm(knots[:-1]+u[:, None]*edge-point, axis=-1)
            self.assertLess(float(dist.min()), 2e-5)

    def test_stopped_and_repeated_knots(self):
        zero = np.zeros((8, 3), np.float32)
        np.testing.assert_array_equal(brake_bank(zero), np.repeat(zero[None], 25, axis=0))
        self.path[4:] = self.path[3]
        self.assertTrue(np.isfinite(brake_bank(self.path)).all())

    def test_invalid_input(self):
        self.path[0, 0] = np.nan
        with self.assertRaises(ValueError):
            brake_bank(self.path)

    def test_strength_and_onset_have_effect(self):
        bank = brake_bank(self.path)
        a = ACTIONS.index((0., .25, 1.))
        b = ACTIONS.index((0., 1.5, 1.))
        c = ACTIONS.index((1.5, 1.5, 1.))
        self.assertLess(bank[b, -1, 0], bank[a, -1, 0])
        self.assertLess(bank[b, -1, 0], bank[c, -1, 0])

    def test_oracle_guards_and_ties(self):
        labels = np.ones((3, 5), np.float32)
        scores = np.array([.7, .9, .8], np.float32)
        direction = np.ones(3, np.float32)
        labels[1, 1] = 0.
        self.assertEqual(int(safe_oracle(labels, scores, direction)), 2)
        direction[2] = 0.
        self.assertEqual(int(safe_oracle(labels, scores, direction)), 0)
        self.assertEqual(int(safe_oracle(np.ones((3, 5)), np.ones(3), np.ones(3))), 0)


class ModelTests(unittest.TestCase):
    def make_inputs(self):
        path = np.stack([np.arange(1, 9)*2., np.zeros(8), np.zeros(8)], -1).astype(np.float32)
        bank = torch.from_numpy(brake_bank(path))[None].repeat(2, 1, 1, 1)
        context = {'bev': torch.randn(2, 256, 8, 8), 'agents': torch.randn(2, 10, 256), 'ego': torch.randn(2, 1, 256)}
        return context, bank

    def test_initial_policy_is_identity(self):
        context, bank = self.make_inputs()
        out = PostSelectionRefiner()(context, bank)
        self.assertEqual(out['gain'].shape, (2, 25))
        self.assertTrue((decide(out) == 0).all())

    def test_no_future_labels_needed_and_backprop(self):
        context, bank = self.make_inputs()
        model = PostSelectionRefiner()
        labels = torch.ones(2, 25, 5)
        scores = torch.ones(2, 25)*.8
        scores[:, 1] = .9
        out = model(context, bank)
        loss = refinement_loss(out, labels, scores, torch.ones(2, 25), torch.ones(2, dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0.)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.step(); opt.zero_grad()
        out = model(context, bank)
        refinement_loss(out, labels, scores, torch.ones(2, 25), torch.ones(2, dtype=torch.long)).backward()
        self.assertGreater(float(model.cross.in_proj_weight.grad.abs().sum()), 0.)

    def test_gain_risk_and_nonfinite_fallback(self):
        output = {'gain': torch.tensor([[0., .2, .1], [0., float('nan'), -.1]]),
                  'unsafe_logits': torch.tensor([[-10., 10., -10.], [-10., -10., -10.]])}
        self.assertEqual(decide(output).tolist(), [2, 0])
        self.assertEqual(decide(output, margin=.15).tolist(), [0, 0])

    def test_validation_identity_available(self):
        from navsim.agents.diffusiondrive.post_selection.training import calibrate
        scores = torch.tensor([[.8, .5], [.8, .7]])
        policy, mode = calibrate(torch.tensor([[0., .1], [0., .1]]),
                                 torch.ones(2, 2)*-10, torch.ones(2, 2, 5), scores, torch.ones(2, 2))
        self.assertFalse(policy['enabled'])
        self.assertEqual(mode.tolist(), [0, 0])


if __name__ == '__main__':
    unittest.main()
