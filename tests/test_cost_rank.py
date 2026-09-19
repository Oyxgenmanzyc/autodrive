"""Meaningful CPU tests, executed on server with launcher test (navhigh)."""
import unittest
import numpy as np
import torch
from torch import nn
from navsim.agents.diffusiondrive.cost_rank.model import FrozenPCS, CostRanker, ranking_loss, select
from navsim.agents.diffusiondrive.cost_rank.data import log_partitions, validate_records
from navsim.agents.diffusiondrive.cost_rank.metrics import outcomes, calibrate, reports


def batch(n=2, k=6):
    return dict(features=torch.randn(n, k, 512).half(), subscores=torch.rand(n, k, 5),
                pcs_scores=torch.rand(n, k), proposals=torch.randn(n, k, 8, 3),
                base_logits=torch.randn(n, k))


class ModelTests(unittest.TestCase):
    def test_zero_initialization_preserves_scores_and_ties(self):
        model, data = CostRanker(), batch()
        data['pcs_scores'][:, :2] = 2.
        residual = model(data)
        self.assertEqual(torch.count_nonzero(residual), 0)
        self.assertTrue(torch.equal(select(data['pcs_scores'], residual), data['pcs_scores'].argmax(-1)))

    def test_labels_and_oof_predictions_cannot_enter_inference(self):
        model, data = CostRanker().eval(), batch()
        nn.init.normal_(model.head[-1].weight, std=.1)
        expected = model(data)
        data.update(labels=torch.full((2, 6, 5), float('nan')), scores=torch.randn(2, 6),
                    oof_scores=torch.full((2, 6), 100.))
        torch.testing.assert_close(model(data), expected, atol=0, rtol=0)

    def test_residual_bound_and_disabled_policy(self):
        model, data = CostRanker(cap=.2), batch()
        nn.init.constant_(model.head[-1].bias, 100.)
        residual = model(data)
        self.assertTrue((residual.abs() <= .2).all())
        self.assertTrue(torch.equal(select(data['pcs_scores'], residual, 0.), data['pcs_scores'].argmax(-1)))

    def test_ranker_can_select_third_candidate(self):
        pcs = torch.tensor([[.9, .85, .7]])
        residual = torch.tensor([[-.2, -.1, .2]])
        self.assertEqual(select(pcs, residual).item(), 2)
        self.assertEqual(select(pcs, residual, 0.).item(), 0)

    def test_single_loss_updates_ranker_but_not_input_features(self):
        torch.manual_seed(12)
        model, data = CostRanker(), batch()
        data['features'] = data['features'].float().requires_grad_()
        data['pcs_scores'] = torch.zeros(2, 6, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        truth = torch.tensor([[1., .8, .5, .2, .1, 0.]]).expand(2, -1)
        before = model.head[-1].weight.detach().clone()
        loss, _ = ranking_loss(data['pcs_scores'], model(data), truth, truth.flip(-1))
        loss.backward()
        self.assertIsNone(data['features'].grad)
        self.assertIsNone(data['pcs_scores'].grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head[-1].weight))

    def test_frozen_scorer_eval_no_gradient_and_hook_cleanup(self):
        class DummyPCS(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(3, 512)
                self.head = nn.Linear(512, 5)
            def forward(self, context):
                features = self.decoder(context['proposals'][:, :, 0])
                metrics = self.head(features).sigmoid()
                return dict(subscores=metrics, scores=metrics.mean(-1))
        source = DummyPCS()
        wrapped = FrozenPCS(source)
        wrapped.train(True)
        self.assertFalse(source.training)
        data = batch()
        out = wrapped(data)
        self.assertFalse(out['features'].requires_grad)
        self.assertTrue(all(not p.requires_grad for p in source.parameters()))
        self.assertEqual(len(source.decoder._forward_hooks), 0)


class LossTests(unittest.TestCase):
    def test_correct_ranking_has_smaller_loss(self):
        truth = torch.tensor([[1., 0.]])
        pcs = torch.zeros_like(truth)
        good = ranking_loss(pcs, torch.tensor([[.2, -.2]]), truth, pcs, top_k=1)[0]
        bad = ranking_loss(pcs, torch.tensor([[-.2, .2]]), truth, pcs, top_k=1)[0]
        self.assertLess(float(good), float(bad))

    def test_large_loss_has_larger_gradient_than_small_loss(self):
        gradients = []
        for gap in (.03, .8):
            residual = torch.zeros(1, 2, requires_grad=True)
            loss, _ = ranking_loss(torch.zeros_like(residual), residual, torch.tensor([[gap, 0.]]),
                                   torch.tensor([[0., 1.]]), top_k=1)
            loss.backward()
            gradients.append(float(residual.grad.abs().sum()))
        self.assertGreater(gradients[1], gradients[0]*10)

    def test_oof_false_high_pair_is_upweighted(self):
        truth = torch.tensor([[1., 0.]])
        correct = ranking_loss(torch.zeros_like(truth), torch.zeros_like(truth), truth, truth, top_k=1)[0]
        wrong, stats = ranking_loss(torch.zeros_like(truth), torch.zeros_like(truth), truth, truth.flip(-1), top_k=1)
        self.assertAlmostEqual(float(wrong/correct), 4., places=5)
        self.assertEqual(int(stats['hard_pairs']), 1)

    def test_no_preference_ties_have_zero_finite_gradient(self):
        residual = torch.randn(2, 3, requires_grad=True)
        truth = torch.ones_like(residual)
        loss, _ = ranking_loss(truth, residual, truth, truth, top_k=2)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(residual.grad.abs().sum()), 0.)

    def test_nonfinite_scores_fail_loudly(self):
        with self.assertRaises(ValueError):
            select(torch.tensor([[float('nan')]]), torch.zeros(1, 1))


class SplitTests(unittest.TestCase):
    def test_log_folds_have_no_leakage_and_ignore_record_order(self):
        records = [dict(token=str(i), log_name=str(i//3), split='train') for i in range(30)]
        folds = log_partitions(records)
        reversed_folds = log_partitions(list(reversed(records)))
        np.testing.assert_array_equal(folds, reversed_folds[::-1])
        for i in range(0, 30, 3):
            self.assertTrue((folds[i:i+3] == folds[i]).all())
        self.assertEqual(set(folds.tolist()), {0, 1, 2})

    def test_cross_split_logs_and_duplicate_tokens_rejected(self):
        with self.assertRaises(ValueError):
            validate_records([dict(token='a', log_name='same', split='train'),
                              dict(token='b', log_name='same', split='val')])
        with self.assertRaises(ValueError):
            validate_records([dict(token='a', log_name='a', split='train'),
                              dict(token='a', log_name='b', split='val')])


class CalibrationTests(unittest.TestCase):
    def data(self):
        return dict(pcs_scores=torch.tensor([[.8, .7], [.8, .7]]),
                    residual=torch.tensor([[-.2, .2], [-.2, .2]]),
                    labels=torch.ones(2, 2, 5), scores=torch.tensor([[.8, .9], [.8, .9]]),
                    direction=torch.ones(2, 2))

    def test_gain_opens_and_loss_disables(self):
        data = self.data()
        policy, _ = calibrate(data)
        self.assertTrue(policy['enabled'])
        data['scores'][:, 1] = .1
        policy, _ = calibrate(data)
        self.assertEqual(policy['alpha'], 0.)

    def test_safety_regression_rejects_pdm_improvement(self):
        data = self.data()
        data['labels'][:, 1, 3] = 0
        policy, _ = calibrate(data)
        self.assertFalse(policy['enabled'])

    def test_audit_cannot_change_calibrated_policy(self):
        data = self.data()
        data['calibration'] = torch.tensor([True, False])
        before = reports(data)['policy']
        data['scores'][1, 1] = 0.
        data['labels'][1, 1] = 0.
        report = reports(data)
        self.assertEqual(report['policy'], before)
        self.assertLess(report['reports']['audit']['policy']['gain_points'], 0.)

    def test_identity_has_exactly_zero_delta(self):
        data = self.data()
        modes = data['pcs_scores'].argmax(-1)
        out = outcomes(modes, modes, data['labels'], data['scores'], data['direction'])
        self.assertEqual(out['gain_points'], 0.)
        self.assertEqual(out['harmful'], 0)


if __name__ == '__main__':
    unittest.main()
