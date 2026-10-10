"""Small structural checks; full official PDM validation runs on the server."""
import unittest

import numpy as np
import torch

from navsim.agents.diffusiondrive.generator_timing.geometry import (
    seed_anchors, select_teachers, teacher_bank,
)
from navsim.agents.diffusiondrive.generator_timing.model import timing_loss


class GeneratorTimingTests(unittest.TestCase):
    def test_cached_half_features_are_promoted_before_generator_controls(self):
        from navsim.planning.script.generator_timing_eval import _candidate_context

        source = {
            'bev': torch.randn(4, 3, 3).half(),
            'agents': torch.randn(2, 4).half(),
            'ego': torch.randn(1, 4).half(),
        }
        proposals = torch.zeros(67, 8, 3)
        logits = torch.zeros(67)
        context = _candidate_context(source, proposals, logits, 'cpu')
        for key, value in context.items():
            self.assertEqual(value.dtype, torch.float32, key)
        for key in source:
            self.assertTrue(torch.equal(context[key][0], source[key].float()))
            self.assertEqual(source[key].dtype, torch.float16)
        # This is the input/weight boundary that failed in the original head.
        output = torch.nn.Conv2d(4, 4, 1)(context['bev'])
        self.assertEqual(tuple(output.shape), (1, 4, 3, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_every_parent_has_distinct_early_and_late_seed(self):
        t = torch.arange(1, 9, dtype=torch.float32)
        anchors = torch.zeros(67, 8, 2)
        anchors[..., 0] = t
        modes = seed_anchors(anchors).reshape(67, 2, 8, 2)
        self.assertEqual(tuple(modes.shape), (67, 2, 8, 2))
        self.assertTrue(torch.all(modes[:, 0, :-1, 0] > anchors[:, :-1, 0]))
        self.assertTrue(torch.all(modes[:, 1, :-1, 0] < anchors[:, :-1, 0]))
        self.assertTrue(torch.equal(modes[..., -1, :], anchors[:, None, -1, :].expand(-1, 2, -1)))

    def test_teacher_never_forces_unsafe_mode(self):
        original = np.zeros((1, 8, 3), np.float32)
        original[0, :, 0] = np.arange(1, 9)
        bank = teacher_bank(original)
        labels = np.ones((5, 5), np.float32)
        labels[1:, 2] = .8
        labels[2, 0] = 0.  # early/strong: unsafe collision
        scores = np.array([.8, .70, .95, .79, .77], np.float32)
        direction = np.ones(5, np.float32)
        targets, valid, weight = select_teachers(original, bank, labels, scores, direction)
        self.assertEqual(targets.shape, (1, 2, 8, 3))
        self.assertFalse(valid[0, 0])
        self.assertEqual(weight[0, 0], 0.)

    def test_loss_uses_only_safe_mode_target(self):
        predicted = torch.zeros(1, 1, 2, 8, 3, requires_grad=True)
        targets = torch.zeros_like(predicted)
        targets[:, :, 0, :, 0] = 1.
        targets[:, :, 1, :, 0] = 100.
        loss = timing_loss(predicted, targets, torch.tensor([[[True, False]]]),
                           torch.ones(1, 1, 2))
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertEqual(float(predicted.grad[:, :, 1].abs().sum()), 0.)
