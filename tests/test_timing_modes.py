import unittest

import numpy as np
import torch

from navsim.agents.diffusiondrive.timing_modes.data import inputs_from_source
from navsim.agents.diffusiondrive.timing_modes.geometry import expand_path, expand_top, top_modes
from navsim.agents.diffusiondrive.timing_modes.metrics import reports
from navsim.agents.diffusiondrive.timing_modes.model import TimingModeValue, choose, safe_targets, value_loss


class TimingGeometryTest(unittest.TestCase):
    def test_identity_reciprocal_modes_and_fixed_endpoint(self):
        path = np.zeros((8, 3), np.float32)
        path[:, 0] = np.arange(1, 9, dtype=np.float32)*2
        bank = expand_path(path)
        self.assertTrue(np.array_equal(bank[0], path))
        self.assertTrue(np.all(np.diff(bank[:, :, 0], axis=1) >= -1e-6))
        self.assertGreater(bank[1, 2, 0], path[2, 0])
        self.assertLess(bank[2, 2, 0], path[2, 0])
        self.assertTrue(np.array_equal(bank[1:3, -1], np.repeat(path[-1][None], 2, axis=0)))
        self.assertLessEqual(bank[3, -1, 0], path[-1, 0])

    def test_top_five_and_flattened_mode_order(self):
        scores = np.zeros(67, np.float32)
        scores[[9, 4]] = 1
        indices = top_modes(scores)
        self.assertEqual(indices.tolist()[:2], [4, 9])
        paths = np.zeros((5, 8, 3), np.float32)
        paths[:, :, 0] = np.arange(5)[:, None]
        bank = expand_top(paths)
        self.assertEqual(bank.shape, (20, 8, 3))
        self.assertTrue(np.array_equal(bank[::4], paths))


class TimingModelTest(unittest.TestCase):
    def test_frozen_input_assembly_and_zero_initialization(self):
        source = dict(features=torch.zeros(67, 512), subscores=torch.zeros(67, 5),
                      pcs_scores=torch.zeros(67), base_logits=torch.zeros(67),
                      proposals=torch.zeros(67, 8, 3))
        source['subscores'][4, 3] = .8
        modes = np.array([4, 2, 1, 0, 3])
        trajectories = np.zeros((20, 8, 3), np.float32)
        batch = inputs_from_source(source, modes, np.zeros(67, np.float32), trajectories)
        self.assertEqual(batch['features'].shape, (20, 512))
        self.assertAlmostEqual(float(batch['subscores'][0, 3]), .8)
        model = TimingModeValue()
        with torch.no_grad():
            output = model({key: value[None] for key, value in batch.items()})
        self.assertTrue(torch.count_nonzero(output) == 0)
        self.assertEqual(int(choose(output, 0)[0]), 0)
        variant = dict(features=torch.ones(20, 512),
                       subscores=torch.full((20, 5), .4), pcs_scores=torch.full((20,), .3))
        rescored = inputs_from_source(source, modes, np.zeros(67, np.float32),
                                      trajectories, variant)
        self.assertAlmostEqual(float(rescored['subscores'][0, 3]), .4)
        self.assertTrue(torch.equal(rescored['features'], variant['features']))

    def test_unsafe_gain_is_not_a_positive_training_target(self):
        labels = torch.ones(1, 20, 5)
        scores = torch.full((1, 20), .8)
        direction = torch.ones(1, 20)
        labels[0, 1, 3] = 0
        scores[0, 1] = .95
        scores[0, 2] = .9
        targets = safe_targets(labels, scores, direction)
        self.assertLess(float(targets[0, 1]), 0)
        self.assertGreater(float(targets[0, 2]), 0)
        prediction = torch.zeros(1, 20, requires_grad=True)
        loss, stats = value_loss(prediction, labels, scores, direction)
        self.assertGreater(int(stats['pairs']), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_calibration_can_enable_but_audit_can_block(self):
        predicted = torch.zeros(4, 20)
        predicted[:, 1] = .3
        labels = torch.ones(4, 20, 5)
        labels[2:, 1, 3] = 0
        scores = torch.full((4, 20), .8)
        scores[:2, 1] = .9
        scores[2:, 1] = 0
        direction = torch.ones(4, 20)
        result = reports(dict(calibration=torch.tensor([True, True, False, False]),
                              predicted=predicted, labels=labels, scores=scores,
                              direction=direction))
        self.assertTrue(result['policy']['enabled'])
        self.assertFalse(result['pass_for_navtest'])
        self.assertEqual(result['reports']['audit']['policy']['new_ttc_failure'], 2)


if __name__ == '__main__':
    unittest.main()
