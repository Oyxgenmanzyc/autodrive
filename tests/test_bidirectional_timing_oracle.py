import unittest
import numpy as np

from navsim.agents.diffusiondrive.timing_oracle.geometry import (
    ACTIONS, BACKLOAD_INDICES, BRAKE_INDICES, FRONTLOAD_INDICES, action_bank, safe_oracle,
)


class TimingGeometryTests(unittest.TestCase):
    def setUp(self):
        x = np.arange(1, 9, dtype=np.float32) * 2
        self.path = np.stack([x, 0.02 * x**2, np.arctan(0.04 * x)], -1).astype(np.float32)

    def test_identity_shape_and_finite(self):
        bank = action_bank(self.path)
        self.assertEqual(bank.shape, (len(ACTIONS), 8, 3))
        np.testing.assert_array_equal(bank[0], self.path)
        self.assertTrue(np.isfinite(bank).all())

    def test_warps_stay_on_path_and_preserve_endpoint(self):
        bank = action_bank(self.path)
        knots = np.vstack([np.zeros((1, 2)), self.path[:, :2]])
        edge = np.diff(knots, axis=0)
        for index in np.concatenate([FRONTLOAD_INDICES[1:], BACKLOAD_INDICES[1:]]):
            np.testing.assert_allclose(bank[index, -1, :2], self.path[-1, :2], atol=1e-6)
            for point in bank[index, :, :2]:
                u = np.clip(((point - knots[:-1]) * edge).sum(-1) / (edge * edge).sum(-1), 0, 1)
                distance = np.linalg.norm(knots[:-1] + u[:, None] * edge - point, axis=-1)
                self.assertLess(float(distance.min()), 2e-5)

    def test_matched_warps_move_progress_in_opposite_directions(self):
        bank = action_bank(self.path)
        self.assertGreater(bank[FRONTLOAD_INDICES[-1], 2, 0], self.path[2, 0])
        self.assertLess(bank[BACKLOAD_INDICES[-1], 2, 0], self.path[2, 0])

    def test_old_brake_actions_never_extend_endpoint(self):
        bank = action_bank(self.path)
        self.assertTrue((bank[BRAKE_INDICES[1:], -1, 0] < self.path[-1, 0]).all())

    def test_safe_oracle_rejects_safety_regression(self):
        labels = np.ones((1, len(ACTIONS), 5), np.float32)
        scores = np.ones((1, len(ACTIONS)), np.float32) * 0.8
        direction = np.ones_like(scores)
        scores[0, 1] = 0.9
        labels[0, 1, 3] = 0.0
        scores[0, FRONTLOAD_INDICES[1]] = 0.85
        selected = int(safe_oracle(labels, scores, direction)[0])
        self.assertEqual(selected, int(FRONTLOAD_INDICES[1]))


if __name__ == "__main__":
    unittest.main()
