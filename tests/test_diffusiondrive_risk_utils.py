import unittest
from types import SimpleNamespace

import numpy as np

from navsim.agents.diffusiondrive.modules.risk_utils import (
    RISK_TOKEN_FIELDS,
    build_gt_history_risk_targets,
    build_history_risk_tokens,
    compute_longitudinal_risk,
)


def _lidar_with_front_points(x: float) -> SimpleNamespace:
    points = np.array(
        [
            [x, x, x, x],
            [0.0, 0.3, -0.3, 0.1],
            [1.0, 1.1, 1.2, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(lidar_pc=points)


def _empty_lidar() -> SimpleNamespace:
    return SimpleNamespace(lidar_pc=np.zeros((6, 0), dtype=np.float32))


def _ego(vx: float = 10.0, ax: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        ego_velocity=np.array([vx, 0.0], dtype=np.float32),
        ego_acceleration=np.array([ax, 0.0], dtype=np.float32),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        risk_history_num_frames=4,
        risk_history_dt=0.5,
        risk_front_x_min=1.0,
        risk_front_x_max=32.0,
        risk_front_y_abs=1.8,
        risk_lidar_min_z=0.2,
        risk_lidar_max_z=3.0,
        risk_lidar_min_points=3,
        risk_lidar_gap_percentile=10.0,
        risk_ego_front_offset=2.0,
        risk_ttc_max=10.0,
        risk_drac_max=6.0,
        lidar_split_height=0.2,
        max_height_lidar=100.0,
        lidar_max_x=32.0,
    )


def _annotations(x: float, vx: float, track_token: str = "lead") -> SimpleNamespace:
    return SimpleNamespace(
        boxes=np.array([[x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32),
        names=["vehicle"],
        velocity_3d=np.array([[vx, 0.0, 0.0]], dtype=np.float32),
        track_tokens=[track_token],
    )


def _frame(x: float, ego_v: float, lead_v: float) -> SimpleNamespace:
    return SimpleNamespace(
        annotations=_annotations(x, lead_v),
        ego_status=SimpleNamespace(
            ego_velocity=np.array([ego_v, 0.0], dtype=np.float32),
            ego_acceleration=np.array([0.0, 0.0], dtype=np.float32),
        ),
    )


class RiskUtilsTest(unittest.TestCase):
    def test_non_closing_relative_velocity_has_zero_drac(self):
        risk = compute_longitudinal_risk(gap=8.0, rel_v=-1.0, ego_v=5.0)

        self.assertEqual(risk["drac"], 0.0)
        self.assertEqual(risk["ttc"], 10.0)

    def test_gap_shrink_generates_positive_relative_velocity(self):
        agent_input = SimpleNamespace(
            ego_statuses=[_ego(), _ego(), _ego(), _ego()],
            lidars=[
                _lidar_with_front_points(14.0),
                _lidar_with_front_points(12.0),
                _lidar_with_front_points(10.0),
                _lidar_with_front_points(8.0),
            ],
        )

        tokens = build_history_risk_tokens(agent_input, _config())
        rel_v_idx = RISK_TOKEN_FIELDS.index("rel_v")
        drac_idx = RISK_TOKEN_FIELDS.index("drac")
        valid_idx = RISK_TOKEN_FIELDS.index("valid")

        self.assertEqual(tokens.shape, (4, len(RISK_TOKEN_FIELDS)))
        self.assertTrue(np.all(tokens[:, valid_idx] == 1.0))
        self.assertAlmostEqual(tokens[-1, rel_v_idx], 4.0)
        self.assertGreater(tokens[-1, drac_idx], 0.0)

    def test_invalid_lidar_does_not_emit_risk(self):
        agent_input = SimpleNamespace(
            ego_statuses=[_ego(), _ego(), _ego(), _ego()],
            lidars=[_empty_lidar(), _empty_lidar(), _empty_lidar(), _empty_lidar()],
        )

        tokens = build_history_risk_tokens(agent_input, _config())
        valid_idx = RISK_TOKEN_FIELDS.index("valid")
        drac_idx = RISK_TOKEN_FIELDS.index("drac")

        self.assertTrue(np.all(tokens[:, valid_idx] == 0.0))
        self.assertTrue(np.all(tokens[:, drac_idx] == 0.0))

    def test_gt_history_targets_emit_auxiliary_labels(self):
        scene = SimpleNamespace(
            scene_metadata=SimpleNamespace(num_history_frames=4),
            frames=[
                _frame(16.0, ego_v=10.0, lead_v=6.0),
                _frame(14.0, ego_v=10.0, lead_v=6.0),
                _frame(12.0, ego_v=10.0, lead_v=6.0),
                _frame(10.0, ego_v=10.0, lead_v=6.0),
            ],
        )

        targets = build_gt_history_risk_targets(scene, _config())

        self.assertEqual(targets["risk_aux_labels"].shape, (3,))
        self.assertEqual(float(targets["risk_aux_valid"]), 1.0)


if __name__ == "__main__":
    unittest.main()
