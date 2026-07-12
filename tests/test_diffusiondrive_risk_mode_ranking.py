import unittest
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.diffusiondrive.modules.risk_mode_ranking import (
    RiskModeRankingHead,
    compute_risk_mode_ranking_loss,
    mine_timing_pairs,
)
from navsim.agents.diffusiondrive.modules.risk_utils import (
    _box_in_origin_frame,
    build_gt_future_front_targets,
)


class RiskModeRankingTest(unittest.TestCase):
    @staticmethod
    def _annotations(boxes, tokens):
        count = len(tokens)
        return SimpleNamespace(
            boxes=np.asarray(boxes, dtype=np.float32),
            names=["vehicle"] * count,
            velocity_3d=np.zeros((count, 3), dtype=np.float32),
            instance_tokens=[f"instance-{idx}" for idx in range(count)],
            track_tokens=tokens,
        )

    def test_box_is_transformed_to_current_ego_frame(self):
        box = np.array([2.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)

        transformed = _box_in_origin_frame(
            box,
            frame_ego_pose=np.array([10.0, 0.0, np.pi / 2]),
            origin_ego_pose=np.array([0.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(transformed[:2], np.array([10.0, 2.0]), atol=1e-5)
        self.assertAlmostEqual(float(transformed[2]), np.pi / 2, places=5)

    def test_future_front_track_stops_when_token_disappears(self):
        current_box = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        frames = [
            SimpleNamespace(
                annotations=self._annotations([current_box], ["front"]),
                ego_status=SimpleNamespace(
                    ego_pose=np.array([0.0, 0.0, 0.0]),
                    ego_velocity=np.array([4.0, 0.0]),
                ),
            ),
            SimpleNamespace(
                annotations=self._annotations([[9.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], ["front"]),
                ego_status=SimpleNamespace(ego_pose=np.array([1.0, 0.0, 0.0])),
            ),
            SimpleNamespace(
                annotations=self._annotations([[8.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], ["front"]),
                ego_status=SimpleNamespace(ego_pose=np.array([2.0, 0.0, 0.0])),
            ),
            SimpleNamespace(
                annotations=self._annotations([[20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], ["other"]),
                ego_status=SimpleNamespace(ego_pose=np.array([3.0, 0.0, 0.0])),
            ),
            SimpleNamespace(
                annotations=self._annotations([[6.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], ["front"]),
                ego_status=SimpleNamespace(ego_pose=np.array([4.0, 0.0, 0.0])),
            ),
        ]
        scene = SimpleNamespace(
            scene_metadata=SimpleNamespace(num_history_frames=1),
            frames=frames,
            get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
                poses=np.stack(
                    [np.array([float(step + 1), 0.0, 0.0]) for step in range(num_trajectory_frames)]
                )
            ),
        )
        config = SimpleNamespace(
            trajectory_sampling=SimpleNamespace(num_poses=4),
            risk_history_dt=0.5,
        )

        targets = build_gt_future_front_targets(scene, config)

        np.testing.assert_array_equal(targets["risk_front_future"][:, 5], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(float(targets["risk_pair_context"][4]), 2.0)

    @staticmethod
    def _pair_inputs():
        dt = 0.5
        early_speed = torch.tensor([5.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        late_speed = torch.tensor([6.0, 6.0, 5.5, 5.0, 5.0, 5.0, 5.0, 5.0])
        poses = torch.zeros(1, 3, 8, 3)
        poses[0, 0, :, 0] = torch.cumsum(early_speed * dt, dim=0)
        poses[0, 1, :, 0] = torch.cumsum(late_speed * dt, dim=0)
        poses[0, 2] = poses[0, 1]
        poses[0, 2, :, 1] = 5.0
        future_front = torch.zeros(1, 8, 6)
        future_front[..., 0] = 30.0
        future_front[..., 3] = 4.0
        future_front[..., 4] = 2.0
        future_front[..., 5] = 1.0
        targets = {
            "risk_front_future": future_front,
            "risk_pair_context": torch.tensor([[6.0, 1.5, 1.0, 2.0, 8.0]]),
            "risk_pair_scene_active": torch.tensor([1.0]),
        }
        return poses, targets

    def test_pair_mining_compares_same_path_brake_timing(self):
        poses, targets = self._pair_inputs()

        mined = mine_timing_pairs(poses, targets, SimpleNamespace())

        self.assertTrue(mined["pair_mask"][0, 0, 1].item())
        self.assertFalse(mined["pair_mask"][0, 0, 2].item())
        self.assertEqual(mined["brake_onset"][0, 0].item(), 0.0)
        self.assertEqual(mined["brake_onset"][0, 1].item(), 2.0)

    def test_ranking_head_is_zero_initialized_and_detached(self):
        head = RiskModeRankingHead(d_model=8)
        mode_feature = torch.randn(2, 3, 8, requires_grad=True)
        risk_memory = torch.randn(2, 4, 8, requires_grad=True)

        delta = head(mode_feature, risk_memory, torch.ones(2, 4))
        delta.sum().backward()

        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertIsNone(mode_feature.grad)
        self.assertIsNone(risk_memory.grad)

    def test_pairwise_loss_only_updates_risk_delta(self):
        poses, targets = self._pair_inputs()
        poses.requires_grad_(True)
        base_logits = torch.tensor([[0.0, 2.0, 1.0]], requires_grad=True)
        risk_delta = torch.zeros(1, 3, requires_grad=True)

        output = compute_risk_mode_ranking_loss(
            poses,
            base_logits,
            risk_delta,
            targets,
            SimpleNamespace(),
        )
        output["risk_mode_ranking_loss"].backward()

        self.assertGreater(output["risk_rank_pair_count"].item(), 0.0)
        self.assertIsNotNone(risk_delta.grad)
        self.assertIsNone(poses.grad)
        self.assertIsNone(base_logits.grad)


if __name__ == "__main__":
    unittest.main()
