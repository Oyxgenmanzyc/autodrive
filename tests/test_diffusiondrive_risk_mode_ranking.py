import unittest
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.diffusiondrive.modules.risk_mode_ranking import (
    RiskModeRankingHead,
    _future_agent_collision,
    compute_risk_mode_ranking_loss,
    mine_timing_pairs,
    select_risk_ranked_mode,
)
from navsim.agents.diffusiondrive.modules.risk_utils import (
    _box_in_origin_frame,
    build_gt_future_agent_targets,
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
        future_agents = build_gt_future_agent_targets(scene, config)
        self.assertEqual(future_agents.shape, (4, 30, 6))
        np.testing.assert_array_equal(future_agents[:, 0, 5], [1.0, 1.0, 0.0, 1.0])

    @staticmethod
    def _pair_inputs():
        dt = 0.5
        safe_speed = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5])
        unsafe_speed = torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
        conservative_speed = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.5, 0.5])
        poses = torch.zeros(1, 3, 8, 3)
        poses[0, 0, :, 0] = torch.cumsum(safe_speed * dt, dim=0)
        poses[0, 1, :, 0] = torch.cumsum(unsafe_speed * dt, dim=0)
        poses[0, 2, :, 0] = torch.cumsum(conservative_speed * dt, dim=0)

        future_front = torch.zeros(1, 8, 6)
        future_front[..., 0] = 18.0
        future_front[..., 3] = 4.0
        future_front[..., 4] = 2.0
        future_front[..., 5] = 1.0
        targets = {
            "risk_front_future": future_front,
            "risk_future_agents": torch.zeros(1, 8, 1, 6),
            "risk_pair_context": torch.tensor([[8.0, 3.0, 2.5, 4.0, 8.0]]),
            "risk_pair_scene_active": torch.tensor([1.0]),
            "trajectory": poses[:, 0].clone(),
        }
        return poses, targets

    def test_pair_mining_builds_single_unsafe_label_and_gt_error(self):
        poses, targets = self._pair_inputs()
        mined = mine_timing_pairs(
            poses,
            targets,
            SimpleNamespace(risk_rank_timing_max_decel_regression=2.0),
        )
        self.assertEqual(mined["unsafe_target"].shape, (1, 3))
        self.assertFalse(mined["unsafe_target"][0, 0].item())
        self.assertTrue(mined["unsafe_target"][0, 1].item())
        self.assertLess(mined["gt_error"][0, 0].item(), mined["gt_error"][0, 2].item())
        self.assertTrue(mined["safety_pair_mask"][0, 0, 1].item())
        self.assertTrue(mined["quality_pair_mask"][0, 0, 2].item())

    def test_relative_safety_gain_uses_timing_inside_qualified_set(self):
        base_logits = torch.tensor([[3.0, 2.9, 2.8]])
        unsafe_logits = torch.tensor([[2.0, -2.0, -1.0]])
        selection = select_risk_ranked_mode(
            base_logits,
            unsafe_logits,
            topk=3,
            timing_logits=torch.tensor([[0.0, 1.0, 2.0]]),
            config=SimpleNamespace(risk_rank_use_inference_path_guard=False),
        )
        # Modes 1 and 2 are safer than base; timing quality selects mode 2.
        self.assertEqual(selection["selected_mode"].item(), 2)
        self.assertEqual(selection["predicted_safe_candidate_count"].item(), 2)
        self.assertFalse(selection["safe_fallback_used"].item())

    def test_no_relative_safety_gain_preserves_base(self):
        selection = select_risk_ranked_mode(
            torch.tensor([[3.0, 2.9, 2.8]]),
            torch.tensor([[0.0, -0.1, 0.1]]),
            topk=3,
            config=SimpleNamespace(risk_rank_use_inference_path_guard=False),
        )
        self.assertEqual(selection["selected_mode"].item(), 0)
        self.assertTrue(selection["safe_fallback_used"].item())

    def test_path_guard_blocks_lateral_path_switch(self):
        base_logits = torch.tensor([[3.0, 2.9, 1.0]])
        unsafe_logits = torch.tensor([[2.0, -2.0, 2.0]])
        poses = torch.zeros(1, 3, 8, 3)
        poses[0, :, :, 0] = torch.arange(1, 9).float()
        poses[0, 1, :, 1] = 2.0
        selection = select_risk_ranked_mode(
            base_logits,
            unsafe_logits,
            topk=3,
            poses=poses,
            config=SimpleNamespace(
                risk_rank_use_inference_path_guard=True,
                risk_rank_inference_lateral_tolerance=0.75,
                risk_rank_inference_heading_tolerance=0.20,
            ),
        )
        self.assertEqual(selection["selected_mode"].item(), 0)

    def test_ranking_heads_preserve_generator_but_connect_risk_memory(self):
        head = RiskModeRankingHead(d_model=8)
        mode_feature = torch.randn(2, 3, 8, requires_grad=True)
        risk_memory = torch.randn(2, 4, 8, requires_grad=True)
        poses = torch.randn(2, 3, 8, 3, requires_grad=True)
        output = head(mode_feature, risk_memory, torch.ones(2, 4), poses=poses)
        (output["unsafe_logits"].sum() + output["timing_logits"].sum()).backward()
        self.assertTrue(
            torch.equal(output["unsafe_logits"], torch.zeros_like(output["unsafe_logits"]))
        )
        self.assertTrue(
            torch.equal(output["timing_logits"], torch.zeros_like(output["timing_logits"]))
        )
        self.assertIsNone(mode_feature.grad)
        self.assertIsNotNone(risk_memory.grad)
        self.assertIsNone(poses.grad)

    def test_loss_updates_both_selector_heads_only(self):
        poses, targets = self._pair_inputs()
        poses.requires_grad_(True)
        base_logits = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        unsafe_logits = torch.zeros(1, 3, requires_grad=True)
        timing_logits = torch.zeros(1, 3, requires_grad=True)
        output = compute_risk_mode_ranking_loss(
            poses,
            base_logits,
            unsafe_logits,
            timing_logits,
            targets,
            SimpleNamespace(risk_rank_timing_max_decel_regression=2.0),
        )
        output["risk_mode_ranking_loss"].backward()
        self.assertIsNotNone(unsafe_logits.grad)
        self.assertIsNotNone(timing_logits.grad)
        self.assertIsNone(poses.grad)
        self.assertIsNone(base_logits.grad)
        self.assertGreater(output["risk_rank_unsafe_rate"].item(), 0.0)

    def test_future_agent_oriented_box_collision_labels_side_agent(self):
        poses = torch.zeros(1, 2, 8, 3)
        poses[0, :, :, 0] = torch.arange(1, 9).float()
        poses[0, 1, :, 1] = 4.0
        agents = torch.zeros(1, 8, 1, 6)
        agents[0, :, 0, 0] = torch.arange(1, 9).float()
        agents[0, :, 0, 1] = 4.0
        agents[0, :, 0, 3:5] = torch.tensor([4.0, 2.0])
        agents[0, :, 0, 5] = 1.0
        collision = _future_agent_collision(poses, agents, SimpleNamespace())
        self.assertFalse(collision[0, 0].any().item())
        self.assertTrue(collision[0, 1].any().item())


if __name__ == "__main__":
    unittest.main()
