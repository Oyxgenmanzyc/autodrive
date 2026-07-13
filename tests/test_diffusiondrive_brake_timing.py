import unittest
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.diffusiondrive.modules.risk_brake_timing import compute_brake_timing_loss
from navsim.agents.diffusiondrive.modules.risk_utils import build_gt_brake_timing_context


def _config():
    return SimpleNamespace(
        risk_history_dt=0.5,
        brake_timing_accel_threshold=-0.5,
        brake_timing_temperature=0.35,
        brake_timing_profile_weight=0.25,
        brake_timing_preparation_time=1.0,
        brake_timing_loss_weight=0.1,
    )


def _braking_trajectory():
    speed = torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5])
    x = torch.cumsum(speed * 0.5, dim=0)
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)


class BrakeTimingLossTest(unittest.TestCase):
    @staticmethod
    def _annotations(boxes, tokens):
        count = len(tokens)
        return SimpleNamespace(
            boxes=np.asarray(boxes, dtype=np.float32),
            names=["vehicle"] * count,
            velocity_3d=np.zeros((count, 3), dtype=np.float32),
            track_tokens=tokens,
        )

    def _inputs(self):
        target = _braking_trajectory().unsqueeze(0)
        poses = target[:, None].repeat(1, 2, 1, 1).clone().requires_grad_(True)
        anchors = target[:, None, :, :2].repeat(1, 2, 1, 1)
        anchors[:, 1, :, 1] = 4.0
        context = torch.tensor([[6.0, 3.0, 2.5, 2.0, 8.0]])
        return poses, anchors, {"trajectory": target, "brake_timing_context": context}

    def test_preparation_zone_activates_loss(self):
        poses, anchors, targets = self._inputs()
        poses.data[:, 0, :, 0] += torch.linspace(0.0, 2.0, 8)
        output = compute_brake_timing_loss(poses, targets, anchors, _config())
        self.assertEqual(float(output["brake_timing_active_rate"]), 1.0)
        self.assertGreater(float(output["brake_timing_loss"]), 0.0)

    def test_normal_scene_has_zero_loss(self):
        poses, anchors, targets = self._inputs()
        targets["brake_timing_context"][:, 1:3] = 8.0
        output = compute_brake_timing_loss(poses, targets, anchors, _config())
        self.assertEqual(float(output["brake_timing_active_rate"]), 0.0)
        self.assertEqual(float(output["brake_timing_loss"]), 0.0)

    def test_gradient_only_reaches_gt_matched_mode(self):
        poses, anchors, targets = self._inputs()
        poses.data[:, 0, :, 0] += torch.linspace(0.0, 2.0, 8)
        output = compute_brake_timing_loss(poses, targets, anchors, _config())
        output["brake_timing_loss"].backward()
        self.assertGreater(float(poses.grad[:, 0].abs().sum()), 0.0)
        self.assertEqual(float(poses.grad[:, 1].abs().sum()), 0.0)

    def test_context_stops_when_front_track_disappears(self):
        box = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        frames = [
            SimpleNamespace(
                annotations=self._annotations([box], ["front"]),
                ego_status=SimpleNamespace(
                    ego_pose=np.array([0.0, 0.0, 0.0]),
                    ego_velocity=np.array([6.0, 0.0]),
                ),
            ),
            SimpleNamespace(
                annotations=self._annotations([box], ["front"]),
                ego_status=SimpleNamespace(ego_pose=np.array([1.0, 0.0, 0.0])),
            ),
            SimpleNamespace(
                annotations=self._annotations([box], ["front"]),
                ego_status=SimpleNamespace(ego_pose=np.array([2.0, 0.0, 0.0])),
            ),
            SimpleNamespace(
                annotations=self._annotations([box], ["other"]),
                ego_status=SimpleNamespace(ego_pose=np.array([3.0, 0.0, 0.0])),
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
            trajectory_sampling=SimpleNamespace(num_poses=3),
            risk_history_dt=0.5,
        )

        context = build_gt_brake_timing_context(scene, config)

        self.assertEqual(float(context[4]), 2.0)
        self.assertGreater(float(context[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
