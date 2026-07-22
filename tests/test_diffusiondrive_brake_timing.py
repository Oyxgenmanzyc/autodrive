import unittest
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.diffusiondrive.modules.risk_brake_timing import (
    _trajectory_dynamics,
    compute_brake_timing_diagnostics,
    compute_brake_timing_loss,
)
from navsim.agents.diffusiondrive.modules.risk_utils import (
    build_gt_brake_timing_context,
    build_gt_temporal_transport_target,
)
from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder,
)


def _config():
    return SimpleNamespace(
        risk_history_dt=0.5,
        brake_timing_accel_threshold=-0.5,
        brake_timing_temperature=0.35,
        brake_timing_profile_weight=0.25,
        brake_timing_preparation_time=1.0,
        brake_timing_loss_weight=0.1,
        transport_progress_weight=1.0,
        transport_terminal_weight=1.0,
        transport_safety_weight=2.0,
        transport_acceleration_weight=0.10,
        transport_jerk_weight=0.05,
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

    @staticmethod
    def _add_transport_targets(targets):
        teacher = targets["trajectory"].clone()
        teacher[:, :, 0] = teacher[:, :, 0] * 0.8
        targets.update(
            {
                "temporal_transport_target": teacher,
                "temporal_transport_upper_s": teacher[:, :, 0] + 0.2,
                "temporal_transport_constraint_mask": torch.ones_like(teacher[:, :, 0]),
                "temporal_transport_valid": torch.ones(teacher.shape[0], 1),
            }
        )

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

    def test_transport_loss_uses_continuous_progress_target(self):
        poses, anchors, targets = self._inputs()
        self._add_transport_targets(targets)
        poses.data[:, 0, :, 0] += 2.0

        output = compute_brake_timing_loss(poses, targets, anchors, _config())

        self.assertEqual(float(output["brake_timing_active_rate"]), 1.0)
        self.assertGreater(float(output["brake_timing_transport_progress_mae"]), 0.0)
        self.assertGreater(float(output["brake_timing_transport_safety_violation"]), 0.0)
        self.assertGreater(float(output["brake_timing_loss"]), 0.0)

    def test_transport_loss_only_updates_gt_matched_mode(self):
        poses, anchors, targets = self._inputs()
        self._add_transport_targets(targets)
        poses.data[:, 0, :, 0] += 2.0
        output = compute_brake_timing_loss(poses, targets, anchors, _config())

        output["brake_timing_loss"].backward()

        self.assertGreater(float(poses.grad[:, 0].abs().sum()), 0.0)
        self.assertEqual(float(poses.grad[:, 1].abs().sum()), 0.0)

    def test_stationary_trajectory_has_finite_zero_displacement_gradient(self):
        poses = torch.zeros(1, 8, 3, requires_grad=True)
        dynamics = _trajectory_dynamics(poses, torch.tensor([0.0]), dt=0.5)

        dynamics["speed"].sum().backward()

        self.assertTrue(torch.isfinite(dynamics["speed"]).all())
        self.assertAlmostEqual(float(dynamics["speed"].abs().sum()), 0.0, places=7)
        self.assertTrue(torch.isfinite(poses.grad).all())
        self.assertEqual(float(poses.grad.abs().sum()), 0.0)

    def test_epoch_statistics_count_only_active_scenes(self):
        poses, anchors, targets = self._inputs()
        poses.data[:, 0, :, 0] += torch.linspace(0.0, 2.0, 8)
        poses = poses.repeat(2, 1, 1, 1)
        anchors = anchors.repeat(2, 1, 1, 1)
        targets = {
            "trajectory": targets["trajectory"].repeat(2, 1, 1),
            "brake_timing_context": targets["brake_timing_context"].repeat(2, 1),
        }
        targets["brake_timing_context"][1, 1:3] = 8.0

        output = compute_brake_timing_loss(poses, targets, anchors, _config())

        self.assertEqual(float(output["brake_timing_scene_count"]), 2.0)
        self.assertEqual(float(output["brake_timing_active_count"]), 1.0)
        self.assertEqual(float(output["brake_timing_active_rate"]), 0.5)
        self.assertAlmostEqual(
            float(output["brake_timing_raw_loss_sum"]),
            float(output["brake_timing_raw_loss"]),
            places=6,
        )

    def test_selected_trajectory_diagnostics_expose_late_braking(self):
        _, _, targets = self._inputs()
        speed = torch.full((8,), 6.0)
        x = torch.cumsum(speed * 0.5, dim=0)
        selected = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1).unsqueeze(0)

        output = compute_brake_timing_diagnostics(selected, targets, _config())

        self.assertEqual(float(output["brake_timing_selected_active_count"]), 1.0)
        self.assertGreater(float(output["brake_timing_selected_onset_abs_error_s"]), 0.0)
        self.assertEqual(float(output["brake_timing_selected_late_rate"]), 1.0)

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

    def test_context_stops_when_front_track_leaves_corridor(self):
        current_box = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        outside_box = [10.0, 3.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        frames = [
            SimpleNamespace(
                annotations=self._annotations([current_box], ["front"]),
                ego_status=SimpleNamespace(
                    ego_pose=np.array([0.0, 0.0, 0.0]),
                    ego_velocity=np.array([6.0, 0.0]),
                ),
            ),
            SimpleNamespace(
                annotations=self._annotations([outside_box], ["front"]),
                ego_status=SimpleNamespace(ego_pose=np.array([1.0, 0.0, 0.0])),
            ),
        ]
        scene = SimpleNamespace(
            scene_metadata=SimpleNamespace(num_history_frames=1),
            frames=frames,
            get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
                poses=np.zeros((num_trajectory_frames, 3), dtype=np.float32)
            ),
        )
        config = SimpleNamespace(
            trajectory_sampling=SimpleNamespace(num_poses=1),
            risk_history_dt=0.5,
        )

        context = build_gt_brake_timing_context(scene, config)

        self.assertEqual(float(context[4]), 0.0)

    def test_transport_target_preserves_gt_path_geometry(self):
        box = [15.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        frames = [
            SimpleNamespace(
                annotations=self._annotations([box], ["front"]),
                ego_status=SimpleNamespace(
                    ego_pose=np.array([0.0, 0.0, 0.0]),
                    ego_velocity=np.array([6.0, 0.0]),
                ),
            )
            for _ in range(9)
        ]
        scene = SimpleNamespace(
            scene_metadata=SimpleNamespace(num_history_frames=1),
            frames=frames,
            get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
                poses=np.stack(
                    [np.array([3.0 * (step + 1), 0.0, 0.0]) for step in range(num_trajectory_frames)]
                )
            ),
        )
        config = SimpleNamespace(
            trajectory_sampling=SimpleNamespace(num_poses=8),
            risk_history_dt=0.5,
            brake_timing_preparation_time=1.0,
            transport_min_gap=1.5,
            transport_time_headway=0.75,
            transport_max_gap=8.0,
            transport_min_front_steps=2,
            transport_min_progress_shift=0.10,
            transport_max_speed=30.0,
            transport_max_accel=10.0,
            transport_max_decel=10.0,
            transport_max_jerk=30.0,
        )

        transport = build_gt_temporal_transport_target(scene, config)

        self.assertEqual(float(transport["valid"][0]), 1.0)
        self.assertGreater(float(transport["target"][0, 0]), 2.0)
        self.assertTrue(np.allclose(transport["target"][:, 1], 0.0))
        self.assertTrue(np.all(np.diff(transport["target"][:, 0]) >= -1e-5))
        self.assertTrue(
            np.all(
                transport["target"][:, 0][transport["constraint_mask"] > 0.5]
                <= transport["upper_s"][transport["constraint_mask"] > 0.5] + 1e-4
            )
        )

    def test_cache_names_separate_incompatible_risk_data(self):
        config = SimpleNamespace(
            use_risk_gate=False,
            use_historical_risk_attention=True,
            use_temporal_risk_cross_attention=False,
            use_risk_shadow_evaluator=False,
            use_soft_risk_rescore=False,
            use_longitudinal_safety_shield=False,
            use_step_brake_timing_loss=True,
            use_endpoint_conditioned_temporal_transport=True,
            use_memory_aux_loss=True,
        )

        self.assertEqual(
            TransfuserFeatureBuilder(config).get_unique_name(),
            "transfuser_feature_risk_history_v1",
        )
        self.assertEqual(
            TransfuserTargetBuilder(config).get_unique_name(),
            "transfuser_target_temporal_transport_v1",
        )


if __name__ == "__main__":
    unittest.main()
