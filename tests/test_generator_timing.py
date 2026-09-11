"""Server tests: timing gradients reach generation; frozen selectors stay fixed."""
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import torch
from torch import nn

from navsim.agents.diffusiondrive.generator_timing.data import (
    SCHEMA, TimingDataset, target_identity, timing_config,
)
from navsim.agents.diffusiondrive.pcs.common import CONTEXT_KEYS, implementation_hashes
from navsim.agents.diffusiondrive.pcs.data import entry_path
from navsim.agents.diffusiondrive.generator_timing.risk_brake_timing import (
    compute_brake_timing_loss, _trajectory_dynamics,
)
from navsim.agents.diffusiondrive.generator_timing.risk_utils import build_gt_brake_timing_context
from navsim.agents.diffusiondrive.generator_timing.training import enable_generator_training, GeneratorTimingModule


def trajectory(speeds):
    result = torch.zeros(1, len(speeds), 3)
    result[0, :, 0] = torch.tensor(speeds).cumsum(0) * 0.5
    return result


class TimingLossTests(unittest.TestCase):
    def setUp(self):
        self.gt = trajectory([9., 8., 7., 6., 5., 4., 3., 2.])
        self.late = trajectory([10., 10., 9., 8., 7., 6., 5., 4.])
        self.targets = {"trajectory": self.gt, "brake_timing_context": torch.tensor([[10., 1., 1., 2., 8.]])}

    def test_late_braking_gets_gradient_and_gt_has_zero_loss(self):
        poses = self.late[:, None].clone().requires_grad_(True)
        loss = compute_brake_timing_loss(poses, self.targets, self.gt[:, None, :, :2], timing_config())
        self.assertEqual(float(loss["brake_timing_active_count"]), 1.)
        self.assertGreater(float(loss["brake_timing_loss"]), 0.)
        loss["brake_timing_loss"].backward()
        self.assertTrue(torch.isfinite(poses.grad).all())
        self.assertGreater(float(poses.grad.abs().sum()), 0.)
        exact = compute_brake_timing_loss(self.gt[:, None], self.targets, self.gt[:, None, :, :2], timing_config())
        self.assertAlmostEqual(float(exact["brake_timing_loss"]), 0.)

    def test_only_gt_matched_anchor_receives_timing_gradient(self):
        poses = self.late[:, None].repeat(1, 2, 1, 1).requires_grad_(True)
        anchors = self.gt[:, None, :, :2].repeat(1, 2, 1, 1)
        anchors[:, 0] += 100
        output = compute_brake_timing_loss(poses, self.targets, anchors, timing_config())
        output["brake_timing_loss"].backward()
        self.assertEqual(float(poses.grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(poses.grad[:, 1].abs().sum()), 0.)

    def test_inactive_and_control_have_no_auxiliary_gradient(self):
        for inactive, weight in ((True, 0.1), (False, 0.0)):
            poses = self.late[:, None].clone().requires_grad_(True)
            targets = {k: v.clone() for k, v in self.targets.items()}
            if inactive:
                targets["brake_timing_context"][:, 4] = 0
            result = compute_brake_timing_loss(poses, targets, self.gt[:, None, :, :2], timing_config(weight))
            result["brake_timing_loss"].backward()
            self.assertEqual(float(poses.grad.abs().sum()), 0.)

    def test_current_speed_is_first_acceleration_boundary(self):
        poses = trajectory([9.] * 8)
        dynamics = _trajectory_dynamics(poses, torch.tensor([10.]), 0.5)
        self.assertAlmostEqual(float(dynamics["acceleration"][0, 0]), -2., places=4)

    def test_stationary_trajectory_has_finite_gradient(self):
        poses = torch.zeros(1, 8, 3, requires_grad=True)
        _trajectory_dynamics(poses, torch.zeros(1), 0.5)["acceleration"].sum().backward()
        self.assertTrue(torch.isfinite(poses.grad).all())

    def test_missing_front_vehicle_is_inactive(self):
        annotations = SimpleNamespace(boxes=[], names=[])
        scene = SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=1),
                                frames=[SimpleNamespace(annotations=annotations)])
        context = build_gt_brake_timing_context(scene, timing_config())
        self.assertEqual(float(context[4]), 0.)


class ToyTrajectoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_anchor = nn.Parameter(torch.ones(1, 8, 2))
        self.plan_reg_branch = nn.Linear(2, 2)
        self.plan_cls_branch = nn.Linear(2, 2)
        self.diff_decoder = nn.Identity()

    def forward_train(self, ego, agents, bev, shape, status, targets=None):
        return {"trajectory_loss": self.plan_reg_branch(ego[..., :2]).sum()}

    def forward_test(self, ego, agents, bev, shape, status, global_img=None):
        return {"proposal_trajectory": ego.new_zeros((len(ego), 1, 8, 3))}


class ToyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout())
        self._trajectory_head = ToyTrajectoryHead()


class FrozenScopeTests(unittest.TestCase):
    def test_only_generator_regression_changes_after_optimizer_step(self):
        model = ToyGenerator()
        enable_generator_training(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        model._trajectory_head.plan_reg_branch(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        for key, value in model.state_dict().items():
            if "plan_reg_branch" in key:
                self.assertFalse(torch.equal(value, before[key]))
            else:
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_training_does_not_update_perception_batchnorm(self):
        module = GeneratorTimingModule(ToyGenerator(), {})
        module.train()
        self.assertFalse(module.generator.encoder.training)
        self.assertTrue(module.generator._trajectory_head.training)
        module.eval()
        self.assertFalse(module.generator._trajectory_head.training)

    def test_cached_context_dispatches_explicit_train_and_eval_paths(self):
        module = GeneratorTimingModule(ToyGenerator(), {})
        context = {
            "bev": torch.zeros(2, 256, 8, 8),
            "agents": torch.zeros(2, 30, 256),
            "ego": torch.zeros(2, 1, 256),
        }
        targets = {"trajectory": torch.zeros(2, 8, 3)}
        train_output = module._forward_head(context, targets)
        self.assertIn("trajectory_loss", train_output)
        self.assertTrue(module.generator._trajectory_head.training)
        eval_output = module._forward_head(context)
        self.assertIn("proposal_trajectory", eval_output)
        self.assertFalse(module.generator._trajectory_head.training)


class FrozenContextDatasetTests(unittest.TestCase):
    def test_training_reads_context_but_not_pdm_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_root = root / "pcs"
            candidate_root.mkdir()
            records = [
                {"token": "train-token", "log_name": "train-log", "split": "train"},
                {"token": "val-token", "log_name": "val-log", "split": "val"},
            ]
            provenance = {"implementation_sha256": implementation_hashes()}
            manifest = {"dataset": "navtrain", "provenance": provenance}
            (candidate_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (candidate_root / "records.json").write_text(json.dumps(records), encoding="utf-8")
            for record in records:
                path = entry_path(candidate_root, record)
                path.parent.mkdir(parents=True, exist_ok=True)
                context = {
                    "proposals": torch.zeros(2, 8, 3), "base_logits": torch.zeros(2),
                    "bev": torch.zeros(256, 8, 8), "agents": torch.zeros(30, 256),
                    "ego": torch.zeros(1, 256),
                }
                self.assertEqual(set(context), set(CONTEXT_KEYS))
                torch.save({"token": record["token"], "provenance": provenance,
                            "context": context, "labels": torch.full((2, 5), 7.)}, path)
            targets = root / "targets.pt"
            torch.save({
                "identity": {"schema": SCHEMA, "implementation": target_identity(),
                             "records": records, "data_root": "unused", "seed": 0},
                "contexts": torch.zeros(2, 5), "trajectories": torch.zeros(2, 8, 3),
                "summary": {"train": {"active_scenes": 0}, "val": {"active_scenes": 0}},
            }, targets)
            dataset = TimingDataset(targets, candidate_root, "train", smoke=True)
            context, target = dataset[0]
            self.assertEqual(set(context), set(CONTEXT_KEYS))
            self.assertEqual(set(target), {"trajectory", "brake_timing_context"})
            self.assertNotIn("labels", target)


if __name__ == "__main__":
    unittest.main()
