"""Run the real V2 constructor/forward against a frozen pre-3.3.1 class.

Only the expensive backbone and task heads are replaced. The decoder, BEV
projection, positional memory and all LAST routing execute as production code.
These are interface regressions, not a replacement for a full NAVSIM smoke test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.diffusiondrive.modules.last_token_selector import LASTTokenSelector

ROOT = Path(__file__).resolve().parents[1]


class TinyBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.low = nn.Conv2d(1, 512, 1)
        self.high = nn.Conv2d(1, 64, 1)

    def forward(self, camera, lidar):
        return self.high(F.interpolate(lidar, size=(16, 16))), self.low(lidar), None


class TinyAgentHead(nn.Module):
    def __init__(self, num_agents, d_ffn, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 5)

    def forward(self, query):
        return {"agent_states": self.proj(query)}


class TinyTrajectoryHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Linear(kwargs["d_model"], 3)

    def forward(self, query, agents, cross, shape, status, **kwargs):
        self.cross = cross.detach().clone()
        return {"trajectory": self.proj(query).repeat(1, 8, 1) + cross.mean((1, 2, 3))[:, None, None]}


def load_model_class(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "V2TransfuserModel")
    namespace = {
        "nn": nn, "torch": torch, "F": F, "Dict": Dict,
        "TransfuserConfig": SimpleNamespace, "TransfuserBackbone": TinyBackbone,
        "AgentHead": TinyAgentHead, "TrajectoryHead": TinyTrajectoryHead,
        "LASTTokenSelector": LASTTokenSelector,
        "linear_relu_ln": lambda *args: [nn.Linear(320, 256), nn.ReLU(), nn.LayerNorm(256)],
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["V2TransfuserModel"]


def config(**overrides):
    values = dict(
        num_bounding_boxes=30, tf_d_model=256, bev_features_channels=64,
        num_bev_classes=7, lidar_resolution_height=32, lidar_resolution_width=32,
        tf_num_head=8, tf_d_ffn=64, tf_dropout=0.0, tf_num_layers=1,
        trajectory_sampling=SimpleNamespace(num_poses=8), plan_anchor_path="unused",
        last_enable=False, last_topk_ratio=0.25, last_sigma_scale=1.0,
        last_eps=1e-6, last_gate_alpha=0.25, last_pre_norm=True,
        last_apply_decoder=True, last_apply_cross_bev=False,
        last_query_injection=False, last_query_scale=0.1, last_debug=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class LASTModelRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.new_class = load_model_class(ROOT / "navsim/agents/diffusiondrive/transfuser_model_v2.py")
        cls.old_class = load_model_class(ROOT / "tests/fixtures/diffusiondrive_main_v2.py")

    def setUp(self):
        torch.manual_seed(331)
        self.old = self.old_class(config()).eval()
        self.new = self.new_class(config()).eval()
        self.new.load_state_dict(self.old.state_dict(), strict=True)
        self.features = {"camera_feature": torch.randn(2, 3, 16, 16),
                         "lidar_feature": torch.randn(2, 1, 8, 8),
                         "status_feature": torch.randn(2, 8)}
        self.calls = {}

        def capture(name):
            def hook(module, args):
                self.calls[name] = tuple(arg.detach().clone() for arg in args)
            return hook

        self.old._tf_decoder.register_forward_pre_hook(capture("old"))
        self.new._tf_decoder.register_forward_pre_hook(capture("new"))

    def compare_baseline(self):
        expected = self.old(self.features)
        actual = self.new(self.features)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_disabled_equivalence_and_strict_old_checkpoint(self):
        self.assertEqual(set(self.old.state_dict()), set(self.new.state_dict()))
        self.compare_baseline()
        self.new._config.last_enable = True
        self.new.load_state_dict(self.old.state_dict(), strict=True)

    def test_alpha_zero_and_all_consumers_off_are_identity(self):
        self.new._config.last_enable = True
        self.new._last_selector.gate_alpha = 0
        self.compare_baseline()
        self.new._last_selector.gate_alpha = 0.25
        self.new._config.last_apply_decoder = False
        self.compare_baseline()

    def test_decoder_gate_does_not_change_cross_memory_or_status(self):
        self.new._config.last_enable = True
        self.old(self.features)
        self.new(self.features)
        torch.testing.assert_close(self.old._trajectory_head.cross,
                                   self.new._trajectory_head.cross, rtol=0, atol=0)
        old_query, old_memory = self.calls["old"]
        new_query, new_memory = self.calls["new"]
        self.assertEqual(new_memory.shape, (2, 65, 256))
        torch.testing.assert_close(old_memory[:, -1], new_memory[:, -1], rtol=0, atol=0)
        torch.testing.assert_close(old_query, new_query, rtol=0, atol=0)
        self.assertFalse(torch.equal(old_memory[:, :-1], new_memory[:, :-1]))

    def test_cross_only_does_not_change_decoder_memory(self):
        self.new._config.last_enable = True
        self.new._config.last_apply_decoder = False
        self.new._config.last_apply_cross_bev = True
        self.old(self.features)
        self.new(self.features)
        torch.testing.assert_close(self.calls["old"][1], self.calls["new"][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.old._trajectory_head.cross, self.new._trajectory_head.cross))

    def test_amp_decoder_only_preserves_cross_memory(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        self.new._config.last_enable = True
        for device in devices:
            self.old.to(device)
            self.new.to(device)
            features = {key: value.to(device) for key, value in self.features.items()}
            dtype = torch.float16 if device == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device, dtype=dtype):
                self.old(features)
                self.new(features)
            torch.testing.assert_close(self.old._trajectory_head.cross,
                                       self.new._trajectory_head.cross, rtol=0, atol=0)

    def test_global_injection_only_changes_trajectory_input_query(self):
        self.new._config.last_enable = True
        self.new._config.last_apply_decoder = False
        self.new._config.last_query_injection = True
        self.old(self.features)
        result = self.new(self.features)
        old_query, old_memory = self.calls["old"]
        new_query, new_memory = self.calls["new"]
        torch.testing.assert_close(old_query[:, 1:], new_query[:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(old_memory, new_memory, rtol=0, atol=0)
        self.assertFalse(torch.equal(old_query[:, :1], new_query[:, :1]))
        result["trajectory"].sum().backward()
        grad = self.new._bev_downscale.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_debug_is_read_only(self):
        self.new._config.last_debug = True
        self.compare_baseline()
        self.new._config.last_enable = True
        result = self.new(self.features)
        for name in ("last_plan_similarity", "last_vote", "last_gate"):
            self.assertEqual(result[name].shape, (2, 8, 8))
            self.assertFalse(result[name].requires_grad)


if __name__ == "__main__":
    unittest.main()
