import unittest
from types import SimpleNamespace

import torch

from navsim.agents.diffusiondrive.modules.risk_attention import (
    HistoricalRiskTemporalSelfAttention,
    TemporalRiskCrossAttention,
)
from navsim.agents.diffusiondrive.modules.risk_gate import select_risk_gated_mode


class RiskAttentionGateTest(unittest.TestCase):
    def test_history_risk_attention_shape(self):
        module = HistoricalRiskTemporalSelfAttention(
            token_dim=12,
            d_model=16,
            num_heads=4,
            num_layers=1,
            history_frames=4,
        )
        tokens = torch.zeros(2, 4, 12)
        tokens[:, :, -1] = 1.0

        memory, logits = module(tokens)

        self.assertEqual(memory.shape, (2, 4, 16))
        self.assertEqual(logits["risk_trend"].shape, (2, 3))
        self.assertEqual(logits["urgency"].shape, (2, 4))
        self.assertEqual(logits["brake_need"].shape, (2, 5))

    def test_temporal_risk_cross_attention_shape(self):
        module = TemporalRiskCrossAttention(d_model=16, num_heads=4)
        traj_feature = torch.zeros(2, 20, 16)
        noisy_traj_points = torch.zeros(2, 20, 8, 2)
        history_risk_memory = torch.zeros(2, 4, 16)

        output = module(traj_feature, noisy_traj_points, history_risk_memory)

        self.assertEqual(output.shape, (2, 20, 16))

    def test_risk_gate_prefers_safe_high_confidence_mode(self):
        config = SimpleNamespace(risk_history_dt=0.5, risk_gate_min_gap=1.0, risk_gate_cls_margin=2.0)
        poses_reg = torch.zeros(1, 3, 8, 3)
        poses_cls = torch.tensor([[0.0, 5.0, 3.0]])
        poses_reg[0, 0, :, 0] = torch.linspace(0.5, 3.0, 8)
        poses_reg[0, 1, :, 0] = torch.linspace(1.0, 12.0, 8)
        poses_reg[0, 2, :, 0] = torch.linspace(0.5, 4.0, 8)
        history_tokens = torch.zeros(1, 4, 12)
        history_tokens[0, -1, 0] = 8.0
        history_tokens[0, -1, 6] = 2.0
        history_tokens[0, -1, 7] = 2.5
        history_tokens[0, -1, -1] = 1.0

        selected = select_risk_gated_mode(poses_reg, poses_cls, history_tokens, config)

        self.assertEqual(selected.item(), 2)


if __name__ == "__main__":
    unittest.main()
