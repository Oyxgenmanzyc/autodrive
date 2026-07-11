import unittest
from types import SimpleNamespace

import torch

from navsim.agents.diffusiondrive.modules.risk_attention import (
    HistoricalRiskTemporalSelfAttention,
    TemporalRiskCrossAttention,
)
from navsim.agents.diffusiondrive.modules.risk_gate import select_risk_gated_mode
from navsim.agents.diffusiondrive.modules.risk_shadow import (
    _delayed_time_risk,
    _sample_drivable_probability,
    _trajectory_dynamics,
    evaluate_risk_shadow,
)


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

    @staticmethod
    def _risk_inputs(agent_gap=8.0):
        poses_reg = torch.zeros(1, 2, 8, 3)
        poses_reg[0, 0, :, 0] = torch.linspace(1.0, 12.0, 8)
        poses_reg[0, 1, :, 0] = torch.linspace(0.5, 7.0, 8)
        poses_cls = torch.tensor([[2.0, 1.8]])
        history_tokens = torch.zeros(1, 4, 12)
        history_tokens[0, -2:, 0] = 8.0
        history_tokens[0, -2:, 2] = 10.0
        history_tokens[0, -2:, 3] = 5.0
        history_tokens[0, -2:, 6] = 1.6
        history_tokens[0, -2:, 7] = 3.0
        history_tokens[0, -2:, -1] = 1.0
        agent_states = torch.zeros(1, 1, 5)
        agent_states[0, 0] = torch.tensor([agent_gap + 4.0, 0.0, 0.0, 4.0, 2.0])
        agent_labels = torch.tensor([[10.0]])
        bev_semantic_map = torch.zeros(1, 7, 128, 256)
        bev_semantic_map[:, 1] = 10.0
        return poses_reg, poses_cls, history_tokens, agent_states, agent_labels, bev_semantic_map

    def test_shadow_uses_dynamic_lead_motion(self):
        inputs = self._risk_inputs()
        _, diagnostics = evaluate_risk_shadow(*inputs, SimpleNamespace())

        self.assertGreater(
            diagnostics["risk_base_dynamic_clearance"].item(),
            diagnostics["risk_base_static_clearance"].item(),
        )
        self.assertEqual(diagnostics["risk_front_source"].item(), 3.0)
        self.assertEqual(diagnostics["risk_shadow_reliable"].item(), 1.0)
        self.assertGreaterEqual(diagnostics["risk_candidate_eligible_count"].item(), 1.0)
        self.assertGreaterEqual(diagnostics["risk_candidate_cost_range"].item(), 0.0)
        self.assertGreaterEqual(diagnostics["risk_candidate_dynamic_clearance_range"].item(), 0.0)

    def test_lateral_non_overlap_removes_longitudinal_collision_cost(self):
        inputs = list(self._risk_inputs())
        inputs[0][..., 1] = 5.0
        _, diagnostics = evaluate_risk_shadow(*inputs, SimpleNamespace())

        self.assertEqual(diagnostics["risk_base_clearance_cost"].item(), 0.0)

    def test_agent_lidar_disagreement_keeps_classifier_mode(self):
        inputs = self._risk_inputs(agent_gap=20.0)
        proposed_mode, diagnostics = evaluate_risk_shadow(*inputs, SimpleNamespace())

        self.assertEqual(proposed_mode.item(), 0)
        self.assertEqual(diagnostics["risk_front_source"].item(), 4.0)
        self.assertEqual(diagnostics["risk_shadow_reliable"].item(), 0.0)

    def test_bev_probability_penalizes_out_of_map_trajectory(self):
        poses = torch.zeros(1, 2, 8, 3)
        poses[0, 0, :, 0] = torch.linspace(0.5, 4.0, 8)
        poses[0, 1, :, 0] = torch.linspace(0.5, 4.0, 8)
        poses[0, 1, :, 1] = 100.0
        bev_semantic_map = torch.zeros(1, 7, 128, 256)
        bev_semantic_map[:, 1] = 10.0

        probability, valid = _sample_drivable_probability(poses, bev_semantic_map, SimpleNamespace())

        self.assertTrue(valid.item())
        self.assertGreater(probability[0, 0].item(), 0.99)
        self.assertEqual(probability[0, 1].item(), 0.0)

    def test_brake_onset_requires_two_consecutive_deceleration_steps(self):
        dt = 0.5
        sustained_speeds = torch.tensor([6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5])
        spike_speeds = torch.tensor([6.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
        poses = torch.zeros(1, 2, 8, 3)
        poses[0, 0, :, 0] = torch.cumsum(sustained_speeds * dt, dim=0)
        poses[0, 1, :, 0] = torch.cumsum(spike_speeds * dt, dim=0)

        dynamics = _trajectory_dynamics(
            poses,
            ego_v=torch.tensor([6.0]),
            ego_a=torch.tensor([0.0]),
            dt=dt,
            config=SimpleNamespace(),
        )

        self.assertLess(dynamics["brake_onset"][0, 0].item(), 4.5)
        self.assertEqual(dynamics["brake_onset"][0, 1].item(), 4.5)

    def test_waiting_one_step_crosses_speed_dependent_t1(self):
        front = {
            "valid": torch.tensor([True]),
            "gap": torch.tensor([12.0]),
            "ego_v": torch.tensor([10.0]),
            "lead_v": torch.tensor([5.0]),
            "lead_a": torch.tensor([0.0]),
        }

        state = _delayed_time_risk(front, SimpleNamespace())

        self.assertGreater(state["ttc"].item(), state["t1"].item())
        self.assertLessEqual(state["delayed_ttc"].item(), state["t1"].item())
        self.assertTrue(state["crossing_t1"].item())
        self.assertTrue(state["trigger"].item())

    def test_non_closing_front_does_not_trigger_time_risk(self):
        front = {
            "valid": torch.tensor([True]),
            "gap": torch.tensor([8.0]),
            "ego_v": torch.tensor([5.0]),
            "lead_v": torch.tensor([5.0]),
            "lead_a": torch.tensor([0.0]),
        }

        state = _delayed_time_risk(front, SimpleNamespace())

        self.assertEqual(state["time_risk"].item(), 0.0)
        self.assertFalse(state["trigger"].item())


if __name__ == "__main__":
    unittest.main()
