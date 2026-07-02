import sys
import types
import inspect
from pathlib import Path

import torch


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _DummyScheduler:
    def __init__(self, *args, **kwargs):
        pass


def _test_sineembed_for_position(pos_tensor, hidden_dim=64):
    half_hidden_dim = hidden_dim // 2
    scale = 2 * torch.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def _load_trajectory_head():
    diffusers_module = _stub_module("diffusers")
    diffusers_module.__path__ = []
    _stub_module("diffusers.schedulers", DDIMScheduler=_DummyScheduler)
    _stub_module("navsim.agents.diffusiondrive.transfuser_config", TransfuserConfig=object)
    _stub_module("navsim.agents.diffusiondrive.transfuser_backbone", TransfuserBackbone=object)
    _stub_module("navsim.agents.diffusiondrive.transfuser_features", BoundingBox2DIndex=object)
    _stub_module("navsim.common.enums", StateSE2Index=object)
    _stub_module(
        "navsim.agents.diffusiondrive.modules.conditional_unet1d",
        ConditionalUnet1D=_DummyModule,
        SinusoidalPosEmb=_DummyModule,
    )
    _stub_module(
        "navsim.agents.diffusiondrive.modules.blocks",
        linear_relu_ln=lambda *args, **kwargs: [],
        bias_init_with_prob=lambda *args, **kwargs: 0.0,
        gen_sineembed_for_position=_test_sineembed_for_position,
        GridSampleCrossBEVAttention=_DummyModule,
    )
    _stub_module("navsim.agents.diffusiondrive.modules.multimodal_loss", LossComputer=_DummyModule)

    from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead

    return TrajectoryHead


def _load_history_planning_adapter():
    _load_trajectory_head()
    from navsim.agents.diffusiondrive.transfuser_model_v2 import HistoryPlanningAdapter

    return HistoryPlanningAdapter


def _load_loss_computer():
    sys.modules.pop("navsim.agents.diffusiondrive.modules.multimodal_loss", None)
    _stub_module("navsim.agents.diffusiondrive.transfuser_config", TransfuserConfig=object)

    from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer

    return LossComputer


def _make_head():
    TrajectoryHead = _load_trajectory_head()
    head = TrajectoryHead.__new__(TrajectoryHead)
    head.temporal_noise_strength = 0.2
    head.temporal_noise_min_scale = 0.88
    head.temporal_noise_max_scale = 1.35
    head.temporal_start_weight = 0.25
    head.temporal_path_weight = 0.40
    head.temporal_velocity_weight = 0.35
    head.temporal_dt = 0.5
    head.temporal_low_speed_delta = 0.20
    head.temporal_accel_delta_tolerance = 0.60
    head.temporal_brake_delta_tolerance = 1.00
    head.temporal_direction_tolerance = 0.45
    head.temporal_lateral_accel_tolerance = 4.89
    head.temporal_turn_change_tolerance = 0.48
    head.temporal_jerk_delta_tolerance = 0.50
    head.temporal_rescore_topk = 5
    head.temporal_rescore_alpha = 0.05
    head.temporal_rescore_cost_clamp = 2.0
    return head


def _make_loss_computer():
    LossComputer = _load_loss_computer()

    class _Config:
        trajectory_cls_weight = 1.0
        trajectory_reg_weight = 1.0
        trajectory_weight = 12.0

    return LossComputer(_Config())


def _straight_x(scale=1.0, steps=8):
    return torch.stack([torch.arange(1, steps + 1).float() * scale, torch.zeros(steps)], dim=-1)


def _straight_y(scale=1.0, steps=8):
    return torch.stack([torch.zeros(steps), torch.arange(1, steps + 1).float() * scale], dim=-1)


def test_history_planning_adapter_inactive_without_previous_trajectory():
    HistoryPlanningAdapter = _load_history_planning_adapter()
    adapter = HistoryPlanningAdapter(d_model=64, num_heads=4, history_steps=3, dropout=0.0)
    traj_feature = torch.randn(2, 20, 64)
    noisy_traj_points = torch.randn(2, 20, 8, 2)

    enhanced, diagnostics = adapter(traj_feature, noisy_traj_points, None)

    assert torch.allclose(enhanced, traj_feature)
    assert diagnostics["history_valid_ratio"] == 0.0
    assert diagnostics["history_delta_norm"] == 0.0
    assert diagnostics["history_feature_delta_norm"] == 0.0
    assert torch.allclose(diagnostics["history_residual_scale"], torch.tensor(0.1))


def test_history_planning_adapter_updates_query_with_valid_three_step_history():
    HistoryPlanningAdapter = _load_history_planning_adapter()
    adapter = HistoryPlanningAdapter(d_model=64, num_heads=4, history_steps=3, dropout=0.0)
    traj_feature = torch.randn(2, 20, 64)
    noisy_traj_points = torch.randn(2, 20, 8, 2)
    previous_trajectory = torch.randn(2, 3, 2)

    enhanced, diagnostics = adapter(traj_feature, noisy_traj_points, previous_trajectory)

    assert enhanced.shape == traj_feature.shape
    assert not torch.allclose(enhanced, traj_feature)
    assert diagnostics["history_valid_ratio"] == 1.0
    assert diagnostics["history_delta_norm"] > 0.0
    assert diagnostics["history_feature_delta_norm"] > 0.0
    assert torch.allclose(diagnostics["history_residual_scale"], torch.tensor(0.1))


def test_history_planning_adapter_no_longer_uses_learnable_gate():
    HistoryPlanningAdapter = _load_history_planning_adapter()
    adapter = HistoryPlanningAdapter(d_model=64, num_heads=4, history_steps=3, dropout=0.0)

    assert not hasattr(adapter, "history_gate")
    assert not any(name == "history_gate" for name, _ in adapter.named_parameters())


def test_temporal_pair_lightning_logs_history_diagnostics():
    source = Path("navsim/planning/training/agent_lightning_module.py").read_text()

    assert '"history_valid_ratio"' in source
    assert '"history_delta_norm"' in source
    assert '"history_feature_delta_norm"' in source
    assert '"history_residual_scale"' in source


def test_trajectory_head_uses_stronger_temporal_rank_weight_for_3_16():
    TrajectoryHead = _load_trajectory_head()

    assert "self.temporal_rank_weight_max = 0.30" in inspect.getsource(TrajectoryHead.__init__)


def test_temporal_noise_scale_returns_none_without_previous():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)

    assert head._temporal_noise_scale(plan_anchor, None) is None


def test_temporal_noise_scale_keeps_anchor_noise_shape():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = torch.zeros(1, 3, 2)
    executed_delta = torch.tensor([[1.0, 0.0]])

    noise_scale = head._temporal_noise_scale(plan_anchor, previous, executed_delta)

    assert noise_scale.shape == (1, 20, 1, 1)


def test_start_speed_penalizes_only_beyond_tolerance():
    head = _make_head()
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = torch.tensor([[1.5, 0.0], [3.0, 0.0], [4.5, 0.0]])
    plan_anchor[:, 1, :3] = torch.tensor([[1.8, 0.0], [3.6, 0.0], [5.4, 0.0]])

    start_cost, path_cost, velocity_cost = head._temporal_compatibility_components(
        plan_anchor,
        previous_trajectory=None,
        previous_ego_delta=executed_delta,
    )

    assert start_cost[0, 1] > start_cost[0, 0]
    assert torch.allclose(path_cost[0, 1], path_cost[0, 0], atol=1e-6)
    assert torch.allclose(velocity_cost[0, 1], velocity_cost[0, 0], atol=1e-6)


def test_start_and_path_penalize_abrupt_direction_change_from_executed_motion():
    head = _make_head()
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    plan_anchor[:, 1, :3] = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])

    start_cost, path_cost, velocity_cost = head._temporal_compatibility_components(
        plan_anchor,
        previous_trajectory=None,
        previous_ego_delta=executed_delta,
    )

    assert start_cost[0, 1] > start_cost[0, 0]
    assert path_cost[0, 1] > path_cost[0, 0]
    assert velocity_cost[0, 1] == velocity_cost[0, 0]


def test_reference_path_affects_near_horizon_without_overriding_start_connection():
    head = _make_head()
    previous = _straight_x(steps=3).unsqueeze(0)
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = previous
    plan_anchor[:, 1, 0] = torch.tensor([1.0, 0.0])
    plan_anchor[:, 1, 1] = torch.tensor([1.0, 1.0])
    plan_anchor[:, 1, 2] = torch.tensor([1.0, 2.0])

    start_cost, path_cost, _ = head._temporal_compatibility_components(
        plan_anchor,
        previous,
        previous_ego_delta=executed_delta,
    )

    assert start_cost[0, 1] == start_cost[0, 0]
    assert path_cost[0, 1] > path_cost[0, 0]


def test_matching_anchor_gets_lower_noise_than_mismatched_anchor():
    head = _make_head()
    previous = _straight_x(steps=3).unsqueeze(0)
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = previous
    plan_anchor[:, 1, :3] = _straight_y(scale=2.0, steps=3)

    noise_scale = head._temporal_noise_scale(plan_anchor, previous, executed_delta)

    assert noise_scale[0, 0, 0, 0] < noise_scale[0, 1, 0, 0]


def test_temporal_rescore_can_change_selection_inside_cls_topk():
    head = _make_head()
    head.temporal_rescore_topk = 3
    head.temporal_rescore_alpha = 1.0
    poses_reg = torch.zeros(1, 5, 8, 3)
    poses_cls = torch.tensor([[1.0, 0.95, 0.90, 0.1, 0.0]])
    temporal_cost = torch.tensor([[2.0, 0.0, 1.0, 0.0, 0.0]])
    head._temporal_compatibility_cost = lambda *args, **kwargs: temporal_cost

    mode_idx, diagnostics = head._select_mode_with_temporal_rescore(poses_reg, poses_cls)

    assert mode_idx.item() == 1
    assert diagnostics["temporal_rescore_active"].item() == 1.0
    assert diagnostics["temporal_rescore_changed"].item() == 1.0


def test_temporal_rescore_does_not_allow_outside_topk_to_win():
    head = _make_head()
    head.temporal_rescore_topk = 3
    head.temporal_rescore_alpha = 1.0
    poses_reg = torch.zeros(1, 5, 8, 3)
    poses_cls = torch.tensor([[1.0, 0.95, 0.90, 0.1, 0.0]])
    temporal_cost = torch.tensor([[2.0, 1.5, 1.0, -10.0, -10.0]])
    head._temporal_compatibility_cost = lambda *args, **kwargs: temporal_cost

    mode_idx, diagnostics = head._select_mode_with_temporal_rescore(poses_reg, poses_cls)

    assert mode_idx.item() in {0, 1, 2}
    assert diagnostics["temporal_rescore_active"].item() == 1.0


def test_near_horizon_reference_ignores_far_anchor_points():
    head = _make_head()
    previous = _straight_x(steps=3).unsqueeze(0)
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = previous
    changed_far_anchor = plan_anchor.clone()
    changed_far_anchor[:, 0, 3:] = _straight_y(steps=5) + torch.tensor([10.0, 0.0])

    costs = head._temporal_compatibility_components(plan_anchor, previous, executed_delta)
    changed_far_costs = head._temporal_compatibility_components(changed_far_anchor, previous, executed_delta)

    for cost, changed_far_cost in zip(costs, changed_far_costs):
        assert torch.allclose(cost, changed_far_cost)


def test_reference_points_after_p2_do_not_change_temporal_cost():
    head = _make_head()
    previous = _straight_x(steps=8).unsqueeze(0)
    changed_far_previous = previous.clone()
    changed_far_previous[:, 3:] = _straight_y(steps=5) + torch.tensor([10.0, 0.0])
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = previous[:, :3]

    costs = head._temporal_compatibility_components(plan_anchor, previous, executed_delta)
    changed_far_costs = head._temporal_compatibility_components(plan_anchor, changed_far_previous, executed_delta)

    for cost, changed_far_cost in zip(costs, changed_far_costs):
        assert torch.allclose(cost, changed_far_cost)


def test_terminal_absolute_position_no_longer_changes_cost():
    head = _make_head()
    previous = _straight_x(steps=3).unsqueeze(0)
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :3] = previous
    changed_terminal_anchor = plan_anchor.clone()
    changed_terminal_anchor[:, 0, -1] = torch.tensor([20.0, 20.0])

    costs = head._temporal_compatibility_components(plan_anchor, previous, executed_delta)
    changed_terminal_costs = head._temporal_compatibility_components(changed_terminal_anchor, previous, executed_delta)

    for cost, changed_terminal_cost in zip(costs, changed_terminal_costs):
        assert torch.allclose(cost, changed_terminal_cost)


def test_accelerate_then_brake_has_larger_reversal_cost_than_recover_after_brake():
    head = _make_head()
    accel_then_brake = torch.tensor([[[1.0, 1.7, 0.5]]])
    brake_then_accel = torch.tensor([[[2.0, 0.8, 1.5]]])

    assert head._reversal_cost(accel_then_brake)[0, 0] > head._reversal_cost(brake_then_accel)[0, 0]


def test_low_speed_turn_segments_do_not_create_false_turn_cost():
    head = _make_head()
    anchor_delta = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]])
    reference_delta = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]])

    assert torch.all(head._turn_cost(anchor_delta, reference_delta) == 0)


def _energy_test_inputs():
    poses_reg = torch.zeros(1, 5, 8, 3)
    plan_anchor = torch.zeros(1, 5, 8, 2)
    target = {"trajectory": torch.zeros(1, 8, 3)}
    for mode_idx, offset in enumerate([0.0, 0.2, 0.4, 4.0, 5.0]):
        plan_anchor[:, mode_idx, :, 0] = offset
        poses_reg[:, mode_idx, :, 0] = offset
    poses_cls = torch.zeros(1, 5)
    return poses_reg, poses_cls, target, plan_anchor


def _energy_context(epoch, temporal_cost=None):
    if temporal_cost is None:
        temporal_cost = torch.tensor([[2.0, 0.0, 1.0, 0.0, 0.0]])
    return {
        "energy_temporal_cost": temporal_cost,
        "energy_comfort_cost": torch.zeros_like(temporal_cost),
        "energy_training_epoch": epoch,
        "energy_topk": 3,
        "energy_gt_weight": 0.00,
        "energy_temporal_weight": 1.00,
        "energy_comfort_weight": 0.00,
        "temporal_rank_weight_max": 0.01,
        "temporal_rank_margin": 0.10,
        "temporal_rank_energy_gap": 0.20,
        "temporal_rank_min_epoch": 50.0,
        "temporal_rank_ramp_epochs": 30.0,
        "temporal_rank_use_comfort": False,
        "temporal_aux_weight_max": 0.0,
    }


def _comfort_energy_context(epoch, temporal_cost=None, comfort_cost=None):
    context = _energy_context(epoch, temporal_cost)
    if comfort_cost is None:
        comfort_cost = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])
    context.update(
        {
            "energy_comfort_cost": comfort_cost,
            "energy_temporal_weight": 0.80,
            "energy_comfort_weight": 0.20,
            "temporal_rank_use_comfort": True,
        }
    )
    return context


def test_temporal_rank_inactive_before_min_epoch_matches_original_loss():
    loss_computer = _make_loss_computer()
    poses_reg, poses_cls, target, plan_anchor = _energy_test_inputs()

    original_loss = loss_computer(poses_reg, poses_cls, target, plan_anchor)
    early_loss = loss_computer(
        poses_reg,
        poses_cls,
        target,
        plan_anchor,
        temporal_context=_energy_context(epoch=49),
    )

    assert torch.allclose(original_loss, early_loss)


def test_temporal_rank_adds_loss_without_changing_classification_target():
    loss_computer = _make_loss_computer()
    poses_reg, poses_cls, target, plan_anchor = _energy_test_inputs()
    poses_cls[:, 1] = -1.0
    poses_cls[:, 0] = 1.0
    dist = torch.linalg.norm(target["trajectory"].unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
    cls_target = torch.argmin(dist, dim=-1)

    energy_info = loss_computer._energy_supervision(
        poses_reg,
        poses_cls,
        target["trajectory"],
        dist,
        cls_target,
        _energy_context(epoch=80),
    )

    assert "cls_soft_target" not in energy_info
    assert energy_info["temporal_rank_loss"] > 0.0
    assert energy_info["temporal_rank_active_ratio"] == 1.0


def test_temporal_rank_does_not_use_mode_outside_gt_topk():
    loss_computer = _make_loss_computer()
    poses_reg, poses_cls, target, plan_anchor = _energy_test_inputs()
    dist = torch.linalg.norm(target["trajectory"].unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
    cls_target = torch.argmin(dist, dim=-1)
    temporal_cost = torch.tensor([[2.0, 2.0, 2.0, 2.0, -10.0]])

    energy_info = loss_computer._energy_supervision(
        poses_reg,
        poses_cls,
        target["trajectory"],
        dist,
        cls_target,
        _energy_context(epoch=80, temporal_cost=temporal_cost),
    )

    assert energy_info["temporal_rank_active_ratio"] == 0.0


def test_temporal_rank_uses_detached_energy_but_backprops_to_logits():
    loss_computer = _make_loss_computer()
    poses_reg, poses_cls, target, plan_anchor = _energy_test_inputs()
    poses_reg.requires_grad_(True)
    poses_cls.requires_grad_(True)
    dist = torch.linalg.norm(target["trajectory"].unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
    cls_target = torch.argmin(dist, dim=-1)

    energy_info = loss_computer._energy_supervision(
        poses_reg,
        poses_cls,
        target["trajectory"],
        dist,
        cls_target,
        _energy_context(epoch=80),
    )
    energy_info["temporal_rank_loss"].backward()

    assert poses_cls.grad is not None
    assert poses_reg.grad is None


def test_temporal_comfort_rank_can_use_comfort_inside_gt_topk():
    loss_computer = _make_loss_computer()
    poses_reg, poses_cls, target, plan_anchor = _energy_test_inputs()
    poses_cls[:, 0] = 0.5
    poses_cls[:, 1] = -0.5
    dist = torch.linalg.norm(target["trajectory"].unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
    cls_target = torch.argmin(dist, dim=-1)
    temporal_cost = torch.tensor([[0.0, 0.0, 0.0, 5.0, 5.0]])
    comfort_cost = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])

    energy_info = loss_computer._energy_supervision(
        poses_reg,
        poses_cls,
        target["trajectory"],
        dist,
        cls_target,
        _comfort_energy_context(epoch=80, temporal_cost=temporal_cost, comfort_cost=comfort_cost),
    )

    assert energy_info["temporal_rank_active_ratio"] == 1.0
    assert energy_info["temporal_rank_bad_cost"] == 0.0
