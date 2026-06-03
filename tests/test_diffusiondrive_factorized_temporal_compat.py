import sys
import types

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


def _load_trajectory_head():
    diffusers_module = _stub_module("diffusers")
    diffusers_module.__path__ = []
    _stub_module("diffusers.schedulers", DDIMScheduler=object)
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
        gen_sineembed_for_position=lambda *args, **kwargs: None,
        GridSampleCrossBEVAttention=_DummyModule,
    )
    _stub_module("navsim.agents.diffusiondrive.modules.multimodal_loss", LossComputer=_DummyModule)

    from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead

    return TrajectoryHead


def _make_head():
    TrajectoryHead = _load_trajectory_head()
    head = TrajectoryHead.__new__(TrajectoryHead)
    head.temporal_noise_strength = 0.2
    head.temporal_noise_min_scale = 0.85
    head.temporal_noise_max_scale = 1.25
    head.temporal_start_weight = 0.25
    head.temporal_path_weight = 0.40
    head.temporal_velocity_weight = 0.35
    head.temporal_reversal_threshold = 0.5
    return head


def _straight_x(scale=1.0, steps=8):
    return torch.stack([torch.arange(1, steps + 1).float() * scale, torch.zeros(steps)], dim=-1)


def _straight_y(scale=1.0, steps=8):
    return torch.stack([torch.zeros(steps), torch.arange(1, steps + 1).float() * scale], dim=-1)


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


def test_start_component_penalizes_same_direction_with_different_speed():
    head = _make_head()
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, 0] = torch.tensor([1.0, 0.0])
    plan_anchor[:, 1, 0] = torch.tensor([2.0, 0.0])

    start_cost, path_cost, velocity_cost = head._temporal_compatibility_components(
        plan_anchor,
        previous_trajectory=None,
        previous_ego_delta=executed_delta,
    )

    assert start_cost[0, 1] > start_cost[0, 0]
    assert torch.allclose(path_cost[0, 1], path_cost[0, 0], atol=1e-6)
    assert velocity_cost[0, 1] > velocity_cost[0, 0]


def test_start_and_path_penalize_abrupt_direction_change_from_executed_motion():
    head = _make_head()
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, 0] = torch.tensor([1.0, 0.0])
    plan_anchor[:, 1, 0] = torch.tensor([0.0, 1.0])

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
    accel_then_brake = torch.tensor([[[1.0, 2.0, 1.0]]])
    brake_then_accel = torch.tensor([[[2.0, 1.0, 2.0]]])

    assert head._reversal_cost(accel_then_brake)[0, 0] > head._reversal_cost(brake_then_accel)[0, 0]


def test_low_speed_turn_segments_do_not_create_false_turn_cost():
    head = _make_head()
    anchor_delta = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]])
    reference_delta = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]])

    assert torch.all(head._turn_cost(anchor_delta, reference_delta) == 0)
