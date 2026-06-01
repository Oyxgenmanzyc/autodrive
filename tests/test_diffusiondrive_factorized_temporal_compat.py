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
    head.temporal_path_weight = 0.45
    head.temporal_velocity_weight = 0.35
    head.temporal_goal_weight = 0.20
    head.temporal_executed_weight = 0.30
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
    previous = torch.zeros(1, 8, 2)

    noise_scale = head._temporal_noise_scale(plan_anchor, previous)

    assert noise_scale.shape == (1, 20, 1, 1)


def test_velocity_component_penalizes_same_path_with_different_speed():
    head = _make_head()
    previous = _straight_x().unsqueeze(0)
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0] = previous
    plan_anchor[:, 1] = _straight_x(scale=2.0)

    path_cost, velocity_cost, _ = head._temporal_compatibility_components(plan_anchor, previous)

    assert velocity_cost[0, 1] > velocity_cost[0, 0]
    assert path_cost[0, 1] == path_cost[0, 0]


def test_path_component_penalizes_different_path_with_similar_speed():
    head = _make_head()
    previous = _straight_x().unsqueeze(0)
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0] = previous
    plan_anchor[:, 1] = _straight_y()

    path_cost, velocity_cost, _ = head._temporal_compatibility_components(plan_anchor, previous)

    assert path_cost[0, 1] > path_cost[0, 0]
    assert velocity_cost[0, 1] == velocity_cost[0, 0]


def test_goal_component_penalizes_final_point_deviation():
    head = _make_head()
    previous = _straight_x().unsqueeze(0)
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0] = previous
    plan_anchor[:, 1] = previous
    plan_anchor[:, 1, -1] = plan_anchor[:, 1, -1] + torch.tensor([5.0, 0.0])

    _, _, goal_cost = head._temporal_compatibility_components(plan_anchor, previous)

    assert goal_cost[0, 1] > goal_cost[0, 0]


def test_matching_anchor_gets_lower_noise_than_mismatched_anchor():
    head = _make_head()
    previous = _straight_x().unsqueeze(0)
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0] = previous
    plan_anchor[:, 1] = _straight_y(scale=2.0)

    noise_scale = head._temporal_noise_scale(plan_anchor, previous)

    assert noise_scale[0, 0, 0, 0] < noise_scale[0, 1, 0, 0]


def test_near_horizon_reference_ignores_far_anchor_points():
    head = _make_head()
    previous = _straight_x(steps=4).unsqueeze(0)
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, :4] = previous
    plan_anchor[:, 0, 4:] = _straight_y(steps=4) + torch.tensor([10.0, 0.0])
    plan_anchor[:, 1, :4] = _straight_y(steps=4)

    path_cost, velocity_cost, goal_cost = head._temporal_compatibility_components(plan_anchor, previous)

    assert path_cost[0, 0] < path_cost[0, 1]
    assert velocity_cost[0, 0] <= velocity_cost[0, 1]
    assert goal_cost[0, 0] < goal_cost[0, 1]


def test_executed_delta_keeps_path_and_velocity_factorized():
    head = _make_head()
    executed_delta = torch.tensor([[1.0, 0.0]])
    plan_anchor = torch.zeros(1, 20, 8, 2)
    plan_anchor[:, 0, 0] = torch.tensor([1.0, 0.0])
    plan_anchor[:, 1, 0] = torch.tensor([2.0, 0.0])
    plan_anchor[:, 2, 0] = torch.tensor([0.0, 1.0])

    path_cost, velocity_cost, goal_cost = head._temporal_compatibility_components(
        plan_anchor,
        previous_trajectory=None,
        previous_ego_delta=executed_delta,
    )

    assert path_cost[0, 1] == path_cost[0, 0]
    assert velocity_cost[0, 1] > velocity_cost[0, 0]
    assert path_cost[0, 2] > path_cost[0, 0]
    assert velocity_cost[0, 2] == velocity_cost[0, 0]
    assert torch.all(goal_cost == 0)
