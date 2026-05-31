import sys
import types
from types import SimpleNamespace

import numpy as np
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


class _LidarIndex:
    POSITION = slice(0, 3)


def _install_stubs():
    diffusers_module = _stub_module("diffusers")
    diffusers_module.__path__ = []
    _stub_module("diffusers.schedulers", DDIMScheduler=object)
    _stub_module("pytorch_lightning", Callback=object)
    _stub_module(
        "omegaconf",
        DictConfig=object,
        OmegaConf=SimpleNamespace(set_struct=lambda *args, **kwargs: None),
        open_dict=lambda *args, **kwargs: None,
    )
    _stub_module("navsim.agents.abstract_agent", AbstractAgent=object)
    _stub_module("navsim.agents.diffusiondrive.transfuser_config", TransfuserConfig=object)
    _stub_module("navsim.agents.diffusiondrive.transfuser_backbone", TransfuserBackbone=object)
    _stub_module(
        "navsim.agents.diffusiondrive.transfuser_features",
        BoundingBox2DIndex=object,
        TransfuserFeatureBuilder=object,
        TransfuserTargetBuilder=object,
    )
    _stub_module("navsim.agents.diffusiondrive.transfuser_callback", TransfuserCallback=object)
    _stub_module("navsim.agents.diffusiondrive.transfuser_loss", transfuser_loss=lambda *args, **kwargs: None)
    _stub_module("navsim.agents.diffusiondrive.modules.scheduler", WarmupCosLR=object)
    _stub_module(
        "navsim.planning.training.abstract_feature_target_builder",
        AbstractFeatureBuilder=object,
        AbstractTargetBuilder=object,
    )
    _stub_module(
        "navsim.common.dataclasses",
        AgentInput=object,
        Trajectory=object,
        SensorConfig=object,
    )
    _stub_module("navsim.common.enums", StateSE2Index=object, LidarIndex=_LidarIndex)
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


def _make_head():
    _install_stubs()
    from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead

    head = TrajectoryHead.__new__(TrajectoryHead)
    head.temporal_noise_strength = 0.2
    head.temporal_noise_min_scale = 0.85
    head.temporal_noise_max_scale = 1.25
    head.temporal_position_weight = 0.35
    head.temporal_delta_weight = 0.35
    head.temporal_speed_weight = 0.15
    head.temporal_heading_weight = 0.15
    return head


def _make_agent():
    _install_stubs()
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent

    agent = TransfuserAgent.__new__(TransfuserAgent)
    agent._history_weight_min = 0.70
    agent._scene_change_dead_zone = 0.8
    agent._config = SimpleNamespace(
        trajectory_sampling=SimpleNamespace(interval_length=0.5),
        lidar_split_height=0.2,
        max_height_lidar=100.0,
    )
    return agent


def _straight_x(steps=7):
    return torch.stack([torch.arange(1, steps + 1).float(), torch.zeros(steps)], dim=-1)


def _summary(
    route=(1.0, 0.0, 0.0),
    speed=8.0,
    accel=0.0,
):
    return {
        "route_command": np.asarray(route, dtype=np.float32),
        "ego_speed": float(speed),
        "ego_lon_accel": float(accel),
    }


def test_history_weight_none_matches_one():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = _straight_x().unsqueeze(0)
    plan_anchor[:, 0, :7] = previous
    plan_anchor[:, 1, :7, 1] = torch.arange(1, 8).float()

    no_gate = head._temporal_noise_scale(plan_anchor, previous, history_weight=None)
    full_gate = head._temporal_noise_scale(plan_anchor, previous, history_weight=1.0)

    assert torch.allclose(no_gate, full_gate)


def test_zero_history_weight_disables_temporal_noise_modulation():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = _straight_x().unsqueeze(0)
    plan_anchor[:, 0, :7] = previous
    plan_anchor[:, 1, :7, 1] = torch.arange(1, 8).float()

    noise_scale = head._temporal_noise_scale(plan_anchor, previous, history_weight=0.0)

    assert torch.allclose(noise_scale, torch.ones_like(noise_scale))


def test_smaller_history_weight_moves_noise_scale_toward_one():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = _straight_x().unsqueeze(0)
    plan_anchor[:, 0, :7] = previous
    plan_anchor[:, 1, :7, 1] = torch.arange(1, 8).float()

    full_gate = head._temporal_noise_scale(plan_anchor, previous, history_weight=1.0)
    weak_gate = head._temporal_noise_scale(plan_anchor, previous, history_weight=0.2)

    assert (weak_gate - 1.0).abs().mean() < (full_gate - 1.0).abs().mean()


def test_route_command_change_lowers_history_weight():
    agent = _make_agent()
    previous = _summary(route=(1.0, 0.0, 0.0))
    current = _summary(route=(0.0, 1.0, 0.0))

    weight = agent._history_weight_from_scene_change(previous, current)

    assert weight < 1.0


def test_small_speed_drop_stays_in_dead_zone():
    agent = _make_agent()
    previous = _summary(speed=8.0)
    current = _summary(speed=7.0)

    weight = agent._history_weight_from_scene_change(previous, current)

    assert weight == 1.0


def test_strong_ego_deceleration_lowers_history_weight_but_keeps_floor():
    agent = _make_agent()
    previous = _summary(speed=10.0, accel=0.0)
    current = _summary(speed=2.0, accel=-8.0)

    weight = agent._history_weight_from_scene_change(previous, current)

    assert weight < 1.0
    assert weight >= 0.70
