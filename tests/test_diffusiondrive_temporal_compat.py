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
    head.temporal_position_weight = 0.35
    head.temporal_delta_weight = 0.35
    head.temporal_speed_weight = 0.15
    head.temporal_heading_weight = 0.15
    return head


def test_temporal_noise_scale_returns_none_without_previous():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)

    assert head._temporal_noise_scale(plan_anchor, None) is None


def test_temporal_noise_scale_accepts_aligned_overlap_window():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = torch.zeros(1, 7, 2)

    noise_scale = head._temporal_noise_scale(plan_anchor, previous)

    assert noise_scale.shape == (1, 20, 1, 1)


def test_matching_anchor_gets_lower_noise_than_mismatched_anchor():
    head = _make_head()
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = torch.stack([torch.arange(1, 8), torch.zeros(7)], dim=-1).float().unsqueeze(0)
    plan_anchor[:, 0, :7] = previous
    plan_anchor[:, 1, :7, 1] = torch.arange(1, 8).float()

    noise_scale = head._temporal_noise_scale(plan_anchor, previous)

    assert noise_scale[0, 0, 0, 0] < noise_scale[0, 1, 0, 0]


def test_position_term_penalizes_spatially_shifted_anchor():
    head = _make_head()
    head.temporal_position_weight = 1.0
    head.temporal_delta_weight = 0.0
    head.temporal_speed_weight = 0.0
    head.temporal_heading_weight = 0.0
    plan_anchor = torch.zeros(1, 20, 8, 2)
    previous = torch.stack([torch.arange(1, 8), torch.zeros(7)], dim=-1).float().unsqueeze(0)
    plan_anchor[:, 0, :7] = previous
    plan_anchor[:, 1, :7] = previous + torch.tensor([4.0, 0.0])

    noise_scale = head._temporal_noise_scale(plan_anchor, previous)

    assert noise_scale[0, 0, 0, 0] < noise_scale[0, 1, 0, 0]
