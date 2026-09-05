import sys
import types

import numpy as np
import pytest
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

    def add_noise(self, original_samples, noise, timesteps):
        return original_samples

    def set_timesteps(self, *args, **kwargs):
        pass

    def step(self, model_output, timestep, sample):
        return types.SimpleNamespace(prev_sample=model_output)


class _ZeroTime(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, timesteps):
        return torch.zeros(len(timesteps), self.dimensions, device=timesteps.device)


class _ZeroLoss(torch.nn.Module):
    def forward(self, poses_reg, poses_cls, targets, plan_anchor):
        return poses_reg.sum() * 0.0 + poses_cls.sum() * 0.0


class _ShapeRecordingDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.noisy_shape = None

    def forward(self, traj_feature, noisy_traj_points, *args, **kwargs):
        self.noisy_shape = tuple(noisy_traj_points.shape)
        heading = torch.zeros(*noisy_traj_points.shape[:-1], 1, device=noisy_traj_points.device)
        poses_reg = torch.cat([noisy_traj_points, heading], dim=-1)
        poses_cls = torch.arange(
            noisy_traj_points.shape[1],
            device=noisy_traj_points.device,
            dtype=noisy_traj_points.dtype,
        ).unsqueeze(0).repeat(noisy_traj_points.shape[0], 1)
        return [poses_reg], [poses_cls]


def _load_model_types():
    sys.modules.pop("navsim.agents.diffusiondrive.transfuser_model_v2", None)
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
        gen_sineembed_for_position=lambda positions, **kwargs: positions,
        GridSampleCrossBEVAttention=_DummyModule,
    )
    _stub_module("navsim.agents.diffusiondrive.modules.multimodal_loss", LossComputer=_DummyModule)

    from navsim.agents.diffusiondrive.transfuser_model_v2 import (
        DiffMotionPlanningRefinementModule,
        TrajectoryHead,
    )

    return DiffMotionPlanningRefinementModule, TrajectoryHead


def _config():
    return types.SimpleNamespace(
        tf_d_model=16,
        tf_num_head=4,
        tf_dropout=0.0,
        tf_d_ffn=32,
    )


def test_refinement_module_preserves_variable_mode_dimension():
    refinement_module, _ = _load_model_types()
    module = refinement_module(embed_dims=16, ego_fut_ts=8)

    for mode_count in (20, 32, 64):
        poses_reg, poses_cls = module(torch.randn(2, mode_count, 16))
        assert poses_reg.shape == (2, mode_count, 8, 3)
        assert poses_cls.shape == (2, mode_count)


@pytest.mark.parametrize("spr_enabled", [False, True])
def test_variable_anchor_bank_smokes_train_and_test(tmp_path, spr_enabled):
    _, trajectory_head = _load_model_types()

    for mode_count in (20, 32, 64):
        anchor_path = tmp_path / f"anchors_{mode_count}.npy"
        np.save(anchor_path, np.zeros((mode_count, 8, 2), dtype=np.float32))
        config = _config()
        config.spr_enabled = spr_enabled
        head = trajectory_head(
            num_poses=8,
            d_ffn=32,
            d_model=16,
            plan_anchor_path=str(anchor_path),
            config=config,
        )
        head.time_mlp = _ZeroTime(16)
        head.loss_computer = _ZeroLoss()
        head.diff_decoder = _ShapeRecordingDecoder()
        ego_query = torch.zeros(2, 1, 16)
        ignored = torch.zeros(2, 1, 16)

        head.train()
        train_output = head.forward_train(
            ego_query,
            ignored,
            ignored,
            (1, 1),
            ignored,
            targets={"trajectory": torch.zeros(2, 8, 3)},
        )
        assert head.ego_fut_mode == mode_count
        assert head.diff_decoder.noisy_shape == (2, mode_count, 8, 2)
        assert train_output["trajectory"].shape == (2, 8, 3)
        assert torch.isfinite(train_output["trajectory_loss"])
        if spr_enabled:
            expected_spr_loss = 2.0 * head.spr_head.get_reconstruction_loss(
                train_output, {"trajectory": torch.zeros(2, 8, 3)}
            )
            torch.testing.assert_close(train_output["trajectory_loss"], expected_spr_loss)
            torch.testing.assert_close(
                train_output["trajectory_loss"], train_output["trajectory_loss_dict"]["spr_loss"]
            )  # Stubbed proposal loss is zero; reconstruction must be counted once.
            train_output["trajectory_loss"].backward()
            assert head.spr_head.trajectory_recon[-1].weight.grad is not None
            assert train_output["proposal_trajectory"].shape == (2, mode_count, 8, 3)
            assert train_output["trajectory"] is train_output["spr_trajectory"]
        else:
            assert head.spr_head is None
            assert "spr_trajectory" not in train_output

        head.eval()
        test_output = head.forward_test(ego_query, ignored, ignored, (1, 1), ignored, None)
        assert head.diff_decoder.noisy_shape == (2, mode_count, 8, 2)
        assert test_output["trajectory"].shape == (2, 8, 3)
        if spr_enabled:
            assert test_output["trajectory"] is test_output["spr_trajectory"]
            head.spr_output = "selector"
            selected = head.forward_test(ego_query, ignored, ignored, (1, 1), ignored, None)
            assert selected["trajectory"] is selected["selector_trajectory"]
            torch.testing.assert_close(selected["trajectory"], selected["proposal_trajectory"][:, -1])


@pytest.mark.parametrize(
    "anchors",
    [
        np.zeros((20, 7, 2), dtype=np.float32),
        np.zeros((20, 8, 3), dtype=np.float32),
        np.full((20, 8, 2), np.nan, dtype=np.float32),
    ],
)
def test_invalid_anchor_bank_is_rejected(tmp_path, anchors):
    _, trajectory_head = _load_model_types()
    anchor_path = tmp_path / "invalid.npy"
    np.save(anchor_path, anchors)

    with pytest.raises(ValueError):
        trajectory_head(
            num_poses=8,
            d_ffn=32,
            d_model=16,
            plan_anchor_path=str(anchor_path),
            config=_config(),
        )
