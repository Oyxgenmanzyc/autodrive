"""Pure Torch checks; run with pytest in the server training environment."""

from types import SimpleNamespace

import pytest
import torch

from navsim.agents.diffusiondrive.modules.proposal_trajectory_utils import cumsum_traj, diff_traj
from navsim.agents.diffusiondrive.modules.scene_proposal_reconstruction import (
    SceneProposalReconstruction,
    validate_spr_checkpoint,
)


def config(detach=True):
    return SimpleNamespace(
        tf_d_model=16, tf_num_head=4, tf_d_ffn=32, tf_dropout=0.0,
        spr_detach_proposals=detach,
    )


def test_coordinate_roundtrip_and_periodic_heading():
    trajectory = torch.randn(5, 8, 3)
    trajectory[..., 2] *= 7
    recovered = cumsum_traj(diff_traj(trajectory))
    torch.testing.assert_close(recovered[..., :2], trajectory[..., :2], atol=5e-6, rtol=1e-5)
    torch.testing.assert_close(recovered[..., 2].sin(), trajectory[..., 2].sin())
    torch.testing.assert_close(recovered[..., 2].cos(), trajectory[..., 2].cos())
    shifted = trajectory.clone()
    shifted[..., 2] += 2 * torch.pi
    torch.testing.assert_close(diff_traj(shifted), diff_traj(trajectory), atol=3e-6, rtol=1e-5)


def test_stationary_trajectory_keeps_upstream_normalization():
    encoded = diff_traj(torch.zeros(1, 8, 3))
    assert (encoded[..., 0] < 0).all()  # Zero displacement is not zero normalized x.
    assert torch.equal(encoded[..., 3], torch.ones(1, 8))
    torch.testing.assert_close(cumsum_traj(encoded), torch.zeros(1, 8, 3), atol=1e-6, rtol=0)


@pytest.mark.parametrize("mode_count", [20, 32, 64])
def test_variable_bank_and_upstream_architecture(mode_count):
    module = SceneProposalReconstruction(config(), mode_count)
    assert len(module.bev_cross_attn.layers) == 3
    assert module.trajectory_recon[0].in_features == mode_count * 16
    assert module.trajectory_recon[0].out_features == 16
    # Non-contiguous input exercises the bank adapter's reshape path.
    proposals = torch.randn(2, mode_count, 3, 8).transpose(-1, -2)
    output = module(proposals, torch.randn(2, 1, 16), torch.randn(2, 4, 16))
    assert output["trajectory"].shape == (2, 8, 3)
    assert output["diff_trajectory"].shape == (2, 8, 4)
    assert torch.isfinite(output["trajectory"]).all()
    target = {"trajectory": torch.randn(2, 8, 3)}
    expected = torch.nn.functional.l1_loss(output["diff_trajectory"].float(), diff_traj(target["trajectory"]))
    torch.testing.assert_close(module.get_reconstruction_loss(output, target), expected)


@pytest.mark.parametrize("detach", [True, False])
def test_gradient_boundary_preserves_scene_learning(detach):
    module = SceneProposalReconstruction(config(detach), 3)
    proposals = torch.randn(2, 3, 8, 3, requires_grad=True)
    ego = torch.randn(2, 1, 16, requires_grad=True)
    agents = torch.randn(2, 4, 16, requires_grad=True)
    output = module(proposals, ego, agents)
    module.get_reconstruction_loss(output, {"trajectory": torch.randn(2, 8, 3)}).backward()
    for tensor in (ego, agents, module.trajectory_recon[-1].weight):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    if detach:
        assert proposals.grad is None
    else:
        assert proposals.grad is not None and torch.isfinite(proposals.grad).all()
        assert proposals.grad.abs().sum() > 0


def test_wrong_bank_size_and_horizon_are_rejected():
    with pytest.raises(ValueError, match="8-pose"):
        SceneProposalReconstruction(config(), 2, num_poses=6)
    module = SceneProposalReconstruction(config(), 2)
    with pytest.raises(ValueError, match="configured"):
        module(torch.zeros(1, 3, 8, 3), torch.zeros(1, 1, 16), torch.zeros(1, 1, 16))


def checkpoint_fixture():
    head = SimpleNamespace(
        spr_head=SceneProposalReconstruction(config(), 2),
        plan_anchor=torch.arange(32, dtype=torch.float32).reshape(2, 8, 2),
    )
    prefix = "_transfuser_model._trajectory_head."
    state = {prefix + "spr_head." + key: value for key, value in head.spr_head.state_dict().items()}
    state[prefix + "plan_anchor"] = head.plan_anchor.clone()
    return head, state


def test_baseline_checkpoint_allowed_only_for_warm_start():
    head, _ = checkpoint_fixture()
    validate_spr_checkpoint(head, {})
    with pytest.raises(ValueError, match="Train 3.1.05"):
        validate_spr_checkpoint(head, {}, require_spr=True)
    validate_spr_checkpoint(SimpleNamespace(spr_head=None), {}, require_spr=True)


def test_partial_checkpoint_rejected_for_training_and_evaluation():
    head, state = checkpoint_fixture()
    state.pop("_transfuser_model._trajectory_head.spr_head.traj_encoder.0.weight")
    for require in (False, True):
        with pytest.raises(ValueError, match="incomplete"):
            validate_spr_checkpoint(head, state, require_spr=require)


def test_checkpoint_requires_exact_bank_order():
    head, state = checkpoint_fixture()
    validate_spr_checkpoint(head, state, require_spr=True)
    key = "_transfuser_model._trajectory_head.plan_anchor"
    state[key] = state[key].flip(0)
    with pytest.raises(ValueError, match="ordering"):
        validate_spr_checkpoint(head, state, require_spr=True)
    state.pop(key)
    with pytest.raises(ValueError, match="ordering"):
        validate_spr_checkpoint(head, state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires server CUDA mixed precision")
def test_cuda_autocast_forward_and_backward():
    module = SceneProposalReconstruction(config(), 20).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        result = module(
            torch.randn(2, 20, 8, 3, device="cuda"),
            torch.randn(2, 1, 16, device="cuda"),
            torch.randn(2, 4, 16, device="cuda"),
        )
        loss = module.get_reconstruction_loss(result, {"trajectory": torch.randn(2, 8, 3, device="cuda")})
    loss.backward()
    assert result["trajectory"].dtype == torch.float32
    assert torch.isfinite(result["trajectory"]).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
