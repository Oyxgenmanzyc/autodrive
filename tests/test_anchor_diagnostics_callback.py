"""Server tests: observers must preserve loss, gradients, RNG and checkpoints."""

import json
from types import SimpleNamespace

import pytest
import torch

from navsim.agents.diffusiondrive.anchor_diagnostics_callback import AnchorDiagnosticsCallback
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer


def _setup(tmp_path):
    loss = LossComputer(SimpleNamespace(trajectory_cls_weight=10.0, trajectory_reg_weight=8.0))
    anchors = torch.zeros(2, 8, 2)
    anchors[1] = 10.0
    head = SimpleNamespace(plan_anchor=anchors, loss_computer=loss)
    module = SimpleNamespace(agent=SimpleNamespace(_transfuser_model=SimpleNamespace(_trajectory_head=head)))
    trainer = SimpleNamespace(default_root_dir=tmp_path, current_epoch=0, global_step=1, global_rank=0, world_size=1, optimizers=[])
    return loss, anchors, module, trainer


def test_callback_preserves_loss_gradients_rng_and_state_dict(tmp_path):
    loss, anchors, module, trainer = _setup(tmp_path)
    target = {"trajectory": torch.zeros(2, 8, 3)}
    reg = torch.ones(2, 2, 8, 3, requires_grad=True)
    logits = torch.tensor([[0.1, 0.5], [0.9, -0.5]], requires_grad=True)
    bank = anchors.unsqueeze(0).repeat(2, 1, 1, 1)
    state_before = {key: value.clone() for key, value in loss.state_dict().items()}
    reference = loss(reg, logits, target, bank)
    grads_before = torch.autograd.grad(reference, (reg, logits))
    rng_before = torch.get_rng_state().clone()
    callback = AnchorDiagnosticsCallback()
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_start(trainer, module, None, 0)
    observed = loss(reg, logits, target, bank)
    grads_after = torch.autograd.grad(observed, (reg, logits))
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    for before, after in zip(grads_before, grads_after):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert set(loss.state_dict()) == set(state_before)
    for key, value in loss.state_dict().items():
        assert torch.equal(value, state_before[key])
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_train_epoch_end(trainer, module)
    report = json.loads((tmp_path / "anchor_diagnostics/epoch_000_rank_0.json").read_text())
    stats = report["layers"]["0"]
    assert stats["winner_counts"] == [2, 0]
    assert stats["selected_counts"] == [1, 1]
    assert stats["top1_anchor_accuracy"] == 0.5
    assert stats["trajectory_cls_loss"] + stats["trajectory_reg_loss"] == pytest.approx(reference.item())
    callback.teardown(trainer, module, "fit")
    assert not loss._forward_hooks


def test_observer_keeps_decoder_layers_separate_and_ignores_validation(tmp_path):
    loss, anchors, module, trainer = _setup(tmp_path)
    callback = AnchorDiagnosticsCallback()
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_start(trainer, module, None, 0)
    args = (torch.zeros(1, 2, 8, 3), torch.zeros(1, 2), {"trajectory": torch.zeros(1, 8, 3)}, anchors[None])
    loss(*args)
    loss(*args)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    loss.eval()
    loss(*args)
    assert set(callback._layers) == {"0", "1"}
    assert callback._layers["1"]["samples_seen"] == 1
    callback.teardown(trainer, module, "fit")


def test_nonfinite_loss_is_recorded_without_being_replaced(tmp_path):
    loss, anchors, module, trainer = _setup(tmp_path)
    callback = AnchorDiagnosticsCallback()
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_start(trainer, module, None, 0)
    result = loss(torch.full((1, 2, 8, 3), float("nan")), torch.zeros(1, 2), {"trajectory": torch.zeros(1, 8, 3)}, anchors[None])
    assert torch.isnan(result)
    assert callback._layers["0"]["nonfinite_batches"] == 1
    callback.on_train_epoch_end(trainer, module)
    report = json.loads((tmp_path / "anchor_diagnostics/epoch_000_rank_0.json").read_text())
    assert report["layers"]["0"]["trajectory_total_loss"] is None
    callback.teardown(trainer, module, "fit")


def test_callback_does_not_call_distributed_collectives(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostics must not synchronize DDP")

    for name in ("all_reduce", "barrier", "all_gather", "broadcast"):
        monkeypatch.setattr(torch.distributed, name, forbidden)
    loss, anchors, module, trainer = _setup(tmp_path)
    trainer.world_size = 4
    trainer.global_rank = 3
    callback = AnchorDiagnosticsCallback()
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_start(trainer, module, None, 0)
    loss(torch.zeros(1, 2, 8, 3), torch.zeros(1, 2), {"trajectory": torch.zeros(1, 8, 3)}, anchors[None])
    callback.on_train_epoch_end(trainer, module)
    assert (tmp_path / "anchor_diagnostics/epoch_000_rank_3.json").exists()
    callback.teardown(trainer, module, "fit")
