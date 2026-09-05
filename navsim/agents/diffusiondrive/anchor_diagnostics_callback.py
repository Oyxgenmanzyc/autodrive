"""Opt-in, detached observers for the unmodified 3.1.02 loss.

No distributed collectives, model buffers, additional forwards, or RNG draws.
One file per epoch/rank; global histograms are merged offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.anchors.diagnostics import empty_layer_stats, summarize_layer
from navsim.agents.diffusiondrive.modules.multimodal_loss import py_sigmoid_focal_loss


class AnchorDiagnosticsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self._handle = None
        self._collect = False
        self._layers = {}
        self._layer_index = 0

    def on_fit_start(self, trainer, pl_module):
        head = pl_module.agent._transfuser_model._trajectory_head
        anchors = head.plan_anchor.detach().cpu().contiguous().numpy()
        self._mode_count = len(anchors)
        self._bank_sha256 = hashlib.sha256(anchors.tobytes()).hexdigest()
        self._run_root = str(Path(trainer.default_root_dir).resolve())
        self._directory = Path(self._run_root) / "anchor_diagnostics"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._handle = head.loss_computer.register_forward_hook(self._observe)

    def on_train_epoch_start(self, trainer, pl_module):
        self._layers = {}
        self._collect = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._layer_index = 0
        self._collect = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect = False

    @torch.no_grad()
    def _observe(self, module, inputs, output):
        # Returning None is essential: a forward hook must not replace the loss.
        if not self._collect or not module.training:
            return
        poses_reg, poses_cls, targets, plan_anchor = inputs
        reg, logits = poses_reg.detach(), poses_cls.detach()
        target, anchors = targets["trajectory"].detach(), plan_anchor.detach()
        layer = str(self._layer_index)
        self._layer_index += 1
        stats = self._layers.setdefault(layer, empty_layer_stats(self._mode_count))
        batch_size = len(reg)
        stats["batches"] += 1
        stats["samples_seen"] += batch_size
        if not all(torch.isfinite(value).all().item() for value in (reg, logits, target, anchors, output.detach())):
            stats["nonfinite_batches"] += 1
            return
        # Exactly the static ADE assignment used in the original LossComputer.
        distances = torch.linalg.norm(target.unsqueeze(1)[..., :2] - anchors, dim=-1).mean(dim=-1)
        winners = distances.argmin(dim=-1)
        onehot = torch.zeros_like(logits)
        onehot.scatter_(1, winners[:, None], 1)
        loss_cls = module.cls_loss_weight * py_sigmoid_focal_loss(logits, onehot)
        indices = winners[:, None, None, None].repeat(1, 1, reg.shape[2], reg.shape[3])
        best_reg = reg.gather(1, indices).squeeze(1)
        loss_reg = module.reg_loss_weight * F.l1_loss(best_reg, target)
        if not torch.isfinite(loss_cls).item() or not torch.isfinite(loss_reg).item():
            stats["nonfinite_batches"] += 1
            return
        selected = logits.argmax(dim=-1)
        stats["finite_samples"] += batch_size
        stats["top1_correct"] += int((selected == winners).sum().item())
        stats["weighted_cls_sum"] += float(loss_cls.item()) * batch_size
        stats["weighted_reg_sum"] += float(loss_reg.item()) * batch_size
        stats["weighted_total_sum"] += float(output.detach().item()) * batch_size
        for key, values in (("winner_counts", winners), ("selected_counts", selected)):
            counts = torch.bincount(values, minlength=self._mode_count).cpu().tolist()
            stats[key] = [a + b for a, b in zip(stats[key], counts)]

    def on_train_epoch_end(self, trainer, pl_module):
        self._collect = False
        report = {
            "schema_version": 1,
            "scope": "rank_local_training_draws",
            "run_root": self._run_root,
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "rank": int(trainer.global_rank),
            "world_size": int(trainer.world_size),
            "mode_count": self._mode_count,
            "bank_sha256": self._bank_sha256,
            "optimizer_group_lrs": [[float(group["lr"]) for group in opt.param_groups] for opt in trainer.optimizers],
            "loss_units": "per-layer weighted losses, before outer trajectory_weight; finite batches only",
            "layers": {key: summarize_layer(value) for key, value in self._layers.items()},
        }
        path = self._directory / f"epoch_{trainer.current_epoch:03d}_rank_{trainer.global_rank}.json"
        # Resumes must not silently replace a previous partial/full epoch report.
        if path.exists():
            logging.getLogger(__name__).warning("Anchor diagnostics already exist, keeping %s", path)
            return
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        logging.getLogger(__name__).info("Rank-local anchor diagnostics: %s", path)

    def teardown(self, trainer, pl_module, stage):
        self._collect = False
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
