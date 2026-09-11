"""Fine-tune the diffusion trajectory head, never PCS/TRV."""
import torch
import pytorch_lightning as pl
from .data import timing_config
from .risk_brake_timing import compute_brake_timing_loss, compute_brake_timing_diagnostics


def enable_generator_training(model):
    model.requires_grad_(False)
    # Keep the anchor, classifier parameters and sensor/context encoder fixed.
    # Shared denoising features can still change logits; weights/decision rules
    # of the external PCS/TRV selector are identical in every experiment arm.
    for name, parameter in model._trajectory_head.named_parameters():
        if name != "plan_anchor" and "plan_cls_branch" not in name:
            parameter.requires_grad_(True)


class GeneratorTimingModule(pl.LightningModule):
    def __init__(self, generator, metadata, lr=2e-5, epochs=10, weight=0.1):
        super().__init__()
        self.generator = generator
        self.metadata = metadata
        self.lr, self.epochs = lr, epochs
        self.timing_config = timing_config(weight)
        enable_generator_training(generator)
        self._poses = None
        self.generator._trajectory_head.diff_decoder.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if self.training:
            self._poses = output[0][-1]

    def train(self, mode=True):
        super().train(mode)
        # eval prevents frozen BatchNorm statistics/dropout changing perception.
        self.generator.eval()
        self.generator._trajectory_head.train(mode)
        return self

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.generator.parameters() if p.requires_grad], lr=self.lr, weight_decay=1e-4,
        )
        return {"optimizer": optimizer, "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)}

    def training_step(self, batch, batch_idx):
        context, targets = batch
        self._poses = None
        output = self._forward_head(context, targets)
        if self._poses is None:
            raise RuntimeError("Training did not produce differentiable decoder candidates")
        poses, self._poses = self._poses, None
        if not torch.isfinite(poses).all():
            raise ValueError("Nonfinite generated training candidates")
        timing = compute_brake_timing_loss(
            poses, targets, self.generator._trajectory_head.plan_anchor.unsqueeze(0), self.timing_config,
        )
        # Correct active-count normalization across DDP ranks, including ranks
        # with no rare braking examples. DDP averages gradients across ranks.
        count = timing["brake_timing_active_count"]
        if torch.distributed.is_initialized():
            global_count = count.clone()
            torch.distributed.all_reduce(global_count)
            scale = torch.distributed.get_world_size() * count / global_count.clamp_min(1)
        else:
            scale = count / count.clamp_min(1)
        auxiliary = timing["brake_timing_loss"] * scale
        main_loss = output["trajectory_loss"]
        loss = main_loss + auxiliary
        for name, value in (("loss", loss), ("trajectory_loss", main_loss), ("timing_loss", auxiliary),
                            ("active_rate", timing["brake_timing_active_rate"])):
            self.log("train/" + name, value, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=len(targets["trajectory"]))
        return loss

    def on_validation_epoch_start(self):
        self._timing_sums = {}

    def _forward_head(self, context, targets=None):
        # The old PCS cache contains the frozen K67 perception/context tensors.
        # status_encoding is retained in the legacy signature but is not read by
        # either decoder layer; pass an explicit zero tensor rather than inventing data.
        bev = context["bev"].float()
        agents = context["agents"].float()
        ego = context["ego"].float()
        status = ego.new_zeros((len(ego), 1, ego.shape[-1]))
        return self.generator._trajectory_head(
            ego, agents, bev, bev.shape[-2:], status, targets=targets,
        )

    def validation_step(self, batch, batch_idx):
        context, targets = batch
        # Same validation noise each epoch/arm for the same DDP batch layout.
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(100000 + batch_idx + self.global_rank * 1000000)
            output = self._forward_head(context)
        proposals = output["proposal_trajectory"]
        distances = torch.linalg.vector_norm(proposals[..., :2] - targets["trajectory"][:, None, :, :2], dim=-1).mean(-1)
        oracle_ade = distances.min(-1).values.mean()
        self.log("val/oracle_ade", oracle_ade, sync_dist=True, batch_size=len(proposals))
        diagnostics = compute_brake_timing_loss(
            proposals, targets, self.generator._trajectory_head.plan_anchor.unsqueeze(0), self.timing_config,
        )
        diagnostics.update(compute_brake_timing_diagnostics(output["trajectory"], targets, self.timing_config))
        for key, value in diagnostics.items():
            if key.endswith("_sum") or key.endswith("_count"):
                self._timing_sums[key] = self._timing_sums.get(key, 0) + value.detach().double()

    def on_validation_epoch_end(self):
        for key in sorted(self._timing_sums):
            value = self._timing_sums[key]
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(value)
        for prefix in ("brake_timing", "brake_timing_selected"):
            count = self._timing_sums[prefix + "_active_count"].clamp_min(1)
            scenes = self._timing_sums[prefix + "_scene_count"].clamp_min(1)
            self.log("val/" + prefix + "_active_rate", self._timing_sums[prefix + "_active_count"] / scenes, sync_dist=True)
            for metric in ("onset_abs_error_s", "late_rate", "early_rate", "pred_jerk_abs", "speed_mae"):
                self.log("val/" + prefix + "_" + metric, self._timing_sums[prefix + "_" + metric + "_sum"] / count, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["generator_timing_metadata"] = self.metadata
