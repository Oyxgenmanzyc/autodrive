"""Fit on train pairs; calibrate on val logs A; select checkpoints on val logs B."""
import csv
import hashlib
import json
from pathlib import Path

import torch
import pytorch_lightning as pl

from .timing_gain import TimingGainSelector, calibrate_policy, choose_action, outcome_metrics
from .veto import load_veto


def calibration_log(log_name):
    return int(hashlib.sha256(log_name.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def quantiles(values):
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": values.mean().item(),
            **{name: torch.quantile(values.float(), q).item()
               for name, q in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))}}


class TimingGainModule(pl.LightningModule):
    def __init__(self, metadata, reference_path, records, lr=3e-4, epochs=10):
        super().__init__()
        reference, _ = load_veto(reference_path, "cpu", metadata["provenance"])
        self.model = TimingGainSelector(reference, metadata["use_timing"])
        self.metadata, self.records = metadata, records
        self.lr, self.epochs = lr, epochs
        self.register_buffer("positive_weights", torch.tensor(metadata["positive_weights"]).float())
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        output = self.model(batch["context"], batch["pcs_mode"], batch["base_mode"], verify_modes=True)
        losses = self.model.loss(output, batch["labels"], batch["scores"], self.positive_weights)
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=name == "loss", on_epoch=True,
                     batch_size=len(batch["scores"]), sync_dist=True)
        self.log("train/pcs_mode_mismatch_rate", output["pcs_mode_mismatch"].float().mean(),
                 on_step=False, on_epoch=True, batch_size=len(batch["scores"]), sync_dist=True)
        return losses["loss"]

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        out = self.model(batch["context"], batch["pcs_mode"], batch["base_mode"], verify_modes=True)
        rows = torch.arange(len(batch["pcs_mode"]), device=self.device)
        pcs, base = batch["pcs_mode"], batch["base_mode"]
        packed = torch.cat([
            batch["index"][:, None], batch["scores"][rows, pcs, None], batch["scores"][rows, base, None],
            (pcs != base)[:, None], out["gain_cost"], out["risks"], out["reference_risks"],
            batch["labels"][rows, pcs], batch["labels"][rows, base],
        ], -1).double()
        if not torch.isfinite(packed).all():
            raise ValueError("Nonfinite validation prediction")
        self.validation_rows.extend(packed.cpu().tolist())

    def on_validation_epoch_end(self):
        rows = self.validation_rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for rank_rows in gathered for row in rank_rows]
        unique = {int(row[0]): row[1:] for row in rows}
        indices = sorted(unique)
        values = torch.tensor([unique[i] for i in indices], dtype=torch.float64)
        if not len(indices):
            raise ValueError("Empty validation set")
        mask = torch.tensor([calibration_log(self.records[i]["log_name"]) for i in indices])
        if not mask.any() or mask.all():
            if not self.metadata["smoke"]:
                raise ValueError("Need both calibration and holdout log groups")
            # Tiny smoke is plumbing only. No calibration or performance claim.
            policy = {"mode": "reference", "reference_thresholds": self.model.reference_thresholds}
        else:
            v = values[mask]
            policy, _, _ = calibrate_policy(
                v[:, 3:5], v[:, 5:8], v[:, 8:11], v[:, 2].bool(), v[:, 0], v[:, 1],
                v[:, 11:16], v[:, 16:21], self.model.reference_thresholds,
            )
        self.model.policy = policy
        self.metadata["policy"] = policy
        ref_policy = {"mode": "reference", "reference_thresholds": self.model.reference_thresholds}
        diagnostics = {"policy": policy, "threshold_source": "navtrain_val_calibration_logs_only",
                       "calibration_count": int(mask.sum()), "holdout_count": int((~mask).sum())}
        for name, group in (("calibration", mask), ("val", ~mask)):
            if not group.any():
                continue
            v = values[group]
            pcs, base, changed = v[:, 0], v[:, 1], v[:, 2].bool()
            veto, utility = choose_action(v[:, 3:5], v[:, 5:8], v[:, 8:11], changed, policy)
            ref_veto, _ = choose_action(v[:, 3:5], v[:, 5:8], v[:, 8:11], changed, ref_policy)
            result = outcome_metrics(veto, pcs, base, v[:, 11:16], v[:, 16:21], changed)
            ref_result = outcome_metrics(ref_veto, pcs, base, v[:, 11:16], v[:, 16:21], changed)
            result.update(reference_pdm=ref_result["pdm"],
                          gain_vs_pcs=result["pdm"]-result["pcs_pdm"],
                          gain_vs_reference=result["pdm"]-ref_result["pdm"], scene_count=len(v))
            for key, value in result.items():
                self.log(f"{name}/{key}", float(value), sync_dist=False,
                         prog_bar=name == "val" and key in ("pdm", "gain_vs_reference"))
            delta = pcs-base
            diagnostics[name] = {"result": result, "reference": ref_result}
            for label, subgroup in (("correct_reference_veto", ref_veto & (delta < -1e-8)),
                                    ("wrong_reference_veto", ref_veto & (delta > 1e-8)),
                                    ("released_reference_veto", ref_veto & ~veto)):
                diagnostics[name][label] = {
                    "actual_delta": quantiles(delta[subgroup]),
                    "predicted_delta": quantiles((v[:, 3]-v[:, 4])[subgroup]),
                    "weighted_utility": quantiles(utility[subgroup]),
                }
            for j, factor in enumerate(("nc", "dac", "ttc")):
                metric = (0, 1, 3)[j]
                positive = v[:, 16+metric] > v[:, 11+metric]
                prediction = v[:, 5+j]
                diagnostics[name][factor] = {
                    "relative_failures": int(positive.sum()),
                    "caught_by_policy": int((positive & veto).sum()),
                    "risk_on_failure": quantiles(prediction[positive]),
                    "risk_on_other": quantiles(prediction[~positive]),
                }
        # A smoke with no holdout still permits checkpoint serialization.
        if self.metadata["smoke"] and mask.all():
            self.log("val/pdm", diagnostics["calibration"]["result"]["pdm"], sync_dist=False)
        if self.trainer.is_global_zero:
            directory = Path(self.trainer.default_root_dir) / "diagnostics"
            directory.mkdir(exist_ok=True)
            stem = f"epoch_{self.current_epoch:02d}"
            (directory / (stem + ".json")).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            with (directory / (stem + ".csv")).open("w", newline="", encoding="utf-8") as stream:
                columns = ["token", "log_name", "calibration", "pcs_pdm", "base_pdm", "changed",
                           "expected_gain", "expected_loss", "nc_risk", "dac_risk", "ttc_risk",
                           "reference_nc", "reference_dac", "reference_ttc",
                           *["pcs_"+k for k in ("nc", "dac", "ep", "ttc", "comfort")],
                           *["base_"+k for k in ("nc", "dac", "ep", "ttc", "comfort")]]
                writer = csv.writer(stream)
                writer.writerow(columns)
                for i, index in enumerate(indices):
                    record = self.records[index]
                    writer.writerow([record["token"], record["log_name"], bool(mask[i]), *values[i].tolist()])
        self.validation_rows = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["timing_gain_metadata"] = self.metadata

    def on_load_checkpoint(self, checkpoint):
        old = checkpoint["timing_gain_metadata"]
        for key in ("schema", "provenance", "pcs_scorer_sha256", "reference_sha256", "use_timing", "positive_weights"):
            if old[key] != self.metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        self.metadata = old
        self.model.policy = old["policy"]
