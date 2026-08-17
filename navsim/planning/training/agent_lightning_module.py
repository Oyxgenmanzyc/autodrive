import pytorch_lightning as pl
import torch

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.modules.finite_trace import (
    assert_tree_finite,
    find_nonfinite_named,
    finite_gradient_norm,
    finite_trace_enabled,
)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._finite_trace_context = "not-started"
        self._brake_timing_epoch_stats = {"train": {}, "val": {}}
        self._risk_rank_epoch_stats = {"train": {}, "val": {}}

    def _reset_brake_timing_epoch_stats(self, logging_prefix: str) -> None:
        self._brake_timing_epoch_stats[logging_prefix] = {}

    def _update_brake_timing_epoch_stats(
        self,
        logging_prefix: str,
        loss_dict: Dict[str, Tensor],
    ) -> None:
        stats = self._brake_timing_epoch_stats[logging_prefix]
        for key, value in loss_dict.items():
            if not key.startswith("brake_timing") or not (key.endswith("_count") or key.endswith("_sum")):
                continue
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            detached = value.detach().float()
            stats[key] = stats.get(key, torch.zeros_like(detached)) + detached

    def _log_brake_timing_epoch_stats(self, logging_prefix: str) -> None:
        """Log exact active-scene means instead of zero-diluted batch averages."""

        stats = self._brake_timing_epoch_stats[logging_prefix]
        if not stats:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in stats.values():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        metric_prefixes = sorted(
            (key[: -len("_active_count")] for key in stats if key.endswith("_active_count")),
            key=len,
            reverse=True,
        )
        for metric_prefix in metric_prefixes:
            scene_count = stats[f"{metric_prefix}_scene_count"].clamp(min=1.0)
            active_count = stats[f"{metric_prefix}_active_count"]
            active_denom = active_count.clamp(min=1.0)
            derived = {
                f"{metric_prefix}_pre_risk_rate": stats[f"{metric_prefix}_pre_risk_count"] / scene_count,
                f"{metric_prefix}_gt_brake_rate": stats[f"{metric_prefix}_gt_brake_count"] / scene_count,
                f"{metric_prefix}_active_rate": active_count / scene_count,
            }
            for key, value in stats.items():
                owner = next((prefix for prefix in metric_prefixes if key.startswith(f"{prefix}_")), None)
                if owner == metric_prefix and key.endswith("_sum"):
                    derived[key[:-4]] = value / active_denom

            for count_name in ("scene_count", "pre_risk_count", "gt_brake_count", "active_count"):
                derived[f"{metric_prefix}_{count_name}"] = stats[f"{metric_prefix}_{count_name}"]

            for key, value in derived.items():
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=key.endswith(("active_rate", "onset_abs_error_s", "late_rate")),
                    sync_dist=False,
                )

    def _reset_risk_rank_epoch_stats(self, logging_prefix: str) -> None:
        self._risk_rank_epoch_stats[logging_prefix] = {}

    def _update_risk_rank_epoch_stats(
        self,
        logging_prefix: str,
        loss_dict: Dict[str, Tensor],
    ) -> None:
        stats = self._risk_rank_epoch_stats[logging_prefix]
        for key, value in loss_dict.items():
            if not key.startswith("risk_rank_stat_"):
                continue
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            detached = value.detach().float()
            stats[key] = stats.get(key, torch.zeros_like(detached)) + detached

    def _log_risk_rank_epoch_stats(self, logging_prefix: str) -> None:
        """Log exact selector outcomes needed to decide whether it is worth keeping."""

        stats = self._risk_rank_epoch_stats[logging_prefix]
        if not stats:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in stats.values():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        reference = next(iter(stats.values()))

        def get(name: str) -> Tensor:
            return stats.get(name, torch.zeros_like(reference))

        scene_count = get("risk_rank_stat_scene_count").clamp(min=1.0)
        valid_mode_count = get("risk_rank_stat_valid_mode_count").clamp(min=1.0)
        unsafe_count = get("risk_rank_stat_unsafe_true_count").clamp(min=1.0)
        safe_count = get("risk_rank_stat_safe_true_count").clamp(min=1.0)
        quality_pair_count = get("risk_rank_stat_quality_pair_count").clamp(min=1.0)
        rescuable_count = get(
            "risk_rank_stat_rescuable_collision_scene_count"
        ).clamp(min=1.0)
        base_noncollision_count = (
            scene_count - get("risk_rank_stat_base_collision_scene_count")
        ).clamp(min=1.0)
        metrics = {
            "risk_rank_exact_active_scene_rate": get("risk_rank_stat_active_scene_count") / scene_count,
            "risk_rank_exact_pair_scene_rate": get("risk_rank_stat_pair_scene_count") / scene_count,
            "risk_rank_exact_unsafe_rate": get("risk_rank_stat_unsafe_mode_count") / valid_mode_count,
            "risk_rank_exact_unsafe_accuracy": get("risk_rank_stat_correct_mode_count") / valid_mode_count,
            "risk_rank_exact_unsafe_recall": get("risk_rank_stat_unsafe_true_positive_count") / unsafe_count,
            "risk_rank_exact_safe_recall": get("risk_rank_stat_safe_true_negative_count") / safe_count,
            "risk_rank_exact_timing_pair_accuracy": get("risk_rank_stat_timing_pair_correct_count") / quality_pair_count,
            "risk_rank_exact_base_timing_pair_accuracy": get("risk_rank_stat_base_quality_pair_correct_count") / quality_pair_count,
            "risk_rank_exact_base_collision_rate": get("risk_rank_stat_base_collision_scene_count") / scene_count,
            "risk_rank_exact_selected_collision_rate": get("risk_rank_stat_selected_collision_scene_count") / scene_count,
            "risk_rank_exact_rescuable_collision_recall": get("risk_rank_stat_rescued_collision_scene_count") / rescuable_count,
            "risk_rank_exact_new_collision_rate": get("risk_rank_stat_new_collision_scene_count") / base_noncollision_count,
            "risk_rank_exact_selection_change_rate": get("risk_rank_stat_selection_change_count") / scene_count,
            "risk_rank_exact_collision_delta": (
                get("risk_rank_stat_selected_collision_scene_count")
                - get("risk_rank_stat_base_collision_scene_count")
            ) / scene_count,
            "risk_rank_exact_rescued_collision_count": get("risk_rank_stat_rescued_collision_scene_count"),
            "risk_rank_exact_new_collision_count": get("risk_rank_stat_new_collision_scene_count"),
        }
        for key, value in metrics.items():
            self.log(
                f"{logging_prefix}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=key.endswith(("collision_delta", "selection_change_rate")),
                sync_dist=False,
            )

    def _set_finite_trace_context(self, features: Dict[str, Tensor], logging_prefix: str, batch_idx: int) -> None:
        tokens = features.get("_cache_token", [])
        self._finite_trace_context = (
            f"phase={logging_prefix} rank={self.global_rank} epoch={self.current_epoch} "
            f"global_step={self.global_step} batch_idx={batch_idx} cache_tokens={tokens}"
        )

    def _report_parameter_state(self) -> None:
        failures = find_nonfinite_named(self.agent.named_parameters())
        if failures:
            for failure in failures[:20]:
                print(f"[DiffusionDrive][finite_trace][PARAMETER] {failure}", flush=True)
        else:
            print(
                "[DiffusionDrive][finite_trace][PARAMETER] all parameters are finite; "
                "the current forward created the first non-finite tensor",
                flush=True,
            )

    def _raise_with_context(self, error: FloatingPointError) -> None:
        print(f"[DiffusionDrive][finite_trace][CONTEXT] {self._finite_trace_context}", flush=True)
        self._report_parameter_state()
        raise error

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
        batch_idx: int,
    ) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        self._set_finite_trace_context(features, logging_prefix, batch_idx)
        try:
            assert_tree_finite("batch.features", features)
            assert_tree_finite("batch.targets", targets)
            prediction = self.agent.forward(features, targets)
            assert_tree_finite("prediction", prediction)
        except FloatingPointError as error:
            self._raise_with_context(error)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        try:
            loss_dict = self.agent.compute_loss(features, targets, prediction)
            assert_tree_finite("loss", loss_dict)
        except FloatingPointError as error:
            self._raise_with_context(error)
        self._update_brake_timing_epoch_stats(logging_prefix, loss_dict)
        self._update_risk_rank_epoch_stats(logging_prefix, loss_dict)
        batch_size = next(value.shape[0] for value in features.values() if torch.is_tensor(value))
        progress_bar_keys = {
            "loss",
            "trajectory_loss",
            "brake_timing_loss",
            "brake_timing_active_rate",
            "brake_timing_onset_abs_error_s",
            "brake_timing_late_rate",
            "risk_mode_ranking_loss",
            "risk_rank_unsafe_accuracy",
            "risk_rank_rescuable_collision_recall",
        }
        for k, v in loss_dict.items():
            if v is not None:
                if k.startswith("risk_rank_stat_"):
                    continue
                if k.startswith("brake_timing"):
                    if k in progress_bar_keys:
                        self.log(
                            f"{logging_prefix}_step/{k}",
                            v,
                            on_step=True,
                            on_epoch=False,
                            prog_bar=True,
                            sync_dist=True,
                            batch_size=batch_size,
                        )
                    continue
                self.log(
                    f"{logging_prefix}/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=k in progress_bar_keys,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val", batch_idx)

    def on_train_epoch_start(self) -> None:
        self._reset_brake_timing_epoch_stats("train")
        self._reset_risk_rank_epoch_stats("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_brake_timing_epoch_stats("val")
        self._reset_risk_rank_epoch_stats("val")

    def on_train_epoch_end(self) -> None:
        self._log_brake_timing_epoch_stats("train")
        self._log_risk_rank_epoch_stats("train")

    def on_validation_epoch_end(self) -> None:
        self._log_brake_timing_epoch_stats("val")
        self._log_risk_rank_epoch_stats("val")

    def on_before_optimizer_step(self, optimizer) -> None:
        """Stop before a non-finite gradient can poison model parameters."""

        if not finite_trace_enabled():
            return
        named_parameters = list(self.agent.named_parameters())
        grad_norm = finite_gradient_norm(named_parameters).to(self.device)
        self.log(
            "train/finite_trace_grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        if bool(torch.isfinite(grad_norm).item()):
            return

        print(f"[DiffusionDrive][finite_trace][CONTEXT] {self._finite_trace_context}", flush=True)
        failures = find_nonfinite_named(
            (name, parameter.grad) for name, parameter in named_parameters if parameter.grad is not None
        )
        for failure in failures[:40]:
            print(f"[DiffusionDrive][finite_trace][GRADIENT] {failure}", flush=True)
        raise FloatingPointError("Non-finite unscaled gradient detected before optimizer.step")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
