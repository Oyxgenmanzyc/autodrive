import pytorch_lightning as pl
import torch

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._risk_rank_epoch_stats = {"train": {}, "val": {}}

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
        stats = self._risk_rank_epoch_stats[logging_prefix]
        if not stats:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in stats.values():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        def get(name: str) -> Tensor:
            value = stats.get(name)
            if value is not None:
                return value
            reference = next(iter(stats.values()))
            return torch.zeros_like(reference)

        scene_count = get("risk_rank_stat_scene_count").clamp(min=1.0)
        pair_scene_count = get("risk_rank_stat_pair_scene_count").clamp(min=1.0)
        valid_mode_count = get("risk_rank_stat_valid_mode_count").clamp(min=1.0)
        unsafe_true_count = get("risk_rank_stat_unsafe_true_count").clamp(min=1.0)
        safe_true_count = get("risk_rank_stat_safe_true_count").clamp(min=1.0)
        quality_pair_count = get("risk_rank_stat_quality_pair_count").clamp(min=1.0)

        derived = {
            "risk_rank_exact_scene_rate": get("risk_rank_stat_active_scene_count") / scene_count,
            "risk_rank_exact_pair_scene_rate": get("risk_rank_stat_pair_scene_count") / scene_count,
            "risk_rank_exact_pairs_per_pair_scene": get("risk_rank_stat_pair_count") / pair_scene_count,
            "risk_rank_exact_unsafe_rate": get("risk_rank_stat_unsafe_mode_count") / valid_mode_count,
            "risk_rank_exact_unsafe_accuracy": get("risk_rank_stat_correct_mode_count") / valid_mode_count,
            "risk_rank_exact_unsafe_recall": (
                get("risk_rank_stat_unsafe_true_positive_count") / unsafe_true_count
            ),
            "risk_rank_exact_safe_recall": (
                get("risk_rank_stat_safe_true_negative_count") / safe_true_count
            ),
            "risk_rank_exact_collision_proxy_rate": (
                get("risk_rank_stat_collision_proxy_count") / valid_mode_count
            ),
            "risk_rank_exact_future_agent_collision_rate": (
                get("risk_rank_stat_future_agent_collision_count") / valid_mode_count
            ),
            "risk_rank_exact_ttc_failure_rate": (
                get("risk_rank_stat_ttc_failure_count") / valid_mode_count
            ),
            "risk_rank_exact_base_quality_pair_accuracy": (
                get("risk_rank_stat_base_quality_pair_correct_count") / quality_pair_count
            ),
            "risk_rank_exact_timing_pair_accuracy": (
                get("risk_rank_stat_timing_pair_correct_count") / quality_pair_count
            ),
            "risk_rank_exact_base_collision_scene_rate": (
                get("risk_rank_stat_base_collision_scene_count") / scene_count
            ),
            "risk_rank_exact_selected_collision_scene_rate": (
                get("risk_rank_stat_selected_collision_scene_count") / scene_count
            ),
            "risk_rank_exact_rescuable_collision_recall": (
                get("risk_rank_stat_rescued_collision_scene_count")
                / get("risk_rank_stat_rescuable_collision_scene_count").clamp(min=1.0)
            ),
            "risk_rank_exact_new_collision_rate": (
                get("risk_rank_stat_new_collision_scene_count") / scene_count
            ),
            "risk_rank_exact_selection_change_rate": (
                get("risk_rank_stat_selection_change_count") / scene_count
            ),
            "risk_rank_exact_rescuable_collision_scene_count": get(
                "risk_rank_stat_rescuable_collision_scene_count"
            ),
            "risk_rank_exact_rescued_collision_scene_count": get(
                "risk_rank_stat_rescued_collision_scene_count"
            ),
            "risk_rank_exact_pair_count": get("risk_rank_stat_pair_count"),
            "risk_rank_exact_pair_scene_count": get("risk_rank_stat_pair_scene_count"),
        }
        for key, value in derived.items():
            self.log(
                f"{logging_prefix}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=key.endswith(("unsafe_accuracy", "unsafe_recall", "pair_scene_rate")),
                sync_dist=False,
            )

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        self._update_risk_rank_epoch_stats(logging_prefix, loss_dict)
        batch_size = next(iter(features.values())).shape[0]
        progress_bar_keys = {
            "loss",
            "trajectory_loss",
            "risk_mode_ranking_loss",
            "risk_rank_scene_rate",
            "risk_rank_pair_scene_rate",
            "risk_rank_pair_count",
            "risk_rank_unsafe_accuracy",
            "risk_rank_unsafe_recall",
        }
        for k, v in loss_dict.items():
            if v is not None:
                if k.startswith("risk_rank_stat_"):
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
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def on_train_epoch_start(self) -> None:
        self._reset_risk_rank_epoch_stats("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_risk_rank_epoch_stats("val")

    def on_train_epoch_end(self) -> None:
        self._log_risk_rank_epoch_stats("train")

    def on_validation_epoch_end(self) -> None:
        self._log_risk_rank_epoch_stats("val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
