import pytorch_lightning as pl
import torch
import inspect

from torch import Tensor
from typing import Any, Dict, Tuple

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

    def _agent_forward(self, features: Dict[str, Tensor], targets: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        forward_signature = inspect.signature(self.agent.forward)
        supported_kwargs = {key: value for key, value in kwargs.items() if key in forward_signature.parameters}
        return self.agent.forward(features, targets, **supported_kwargs)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self._agent_forward(features, targets, training_epoch=self.current_epoch)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
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

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class TemporalPairAgentLightningModule(AgentLightningModule):
    """Lightning wrapper for train-time prev -> current temporal-pair rollout."""

    def __init__(self, agent: AbstractAgent):
        super().__init__(agent)
        self._energy_loss_history = []
        self._energy_loss_epoch_sum = 0.0
        self._energy_loss_epoch_count = 0
        self._energy_start_epoch = None
        self._energy_ramp_span = 15
        self._energy_loss_threshold = 15.0
        self._energy_loss_relative_threshold = 0.05
        self._energy_force_start_epoch = 85

    @staticmethod
    def _should_start_energy_from_history(
        loss_history,
        loss_threshold: float = 15.0,
        relative_threshold: float = 0.05,
    ) -> bool:
        if len(loss_history) < 2:
            return False
        previous_loss = float(loss_history[-2])
        current_loss = float(loss_history[-1])
        relative_change = abs(current_loss - previous_loss) / max(abs(previous_loss), 1e-6)
        return current_loss < loss_threshold and relative_change < relative_threshold

    def _maybe_start_energy_schedule(self) -> None:
        if self._energy_start_epoch is not None:
            return

        current_epoch = int(self.current_epoch)
        if current_epoch >= self._energy_force_start_epoch:
            self._energy_start_epoch = self._energy_force_start_epoch
            return

        if self._should_start_energy_from_history(
            self._energy_loss_history,
            self._energy_loss_threshold,
            self._energy_loss_relative_threshold,
        ):
            self._energy_start_epoch = current_epoch

    def _energy_ramp_override(self) -> Tensor:
        self._maybe_start_energy_schedule()
        if self._energy_start_epoch is None:
            return torch.zeros((), device=self.device, dtype=torch.float32)

        ramp = (float(self.current_epoch) - float(self._energy_start_epoch)) / float(self._energy_ramp_span)
        ramp = min(max(ramp, 0.0), 1.0)
        return torch.tensor(ramp, device=self.device, dtype=torch.float32)

    def _accumulate_original_trajectory_loss(self, loss_dict: Dict[str, Tensor], batch_size: int) -> None:
        original_loss = loss_dict.get("trajectory_original_loss")
        if original_loss is None:
            original_loss = loss_dict.get("trajectory_loss")
        if original_loss is None:
            return

        loss_value = original_loss.detach().float().mean().item()
        self._energy_loss_epoch_sum += loss_value * batch_size
        self._energy_loss_epoch_count += batch_size

    def on_train_epoch_start(self) -> None:
        self._maybe_start_energy_schedule()

    def on_train_epoch_end(self) -> None:
        if self._energy_loss_epoch_count > 0:
            epoch_loss = self._energy_loss_epoch_sum / self._energy_loss_epoch_count
            self._energy_loss_history.append(epoch_loss)
            self.log(
                "train/energy_schedule_trajectory_loss_epoch",
                torch.tensor(epoch_loss, device=self.device),
                prog_bar=False,
                sync_dist=True,
            )

        self._energy_loss_epoch_sum = 0.0
        self._energy_loss_epoch_count = 0

    def _log_loss_dict(self, loss_dict: Dict[str, Tensor], logging_prefix: str, batch_size: int) -> Tensor:
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss_dict["loss"]

    @staticmethod
    def _batch_size(features: Dict[str, Tensor]) -> int:
        first_tensor = next(iter(features.values()))
        return first_tensor.shape[0]

    @staticmethod
    def _build_temporal_reference(previous_trajectory: Tensor, previous_ego_pose: Tensor) -> Tensor:
        if previous_trajectory is None or previous_trajectory.shape[-2] < 3:
            return None

        previous_trajectory = previous_trajectory[..., :3, :2]
        previous_ego_pose = previous_ego_pose.to(
            device=previous_trajectory.device,
            dtype=previous_trajectory.dtype,
        )
        previous_xy = previous_trajectory[..., :2]
        previous_heading = previous_ego_pose[..., 2]
        cos_h = torch.cos(previous_heading)
        sin_h = torch.sin(previous_heading)
        rotation = torch.stack(
            [
                torch.stack([cos_h, -sin_h], dim=-1),
                torch.stack([sin_h, cos_h], dim=-1),
            ],
            dim=-2,
        )
        previous_xy_in_current = torch.bmm(previous_xy, rotation.transpose(1, 2)) + previous_ego_pose[:, None, :2]
        return previous_xy_in_current.detach()

    def _step(self, batch: Dict[str, Any], logging_prefix: str) -> Tensor:
        prev_targets = batch["prev_targets"]
        curr_features = batch["curr_features"]
        curr_targets = batch["curr_targets"]
        reference_tensor = prev_targets["trajectory"]
        previous_ego_delta = batch["pair_metadata"]["previous_ego_delta"].to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        previous_ego_pose = batch["pair_metadata"]["previous_ego_pose"].to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

        previous_trajectory = self._build_temporal_reference(
            reference_tensor,
            previous_ego_pose,
        )
        energy_ramp_override = self._energy_ramp_override() if logging_prefix == "train" else None

        curr_prediction = self._agent_forward(
            curr_features,
            curr_targets,
            previous_trajectory=previous_trajectory,
            previous_ego_delta=previous_ego_delta,
            training_epoch=self.current_epoch,
            energy_ramp_override=energy_ramp_override,
        )
        loss_dict = self.agent.compute_loss(curr_features, curr_targets, curr_prediction)
        batch_size = self._batch_size(curr_features)
        if logging_prefix == "train":
            self._accumulate_original_trajectory_loss(loss_dict, batch_size)
            if energy_ramp_override is not None:
                self.log(
                    "train/energy_ramp_override",
                    energy_ramp_override,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        self.log(f"{logging_prefix}/temporal_pair_active", torch.ones((), device=self.device), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        for metric_name in (
            "temporal_start_cost",
            "temporal_path_cost",
            "temporal_velocity_cost",
            "history_valid_ratio",
            "history_delta_norm",
            "history_feature_delta_norm",
            "history_residual_scale",
        ):
            if metric_name in curr_prediction:
                self.log(
                    f"{logging_prefix}/{metric_name}",
                    curr_prediction[metric_name],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        return self._log_loss_dict(loss_dict, logging_prefix, batch_size)
