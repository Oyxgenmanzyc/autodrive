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

        curr_prediction = self._agent_forward(
            curr_features,
            curr_targets,
            previous_trajectory=previous_trajectory,
            previous_ego_delta=previous_ego_delta,
            training_epoch=self.current_epoch,
        )
        loss_dict = self.agent.compute_loss(curr_features, curr_targets, curr_prediction)
        batch_size = self._batch_size(curr_features)
        self.log(f"{logging_prefix}/temporal_pair_active", torch.ones((), device=self.device), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        for metric_name in ("temporal_start_cost", "temporal_path_cost", "temporal_velocity_cost"):
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
