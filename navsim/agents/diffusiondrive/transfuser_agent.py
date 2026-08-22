from typing import Any, List, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel as TransfuserModel

from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback 
from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(config)
        self._previous_trajectory: Optional[np.ndarray] = None
        self._temporal_reset_distance = 5.0
        self._last_temporal_reference_active = False
        self._last_previous_ego_delta_active = False
        self._last_temporal_rescore_active = 0.0
        self._last_temporal_rescore_changed = 0.0
        self._last_temporal_rescore_selected_cost = 0.0
        self._last_temporal_rescore_base_cost = 0.0
        self._last_temporal_rescore_topk_min_cost = 0.0
        self._last_temporal_rescore_cls_margin = 0.0
        self._last_temporal_rescore_selected_mode = 0.0
        self._last_temporal_rescore_base_mode = 0.0
        self.init_from_pretrained()

    @staticmethod
    def _without_checkpoint_anchor(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the anchor bank selected by config instead of restoring a checkpoint asset."""
        return {key: value for key, value in state_dict.items() if not key.endswith("plan_anchor")}

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            state_dict = self._without_checkpoint_anchor(state_dict)
            
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            unexpected_missing = [key for key in missing_keys if not key.endswith("plan_anchor")]
            if unexpected_missing:
                print(f"Missing keys when loading pretrained weights: {unexpected_missing}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
            print(f"Loaded checkpoint from: {self._checkpoint_path}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(self._without_checkpoint_anchor(state_dict), strict=False)
        print(f"Initialized agent from checkpoint: {self._checkpoint_path}")


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]=None,
        previous_trajectory: Optional[torch.Tensor]=None,
        previous_ego_delta: Optional[torch.Tensor]=None,
        training_epoch: Optional[int]=None,
        energy_ramp_override: Optional[torch.Tensor]=None,
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(
            features,
            targets=targets,
            previous_trajectory=previous_trajectory,
            previous_ego_delta=previous_ego_delta,
            training_epoch=training_epoch,
            energy_ramp_override=energy_ramp_override,
        )

    def reset_temporal_context(self) -> None:
        """Clears cached inference trajectory before starting an unrelated scene."""
        self._previous_trajectory = None
        self._last_temporal_reference_active = False
        self._last_previous_ego_delta_active = False
        self._last_temporal_rescore_active = 0.0
        self._last_temporal_rescore_changed = 0.0
        self._last_temporal_rescore_selected_cost = 0.0
        self._last_temporal_rescore_base_cost = 0.0
        self._last_temporal_rescore_topk_min_cost = 0.0
        self._last_temporal_rescore_cls_margin = 0.0
        self._last_temporal_rescore_selected_mode = 0.0
        self._last_temporal_rescore_base_mode = 0.0

    def _build_temporal_reference(self, agent_input: AgentInput) -> Optional[np.ndarray]:
        if self._previous_trajectory is None or len(agent_input.ego_statuses) < 2:
            return None

        previous_ego_pose = agent_input.ego_statuses[-2].ego_pose
        previous_xy = self._previous_trajectory[:, :2]
        cos_h = np.cos(previous_ego_pose[2])
        sin_h = np.sin(previous_ego_pose[2])
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        previous_xy_in_current = previous_xy @ rotation.T + previous_ego_pose[:2].astype(np.float32)

        if previous_xy_in_current.shape[0] < 3:
            return None
        if np.linalg.norm(previous_xy_in_current[0]) > self._temporal_reset_distance:
            return None

        near_horizon_points = 3
        temporal_reference = previous_xy_in_current[:near_horizon_points]
        return temporal_reference.astype(np.float32)

    def _build_previous_ego_delta(self, agent_input: AgentInput) -> Optional[np.ndarray]:
        if len(agent_input.ego_statuses) < 2:
            return None

        previous_ego_pose = agent_input.ego_statuses[-2].ego_pose
        return (-previous_ego_pose[:2]).astype(np.float32)

    def get_temporal_debug_info(self) -> Dict[str, Any]:
        return {
            "temporal_reference_active": self._last_temporal_reference_active,
            "previous_ego_delta_active": self._last_previous_ego_delta_active,
            "temporal_rescore_active": self._last_temporal_rescore_active,
            "temporal_rescore_changed": self._last_temporal_rescore_changed,
            "temporal_rescore_selected_cost": self._last_temporal_rescore_selected_cost,
            "temporal_rescore_base_cost": self._last_temporal_rescore_base_cost,
            "temporal_rescore_topk_min_cost": self._last_temporal_rescore_topk_min_cost,
            "temporal_rescore_cls_margin": self._last_temporal_rescore_cls_margin,
            "temporal_rescore_selected_mode": self._last_temporal_rescore_selected_mode,
            "temporal_rescore_base_mode": self._last_temporal_rescore_base_mode,
        }

    @staticmethod
    def _prediction_scalar(predictions: Dict[str, torch.Tensor], key: str) -> float:
        value = predictions.get(key)
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes trajectory while passing the previous prediction as a temporal continuity reference.
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features = {k: v.unsqueeze(0) for k, v in features.items()}
        previous_trajectory = self._build_temporal_reference(agent_input)
        previous_trajectory_tensor = None
        if previous_trajectory is not None:
            previous_trajectory_tensor = torch.tensor(previous_trajectory).unsqueeze(0)
        previous_ego_delta = self._build_previous_ego_delta(agent_input)
        previous_ego_delta_tensor = None
        if previous_ego_delta is not None:
            previous_ego_delta_tensor = torch.tensor(previous_ego_delta).unsqueeze(0)
        self._last_temporal_reference_active = previous_trajectory_tensor is not None
        self._last_previous_ego_delta_active = previous_ego_delta_tensor is not None

        with torch.no_grad():
            predictions = self.forward(
                features,
                previous_trajectory=previous_trajectory_tensor,
                previous_ego_delta=previous_ego_delta_tensor,
            )
            self._last_temporal_rescore_active = self._prediction_scalar(predictions, "temporal_rescore_active")
            self._last_temporal_rescore_changed = self._prediction_scalar(predictions, "temporal_rescore_changed")
            self._last_temporal_rescore_selected_cost = self._prediction_scalar(predictions, "temporal_rescore_selected_cost")
            self._last_temporal_rescore_base_cost = self._prediction_scalar(predictions, "temporal_rescore_base_cost")
            self._last_temporal_rescore_topk_min_cost = self._prediction_scalar(predictions, "temporal_rescore_topk_min_cost")
            self._last_temporal_rescore_cls_margin = self._prediction_scalar(predictions, "temporal_rescore_cls_margin")
            self._last_temporal_rescore_selected_mode = self._prediction_scalar(predictions, "temporal_rescore_selected_mode")
            self._last_temporal_rescore_base_mode = self._prediction_scalar(predictions, "temporal_rescore_base_mode")
            poses = predictions["trajectory"].squeeze(0).numpy()

        self._previous_trajectory = poses.copy()
        return Trajectory(poses)
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_temporal_optimization_parameters(self):
        """Returns the trajectory-head parameters refined by temporal supervision."""
        return self._transfuser_model._trajectory_head.parameters()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
