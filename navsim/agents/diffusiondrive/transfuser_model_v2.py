from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )


    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]=None,
        previous_trajectory: Optional[torch.Tensor]=None,
        previous_ego_delta: Optional[torch.Tensor]=None,
        training_epoch: Optional[int]=None,
        energy_ramp_override: Optional[torch.Tensor]=None,
    ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            targets=targets,
            global_img=None,
            previous_trajectory=previous_trajectory,
            previous_ego_delta=previous_ego_delta,
            training_epoch=training_epoch,
            energy_ramp_override=energy_ramp_override,
        )
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list


class HistoryPlanningAdapter(nn.Module):
    """BridgeAD-style short history query adapter for DiffusionDrive planning modes."""

    def __init__(self, d_model: int, num_heads: int, history_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.history_steps = history_steps
        self.current_step_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 64),
            nn.Linear(d_model, d_model),
        )
        self.history_step_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 64),
            nn.Linear(d_model, d_model),
        )
        self.history_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.step_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mode_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_step = nn.LayerNorm(d_model)
        self.norm_mode = nn.LayerNorm(d_model)
        self.history_gate = nn.Parameter(torch.tensor(0.1))

    def _inactive_metrics(self, traj_feature):
        return {
            "history_valid_ratio": traj_feature.new_tensor(0.0),
            "history_gate": self.history_gate.detach(),
            "history_delta_norm": traj_feature.new_tensor(0.0),
        }

    def forward(self, traj_feature, noisy_traj_points, previous_trajectory):
        if previous_trajectory is None:
            return traj_feature, self._inactive_metrics(traj_feature)

        if previous_trajectory.dim() == 2:
            previous_trajectory = previous_trajectory.unsqueeze(0)
        if previous_trajectory.dim() < 3 or previous_trajectory.shape[0] != traj_feature.shape[0]:
            return traj_feature, self._inactive_metrics(traj_feature)

        steps = min(self.history_steps, noisy_traj_points.shape[-2], previous_trajectory.shape[-2])
        if steps < self.history_steps:
            return traj_feature, self._inactive_metrics(traj_feature)

        previous_trajectory = previous_trajectory.to(device=traj_feature.device, dtype=traj_feature.dtype)
        current_points = noisy_traj_points[:, :, :steps, :2]
        history_points = previous_trajectory[:, :steps, :2]
        if not torch.isfinite(current_points).all() or not torch.isfinite(history_points).all():
            return traj_feature, self._inactive_metrics(traj_feature)

        bs, num_modes, _, d_model = (
            traj_feature.shape[0],
            traj_feature.shape[1],
            steps,
            traj_feature.shape[-1],
        )
        current_embed = gen_sineembed_for_position(current_points, hidden_dim=64)
        current_step_query = self.current_step_encoder(current_embed) + traj_feature.unsqueeze(2)
        history_embed = gen_sineembed_for_position(history_points, hidden_dim=64)
        history_step_query = self.history_step_encoder(history_embed)

        q = current_step_query.reshape(bs * num_modes, steps, d_model)
        kv = history_step_query.unsqueeze(1).expand(bs, num_modes, steps, d_model)
        kv = kv.reshape(bs * num_modes, steps, d_model)

        history_context = self.history_attn(q, kv, kv)[0]
        step_query = self.norm_step(q + self.dropout(history_context))
        step_context = self.step_self_attn(step_query, step_query, step_query)[0]
        step_query = self.norm_step(step_query + self.dropout(step_context))
        step_query = step_query.reshape(bs, num_modes, steps, d_model)

        mode_delta = step_query.mean(dim=2)
        mode_context = self.mode_self_attn(mode_delta, mode_delta, mode_delta)[0]
        mode_delta = self.norm_mode(mode_delta + self.dropout(mode_context))
        enhanced = traj_feature + self.history_gate * mode_delta

        diagnostics = {
            "history_valid_ratio": traj_feature.new_tensor(1.0),
            "history_gate": self.history_gate.detach(),
            "history_delta_norm": mode_delta.detach().norm(dim=-1).mean(),
        }
        return enhanced, diagnostics


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20
        self.temporal_noise_strength = 0.2
        self.temporal_noise_min_scale = 0.88
        self.temporal_noise_max_scale = 1.35
        self.temporal_start_weight = 0.25
        self.temporal_path_weight = 0.40
        self.temporal_velocity_weight = 0.35
        self.temporal_dt = 0.5
        self.temporal_low_speed_delta = 0.20
        self.temporal_accel_delta_tolerance = 0.60
        self.temporal_brake_delta_tolerance = 1.00
        self.temporal_direction_tolerance = 0.45
        self.temporal_lateral_accel_tolerance = 4.89
        self.temporal_turn_change_tolerance = 0.48
        self.temporal_jerk_delta_tolerance = 0.50
        self.energy_topk = 3
        self.energy_temperature = 0.5
        self.energy_start_epoch = 70
        self.energy_full_epoch = 85
        self.energy_gt_weight = 0.00
        self.energy_temporal_weight = 0.80
        self.energy_comfort_weight = 0.20
        self.temporal_rank_weight_max = 0.30
        self.temporal_rank_margin = 0.10
        self.temporal_rank_energy_gap = 0.20
        self.temporal_rank_min_epoch = 50.0
        self.temporal_rank_ramp_epochs = 30.0
        self.temporal_rank_use_comfort = True
        self.temporal_aux_weight_max = 0.0
        self.temporal_rescore_topk = 5
        self.temporal_rescore_alpha = 0.05
        self.temporal_rescore_cost_clamp = 2.0

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )


        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.history_planning_adapter = HistoryPlanningAdapter(
            d_model=d_model,
            num_heads=config.tf_num_head,
            history_steps=3,
            dropout=config.tf_dropout,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def _point_deltas(self, points):
        origin = torch.zeros_like(points[..., :1, :])
        previous_points = torch.cat([origin, points[..., :-1, :]], dim=-2)
        return points - previous_points

    def _angle_error(self, anchor_angle, previous_angle):
        return torch.atan2(
            torch.sin(anchor_angle - previous_angle),
            torch.cos(anchor_angle - previous_angle),
        ).abs() / np.pi

    def _angle_error_rad(self, anchor_angle, reference_angle):
        return torch.atan2(
            torch.sin(anchor_angle - reference_angle),
            torch.cos(anchor_angle - reference_angle),
        ).abs()

    def _positive_violation(self, value, tolerance, scale=None):
        if scale is None:
            scale = tolerance
        return (torch.relu(value - tolerance) / (scale + 1e-6)).clamp(max=2.0)

    def _signed_delta_violation(self, delta):
        accel_tol = self.temporal_accel_delta_tolerance
        brake_tol = self.temporal_brake_delta_tolerance
        accel_excess = self._positive_violation(delta, accel_tol)
        brake_excess = self._positive_violation(-delta, brake_tol)
        return torch.where(delta >= 0, accel_excess, brake_excess)

    def _dynamic_direction_tolerance(self, reference_speed):
        speed_mps = reference_speed / self.temporal_dt
        dynamic_tolerance = (
            self.temporal_dt
            * self.temporal_lateral_accel_tolerance
            / torch.clamp(speed_mps, min=1.0)
        )
        return torch.minimum(
            torch.full_like(reference_speed, self.temporal_direction_tolerance),
            dynamic_tolerance,
        )

    def _cosine_direction_cost(self, anchor_delta, reference_delta):
        eps = 1e-6
        anchor_speed = torch.linalg.norm(anchor_delta, dim=-1)
        reference_speed = torch.linalg.norm(reference_delta, dim=-1)
        cosine = (anchor_delta * reference_delta).sum(dim=-1) / (anchor_speed * reference_speed + eps)
        direction_cost = 0.5 * (1.0 - cosine.clamp(-1.0, 1.0))
        return torch.where(
            (anchor_speed > eps) & (reference_speed > eps),
            direction_cost,
            torch.zeros_like(direction_cost),
        )

    def _direction_violation_cost(self, anchor_delta, reference_delta):
        eps = 1e-6
        anchor_speed = torch.linalg.norm(anchor_delta, dim=-1)
        reference_speed = torch.linalg.norm(reference_delta, dim=-1)
        anchor_heading = torch.atan2(anchor_delta[..., 1], anchor_delta[..., 0])
        reference_heading = torch.atan2(reference_delta[..., 1], reference_delta[..., 0])
        angle_error = self._angle_error_rad(anchor_heading, reference_heading)
        tolerance = self._dynamic_direction_tolerance(reference_speed)
        direction_violation = self._positive_violation(angle_error, tolerance)
        valid = (
            (anchor_speed > self.temporal_low_speed_delta)
            & (reference_speed > self.temporal_low_speed_delta)
        )
        return torch.where(valid, direction_violation, torch.zeros_like(direction_violation))

    def _turn_values(self, deltas):
        heading = torch.atan2(deltas[..., 1], deltas[..., 0])
        return torch.atan2(
            torch.sin(heading[..., 1:] - heading[..., :-1]),
            torch.cos(heading[..., 1:] - heading[..., :-1]),
        )

    def _turn_cost(self, anchor_delta, reference_delta):
        eps = 1e-6
        anchor_speed = torch.linalg.norm(anchor_delta, dim=-1)
        reference_speed = torch.linalg.norm(reference_delta, dim=-1)
        if anchor_delta.shape[-2] < 2:
            return torch.zeros_like(anchor_speed[..., 0])

        anchor_heading = torch.atan2(anchor_delta[..., 1], anchor_delta[..., 0])
        reference_heading = torch.atan2(reference_delta[..., 1], reference_delta[..., 0])
        anchor_turn = torch.atan2(
            torch.sin(anchor_heading[..., 1:] - anchor_heading[..., :-1]),
            torch.cos(anchor_heading[..., 1:] - anchor_heading[..., :-1]),
        )
        reference_turn = torch.atan2(
            torch.sin(reference_heading[..., 1:] - reference_heading[..., :-1]),
            torch.cos(reference_heading[..., 1:] - reference_heading[..., :-1]),
        )
        turn_error = self._angle_error(anchor_turn, reference_turn)
        turn_valid = (
            (anchor_speed[..., 1:] > eps)
            & (anchor_speed[..., :-1] > eps)
            & (reference_speed[..., 1:] > eps)
            & (reference_speed[..., :-1] > eps)
        )
        turn_error = torch.where(turn_valid, turn_error, torch.zeros_like(turn_error))
        return turn_error.mean(dim=-1)

    def _turn_violation_cost(self, anchor_delta, reference_delta):
        anchor_speed = torch.linalg.norm(anchor_delta, dim=-1)
        reference_speed = torch.linalg.norm(reference_delta, dim=-1)
        if anchor_delta.shape[-2] < 2:
            return torch.zeros_like(anchor_speed[..., 0])

        anchor_turn = self._turn_values(anchor_delta)
        reference_turn = self._turn_values(reference_delta)
        turn_error = self._angle_error_rad(anchor_turn, reference_turn)
        turn_violation = self._positive_violation(turn_error, self.temporal_turn_change_tolerance)
        turn_valid = (
            (anchor_speed[..., 1:] > self.temporal_low_speed_delta)
            & (anchor_speed[..., :-1] > self.temporal_low_speed_delta)
            & (reference_speed[..., 1:] > self.temporal_low_speed_delta)
            & (reference_speed[..., :-1] > self.temporal_low_speed_delta)
        )
        turn_violation = torch.where(turn_valid, turn_violation, torch.zeros_like(turn_violation))
        return turn_violation.mean(dim=-1)

    def _connection_turn_violation(self, executed_delta, anchor_delta):
        anchor_speed = torch.linalg.norm(anchor_delta, dim=-1)
        executed_speed = torch.linalg.norm(executed_delta, dim=-1)
        if anchor_delta.shape[-2] < 2:
            return torch.zeros_like(anchor_speed[..., 0])

        executed_expanded = executed_delta.unsqueeze(1).expand(*anchor_delta.shape[:-2], 1, 2)
        connection_deltas = torch.cat([executed_expanded, anchor_delta[..., :2, :]], dim=-2)
        connection_turn = self._turn_values(connection_deltas)
        turn_change = self._angle_error_rad(connection_turn[..., 1:], connection_turn[..., :-1])
        turn_violation = self._positive_violation(turn_change, self.temporal_turn_change_tolerance).mean(dim=-1)
        valid = (
            (executed_speed.unsqueeze(1).expand_as(anchor_speed[..., :1]) > self.temporal_low_speed_delta)
            & (anchor_speed[..., :1] > self.temporal_low_speed_delta)
            & (anchor_speed[..., 1:2] > self.temporal_low_speed_delta)
        ).squeeze(-1)
        return torch.where(valid, turn_violation, torch.zeros_like(turn_violation))

    def _reversal_cost(self, anchor_speed):
        if anchor_speed.shape[-1] < 3:
            return torch.zeros_like(anchor_speed[..., 0])

        accel = anchor_speed[..., 1:] - anchor_speed[..., :-1]
        accel_then_brake = (accel[..., :-1] > self.temporal_accel_delta_tolerance) & (
            accel[..., 1:] < -self.temporal_brake_delta_tolerance
        )
        brake_then_accel = (accel[..., :-1] < -self.temporal_brake_delta_tolerance) & (
            accel[..., 1:] > self.temporal_accel_delta_tolerance
        )
        reversal = accel_then_brake.to(anchor_speed.dtype) + 0.5 * brake_then_accel.to(anchor_speed.dtype)
        return reversal.mean(dim=-1)

    def _temporal_compatibility_components(self, plan_anchor, previous_trajectory, previous_ego_delta=None):
        start_cost = None
        path_cost = None
        velocity_cost = None
        previous_delta = None
        anchor_delta = None
        accel_cost_terms = []
        path_cost_terms = []

        if plan_anchor.shape[-2] >= 3:
            anchor_xy = plan_anchor[..., :3, :2]
            anchor_delta = self._point_deltas(anchor_xy)
            anchor_speed = torch.linalg.norm(anchor_delta, dim=-1)
            if anchor_speed.shape[-1] >= 2:
                start_accel = anchor_speed[..., 1] - anchor_speed[..., 0]
                accel_cost_terms.append(self._signed_delta_violation(start_accel))
            reversal_cost = self._reversal_cost(anchor_speed)
        else:
            reversal_cost = None

        if previous_trajectory is not None:
            previous_trajectory = previous_trajectory.to(device=plan_anchor.device, dtype=plan_anchor.dtype)
            if torch.isfinite(previous_trajectory).all() and previous_trajectory.shape[-2] >= 3 and plan_anchor.shape[-2] >= 3:
                previous_xy = previous_trajectory[..., :3, :2]
                previous_delta = (previous_xy[..., 1:, :] - previous_xy[..., :-1, :]).unsqueeze(1)
                anchor_ref_delta = anchor_delta[..., 1:3, :]

                anchor_ref_speed = torch.linalg.norm(anchor_ref_delta, dim=-1)
                previous_speed = torch.linalg.norm(previous_delta, dim=-1)

                ref_turn_cost = self._turn_violation_cost(anchor_ref_delta, previous_delta)
                path_cost_terms.append(("ref", ref_turn_cost))

                if previous_delta.shape[-2] > 1:
                    anchor_speed_delta = anchor_ref_speed[..., 1:] - anchor_ref_speed[..., :-1]
                    previous_speed_delta = previous_speed[..., 1:] - previous_speed[..., :-1]
                    accel_delta_error = anchor_speed_delta - previous_speed_delta
                    accel_cost_terms.append(self._signed_delta_violation(accel_delta_error).mean(dim=-1))

        if previous_ego_delta is not None:
            previous_ego_delta = previous_ego_delta.to(device=plan_anchor.device, dtype=plan_anchor.dtype)
            if previous_ego_delta.dim() == 1:
                previous_ego_delta = previous_ego_delta.unsqueeze(0)

            if torch.isfinite(previous_ego_delta).all() and previous_ego_delta.shape[-1] >= 2:
                executed_delta = previous_ego_delta[..., :2].unsqueeze(1)
                anchor_start_delta = plan_anchor[..., 0, :2]
                anchor_start_speed = torch.linalg.norm(anchor_start_delta, dim=-1)
                executed_speed = torch.linalg.norm(executed_delta, dim=-1)
                start_direction_cost = self._direction_violation_cost(anchor_start_delta, executed_delta)
                start_speed_cost = self._signed_delta_violation(anchor_start_speed - executed_speed)
                start_cost = 0.6 * start_direction_cost + 0.4 * start_speed_cost

                if anchor_delta is not None:
                    connection_turn_cost = self._connection_turn_violation(executed_delta, anchor_delta)
                    path_cost_terms.append(("connection", connection_turn_cost))

        connection_path_cost = None
        ref_path_cost = None
        for name, value in path_cost_terms:
            if name == "connection":
                connection_path_cost = value if connection_path_cost is None else 0.5 * (connection_path_cost + value)
            elif name == "ref":
                ref_path_cost = value if ref_path_cost is None else 0.5 * (ref_path_cost + value)

        if connection_path_cost is not None and ref_path_cost is not None:
            path_cost = 0.6 * connection_path_cost + 0.4 * ref_path_cost
        elif connection_path_cost is not None:
            path_cost = connection_path_cost
        elif ref_path_cost is not None:
            path_cost = ref_path_cost

        if accel_cost_terms:
            accel_cost = torch.stack(accel_cost_terms, dim=0).mean(dim=0)
            if reversal_cost is None:
                velocity_cost = accel_cost
            else:
                velocity_cost = 0.7 * accel_cost + 0.3 * reversal_cost
        elif reversal_cost is not None:
            velocity_cost = reversal_cost

        if start_cost is None and path_cost is not None:
            start_cost = torch.zeros_like(path_cost)

        if start_cost is None or path_cost is None or velocity_cost is None:
            return None, None, None

        return start_cost, path_cost, velocity_cost

    def _temporal_compatibility_cost(self, plan_anchor, previous_trajectory, previous_ego_delta=None):
        start_cost, path_cost, velocity_cost = self._temporal_compatibility_components(
            plan_anchor,
            previous_trajectory,
            previous_ego_delta,
        )
        if start_cost is None:
            return None

        return (
            self.temporal_start_weight * start_cost.clamp(max=2.0)
            + self.temporal_path_weight * path_cost.clamp(max=2.0)
            + self.temporal_velocity_weight * velocity_cost.clamp(max=2.0)
        )

    def _trajectory_comfort_cost(self, candidates):
        xy = candidates[..., :2]
        deltas = self._point_deltas(xy)
        speed = torch.linalg.norm(deltas, dim=-1)
        if speed.shape[-1] >= 4:
            accel = speed[..., 1:] - speed[..., :-1]
            jerk = accel[..., 1:] - accel[..., :-1]
            jerk_cost = self._positive_violation(jerk.abs(), self.temporal_jerk_delta_tolerance).mean(dim=-1)
        else:
            jerk_cost = torch.zeros_like(speed[..., 0])

        if deltas.shape[-2] >= 4:
            eps = 1e-6
            heading = torch.atan2(deltas[..., 1], deltas[..., 0])
            turn = torch.atan2(
                torch.sin(heading[..., 1:] - heading[..., :-1]),
                torch.cos(heading[..., 1:] - heading[..., :-1]),
            )
            turn_delta = torch.atan2(
                torch.sin(turn[..., 1:] - turn[..., :-1]),
                torch.cos(turn[..., 1:] - turn[..., :-1]),
            ).abs()
            valid = (speed[..., 2:] > eps) & (speed[..., 1:-1] > eps) & (speed[..., :-2] > eps)
            turn_delta = self._positive_violation(turn_delta, self.temporal_turn_change_tolerance)
            turn_delta = torch.where(valid, turn_delta, torch.zeros_like(turn_delta))
            turn_smooth_cost = turn_delta.mean(dim=-1)
        else:
            turn_smooth_cost = torch.zeros_like(speed[..., 0])

        return 0.5 * jerk_cost.clamp(max=2.0) + 0.5 * turn_smooth_cost.clamp(max=2.0)

    def _energy_loss_context(
        self,
        candidates,
        previous_trajectory,
        previous_ego_delta=None,
        training_epoch=None,
        energy_ramp_override=None,
    ):
        start_cost, path_cost, velocity_cost = self._temporal_compatibility_components(
            candidates,
            previous_trajectory,
            previous_ego_delta,
        )
        if start_cost is None:
            return None

        temporal_cost = (
            self.temporal_start_weight * start_cost.clamp(max=2.0)
            + self.temporal_path_weight * path_cost.clamp(max=2.0)
            + self.temporal_velocity_weight * velocity_cost.clamp(max=2.0)
        )
        return {
            "energy_temporal_cost": temporal_cost,
            "energy_start_cost": start_cost,
            "energy_path_cost": path_cost,
            "energy_velocity_cost": velocity_cost,
            "energy_comfort_cost": self._trajectory_comfort_cost(candidates),
            "energy_training_epoch": training_epoch,
            "energy_ramp_override": energy_ramp_override,
            "energy_topk": self.energy_topk,
            "energy_temperature": self.energy_temperature,
            "energy_start_epoch": self.energy_start_epoch,
            "energy_full_epoch": self.energy_full_epoch,
            "energy_gt_weight": self.energy_gt_weight,
            "energy_temporal_weight": self.energy_temporal_weight,
            "energy_comfort_weight": self.energy_comfort_weight,
            "temporal_rank_weight_max": self.temporal_rank_weight_max,
            "temporal_rank_margin": self.temporal_rank_margin,
            "temporal_rank_energy_gap": self.temporal_rank_energy_gap,
            "temporal_rank_min_epoch": self.temporal_rank_min_epoch,
            "temporal_rank_ramp_epochs": self.temporal_rank_ramp_epochs,
            "temporal_rank_use_comfort": self.temporal_rank_use_comfort,
            "temporal_aux_weight_max": self.temporal_aux_weight_max,
        }

    def _temporal_noise_scale(self, plan_anchor, previous_trajectory, previous_ego_delta=None):
        temporal_cost = self._temporal_compatibility_cost(plan_anchor, previous_trajectory, previous_ego_delta)
        if temporal_cost is None:
            return None

        centered_cost = temporal_cost - temporal_cost.mean(dim=1, keepdim=True)
        noise_scale = 1.0 + self.temporal_noise_strength * torch.tanh(centered_cost)
        noise_scale = noise_scale.clamp(self.temporal_noise_min_scale, self.temporal_noise_max_scale)
        return noise_scale[:, :, None, None]

    def _select_mode_with_temporal_rescore(self, poses_reg, poses_cls, previous_trajectory=None, previous_ego_delta=None):
        base_mode_idx = poses_cls.argmax(dim=-1)
        diagnostics = {
            "temporal_rescore_active": torch.zeros((), device=poses_cls.device, dtype=poses_cls.dtype),
            "temporal_rescore_changed": torch.zeros((), device=poses_cls.device, dtype=poses_cls.dtype),
        }

        temporal_cost = self._temporal_compatibility_cost(
            poses_reg[..., :2],
            previous_trajectory,
            previous_ego_delta,
        )
        if temporal_cost is None:
            return base_mode_idx, diagnostics

        if temporal_cost.shape != poses_cls.shape or not torch.isfinite(temporal_cost).all():
            return base_mode_idx, diagnostics

        topk = max(1, min(int(self.temporal_rescore_topk), poses_cls.shape[1]))
        topk_cls, topk_idx = torch.topk(poses_cls, k=topk, dim=-1, largest=True)
        topk_cost = torch.gather(temporal_cost, 1, topk_idx).clamp(max=self.temporal_rescore_cost_clamp)
        cost_mean = topk_cost.mean(dim=1, keepdim=True)
        cost_std = topk_cost.std(dim=1, unbiased=False, keepdim=True)
        normalized_cost = ((topk_cost - cost_mean) / (cost_std + 1e-6)).clamp(-2.0, 2.0)
        rescored_logits = topk_cls - float(self.temporal_rescore_alpha) * normalized_cost
        selected_topk_pos = rescored_logits.argmax(dim=-1, keepdim=True)
        mode_idx = torch.gather(topk_idx, 1, selected_topk_pos).squeeze(1)

        if poses_cls.shape[1] >= 2:
            top2_cls = torch.topk(poses_cls, k=2, dim=-1, largest=True).values
            cls_margin = top2_cls[:, 0] - top2_cls[:, 1]
        else:
            cls_margin = torch.zeros_like(base_mode_idx, dtype=poses_cls.dtype)

        selected_cost = torch.gather(temporal_cost, 1, mode_idx.unsqueeze(1)).squeeze(1)
        base_cost = torch.gather(temporal_cost, 1, base_mode_idx.unsqueeze(1)).squeeze(1)
        diagnostics = {
            "temporal_rescore_active": torch.ones((), device=poses_cls.device, dtype=poses_cls.dtype),
            "temporal_rescore_changed": (mode_idx != base_mode_idx).to(poses_cls.dtype).mean().detach(),
            "temporal_rescore_selected_cost": selected_cost.mean().detach(),
            "temporal_rescore_base_cost": base_cost.mean().detach(),
            "temporal_rescore_topk_min_cost": topk_cost.min(dim=1).values.mean().detach(),
            "temporal_rescore_cls_margin": cls_margin.mean().detach(),
            "temporal_rescore_selected_mode": mode_idx.to(poses_cls.dtype).mean().detach(),
            "temporal_rescore_base_mode": base_mode_idx.to(poses_cls.dtype).mean().detach(),
        }
        return mode_idx, diagnostics

    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,previous_trajectory=None,previous_ego_delta=None,training_epoch=None,energy_ramp_override=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,previous_trajectory,previous_ego_delta,training_epoch,energy_ramp_override)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img,previous_trajectory,previous_ego_delta)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,previous_trajectory=None,previous_ego_delta=None,training_epoch=None,energy_ramp_override=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
        temporal_metrics = {}
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        traj_feature, history_metrics = self.history_planning_adapter(
            traj_feature,
            noisy_traj_points,
            previous_trajectory,
        )
        temporal_metrics.update(history_metrics)
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)


        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
        temporal_context = self._energy_loss_context(
            poses_reg_list[-1][..., :2],
            previous_trajectory,
            previous_ego_delta,
            training_epoch,
            energy_ramp_override,
        )

        trajectory_loss_dict = {}
        ret_traj_loss = 0
        ret_original_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss_output = self.loss_computer(
                poses_reg,
                poses_cls,
                targets,
                plan_anchor,
                temporal_context=temporal_context,
            )
            if isinstance(trajectory_loss_output, tuple):
                trajectory_loss, energy_loss_dict = trajectory_loss_output
                original_trajectory_loss = energy_loss_dict.get(
                    "trajectory_original_loss",
                    trajectory_loss.detach(),
                )
                for key, value in energy_loss_dict.items():
                    if key != "trajectory_scaled_energy_aux_loss":
                        trajectory_loss_dict[f"{key}_{idx}"] = value
            else:
                trajectory_loss = trajectory_loss_output
                original_trajectory_loss = trajectory_loss.detach()
            trajectory_loss_dict[f"trajectory_original_loss_{idx}"] = original_trajectory_loss.detach()
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss
            ret_original_traj_loss += original_trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        trajectory_loss_dict["trajectory_original_loss"] = ret_original_traj_loss.detach()
        output = {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}
        output.update(temporal_metrics)
        return output

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img,previous_trajectory=None,previous_ego_delta=None) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        history_metrics = {}
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
            traj_feature, history_metrics = self.history_planning_adapter(
                traj_feature,
                noisy_traj_points,
                previous_trajectory,
            )

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        mode_idx, temporal_diagnostics = self._select_mode_with_temporal_rescore(
            poses_reg,
            poses_cls,
            previous_trajectory,
            previous_ego_delta,
        )
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        output = {"trajectory": best_reg}
        output.update(temporal_diagnostics)
        output.update(history_metrics)
        return output
