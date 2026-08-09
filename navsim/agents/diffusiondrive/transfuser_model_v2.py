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
from navsim.agents.diffusiondrive.modules.risk_attention import (
    HistoricalRiskTemporalSelfAttention,
    TemporalRiskCrossAttention,
)
from navsim.agents.diffusiondrive.modules.risk_gate import select_risk_gated_mode
from navsim.agents.diffusiondrive.modules.risk_shadow import evaluate_risk_shadow
from navsim.agents.diffusiondrive.modules.risk_mode_ranking import (
    RiskModeRankingHead,
    compute_risk_mode_ranking_loss,
    select_risk_ranked_mode,
)
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


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        history_risk_tokens: Optional[torch.Tensor] = features.get("history_risk_tokens")

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

        agents = self._agent_head(agents_query)

        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            targets=targets,
            global_img=None,
            history_risk_tokens=history_risk_tokens,
            agent_states=agents["agent_states"],
            agent_labels=agents["agent_labels"],
            bev_semantic_map=bev_semantic_map,
        )
        output.update(trajectory)

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

        invalid_batch = torch.nonzero(
            ~torch.isfinite(agent_queries).flatten(1).all(dim=1),
            as_tuple=False,
        ).flatten()
        if invalid_batch.numel() > 0:
            print(
                "[DiffusionDrive][agent_head] Sanitizing non-finite agent queries "
                f"for batch indices={invalid_batch.detach().cpu().tolist()}",
                flush=True,
            )
        agent_queries = torch.nan_to_num(agent_queries, nan=0.0, posinf=0.0, neginf=0.0)
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)
        agent_states = torch.nan_to_num(agent_states, nan=0.0, posinf=0.0, neginf=0.0)
        agent_labels = torch.nan_to_num(agent_labels, nan=0.0, posinf=0.0, neginf=0.0)

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
        self.use_temporal_risk_cross_attention = config.use_temporal_risk_cross_attention
        if self.use_temporal_risk_cross_attention:
            self.temporal_risk_attention = TemporalRiskCrossAttention(
                d_model=config.tf_d_model,
                num_heads=config.tf_num_head,
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
                history_risk_memory=None,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)

        if self.use_temporal_risk_cross_attention and history_risk_memory is not None:
            traj_feature = self.temporal_risk_attention(traj_feature, noisy_traj_points, history_risk_memory)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls, traj_feature
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
                history_risk_memory=None,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        final_traj_feature = None
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls, final_traj_feature = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                history_risk_memory,
                global_img,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list, final_traj_feature

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
        self._config = config
        self.use_historical_risk_attention = config.use_historical_risk_attention
        if self.use_historical_risk_attention:
            self.history_risk_encoder = HistoricalRiskTemporalSelfAttention(
                token_dim=config.risk_token_dim,
                d_model=d_model,
                num_heads=config.tf_num_head,
                num_layers=config.risk_attention_layers,
                history_frames=config.risk_history_num_frames,
            )
        self.use_risk_aware_cls = config.use_risk_aware_cls
        if self.use_risk_aware_cls:
            if not self.use_historical_risk_attention:
                raise ValueError("use_risk_aware_cls requires use_historical_risk_attention")
            self.risk_mode_ranking_head = RiskModeRankingHead(
                d_model, config.trajectory_sampling.num_poses
            )
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
    def _encode_history_risk(self, history_risk_tokens, targets=None):
        if not self.use_historical_risk_attention or history_risk_tokens is None:
            return None, None
        history_risk_memory, risk_aux_logits = self.history_risk_encoder(history_risk_tokens)
        risk_aux_output = None
        if self._config.use_memory_aux_loss and targets is not None:
            risk_aux_output = self.history_risk_encoder.compute_aux_outputs(
                risk_aux_logits,
                targets,
                self._config.memory_aux_loss_weight,
            )
        return history_risk_memory, risk_aux_output

    def forward(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        history_risk_tokens=None,
        agent_states=None,
        agent_labels=None,
        bev_semantic_map=None,
    ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                targets,
                global_img,
                history_risk_tokens,
            )
        else:
            return self.forward_test(
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
                history_risk_tokens,
                agent_states,
                agent_labels,
                bev_semantic_map,
                targets,
            )


    def forward_train(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        history_risk_tokens=None,
    ) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        history_risk_memory, risk_aux_output = self._encode_history_risk(history_risk_tokens, targets)
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
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
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)


        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list, final_traj_feature = self.diff_decoder(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_encoding,
            history_risk_memory,
            global_img,
        )

        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        ranking_output: Dict[str, torch.Tensor] = {}
        selection_logits = poses_cls_list[-1]
        if self.use_risk_aware_cls:
            history_valid = history_risk_tokens[..., -1] if history_risk_tokens is not None else None
            risk_rank_logits = self.risk_mode_ranking_head(
                final_traj_feature,
                history_risk_memory,
                history_valid,
                poses=poses_reg_list[-1],
                dt=self._config.risk_history_dt,
            )
            unsafe_logits = risk_rank_logits["unsafe_logits"]
            timing_logits = risk_rank_logits["timing_logits"]
            ranking_output = compute_risk_mode_ranking_loss(
                poses_reg_list[-1],
                poses_cls_list[-1],
                unsafe_logits,
                timing_logits,
                targets,
                self._config,
            )
            ranking_selection = select_risk_ranked_mode(
                poses_cls_list[-1],
                unsafe_logits,
                self._config.risk_rank_inference_topk,
                timing_logits=timing_logits,
                poses=poses_reg_list[-1],
                config=self._config,
            )
            mode_idx = ranking_selection["selected_mode"]
        else:
            mode_idx = selection_logits.argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        output = {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}
        if risk_aux_output is not None:
            output.update(risk_aux_output)
            output["trajectory_loss_dict"]["memory_aux_loss"] = risk_aux_output["memory_aux_loss"]
        output.update(ranking_output)
        return output

    def forward_test(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        history_risk_tokens=None,
        agent_states=None,
        agent_labels=None,
        bev_semantic_map=None,
        targets=None,
    ) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        history_risk_memory, risk_aux_output = self._encode_history_risk(
            history_risk_tokens, targets
        )
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
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

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
            poses_reg_list, poses_cls_list, final_traj_feature = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                history_risk_memory,
                global_img,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        risk_diagnostics: Dict[str, torch.Tensor] = {}
        ranking_output: Dict[str, torch.Tensor] = {}
        selection_logits = poses_cls
        base_mode = poses_cls.argmax(dim=-1)
        if self.use_risk_aware_cls:
            history_valid = history_risk_tokens[..., -1] if history_risk_tokens is not None else None
            risk_rank_logits = self.risk_mode_ranking_head(
                final_traj_feature,
                history_risk_memory,
                history_valid,
                poses=poses_reg,
                dt=self._config.risk_history_dt,
            )
            unsafe_logits = risk_rank_logits["unsafe_logits"]
            timing_logits = risk_rank_logits["timing_logits"]
            ranking_selection = select_risk_ranked_mode(
                poses_cls,
                unsafe_logits,
                self._config.risk_rank_inference_topk,
                timing_logits=timing_logits,
                poses=poses_reg,
                config=self._config,
            )
            selection_logits = ranking_selection["adjusted_logits"]
            raw_mode = ranking_selection["selected_mode"]
            risk_diagnostics.update(
                {
                    "risk_rank_base_mode": base_mode.float(),
                    "risk_rank_unconstrained_mode": ranking_selection[
                        "unconstrained_mode"
                    ].float(),
                    "risk_rank_unconstrained_topk_mode": ranking_selection[
                        "unconstrained_topk_mode"
                    ].float(),
                    "risk_rank_selected_mode": raw_mode.float(),
                    "risk_rank_selected_base_rank": ranking_selection[
                        "selected_base_rank"
                    ].float(),
                    "risk_rank_guard_blocked": ranking_selection[
                        "guard_blocked"
                    ].float(),
                    "risk_rank_path_guard_blocked": ranking_selection[
                        "path_guard_blocked"
                    ].float(),
                    "risk_rank_eligible_candidate_count": ranking_selection[
                        "eligible_candidate_count"
                    ].float(),
                    "risk_rank_predicted_safe_candidate_count": ranking_selection[
                        "predicted_safe_candidate_count"
                    ].float(),
                    "risk_rank_qualified_candidate_count": ranking_selection[
                        "qualified_candidate_count"
                    ].float(),
                    "risk_rank_safe_fallback_used": ranking_selection[
                        "safe_fallback_used"
                    ].float(),
                    "risk_rank_base_unsafe_probability": ranking_selection[
                        "base_unsafe_probability"
                    ],
                    "risk_rank_selected_unsafe_probability": ranking_selection[
                        "selected_unsafe_probability"
                    ],
                    "risk_rank_selected_unsafe_probability_gain": ranking_selection[
                        "selected_unsafe_probability_gain"
                    ],
                    "risk_rank_selected_timing_logit": ranking_selection[
                        "selected_timing_logit"
                    ],
                    "risk_rank_proposed_lateral_distance": ranking_selection[
                        "proposed_lateral_distance"
                    ],
                    "risk_rank_proposed_heading_distance": ranking_selection[
                        "proposed_heading_distance"
                    ],
                    "risk_rank_proposed_progress_delta": ranking_selection[
                        "proposed_progress_delta"
                    ],
                    "risk_rank_selected_lateral_distance": ranking_selection[
                        "selected_lateral_distance"
                    ],
                    "risk_rank_selected_heading_distance": ranking_selection[
                        "selected_heading_distance"
                    ],
                    "risk_rank_selected_progress_delta": ranking_selection[
                        "selected_progress_delta"
                    ],
                    "risk_rank_unconstrained_selection_changed": (
                        ranking_selection["unconstrained_mode"] != base_mode
                    ).float(),
                    "risk_rank_selection_changed": (raw_mode != base_mode).float(),
                    "risk_rank_unsafe_probability_mean": unsafe_logits.sigmoid().mean(dim=-1),
                    "risk_rank_unsafe_probability_max": unsafe_logits.sigmoid().max(dim=-1).values,
                    "risk_rank_unsafe_probability_span": (
                        unsafe_logits.sigmoid().max(dim=-1).values
                        - unsafe_logits.sigmoid().min(dim=-1).values
                    ),
                }
            )
            if targets is not None:
                ranking_output = compute_risk_mode_ranking_loss(
                    poses_reg,
                    poses_cls,
                    unsafe_logits,
                    timing_logits,
                    targets,
                    self._config,
                )
        else:
            raw_mode = base_mode
        shadow_enabled = self._config.use_risk_shadow_evaluator or self._config.use_soft_risk_rescore
        if shadow_enabled and history_risk_tokens is not None:
            proposed_mode, shadow_diagnostics = evaluate_risk_shadow(
                poses_reg,
                selection_logits,
                history_risk_tokens,
                agent_states,
                agent_labels,
                bev_semantic_map,
                self._config,
            )
            risk_diagnostics.update(shadow_diagnostics)
            # Shadow mode is deliberately output-neutral. Soft re-ranking is a separate opt-in.
            mode_idx = proposed_mode if self._config.use_soft_risk_rescore else raw_mode
        elif self._config.use_risk_gate and history_risk_tokens is not None:
            mode_idx = select_risk_gated_mode(poses_reg, selection_logits, history_risk_tokens, self._config)
        else:
            mode_idx = raw_mode
        gather_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, gather_idx).squeeze(1)
        output = {"trajectory": best_reg}
        if risk_aux_output is not None:
            output.update(risk_aux_output)
        output.update(ranking_output)
        if risk_diagnostics:
            risk_diagnostics["risk_selected_mode"] = mode_idx.float()
            risk_diagnostics["risk_selection_changed"] = (mode_idx != raw_mode).float()
            rank_changed = raw_mode != base_mode
            if bool(rank_changed.any().item()):
                # Positive counterfactual_score_delta means risk ranking hurt PDM.
                risk_diagnostics["risk_counterfactual_active"] = rank_changed.float()
                risk_diagnostics["risk_counterfactual_mode"] = base_mode.float()
                risk_diagnostics["risk_counterfactual_source"] = torch.ones_like(base_mode).float()
            if "risk_counterfactual_mode" in risk_diagnostics:
                counterfactual_mode = risk_diagnostics["risk_counterfactual_mode"].long()
                counterfactual_idx = counterfactual_mode[..., None, None, None].repeat(
                    1, 1, self._num_poses, 3
                )
                output["risk_counterfactual_trajectory"] = torch.gather(
                    poses_reg, 1, counterfactual_idx
                ).squeeze(1)
            output.update(risk_diagnostics)
        return output
