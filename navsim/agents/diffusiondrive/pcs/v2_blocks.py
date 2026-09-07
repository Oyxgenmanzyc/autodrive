# Adapted from hustvl/DiffusionDriveV2, commit 1cd12a1e155c34dcc471261835444c8d5587580b.
# Original code is MIT licensed; see LICENSE_DiffusionDriveV2.txt in this directory.
# These are the upstream single-stage scoring building blocks; no RL or fine scorer.
import torch
from torch import nn
def gen_sineembed_for_position_1d(theta, hidden_dim):
    dim_t = torch.arange(hidden_dim, device=theta.device).float()
    dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim)
    emb   = theta[..., None] / dim_t
    emb   = torch.stack([emb.sin(), emb.cos()], dim=-1)  # (..., num_feats, 2)
    return emb.flatten(-2)


class GridSampleCrossBEVAttentionScorer(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttentionScorer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, embed_dims, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        bs, num_queries, num_points, _ = traj_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries

class ScorerTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        tf_d_model: int = 512
        tf_d_ffn: int = 2048
        tf_num_layers: int = 6
        tf_num_head: int = 16
        tf_dropout: float = 0.1

        self.cross_bev_attention = GridSampleCrossBEVAttentionScorer(
            tf_d_model,
            tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.agent_input = nn.Linear(256, tf_d_model)
        self.ego_input = nn.Linear(256, tf_d_model)
        self.cross_agent_attention = nn.MultiheadAttention(
            tf_d_model,
            tf_num_head,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            tf_d_model,
            tf_num_head,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            tf_d_model, tf_num_head,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(tf_d_model, tf_d_ffn),
            nn.ReLU(),
            nn.Linear(tf_d_ffn, tf_d_model),
        )
        self.norm1 = nn.LayerNorm(tf_d_model)
        self.norm2 = nn.LayerNorm(tf_d_model)
        self.norm3 = nn.LayerNorm(tf_d_model)
        self.norm4 = nn.LayerNorm(tf_d_model)
        # self.time_modulation = ModulationLayer(config.tf_d_model,256)

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
        agents_query = self.agent_input(agents_query)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        traj_feature = traj_feature + self.dropout1(self.self_attn(traj_feature, traj_feature, traj_feature)[0])
        traj_feature = self.norm2(traj_feature)

        # 4.5 cross attention with  ego query
        ego_query = self.ego_input(ego_query)
        traj_feature = traj_feature + self.dropout2(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm3(traj_feature)

        # 4.6 feedforward network
        traj_feature = self.norm4(self.ffn(traj_feature))

        return traj_feature
