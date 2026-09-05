# SPDX-License-Identifier: Apache-2.0
# Adapted from wjl2244/MeanFuser, revision 8de8ba6244834645192e318dcc437d124cfd6872.
# See third_party/meanfuser/NOTICE.md and LICENSE for attribution and changes.
# Modified: class/interface, dynamic K, ego memory, detached proposals and FP32 transforms.
import torch
import torch.nn as nn
from navsim.agents.diffusiondrive.modules.proposal_trajectory_utils import (
    cumsum_traj, diff_traj, HORIZON, ACTION_DIM_DELTA,
)


def validate_spr_checkpoint(head, state_dict, require_spr=False):
    """Allow baseline warm starts, but reject incomplete SPR or changed bank order."""
    if head.spr_head is None:
        return
    prefix = "_transfuser_model._trajectory_head.spr_head."
    expected = {prefix + key for key in head.spr_head.state_dict()}
    present = {key for key in state_dict if key.startswith(prefix)}
    if not present and not require_spr:
        return
    missing = expected - present
    if missing:
        raise ValueError(
            "SPR checkpoint is incomplete. Train 3.1.05 first, or use "
            "diffusiondrive_agent for a baseline checkpoint. Missing: " + ", ".join(sorted(missing))
        )
    anchor = state_dict.get("_transfuser_model._trajectory_head.plan_anchor")
    current = head.plan_anchor.detach().cpu()
    if anchor is None or not torch.equal(anchor.detach().cpu().to(current.dtype), current):
        raise ValueError("SPR checkpoint requires the same anchor bank values and ordering as training")


class SceneProposalReconstruction(nn.Module):
    """MeanFuser reconstruction adapted to DiffusionDrive's ordered proposal bank."""

    def __init__(self, config, num_proposals, num_poses=HORIZON):
        super().__init__()
        self._config = config
        self.num_proposals = num_proposals
        self.num_poses = num_poses
        if num_proposals < 1 or num_poses != HORIZON:
            raise ValueError("SPR requires a nonempty proposal bank and the NAVSIM 8-pose horizon")

        self.traj_encoder = nn.Sequential(
            nn.Linear(HORIZON * ACTION_DIM_DELTA, config.tf_d_model*2, bias=False),
            nn.ReLU(),
            nn.Linear(config.tf_d_model*2, config.tf_d_model, bias=False),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.bev_cross_attn = nn.TransformerDecoder(
            decoder_layer, 3)

        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.trajectory_recon = nn.Sequential(
            nn.Linear(num_proposals * config.tf_d_model, config.tf_d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.tf_d_model, config.tf_d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.tf_d_model, HORIZON * ACTION_DIM_DELTA, bias=False),
        )

        self.loss_fn = nn.L1Loss(reduction='mean')

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ModuleList):
            for submodule in module:
                self._init_weights(submodule)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_reconstruction_loss(self, predictions, targets):
        pred_delta_traj = predictions["diff_trajectory"]

        pred_delta_traj = pred_delta_traj.float()
        gt_delta_trajectory = diff_traj(targets["trajectory"].float())

        reconstruction_loss = self.loss_fn(pred_delta_traj, gt_delta_trajectory)
        return reconstruction_loss

    def forward(self, proposals, ego_query, agents_query):
        bs, num_proposals, num_poses, dim = proposals.shape
        if (num_proposals, num_poses, dim) != (self.num_proposals, self.num_poses, 3):
            raise ValueError("SPR proposals must match the configured [B, K, 8, 3] bank")
        # Upstream reconstruction receives no-grad samples; retain that boundary.
        if getattr(self._config, "spr_detach_proposals", True):
            proposals = proposals.detach()
        # Keep the upstream coordinate transform in FP32 under mixed precision.
        trajectorys = diff_traj(proposals.float().reshape(-1, num_poses, dim))
        trajectorys = trajectorys.reshape(bs, num_proposals, num_poses, ACTION_DIM_DELTA)
        context_query = torch.cat([ego_query, agents_query], dim=1)

        embedded_vocab = self.traj_encoder(trajectorys.reshape(bs, num_proposals, -1))
        cross_attn_output = self.bev_cross_attn(embedded_vocab, context_query).contiguous()

        cross_attn_output = self.norm1(cross_attn_output)
        embedded_vocab = embedded_vocab + cross_attn_output
        waypoints = self.trajectory_recon(embedded_vocab.reshape(bs, -1)).reshape(bs, num_poses, -1)
        # Avoid FP16 cumulative-position error; loss supervises delta/sin/cos,
        # so it does not backpropagate through atan2 near the zero vector.
        trajectory = cumsum_traj(waypoints.float())
        return {"trajectory": trajectory, "diff_trajectory": waypoints}
