from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoricalRiskTemporalSelfAttention(nn.Module):
    """Encode explicit history risk tokens into temporal risk memory."""

    def __init__(self, token_dim: int, d_model: int, num_heads: int, num_layers: int, history_frames: int):
        super().__init__()
        self.token_dim = token_dim
        self.history_frames = history_frames
        self.token_embedding = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, history_frames, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.risk_trend_head = nn.Linear(d_model, 3)
        self.urgency_head = nn.Linear(d_model, 4)
        self.brake_need_head = nn.Linear(d_model, 5)

    def forward(self, history_risk_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        history_risk_tokens = torch.nan_to_num(history_risk_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = history_risk_tokens[..., -1:].clamp(0.0, 1.0)
        x = (self.token_embedding(history_risk_tokens) + self.time_embedding[:, : history_risk_tokens.shape[1]]) * valid
        memory = torch.nan_to_num(self.encoder(x) * valid, nan=0.0, posinf=0.0, neginf=0.0)

        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = memory.sum(dim=1) / denom
        logits = {
            "risk_trend": self.risk_trend_head(pooled),
            "urgency": self.urgency_head(pooled),
            "brake_need": self.brake_need_head(pooled),
        }
        return memory, logits

    @staticmethod
    def compute_aux_loss(
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weight: float,
    ) -> Optional[torch.Tensor]:
        if "risk_aux_labels" not in targets or "risk_aux_valid" not in targets:
            return None

        labels = targets["risk_aux_labels"].to(next(iter(logits.values())).device)
        valid = targets["risk_aux_valid"].to(labels.device).float()
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if valid.ndim == 0:
            valid = valid.unsqueeze(0)

        valid_sum = valid.sum()
        if valid_sum.item() <= 0:
            return next(iter(logits.values())).sum() * 0.0

        losses = []
        for label_idx, name in enumerate(("risk_trend", "urgency", "brake_need")):
            raw_loss = F.cross_entropy(logits[name], labels[:, label_idx].long(), reduction="none")
            losses.append((raw_loss * valid).sum() / valid_sum.clamp(min=1.0))
        return weight * sum(losses)


class TemporalRiskCrossAttention(nn.Module):
    """Let step-level trajectory queries attend to history risk memory."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.step_embedding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        traj_feature: torch.Tensor,
        noisy_traj_points: torch.Tensor,
        history_risk_memory: torch.Tensor,
    ) -> torch.Tensor:
        traj_feature = torch.nan_to_num(traj_feature, nan=0.0, posinf=0.0, neginf=0.0)
        noisy_traj_points = torch.nan_to_num(noisy_traj_points, nan=0.0, posinf=0.0, neginf=0.0)
        history_risk_memory = torch.nan_to_num(history_risk_memory, nan=0.0, posinf=0.0, neginf=0.0)
        bs, num_mode, num_step, _ = noisy_traj_points.shape
        step_query = traj_feature[:, :, None, :] + self.step_embedding(noisy_traj_points)
        flat_query = step_query.reshape(bs, num_mode * num_step, -1)
        risk_context = self.cross_attention(flat_query, history_risk_memory, history_risk_memory)[0]
        risk_context = torch.nan_to_num(risk_context, nan=0.0, posinf=0.0, neginf=0.0)
        risk_context = risk_context.reshape(bs, num_mode, num_step, -1).mean(dim=2)
        return self.norm(traj_feature + risk_context)
