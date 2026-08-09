from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


AUX_LABEL_NAMES = (
    "ttc_worsening",
    "drac_worsening",
    "urgency",
)


class HistoricalRiskTemporalSelfAttention(nn.Module):
    """Encode raw history risk tokens into temporal memory and auxiliary predictions."""

    def __init__(
        self,
        token_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        history_frames: int,
    ):
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
        self.ttc_worsening_head = nn.Linear(d_model, 3)
        self.drac_worsening_head = nn.Linear(d_model, 3)
        self.urgency_head = nn.Linear(d_model, 4)

    def forward(
        self,
        history_risk_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        history_risk_tokens = torch.nan_to_num(
            history_risk_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        valid = history_risk_tokens[..., -1:].clamp(0.0, 1.0)
        x = self.token_embedding(history_risk_tokens)
        x = (x + self.time_embedding[:, : history_risk_tokens.shape[1]]) * valid

        padding_mask = valid.squeeze(-1) <= 0.5
        # Transformer attention cannot accept an all-masked row.  Keep one zero token
        # visible for that row; pooled memory is still forced to zero below.
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        memory = self.encoder(x, src_key_padding_mask=padding_mask)
        memory = torch.nan_to_num(memory * valid, nan=0.0, posinf=0.0, neginf=0.0)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = memory.sum(dim=1) / denom

        logits = {
            "ttc_worsening": self.ttc_worsening_head(pooled),
            "drac_worsening": self.drac_worsening_head(pooled),
            "urgency": self.urgency_head(pooled),
        }
        return memory, logits

    @staticmethod
    def compute_aux_outputs(
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weight: float,
    ) -> Optional[Dict[str, torch.Tensor]]:
        required = {"risk_aux_labels", "risk_aux_label_valid"}
        if not required.issubset(targets):
            return None

        device = logits["urgency"].device
        labels = targets["risk_aux_labels"].to(device)
        label_valid = targets["risk_aux_label_valid"].to(device).float()

        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if label_valid.ndim == 1:
            label_valid = label_valid.unsqueeze(0)
        total = logits["urgency"].sum() * 0.0
        outputs: Dict[str, torch.Tensor] = {}
        for index, name in enumerate(AUX_LABEL_NAMES):
            valid = label_valid[:, index]
            raw = F.cross_entropy(logits[name], labels[:, index].long(), reduction="none")
            loss = (raw * valid).sum() / valid.sum().clamp(min=1.0)
            total = total + loss

            prediction = logits[name].argmax(dim=-1)
            correct = ((prediction == labels[:, index]) * (valid > 0.5)).float().sum()
            count = (valid > 0.5).float().sum()
            ordinal_error = (
                (prediction.float() - labels[:, index].float()).abs() * valid
            ).sum()
            outputs[f"risk_aux_{name}_accuracy"] = correct / count.clamp(min=1.0)
            outputs[f"risk_aux_{name}_ordinal_mae"] = ordinal_error / count.clamp(min=1.0)
            outputs[f"risk_aux_{name}_valid_rate"] = valid.mean()
            class_count = logits[name].shape[-1]
            for class_index in range(class_count):
                outputs[f"risk_aux_{name}_target_class_{class_index}_rate"] = (
                    ((labels[:, index] == class_index).float() * valid).sum()
                    / count.clamp(min=1.0)
                )

        outputs["memory_aux_loss"] = float(weight) * total
        return outputs

    @staticmethod
    def compute_aux_loss(
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weight: float,
    ) -> Optional[torch.Tensor]:
        """Backward-compatible scalar interface used by older callers/tests."""

        outputs = HistoricalRiskTemporalSelfAttention.compute_aux_outputs(
            logits, targets, weight
        )
        return None if outputs is None else outputs["memory_aux_loss"]


class TemporalRiskCrossAttention(nn.Module):
    """Optional step-level risk injection; disabled in the 3.21 selector setup."""

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
        noisy_traj_points = torch.nan_to_num(
            noisy_traj_points, nan=0.0, posinf=0.0, neginf=0.0
        )
        history_risk_memory = torch.nan_to_num(
            history_risk_memory, nan=0.0, posinf=0.0, neginf=0.0
        )
        batch_size, num_modes, num_steps, _ = noisy_traj_points.shape
        step_query = traj_feature[:, :, None, :] + self.step_embedding(noisy_traj_points)
        flat_query = step_query.reshape(batch_size, num_modes * num_steps, -1)
        risk_context = self.cross_attention(
            flat_query, history_risk_memory, history_risk_memory
        )[0]
        risk_context = torch.nan_to_num(
            risk_context, nan=0.0, posinf=0.0, neginf=0.0
        )
        risk_context = risk_context.reshape(
            batch_size, num_modes, num_steps, -1
        ).mean(dim=2)
        return self.norm(traj_feature + risk_context)
