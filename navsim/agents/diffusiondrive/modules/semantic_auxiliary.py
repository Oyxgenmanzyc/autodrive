"""DiffE2E-style hybrid trajectory and future-speed supervision."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class HybridSpeedSemanticDecoder(nn.Module):
    """Jointly refine trajectory-mode tokens and one supervised speed token.

    The supervision-query, Transformer encoder, and supervision-head structure is
    adapted from ``bdjukic/DiffE2E@4985f8b/hybrid_decoder.py``. Trajectory and
    semantic type encodings replace its fixed-length positional table so adaptive
    anchor banks can retain a variable number of modes.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_speed_classes: int = 4,
    ) -> None:
        super().__init__()
        self.supervision_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos_encoding = nn.Parameter(torch.randn(1, 2, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.sup_head = nn.Linear(feature_dim, num_speed_classes)

    def forward(self, trajectory_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_modes, _ = trajectory_tokens.shape
        trajectory_tokens = trajectory_tokens + self.pos_encoding[:, :1]
        supervision_token = self.supervision_query.expand(batch_size, -1, -1)
        supervision_token = supervision_token + self.pos_encoding[:, 1:]

        hybrid_tokens = torch.cat([trajectory_tokens, supervision_token], dim=1)
        hybrid_tokens = self.transformer(hybrid_tokens)

        trajectory_tokens = hybrid_tokens[:, :num_modes]
        speed_logits = self.sup_head(hybrid_tokens[:, num_modes])
        return trajectory_tokens, speed_logits


def future_speed_targets(
    trajectory: torch.Tensor,
    interval_length: float,
    thresholds: Sequence[float],
) -> torch.Tensor:
    """Convert a future ego trajectory into four ordered mean-speed classes."""
    if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
        raise ValueError("trajectory must have shape [B, T, D] with D >= 2")
    if trajectory.shape[1] < 2:
        raise ValueError("trajectory must contain at least two future poses")
    if interval_length <= 0:
        raise ValueError("interval_length must be positive")
    if len(thresholds) != 3 or any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("thresholds must contain three strictly increasing values")

    displacement = trajectory[:, 1:, :2] - trajectory[:, :-1, :2]
    mean_speed = torch.linalg.vector_norm(displacement, dim=-1).mean(dim=-1) / interval_length
    boundaries = mean_speed.new_tensor(tuple(thresholds))
    return torch.bucketize(mean_speed, boundaries, right=True)
