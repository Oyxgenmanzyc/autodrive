"""Conservative base-relative scoring built on the frozen PCS representation."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.diffusiondrive.modules.blocks import (
    gen_sineembed_for_position, linear_relu_ln,
)
from .model import combine_subscores
from .v2_blocks import ScorerTransformerDecoderLayer, gen_sineembed_for_position_1d


class ConservativeAdvantageScorer(nn.Module):
    """Predict PDM plus base-relative advantage, win probability, and tail risk."""

    def __init__(
        self,
        num_poses=8,
        lidar_max_x=32.0,
        lidar_max_y=32.0,
        min_delta=0.01,
        min_win_probability=0.55,
        max_catastrophic_risk=0.10,
        risk_penalty=0.50,
    ):
        super().__init__()
        self.settings = dict(
            num_poses=num_poses,
            lidar_max_x=lidar_max_x,
            lidar_max_y=lidar_max_y,
            min_delta=min_delta,
            min_win_probability=min_win_probability,
            max_catastrophic_risk=max_catastrophic_risk,
            risk_penalty=risk_penalty,
        )
        config = SimpleNamespace(
            num_poses=num_poses, lidar_max_x=lidar_max_x, lidar_max_y=lidar_max_y,
        )
        self.encoder = nn.Sequential(
            *linear_relu_ln(256, 1, 1, num_poses * 128),
            nn.Linear(256, 512),
        )
        self.decoder = ScorerTransformerDecoderLayer(num_poses, 256, 1024, config)
        self.metric_heads = nn.ModuleList([
            nn.Sequential(*linear_relu_ln(512, 2 if i == 2 else 1, 2), nn.Linear(512, 1))
            for i in range(5)
        ])
        # candidate, base, and their residual keep the decision explicitly relative.
        self.decision_head = nn.Sequential(
            nn.Linear(3 * 512, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, 3),
        )

    def forward(self, context):
        poses = context["proposals"].detach().float()
        xy = poses[..., :2]
        position = gen_sineembed_for_position(xy, hidden_dim=64).flatten(-2)
        heading = gen_sineembed_for_position_1d(poses[..., 2], hidden_dim=32).flatten(-2)
        features = self.encoder(torch.cat([position, heading], dim=-1))
        bev = context["bev"].detach().to(torch.float16).float()
        agents = context["agents"].detach().to(torch.float16).float()
        ego = context["ego"].detach().to(torch.float16).float()
        features = self.decoder(features, xy, bev, bev.shape[-2:], agents, ego, None, None)

        metric_logits = torch.cat([head(features) for head in self.metric_heads], dim=-1).float()
        subscores = metric_logits.sigmoid()
        scores = combine_subscores(subscores)

        base = context["base_logits"].argmax(-1)
        rows = torch.arange(len(base), device=base.device)
        base_features = features[rows, base].unsqueeze(1).expand_as(features)
        decision = self.decision_head(torch.cat(
            [features, base_features, features - base_features], dim=-1,
        )).float()
        predicted_delta = decision[..., 0].tanh()
        win_logits = decision[..., 1]
        catastrophic_logits = decision[..., 2]
        win_probability = win_logits.sigmoid()
        catastrophic_risk = catastrophic_logits.sigmoid()

        # Base is an explicit zero-advantage action, independent of head calibration.
        predicted_delta = predicted_delta.scatter(1, base[:, None], 0.0)
        win_probability = win_probability.scatter(1, base[:, None], 0.5)
        catastrophic_risk = catastrophic_risk.scatter(1, base[:, None], 0.0)
        utility = predicted_delta - self.settings["risk_penalty"] * catastrophic_risk
        return {
            "logits": metric_logits,
            "subscores": subscores,
            "scores": scores,
            "predicted_delta": predicted_delta,
            "win_logits": win_logits,
            "win_probability": win_probability,
            "catastrophic_logits": catastrophic_logits,
            "catastrophic_risk": catastrophic_risk,
            "utility": utility,
        }

    @staticmethod
    def losses(
        output,
        context,
        labels,
        scores,
        tie_margin=1e-4,
        catastrophic_delta=-0.25,
        switch_cost=0.01,
        catastrophe_penalty=1.0,
        policy_temperature=0.05,
    ):
        metric_loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
        base = context["base_logits"].argmax(-1)
        rows = torch.arange(len(base), device=base.device)
        base_scores = scores[rows, base].unsqueeze(1)
        true_delta = scores - base_scores

        delta_weight = 1.0 + 4.0 * true_delta.abs()
        delta_loss = (
            F.smooth_l1_loss(output["predicted_delta"], true_delta, reduction="none")
            * delta_weight
        ).mean()

        decisive = true_delta.abs() > tie_margin
        win_target = (true_delta > tie_margin).float()
        win_raw = F.binary_cross_entropy_with_logits(
            output["win_logits"], win_target, reduction="none",
        )
        win_loss = (win_raw * decisive).sum() / decisive.sum().clamp_min(1)

        # Absolute failures and severe regret are both unacceptable switch actions.
        catastrophic = (
            (labels[..., 0] <= 0)
            | (labels[..., 1] <= 0)
            | (labels[..., 3] <= 0)
            | (true_delta <= catastrophic_delta)
        )
        catastrophic_loss = F.binary_cross_entropy_with_logits(
            output["catastrophic_logits"], catastrophic.float(),
        )

        non_base = torch.ones_like(true_delta, dtype=torch.bool)
        non_base[rows, base] = False
        target_utility = (
            true_delta
            - switch_cost * non_base
            - catastrophe_penalty * catastrophic.float()
        )
        target_policy = F.softmax(target_utility / policy_temperature, dim=-1).detach()
        predicted_policy = F.log_softmax(output["utility"] / policy_temperature, dim=-1)
        policy_loss = F.kl_div(predicted_policy, target_policy, reduction="batchmean")

        total = (
            metric_loss
            + 2.0 * delta_loss
            + 0.5 * win_loss
            + catastrophic_loss
            + 0.5 * policy_loss
        )
        return {
            "loss": total,
            "metric": metric_loss,
            "delta": delta_loss,
            "win": win_loss,
            "catastrophic": catastrophic_loss,
            "policy": policy_loss,
        }


def select_candidates(context, output):
    """Select a learned action only when all conservative switch tests pass."""
    base = context["base_logits"].argmax(-1)
    raw_selected = output["utility"].argmax(-1)
    rows = torch.arange(len(base), device=base.device)
    predicted_delta = output["predicted_delta"][rows, raw_selected]
    win_probability = output["win_probability"][rows, raw_selected]
    catastrophic_risk = output["catastrophic_risk"][rows, raw_selected]
    # Normal model outputs carry thresholds separately via the caller below.
    thresholds = output.get("thresholds")
    if thresholds is None:
        thresholds = {
            "min_delta": 0.01,
            "min_win_probability": 0.55,
            "max_catastrophic_risk": 0.10,
        }
    switch = (
        (raw_selected != base)
        & (predicted_delta >= thresholds["min_delta"])
        & (win_probability >= thresholds["min_win_probability"])
        & (catastrophic_risk <= thresholds["max_catastrophic_risk"])
    )
    selected = torch.where(switch, raw_selected, base)
    return {
        "trajectory": context["proposals"][rows, selected],
        "selector_trajectory": context["proposals"][rows, base],
        "selected_mode": selected,
        "raw_selected_mode": raw_selected,
        "base_mode": base,
        "switched": switch,
        "predicted_delta": predicted_delta,
        "win_probability": win_probability,
        "catastrophic_risk": catastrophic_risk,
    }


def scorer_output(model, context):
    """Attach checkpointed decision thresholds without global mutable state."""
    output = model(context)
    output["thresholds"] = {
        key: model.settings[key]
        for key in ("min_delta", "min_win_probability", "max_catastrophic_risk")
    }
    return output


def load_conservative_scorer(path, device, expected=None):
    from .common import load_torch

    checkpoint = load_torch(path)
    metadata = checkpoint["pcs_metadata"]
    if metadata.get("scorer_type") != "conservative_advantage_v1":
        raise ValueError("Checkpoint is not a 3.1.05_2 conservative scorer")
    if expected is not None and metadata["provenance"] != expected:
        raise ValueError("Scorer checkpoint does not match baseline/anchor/seed")
    scorer = ConservativeAdvantageScorer(**metadata["settings"])
    prefix = "head."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    scorer.load_state_dict(state, strict=True)
    return scorer.eval().to(device), metadata


def initialize_from_pcs(model, path, expected):
    """Warm-start the shared representation and metric heads from 3.1.05 PCS."""
    from .common import load_torch

    checkpoint = load_torch(path)
    metadata = checkpoint["pcs_metadata"]
    if metadata["provenance"] != expected:
        raise ValueError("Initial PCS checkpoint does not match candidate cache provenance")
    converted = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("head.encoder."):
            converted[key[len("head."):]] = value
        elif key.startswith("head.decoder."):
            converted[key[len("head."):]] = value
        elif key.startswith("head.heads."):
            converted["metric_heads." + key[len("head.heads."):]] = value
    expected_keys = {
        key for key in model.state_dict()
        if not key.startswith("decision_head.")
    }
    if set(converted) != expected_keys:
        missing = sorted(expected_keys - set(converted))
        unexpected = sorted(set(converted) - expected_keys)
        raise ValueError(
            f"Initial PCS weights are incompatible; missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    result = model.load_state_dict(converted, strict=False)
    if result.unexpected_keys or any(
        not key.startswith("decision_head.") for key in result.missing_keys
    ):
        raise ValueError("Unexpected PCS warm-start state mismatch")
    return metadata
