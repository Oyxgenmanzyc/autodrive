"""Triple-Risk Veto (TRV): preserve PCS ranking and reject unsafe switches."""
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
from .common import load_torch
from .model import PDMCSHead, combine_subscores
from .v2_blocks import gen_sineembed_for_position_1d


RISK_NAMES = ("nc", "dac", "ttc")
RISK_METRIC_INDICES = (0, 1, 3)
DEFAULT_THRESHOLD_GRID = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0)


def relative_risk_targets(labels, selected_mode, base_mode):
    """Return soft NC/DAC/TTC degradation relative to the fallback trajectory."""
    rows = torch.arange(len(selected_mode), device=selected_mode.device)
    selected = labels[rows, selected_mode][:, RISK_METRIC_INDICES]
    base = labels[rows, base_mode][:, RISK_METRIC_INDICES]
    return (base - selected).clamp(0.0, 1.0), selected, base


def calibrate_thresholds(
    risks, pcs_scores, base_scores, changed=None, grid=DEFAULT_THRESHOLD_GRID,
):
    """Choose per-factor thresholds by validation PDM; no-veto is always available."""
    risks = risks.detach().float().cpu()
    pcs_scores = pcs_scores.detach().float().cpu()
    base_scores = base_scores.detach().float().cpu()
    changed = (
        torch.ones_like(pcs_scores, dtype=torch.bool)
        if changed is None else changed.detach().bool().cpu()
    )
    best = None
    for thresholds in product(grid, repeat=3):
        threshold = torch.tensor(thresholds).view(1, 3)
        vetoed = (risks > threshold).any(-1) & changed
        final = torch.where(vetoed, base_scores, pcs_scores)
        new_zero = ((base_scores > 0) & (final == 0)).sum().item()
        key = (final.mean().item(), -new_zero, -vetoed.sum().item())
        if best is None or key > best[0]:
            best = (key, thresholds, vetoed)
    key, thresholds, vetoed = best
    final = torch.where(vetoed, base_scores, pcs_scores)
    return list(thresholds), {
        "pdm": final.mean().item(),
        "pcs_pdm": pcs_scores.mean().item(),
        "base_pdm": base_scores.mean().item(),
        "accepted_switch_count": (changed & ~vetoed).sum().item(),
        "vetoed_count": vetoed.sum().item(),
        "changed_count": changed.sum().item(),
        "new_zero_count": ((base_scores > 0) & (final == 0)).sum().item(),
        "rescued_count": ((base_scores == 0) & (final > 0)).sum().item(),
    }


class TripleRiskVeto(nn.Module):
    """Frozen PCS proposer plus independent NC, DAC, and TTC relative-risk heads."""

    def __init__(self, proposer, thresholds=(1.0, 1.0, 1.0), hidden_dim=256):
        super().__init__()
        self.proposer = proposer.requires_grad_(False).eval()
        self.thresholds = [float(value) for value in thresholds]
        self.hidden_dim = hidden_dim
        pair_dim = 3 * 512 + 3 * 5 + 5
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pair_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                nn.Dropout(0.10), nn.Linear(hidden_dim, 1),
            )
            for _ in RISK_NAMES
        ])

    def train(self, mode=True):
        super().train(mode)
        self.proposer.eval()
        return self

    def _frozen_pcs(self, context):
        device_type = context["proposals"].device.type
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            poses = context["proposals"].detach().float()
            xy = poses[..., :2]
            position = gen_sineembed_for_position(xy, hidden_dim=64).flatten(-2)
            heading = gen_sineembed_for_position_1d(
                poses[..., 2], hidden_dim=32,
            ).flatten(-2)
            features = self.proposer.encoder(torch.cat([position, heading], dim=-1))
            bev = context["bev"].detach().to(torch.float16).float()
            agents = context["agents"].detach().to(torch.float16).float()
            ego = context["ego"].detach().to(torch.float16).float()
            features = self.proposer.decoder(
                features, xy, bev, bev.shape[-2:], agents, ego, None, None,
            )
            logits = torch.cat([head(features) for head in self.proposer.heads], dim=-1)
            subscores = logits.float().sigmoid()
            scores = combine_subscores(subscores)
        return features.float(), subscores, scores

    def forward(self, context, selected_mode=None, base_mode=None, verify_modes=False):
        features, subscores, pcs_scores = self._frozen_pcs(context)
        online_selected = pcs_scores.argmax(-1)
        online_base = context["base_logits"].argmax(-1)
        if selected_mode is None:
            selected_mode = online_selected
        if base_mode is None:
            base_mode = online_base
        selected_mode = selected_mode.long()
        base_mode = base_mode.long()
        # Cached PCS decisions are authoritative during training. Near-tied PCS
        # scores can flip argmax when GEMM batch shapes change, even though the
        # checkpoint and inputs are identical. The base argmax is read directly
        # from cached logits and therefore remains a strict integrity check.
        pcs_mode_mismatch = selected_mode != online_selected
        if verify_modes and not torch.equal(base_mode, online_base):
            raise ValueError("Pair cache base modes differ from cached base logits")

        rows = torch.arange(len(selected_mode), device=selected_mode.device)
        candidate_feature = features[rows, selected_mode]
        base_feature = features[rows, base_mode]
        candidate_metrics = subscores[rows, selected_mode]
        base_metrics = subscores[rows, base_mode]
        original_logits = context["base_logits"].detach().float()
        original_probability = original_logits.softmax(-1)
        policy_features = torch.stack([
            original_probability[rows, selected_mode],
            original_probability[rows, base_mode],
            original_logits[rows, selected_mode] - original_logits[rows, base_mode],
            pcs_scores[rows, selected_mode],
            pcs_scores[rows, base_mode],
        ], dim=-1)
        pair_features = torch.cat([
            candidate_feature, base_feature, candidate_feature - base_feature,
            candidate_metrics, base_metrics, candidate_metrics - base_metrics,
            policy_features,
        ], dim=-1)
        risk_logits = torch.cat([head(pair_features) for head in self.risk_heads], dim=-1)
        risks = risk_logits.float().sigmoid()
        thresholds = torch.tensor(
            self.thresholds, device=risks.device, dtype=risks.dtype,
        ).view(1, 3)
        vetoed = (risks > thresholds).any(-1) & (selected_mode != base_mode)
        final_mode = torch.where(vetoed, base_mode, selected_mode)
        return {
            "risk_logits": risk_logits.float(),
            "risks": risks,
            "pcs_scores": pcs_scores,
            "pcs_mode": selected_mode,
            "pcs_mode_mismatch": pcs_mode_mismatch,
            "base_mode": base_mode,
            "final_mode": final_mode,
            "vetoed": vetoed,
        }

    @staticmethod
    def loss(output, labels, scores):
        selected_mode, base_mode = output["pcs_mode"], output["base_mode"]
        targets, selected_labels, base_labels = relative_risk_targets(
            labels, selected_mode, base_mode,
        )
        rows = torch.arange(len(selected_mode), device=selected_mode.device)
        selected_scores = scores[rows, selected_mode]
        base_scores = scores[rows, base_mode]
        new_zero = (selected_labels <= 0) & (base_labels > 0)
        hard_negative = (selected_scores < base_scores).float().unsqueeze(-1)
        probability = output["risk_logits"].sigmoid()
        focal = (
            targets * (1 - probability).pow(2)
            + (1 - targets) * probability.pow(2)
        )
        weight = 1.0 + 7.0 * targets + 12.0 * new_zero.float() + 2.0 * hard_negative
        raw = F.binary_cross_entropy_with_logits(
            output["risk_logits"], targets, reduction="none",
        )
        per_factor = (raw * focal * weight).mean(0)
        return {
            "loss": per_factor.sum(),
            **{name: per_factor[index] for index, name in enumerate(RISK_NAMES)},
        }


def load_veto(path, device, expected_provenance=None):
    checkpoint = load_torch(path)
    metadata = checkpoint["veto_metadata"]
    if metadata.get("schema") != "triple_risk_veto_v1":
        raise ValueError("Checkpoint is not a Triple-Risk Veto model")
    if expected_provenance is not None and metadata["provenance"] != expected_provenance:
        raise ValueError("Veto checkpoint does not match baseline/anchor/seed")
    proposer = PDMCSHead(**metadata["pcs_settings"])
    model = TripleRiskVeto(
        proposer, thresholds=metadata["thresholds"], hidden_dim=metadata["hidden_dim"],
    )
    prefix = "model."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), metadata
