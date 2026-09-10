"""3.1.05_3: compare expected gain/cost with scene-conditioned brake timing.

Timing representation adapted from user-supplied best_3.22/risk_brake_timing.py.
No expert trajectory, future actor annotation, or PDM label enters forward().
"""
from copy import deepcopy
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .common import load_torch
from .veto import TripleRiskVeto, relative_risk_targets


SCHEMA = "timing_weighted_gain_v1"
TIMING_DIM = 8 * 11


def timing_features(poses, dt=0.5):
    """Ordered 0.5s profiles; unknown initial ego speed is NOT invented.

    Cache contains poses, not raw current velocity. Acceleration of interval 0
    is masked out; only subsequent interval differences are observed. Onset is
    therefore a trajectory-relative diagnostic, not an absolute TTC estimate.
    """
    if poses.shape[-2:] != (8, 3) or dt <= 0:
        raise ValueError("Timing features require eight SE2 poses and positive dt")
    xy = poses.detach().float()[..., :2]
    displacement = torch.diff(xy, dim=-2, prepend=torch.zeros_like(xy[..., :1, :]))
    speed = (torch.sqrt(displacement.square().sum(-1) + 1e-12) - 1e-6).clamp_min(0) / dt
    accel = torch.cat([torch.zeros_like(speed[..., :1]), torch.diff(speed, dim=-1) / dt], -1)
    jerk = torch.cat([torch.zeros_like(speed[..., :2]), torch.diff(accel[..., 1:], dim=-1) / dt], -1)
    valid = torch.ones_like(speed)
    valid[..., 0] = 0
    brake = torch.sigmoid((-0.5 - accel) / 0.35) * valid
    # Two consecutive braking intervals, so a single spike is not an onset.
    sustained = torch.cat([brake[..., :-1] * brake[..., 1:], torch.zeros_like(brake[..., :1])], -1)
    cumulative = sustained.cummax(-1).values
    time = torch.arange(1, 9, device=xy.device).float().expand_as(speed) / 8
    return torch.stack([
        xy[..., 0] / 50, xy[..., 1] / 20, speed / 20,
        accel.clamp(-20, 20) / 10, jerk.clamp(-40, 40) / 20,
        brake, sustained, cumulative, poses.detach().float()[..., 2].sin(),
        time, valid,
    ], -1).flatten(-2)


def choose_action(gain_cost, risks, reference_risks, changed, policy):
    """No outcome labels here. A high individual risk is not an automatic veto."""
    gain, cost = gain_cost.unbind(-1)
    weight = risks.new_tensor(policy.get("risk_weights", (0., 0., 0.)))
    utility = gain - float(policy.get("loss_multiplier", 1.0)) * cost - (risks * weight).sum(-1)
    mode = policy["mode"]
    if mode == "weighted":
        vetoed = utility < float(policy["threshold"])
    elif mode == "reference":
        vetoed = (reference_risks > risks.new_tensor(policy["reference_thresholds"])).any(-1)
    elif mode == "pcs":
        vetoed = torch.zeros_like(changed)
    else:
        raise ValueError(f"Unknown decision mode: {mode}")
    return vetoed & changed, utility


def outcome_metrics(vetoed, pcs, base, pcs_labels, base_labels, changed):
    final = torch.where(vetoed, base, pcs)
    final_labels = torch.where(vetoed[:, None], base_labels, pcs_labels)
    return {
        "pdm": final.mean().item(), "pcs_pdm": pcs.mean().item(),
        "base_pdm": base.mean().item(), "vetoed_count": int(vetoed.sum()),
        "new_zero_count": int(((base > 0) & (final == 0)).sum()),
        "rescued_count": int(((base == 0) & (final > 0)).sum()),
        "changed_count": int(changed.sum()),
        "accepted_switch_count": int((changed & ~vetoed).sum()),
        "nc": final_labels[:, 0].mean().item(),
        "dac": final_labels[:, 1].mean().item(),
        "ttc": final_labels[:, 3].mean().item(),
    }


def calibrate_policy(gain_cost, risks, reference_risks, changed, pcs, base,
                     pcs_labels, base_labels, reference_thresholds):
    """Call ONLY on navtrain calibration logs; retain a reference fallback.

    The grid uses PDM units, not a navtest-derived rescue threshold. Select by
    final calibration PDM, subject to no extra zero scores or NC/DAC/TTC mean
    regression versus the original TRV on these SAME scenes.
    """
    reference = {"mode": "reference", "reference_thresholds": list(reference_thresholds)}
    mask, _ = choose_action(gain_cost, risks, reference_risks, changed, reference)
    reference_result = outcome_metrics(mask, pcs, base, pcs_labels, base_labels, changed)
    best_policy, best_result = reference, reference_result
    best_key = (reference_result["pdm"], -reference_result["new_zero_count"], -reference_result["vetoed_count"])
    policies = [{"mode": "pcs"}]
    for loss_weight, weights in product(
        (1.0, 1.5, 2.0),
        ((0., 0., 0.), (0.05, 0.05, 0.02), (0.1, 0.1, 0.04)),
    ):
        utility = gain_cost[:, 0] - loss_weight * gain_cost[:, 1] - (risks * risks.new_tensor(weights)).sum(-1)
        thresholds = {-.05, -.02, -.01, 0., .01, .02, .05}
        # Measure the *predicted* utility distribution of correctly rescued and
        # incorrectly blocked decisions in calibration, not just actual PDM.
        for group in (mask & (pcs < base), mask & (pcs > base)):
            if group.any():
                thresholds.update(torch.quantile(utility[group].float(), torch.tensor([.1, .25, .5, .75, .9])).tolist())
        for threshold in sorted(thresholds):
            policies.append({"mode": "weighted", "loss_multiplier": loss_weight,
                             "risk_weights": list(weights), "threshold": threshold})
    for policy in policies:
        vetoed, _ = choose_action(gain_cost, risks, reference_risks, changed, policy)
        result = outcome_metrics(vetoed, pcs, base, pcs_labels, base_labels, changed)
        if result["new_zero_count"] > reference_result["new_zero_count"]:
            continue
        if any(result[k] + 1e-10 < reference_result[k] for k in ("nc", "dac", "ttc")):
            continue
        key = (result["pdm"], -result["new_zero_count"], -result["vetoed_count"])
        if key > best_key:
            best_policy, best_result, best_key = policy, result, key
    return best_policy, best_result, reference_result


class TimingGainSelector(TripleRiskVeto):
    def __init__(self, reference, use_timing=True):
        super().__init__(reference.proposer, reference.thresholds, reference.hidden_dim)
        self.use_timing = bool(use_timing)
        self.reference_thresholds = list(reference.thresholds)
        self.reference_heads = deepcopy(reference.risk_heads).requires_grad_(False).eval()
        pair_dim = 3 * 512 + 3 * 5 + 5
        extra = 3 * TIMING_DIM if self.use_timing else 0
        self.risk_heads = deepcopy(reference.risk_heads)
        if extra:
            for old, new in zip(reference.risk_heads, self.risk_heads):
                new[0] = nn.Linear(pair_dim + extra, reference.hidden_dim)
                with torch.no_grad():
                    new[0].weight.zero_()
                    new[0].weight[:, :pair_dim].copy_(old[0].weight)
                    new[0].bias.copy_(old[0].bias)
        self.gain_head = nn.Sequential(
            nn.Linear(pair_dim + extra, reference.hidden_dim), nn.ReLU(),
            nn.LayerNorm(reference.hidden_dim), nn.Linear(reference.hidden_dim, 2),
        )
        nn.init.constant_(self.gain_head[-1].bias, -3.0)
        self.policy = {"mode": "reference", "reference_thresholds": self.reference_thresholds}

    def train(self, mode=True):
        super().train(mode)
        self.reference_heads.eval()
        return self

    def forward(self, context, selected_mode=None, base_mode=None, verify_modes=False):
        features, subscores, scores = self._frozen_pcs(context)
        online = scores.argmax(-1)
        online_base = context["base_logits"].argmax(-1)
        selected = online if selected_mode is None else selected_mode.long()
        base = online_base if base_mode is None else base_mode.long()
        if verify_modes and not torch.equal(base, online_base):
            raise ValueError("Cached base differs from original selector")
        rows = torch.arange(len(base), device=base.device)
        logits = context["base_logits"].detach().float()
        prob = logits.softmax(-1)
        pair = torch.cat([
            features[rows, selected], features[rows, base], features[rows, selected] - features[rows, base],
            subscores[rows, selected], subscores[rows, base], subscores[rows, selected] - subscores[rows, base],
            torch.stack([prob[rows, selected], prob[rows, base], logits[rows, selected] - logits[rows, base],
                         scores[rows, selected], scores[rows, base]], -1),
        ], -1)
        with torch.no_grad(), torch.autocast(device_type=pair.device.type, enabled=False):
            reference_risks = torch.cat([head(pair.float()) for head in self.reference_heads], -1).sigmoid()
        if self.use_timing:
            timing = timing_features(context["proposals"])
            pair = torch.cat([pair, timing[rows, selected], timing[rows, base],
                              timing[rows, selected] - timing[rows, base]], -1)
        risk_logits = torch.cat([head(pair) for head in self.risk_heads], -1).float()
        risks = risk_logits.sigmoid()
        gain_cost = self.gain_head(pair).float().sigmoid()
        changed = selected != base
        vetoed, utility = choose_action(gain_cost, risks, reference_risks, changed, self.policy)
        return {
            "risk_logits": risk_logits, "risks": risks, "gain_cost": gain_cost,
            "reference_risks": reference_risks, "utility": utility,
            "pcs_scores": scores, "pcs_mode": selected, "base_mode": base,
            "pcs_mode_mismatch": online != selected,
            "vetoed": vetoed, "final_mode": torch.where(vetoed, base, selected),
        }

    @staticmethod
    def loss(output, labels, scores, positive_weights):
        targets, _, _ = relative_risk_targets(labels, output["pcs_mode"], output["base_mode"])
        rows = torch.arange(len(scores), device=scores.device)
        delta = scores[rows, output["pcs_mode"]] - scores[rows, output["base_mode"]]
        changed = output["pcs_mode"] != output["base_mode"]
        mask = changed.float()
        denominator = mask.sum().clamp_min(1)
        # MSE estimates mean positive/negative PDM magnitude, including zeros.
        # Do not severity-weight this objective: that would distort expected gain.
        truth = torch.stack([delta.clamp_min(0), (-delta).clamp_min(0)], -1)
        gain_loss = ((output["gain_cost"] - truth).square().sum(-1) * mask).sum() / denominator
        # Positive weights come ONLY from train pair counts. Weighted BCE scores
        # are not claimed to be calibrated probabilities; val sets utility costs.
        raw = F.binary_cross_entropy_with_logits(
            output["risk_logits"], targets, pos_weight=positive_weights, reduction="none",
        )
        risk_loss = (raw * mask[:, None]).sum(0) / denominator
        return {"loss": gain_loss + 0.1 * risk_loss.sum(), "gain_loss": gain_loss,
                **{name: risk_loss[i] for i, name in enumerate(("nc", "dac", "ttc"))}}


def load_timing_gain(path, device, expected_provenance=None):
    checkpoint = load_torch(path)
    metadata = checkpoint["timing_gain_metadata"]
    if metadata["schema"] != SCHEMA:
        raise ValueError("Not a 3.1.05_3 checkpoint")
    if expected_provenance is not None and metadata["provenance"] != expected_provenance:
        raise ValueError("Candidate provenance mismatch")
    from .model import PDMCSHead
    reference = TripleRiskVeto(PDMCSHead(**metadata["pcs_settings"]),
                               metadata["reference_thresholds"], metadata["hidden_dim"])
    model = TimingGainSelector(reference, metadata["use_timing"])
    state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.policy = metadata["policy"]
    return model.to(device).eval(), metadata
