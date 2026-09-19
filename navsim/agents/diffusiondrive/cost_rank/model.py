"""A single ranking objective; the original PCS is never optimized here."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenPCS(nn.Module):
    """Extract original decoder features without editing cache-provenance files."""
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, context):
        captured = []
        handle = self.scorer.decoder.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach()))
        try:
            with torch.autocast(device_type=context['proposals'].device.type, enabled=False):
                output = self.scorer(context)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one PCS decoder invocation')
        return dict(features=captured[0].half(), subscores=output['subscores'].float(),
                    pcs_scores=output['scores'].float(),
                    proposals=context['proposals'].detach().float(),
                    base_logits=context['base_logits'].detach().float())


class CostRanker(nn.Module):
    """Small independent contextual ranker; no metric, gate or risk loss/head."""
    def __init__(self, width=128, cap=.2):
        super().__init__()
        if cap <= 0:
            raise ValueError('Residual cap must be positive')
        self.settings = dict(width=width, cap=cap)
        self.cap = cap
        self.feature_norm = nn.LayerNorm(512)
        # 512 frozen features + 5 metrics + PCS score + base probability + 8*4 geometry.
        self.embed = nn.Sequential(nn.Linear(551, width), nn.ReLU(), nn.LayerNorm(width))
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        # Only deployable inputs. labels/scores/oof_scores are never read here.
        poses = batch['proposals'].detach().float()
        geometry = torch.cat([poses[..., :2]/poses.new_tensor([60., 30.]),
                              poses[..., 2:3].sin(), poses[..., 2:3].cos()], -1).flatten(-2)
        inputs = torch.cat([
            self.feature_norm(batch['features'].detach().float()),
            batch['subscores'].detach().float(), batch['pcs_scores'].detach().float()[..., None],
            batch['base_logits'].detach().float().softmax(-1)[..., None], geometry.clamp(-5., 5.),
        ], -1)
        x = self.embed(inputs)
        attended, _ = self.attention(x, x, x, need_weights=False)
        return self.cap*self.head(self.norm(x+attended)).squeeze(-1).float().tanh()


def select(pcs_scores, residual, alpha=1.):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    if not torch.isfinite(pcs_scores).all() or not torch.isfinite(residual).all():
        raise ValueError('Nonfinite ranking predictions')
    # alpha=0 explicitly retains original tie breaking.
    return (pcs_scores.float()+alpha*residual.float()).argmax(-1)


def ranking_loss(pcs_scores, residual, truth, oof_scores, min_gap=.02,
                 temperature=.05, top_k=5, hard_weight=3., cost_cap=5.):
    """One cost-weighted pairwise softplus loss over all same-scene candidates.

    Ordered pair (i,j): truth[i] > truth[j]. Upweight large actual PDM losses
    and false-high predictions from an out-of-fold teacher. All eligible normal
    pairs retain positive weight. Labels/teacher scores affect ONLY this loss.
    """
    if temperature <= 0 or min_gap <= 0 or not 1 <= top_k <= truth.shape[1]:
        raise ValueError('Invalid ranking loss settings')
    truth, oof_scores = truth.detach().float(), oof_scores.detach().float()
    if not all(torch.isfinite(x).all() for x in (truth, oof_scores, pcs_scores, residual)):
        raise ValueError('Nonfinite training batch')
    gap = truth[:, :, None]-truth[:, None, :]
    valid = gap >= min_gap
    top = torch.zeros_like(truth, dtype=torch.bool)
    top.scatter_(1, oof_scores.topk(top_k, dim=-1).indices, True)
    wrong_order = oof_scores[:, None, :] >= oof_scores[:, :, None]
    hard = top[:, None, :] & wrong_order
    cost = (gap/.1).clamp(0., cost_cap)*(1.+hard_weight*hard.float())
    predicted = pcs_scores.detach().float()+residual.float()
    diff = predicted[:, :, None]-predicted[:, None, :]
    pair_loss = F.softplus(-diff/temperature)*cost*valid
    # Each scene contributes once; large-regret pairs still receive larger cost.
    counts = valid.sum((1, 2))
    scene_loss = pair_loss.sum((1, 2))/counts.clamp_min(1)
    active = counts > 0
    loss = scene_loss.sum()/active.sum().clamp_min(1)
    return loss, dict(active_scenes=active.sum(), pairs=valid.sum(), hard_pairs=(valid & hard).sum())
