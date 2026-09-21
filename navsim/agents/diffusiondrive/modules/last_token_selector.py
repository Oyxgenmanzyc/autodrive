"""Parameter-free LAST-inspired selection for spatial tokens.

Reference: https://github.com/ChengShiest/LAST-ViT (cls_pretrain/conf.py).
This adaptation keeps every spatial token and uses channel-wise selection votes
as a bounded residual gate instead of replacing spatial memory by a CLS token.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LASTTokenSelector(nn.Module):
    """Select along N for each channel of [B, N, C]; never reorder tokens.

    Top-k indices/votes are discrete (no gradient through the ranking). Gradients
    still flow through the original values in both the gate and global gather.
    No parameters or persistent buffers are added to existing checkpoints.
    """

    def __init__(self, topk_ratio=0.25, sigma_scale=1.0, eps=1e-6,
                 gate_alpha=0.25, pre_norm=True):
        super().__init__()
        for name, value in (("topk_ratio", topk_ratio), ("sigma_scale", sigma_scale),
                            ("eps", eps), ("gate_alpha", gate_alpha)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0, 1]")
        if sigma_scale <= 0 or eps <= 0:
            raise ValueError("sigma_scale and eps must be positive")
        if not 0 <= gate_alpha < 1:
            raise ValueError("gate_alpha must be in [0, 1) to retain all tokens")
        self.topk_ratio = topk_ratio
        self.sigma_scale = sigma_scale
        self.eps = eps
        self.gate_alpha = gate_alpha
        self.pre_norm = pre_norm

    def gaussian_kernel_1d(self, channels, device):
        # Center exactly on the fftshift DC bin, including odd channel counts.
        # Upstream's arange(-C//2+1, C//2+1) offsets DC by one for even C.
        coord = torch.arange(channels, device=device, dtype=torch.float32) - channels // 2
        sigma = math.sqrt(channels) * self.sigma_scale
        return torch.exp(-0.5 * (coord / sigma).square()).view(1, 1, channels)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] == 0:
            raise ValueError("LAST expects nonempty spatial tokens [B, N, C]")
        if not x.is_floating_point():
            raise ValueError("LAST expects floating-point features")
        _, tokens, channels = x.shape
        k = max(1, min(tokens, round(tokens * self.topk_ratio)))

        with torch.autocast(device_type=x.device.type, enabled=False):
            # oneMKL/cuFFT require a materialized layout for zero-stride views
            # such as expand(); normal permuted BEV tokens are handled too.
            score_x = x.float().contiguous()
            if self.pre_norm:
                score_x = F.layer_norm(score_x, (channels,))
            spectrum = torch.fft.fftshift(torch.fft.fft(score_x, dim=-1), dim=-1)
            spectrum = spectrum * self.gaussian_kernel_1d(channels, x.device)
            lowpass = torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), dim=-1).real
            stability = score_x / ((lowpass - score_x).abs() + self.eps)
            stability = torch.nan_to_num(stability, nan=0.0, posinf=0.0, neginf=0.0)
            indices = stability.topk(k, dim=1, largest=True).indices
            mask = torch.zeros_like(stability, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            vote = mask.float().mean(dim=-1)
            # Population std stays finite even for a single spatial token.
            std = vote.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            vote_norm = (vote - vote.mean(dim=1, keepdim=True)) / std
            gate = 1.0 + self.gate_alpha * torch.tanh(vote_norm)

        # Gather original features, not normalized/low-pass features; do not detach.
        global_token = x.gather(1, indices).float().mean(dim=1).to(x.dtype)
        gated_x = x * gate.unsqueeze(-1).to(x.dtype)
        return gated_x, global_token, {
            "stability": stability,
            "vote": vote,
            "gate": gate,
            "indices": indices,
        }
