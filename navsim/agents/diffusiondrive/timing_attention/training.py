"""Train only the added timing-query/attention parameters."""
import torch
import pytorch_lightning as pl
from .model import bind_timing_inputs, mutable_parameter, history_validity
from .loss import timing_strength_loss, distributed_active_mean


class TimingAttentionModule(pl.LightningModule):
    def __init__(self, generator, metadata):
        super().__init__()
        self.generator, self.metadata = generator, metadata
        self._poses = None
        self.generator._trajectory_head.diff_decoder.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._poses = output[0][-1]

    def train(self, mode=True):
        super().train(mode)
        # Original dropout and BatchNorm stay frozen/eval. Attention has no dropout.
        self.generator.eval()
        return self

    def configure_optimizers(self):
        named = [(n, p) for n, p in self.generator.named_parameters() if p.requires_grad]
        if not named or any(not mutable_parameter(n) for n, _ in named):
            raise ValueError('Only timing-query and attention parameters may be optimized')
        optimizer = torch.optim.AdamW([p for _, p in named], lr=self.metadata['lr'], weight_decay=1e-4)
        return {'optimizer': optimizer, 'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.metadata['epochs'])}

    def _forward(self, context, targets=None):
        head = self.generator._trajectory_head
        bev, ego, agents = (context[k].float() for k in ('bev', 'ego', 'agents'))
        args = (ego, agents, bev, bev.shape[-2:], ego.new_zeros((len(ego), 1, ego.shape[-1])))
        self._poses = None
        with bind_timing_inputs(self.generator, context['timing_history'], context['timing_ego_speed']):
            if targets is not None:
                output = head.forward_train(*args, targets=targets)
            else:
                output = head.forward_test(*args, global_img=None)
        poses, self._poses = self._poses, None
        if poses is None or not torch.isfinite(poses).all():
            raise ValueError('Missing/nonfinite differentiable generated trajectories')
        return output, poses

    def training_step(self, batch, batch_idx):
        context, targets = batch
        output, poses = self._forward(context, targets)
        _, enabled = history_validity(context['timing_history'])
        with torch.autocast(device_type=poses.device.type, enabled=False):
            sums, stats = timing_strength_loss(poses, targets, self.generator._trajectory_head.plan_anchor,
                                                enabled, context['timing_ego_speed'])
            losses = {key: distributed_active_mean(value, stats['active_count']) for key, value in sums.items()}
            loss = output['trajectory_loss'].float()
            for key in ('timing', 'strength', 'jerk'):
                loss = loss + self.metadata[key + '_weight'] * losses[key]
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite timing-attention loss')
        for key, value in dict(loss=loss, trajectory=output['trajectory_loss'], **losses,
                               enabled_rate=enabled.float().mean(), active_rate=stats['active_count']/len(enabled)).items():
            self.log('train/'+key, value, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(enabled))
        return loss

    def on_validation_epoch_start(self):
        self._stats = {}

    def validation_step(self, batch, batch_idx):
        context, targets = batch
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(100000 + batch_idx + self.global_rank * 1000000)
            output, poses = self._forward(context)
        _, enabled = history_validity(context['timing_history'])
        _, stats = timing_strength_loss(poses, targets, self.generator._trajectory_head.plan_anchor,
                                        enabled, context['timing_ego_speed'])
        distances = torch.linalg.vector_norm(poses[..., :2] - targets['trajectory'][:, None, :, :2], dim=-1).mean(-1)
        stats['oracle_ade_sum'] = distances.min(-1).values.sum()
        stats['enabled_count'] = enabled.float().sum()
        ratios = torch.stack([x.timing_adapter.last_ratio for x in self.generator._trajectory_head.diff_decoder.layers])
        stats['residual_ratio_sum'] = ratios.mean((0, 2)).sum()
        for key, value in stats.items():
            self._stats[key] = self._stats.get(key, 0) + value.detach().double()

    def on_validation_epoch_end(self):
        for key in sorted(self._stats):
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(self._stats[key])
        count = self._stats['active_count'].clamp_min(1.)
        scenes = self._stats['scene_count'].clamp_min(1.)
        for key, value in self._stats.items():
            if key in ('oracle_ade_sum', 'residual_ratio_sum'):
                value = value / scenes
            elif key.endswith('_sum') or key == 'late_count':
                value = value / count
            self.log('val/'+key.replace('_sum', ''), value, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['timing_attention_metadata'] = self.metadata
