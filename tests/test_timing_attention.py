"""Run in navhigh: python -m unittest discover -s tests -p test_timing_attention.py."""
import copy
import unittest
import torch
from torch import nn
from navsim.agents.diffusiondrive.timing_attention.model import (
    TimingQueryEncoder, StepTimingAttention, TimingDecoderLayer, history_validity,
    install_attention, mutable_parameter, bind_timing_inputs,
)
from navsim.agents.diffusiondrive.timing_attention.loss import timing_strength_loss


def history(batch=2):
    data = torch.zeros(batch, 4, 12)
    data[..., 0] = 10.
    data[..., 1] = 4.
    data[..., 2] = 10.
    data[..., -1] = 1.
    return data


def trajectory(speeds):
    result = torch.zeros(1, len(speeds), 3)
    result[0, :, 0] = torch.tensor(speeds).cumsum(0) * .5
    return result


class TimingAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.query = TimingQueryEncoder(32)
        self.adapter = StepTimingAttention(32, 4)
        self.x = torch.randn(2, 3, 32)
        self.poses = trajectory([10.,9.,8.,7.,6.,5.,4.,3.])[:,None].repeat(2,3,1,1)
        self.speed = torch.tensor([10.,10.])
        self.h = history()

    def output(self):
        q, enabled = self.query(self.h, self.speed)
        return self.adapter(self.x, self.poses, q, enabled, self.speed)

    def test_zero_initialization_exact_identity(self):
        torch.testing.assert_close(self.output(), self.x, rtol=0, atol=0)

    def test_no_current_risk_returns_identity_even_after_learning(self):
        with torch.no_grad():
            self.adapter.output.weight.normal_()
            self.adapter.output.bias.fill_(.5)
        self.h[:, -1, -1] = 0
        torch.testing.assert_close(self.output(), self.x, rtol=0, atol=0)

    def test_residual_bound_after_large_parameter_update(self):
        with torch.no_grad():
            self.adapter.output.weight.normal_(0, 10)
        delta = self.output() - self.x
        ratio = delta.square().mean(-1).sqrt() / self.x.square().mean(-1).sqrt()
        self.assertTrue((ratio <= .100001).all())

    def test_braking_demand_changes_with_wait_time_and_closing_speed(self):
        d, _ = self.query.descriptors(self.h, self.speed)
        self.assertGreater(float(d[0,0,7]), float(d[0,-1,7]))  # TTC margin
        self.assertLess(float(d[0,0,8]), float(d[0,-1,8]))     # required decel
        other = self.h.clone()
        other[...,1] = 0
        quiet, _ = self.query.descriptors(other, self.speed)
        self.assertEqual(float(quiet[...,8].abs().sum()), 0.)
        self.assertFalse(torch.equal(d, quiet))

    def test_queries_have_physical_step_identity_and_bounded_units(self):
        d, _ = self.query.descriptors(self.h*1000, self.speed*1000)
        self.assertTrue(torch.isfinite(d).all())
        self.assertLessEqual(float(d.abs().max()), 2.)
        q, _ = self.query(self.h, self.speed)
        self.assertFalse(torch.equal(q[:,0],q[:,-1]))

    def test_candidates_do_not_mix_in_attention(self):
        with torch.no_grad():
            self.adapter.output.weight.normal_(0,.01)
        original = self.output()
        self.poses[:,1] += 7.
        changed = self.output()
        torch.testing.assert_close(original[:,0], changed[:,0], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[:,1], changed[:,1]))

    def test_query_and_attention_get_gradient_after_zero_output_opens(self):
        optimizer = torch.optim.SGD(list(self.adapter.parameters())+list(self.query.parameters()), lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            (self.output()-2*self.x).square().mean().backward()
            optimizer.step()
        self.assertGreater(float(self.query.embedding[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.adapter.attention.in_proj_weight.grad.abs().sum()), 0.)

    def test_invalid_nonfinite_history_does_not_inject(self):
        self.h[:,-1,0] = float('nan')
        self.assertFalse(history_validity(self.h)[1].any())
        torch.testing.assert_close(self.output(), self.x, rtol=0, atol=0)


class StrengthTests(unittest.TestCase):
    def test_strength_supervision_distinguishes_strong_braking(self):
        gt = trajectory([9.,8.,7.,6.,5.,4.,3.,2.])
        weak = trajectory([9.7,9.4,9.1,8.8,8.5,8.2,7.9,7.6])
        targets = {'trajectory': gt, 'brake_timing_context': torch.tensor([[10.,1.,1.,2.,8.]])}
        proposals = weak[:,None].repeat(1,2,1,1).requires_grad_(True)
        anchors = gt[...,:2].repeat(2,1,1)
        anchors[1] += 50
        losses, stats = timing_strength_loss(proposals, targets, anchors, torch.tensor([True]), torch.tensor([10.]))
        self.assertGreater(float(losses['strength']),0.)
        losses['strength'].backward()
        self.assertTrue(torch.isfinite(proposals.grad).all())
        self.assertGreater(float(proposals.grad[:,0].abs().sum()),0.)
        self.assertEqual(float(proposals.grad[:,1].abs().sum()),0.)
        exact, _ = timing_strength_loss(gt[:,None], targets, anchors[:1], torch.tensor([True]), torch.tensor([10.]))
        self.assertLess(float(exact['strength']),1e-8)

    def test_no_valid_input_has_no_auxiliary_gradient(self):
        gt = trajectory([9.,8.,7.,6.,5.,4.,3.,2.])
        pred = (gt[:,None]+.1).requires_grad_(True)
        targets = {'trajectory':gt, 'brake_timing_context':torch.tensor([[10.,1.,1.,2.,8.]])}
        losses, _ = timing_strength_loss(pred, targets, gt[...,:2], torch.tensor([False]), torch.tensor([10.]))
        sum(losses.values()).backward()
        self.assertEqual(float(pred.grad.abs().sum()),0.)


class DecoderTests(unittest.TestCase):
    def test_real_decoder_initial_identity_and_frozen_originals(self):
        # Use the real generator decoder with a deterministic BEV stand-in;
        # checkpoint/sensor-level equivalence is separately checked by check-init.
        from types import SimpleNamespace
        from navsim.agents.diffusiondrive.transfuser_model_v2 import CustomTransformerDecoderLayer
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        config = TransfuserConfig()
        layer = CustomTransformerDecoderLayer(8,256,1024,config).eval()
        class BEV(nn.Module):
            def forward(self, feature, *args):
                return feature
        layer.cross_bev_attention = BEV()
        layer.requires_grad_(False)
        modified = TimingDecoderLayer(copy.deepcopy(layer),.1).eval()
        args = (torch.randn(2,3,256), torch.randn(2,3,8,2), torch.zeros(2,256,8,8),
                (8,8), torch.randn(2,4,256), torch.randn(2,1,256), torch.randn(2,1,256), torch.zeros(2,1,256))
        modified.runtime_timing = (torch.randn(2,8,256), torch.ones(2,dtype=torch.bool), torch.ones(2)*10)
        old, logits = layer(*args)
        new, new_logits = modified(*args)
        torch.testing.assert_close(new,old,atol=0,rtol=0)
        torch.testing.assert_close(new_logits,logits,atol=0,rtol=0)
        before = {n:p.detach().clone() for n,p in modified.named_parameters() if not n.startswith('timing_adapter.')}
        optimizer = torch.optim.SGD([p for p in modified.parameters() if p.requires_grad],lr=.1)
        new.square().mean().backward()
        optimizer.step()
        for n,p in modified.named_parameters():
            if n in before:
                torch.testing.assert_close(p,before[n],atol=0,rtol=0)


class InputBoundaryTests(unittest.TestCase):
    def test_builder_needs_only_inference_inputs(self):
        import numpy as np
        from types import SimpleNamespace
        from navsim.agents.diffusiondrive.timing_attention.data import risk_input
        # No scene annotations, future boxes, future ego trajectory or labels exist.
        ego = [SimpleNamespace(ego_velocity=np.array([10.,0.]), ego_acceleration=np.array([0.,0.])) for _ in range(4)]
        clouds = []
        for distance in (24.,23.,22.,21.):
            points = np.zeros((6,10),dtype=np.float32)
            points[0,:] = distance
            points[2,:] = 1.
            clouds.append(SimpleNamespace(lidar_pc=points))
        tokens, speed = risk_input(SimpleNamespace(ego_statuses=ego,lidars=clouds))
        self.assertEqual(tokens.shape,(4,12))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(float(speed),10.)
        self.assertTrue(history_validity(tokens[None])[1].item())

    def test_input_cache_checks_token_order(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from navsim.agents.diffusiondrive.timing_attention.data import read_inputs, input_sources, SCHEMA
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{'token':'first'},{'token':'second'}]
            (root/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA,sources=input_sources(),records=records,target_sha256='gt')),encoding='utf-8')
            torch.save(dict(tokens=['second','first'],history=history(),ego_speed=torch.ones(2)),root/'block_0000.pt')
            with self.assertRaisesRegex(ValueError,'token order'):
                read_inputs(root,records,'gt')


if __name__ == '__main__':
    unittest.main()
