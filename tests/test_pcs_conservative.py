"""Run on navhigh: python -m unittest discover -s tests -p 'test_pcs*.py'"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from navsim.agents.diffusiondrive.pcs.conservative import (
    ConservativeAdvantageScorer, initialize_from_pcs, scorer_output, select_candidates,
)
from navsim.agents.diffusiondrive.pcs.model import PDMCSHead
from navsim.agents.diffusiondrive.pcs.gtrs import (
    GTRSAugmentedDataset, load_vocabulary, stratified_indices,
)


class ConservativeDecisionTests(unittest.TestCase):
    def test_switch_requires_gain_win_and_low_risk(self):
        proposals = torch.randn(1, 3, 8, 3)
        context = {
            "proposals": proposals,
            "base_logits": torch.tensor([[2.0, 0.0, 0.0]]),
        }
        output = {
            "utility": torch.tensor([[0.0, 0.2, 0.1]]),
            "predicted_delta": torch.tensor([[0.0, 0.2, 0.1]]),
            "win_probability": torch.tensor([[0.5, 0.8, 0.7]]),
            "catastrophic_risk": torch.tensor([[0.0, 0.05, 0.05]]),
            "thresholds": {
                "min_delta": 0.01,
                "min_win_probability": 0.55,
                "max_catastrophic_risk": 0.10,
            },
        }
        selected = select_candidates(context, output)
        self.assertEqual(int(selected["raw_selected_mode"]), 1)
        self.assertEqual(int(selected["selected_mode"]), 1)
        output["catastrophic_risk"][0, 1] = 0.2
        selected = select_candidates(context, output)
        self.assertEqual(int(selected["selected_mode"]), 0)
        self.assertFalse(bool(selected["switched"]))

    def test_multitask_losses_are_finite_and_backpropagate(self):
        batch, modes = 2, 4
        context = {"base_logits": torch.zeros(batch, modes)}
        labels = torch.ones(batch, modes, 5)
        labels[:, 1, 1] = 0
        labels[:, 2, 3] = 0
        scores = labels[..., 0] * labels[..., 1] * (
            5 * labels[..., 2] + 5 * labels[..., 3] + 2 * labels[..., 4]
        ) / 12
        logits = torch.zeros(batch, modes, 5, requires_grad=True)
        delta = torch.zeros(batch, modes, requires_grad=True)
        win = torch.zeros(batch, modes, requires_grad=True)
        risk = torch.zeros(batch, modes, requires_grad=True)
        output = {
            "logits": logits,
            "predicted_delta": delta,
            "win_logits": win,
            "catastrophic_logits": risk,
            "utility": delta - risk.sigmoid() * 0.5,
        }
        losses = ConservativeAdvantageScorer.losses(output, context, labels, scores)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["loss"].backward()
        self.assertTrue(all(value.grad is not None for value in (logits, delta, win, risk)))


class GTRSAugmentationTests(unittest.TestCase):
    def test_stratified_sampling_is_deterministic_and_keeps_hard_failures(self):
        labels = np.ones((32, 5), dtype=np.float32)
        labels[:10, 1] = 0
        scores = labels[:, 0] * labels[:, 1] * (
            5 * labels[:, 2] + 5 * labels[:, 3] + 2 * labels[:, 4]
        ) / 12
        first = stratified_indices(labels, scores, 1.0, 16, seed=7)
        second = stratified_indices(labels, scores, 1.0, 16, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 16)
        self.assertGreaterEqual(np.isin(first, np.arange(10)).sum(), 8)

    def test_vocabulary_downsamples_40_steps_to_eight(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vocab.npy"
            np.save(path, np.zeros((4, 40, 3), dtype=np.float32))
            self.assertEqual(load_vocabulary(path).shape, (4, 8, 3))

    def test_dataset_appends_gtrs_without_changing_original_base(self):
        class FakeBase:
            provenance = {"seed": 0}
            records = [{"token": "token-a"}]
            manifest = {"settings": {"num_poses": 8}}

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {
                    "context": {
                        "proposals": torch.zeros(2, 8, 3),
                        "base_logits": torch.tensor([0.0, 2.0]),
                        "bev": torch.zeros(256, 2, 2),
                        "agents": torch.zeros(2, 256),
                        "ego": torch.zeros(1, 256),
                    },
                    "labels": torch.ones(2, 5),
                    "scores": torch.ones(2),
                    "index": 0,
                }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vocab = root / "vocab.npy"
            np.save(vocab, np.ones((4, 8, 3), dtype=np.float32))
            np.save(root / "indices.npy", np.array([[1, 3]], dtype=np.uint16))
            np.save(root / "labels.npy", np.ones((1, 2, 5), dtype=np.float16))
            np.save(root / "scores.npy", np.ones((1, 2), dtype=np.float16))
            np.save(root / "completed.npy", np.ones(1, dtype=np.bool_))
            (root / "tokens.json").write_text('["token-a"]', encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "schema": "pcs_gtrs_stratified_v1",
                "candidate_provenance": {"seed": 0},
                "vocabulary_path": str(vocab),
            }), encoding="utf-8")
            dataset = GTRSAugmentedDataset(FakeBase(), root)
            item = dataset[0]
            self.assertEqual(item["context"]["proposals"].shape, (4, 8, 3))
            self.assertEqual(item["labels"].shape, (4, 5))
            self.assertEqual(int(item["context"]["base_logits"].argmax()), 1)


class ConservativeNetworkTests(unittest.TestCase):
    def test_original_pcs_weights_initialize_only_shared_layers(self):
        original = PDMCSHead()
        target = ConservativeAdvantageScorer()
        before = {
            key: value.clone() for key, value in target.decision_head.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pcs.ckpt"
            torch.save({
                "pcs_metadata": {"provenance": {"seed": 0}},
                "state_dict": {
                    "head." + key: value for key, value in original.state_dict().items()
                },
            }, path)
            initialize_from_pcs(target, path, {"seed": 0})
        for key, value in original.encoder.state_dict().items():
            torch.testing.assert_close(target.encoder.state_dict()[key], value)
        for key, value in original.heads.state_dict().items():
            torch.testing.assert_close(target.metric_heads.state_dict()[key], value)
        for key, value in before.items():
            torch.testing.assert_close(target.decision_head.state_dict()[key], value)

    def test_forward_shapes_and_context_gradients_are_isolated(self):
        torch.set_num_threads(2)
        model = ConservativeAdvantageScorer()
        context = {
            "proposals": torch.randn(1, 5, 8, 3, requires_grad=True),
            "base_logits": torch.randn(1, 5),
            "bev": torch.randn(1, 256, 8, 8, requires_grad=True),
            "agents": torch.randn(1, 30, 256, requires_grad=True),
            "ego": torch.randn(1, 1, 256, requires_grad=True),
        }
        output = scorer_output(model, context)
        self.assertEqual(output["scores"].shape, (1, 5))
        self.assertEqual(output["predicted_delta"].shape, (1, 5))
        output["utility"].sum().backward()
        for key in ("proposals", "bev", "agents", "ego"):
            self.assertIsNone(context[key].grad)


if __name__ == "__main__":
    unittest.main()
