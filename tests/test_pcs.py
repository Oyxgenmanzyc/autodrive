"""Run on navhigh: python -m unittest discover -s tests -p test_pcs.py"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from navsim.agents.diffusiondrive.pcs.model import PDMCSHead, combine_subscores, select_candidates
from navsim.agents.diffusiondrive.pcs.scoring import pairwise_subscores
from navsim.agents.diffusiondrive.pcs.common import token_seed, write_new_json, load_scorer
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset


class PCSScoringTests(unittest.TestCase):
    def test_progress_uses_individual_reference_and_keeps_soft_collision_label(self):
        # Reference progresses 10m, candidates 5m/20m/8m. Third has NC=0.5.
        fake = SimpleNamespace(
            _multi_metrics=np.array([[1., 1., 1., .5], [1., 1., 1., 1.]]),
            _weighted_metrics=np.ones((4, 4)),
            _progress_raw=np.array([10., 5., 20., 8.]),
            _config=SimpleNamespace(
                progress_distance_threshold=5., weighted_metrics_array=np.array([5., 5., 2., 0.]),
            ),
        )
        original = fake._weighted_metrics.copy()
        labels, scores, _ = pairwise_subscores(fake)
        np.testing.assert_allclose(labels[:, 2], [.5, 1., .4])
        self.assertEqual(labels[2, 0], .5)
        np.testing.assert_array_equal(fake._weighted_metrics, original)
        expected = labels[:, 0] * labels[:, 1] * (5*labels[:, 2] + 5*labels[:, 3] + 2*labels[:, 4]) / 12
        np.testing.assert_allclose(scores, expected)

    def test_stopped_reference_and_zero_gate_no_nan(self):
        fake = SimpleNamespace(
            _multi_metrics=np.array([[1., 0., 1.], [1., 1., 1.]]),
            _weighted_metrics=np.ones((4, 3)),
            _progress_raw=np.zeros(3),
            _config=SimpleNamespace(
                progress_distance_threshold=5., weighted_metrics_array=np.array([5., 5., 2., 0.]),
            ),
        )
        labels, scores, _ = pairwise_subscores(fake)
        np.testing.assert_array_equal(labels[:, 2], [0., 1.])
        np.testing.assert_array_equal(scores, [0., 1.])

    def test_selection_preserves_coordinates_and_original_counterfactual(self):
        proposals = torch.randn(2, 67, 8, 3)
        context = {"proposals": proposals, "base_logits": torch.zeros(2, 67)}
        context["base_logits"][:, 12] = 1
        scores = torch.zeros(2, 67)
        scores[:, 56] = 2
        result = select_candidates(context, scores)
        self.assertTrue(torch.equal(result["trajectory"], proposals[:, 56]))
        self.assertTrue(torch.equal(result["selector_trajectory"], proposals[:, 12]))


class PCSNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def context(self, modes):
        return {
            "proposals": torch.randn(1, modes, 8, 3, requires_grad=True),
            "base_logits": torch.randn(1, modes),
            "bev": torch.randn(1, 256, 8, 8, requires_grad=True),
            "agents": torch.randn(1, 30, 256, requires_grad=True),
            "ego": torch.randn(1, 1, 256, requires_grad=True),
        }

    def test_k67_gradients_do_not_reach_generator_and_labels_are_unchanged(self):
        head = PDMCSHead()
        context = self.context(67)
        output = head(context)
        self.assertEqual(tuple(output["scores"].shape), (1, 67))
        labels = torch.rand(1, 67, 5)
        labels[:, :, 0] = .5
        before = labels.clone()
        head.loss(output, labels).backward()
        self.assertTrue(torch.equal(before, labels))
        for key in ("proposals", "bev", "agents", "ego"):
            self.assertIsNone(context[key].grad)
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in head.parameters()))
        self.assertTrue(torch.isfinite(output["scores"]).all())

    def test_evaluation_is_equivariant_to_candidate_permutation(self):
        head = PDMCSHead().eval()
        context = self.context(7)
        order = torch.tensor([6, 2, 1, 4, 0, 5, 3])
        permuted = dict(context)
        for key in ("proposals", "base_logits"):
            permuted[key] = context[key][:, order]
        with torch.no_grad():
            original = head(context)["scores"]
            reordered = head(permuted)["scores"]
        torch.testing.assert_close(reordered, original[:, order], rtol=2e-5, atol=2e-6)

    def test_checkpoint_roundtrip_has_identical_scores(self):
        head = PDMCSHead().eval()
        context = self.context(3)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.ckpt"
            torch.save({
                "pcs_metadata": {"provenance": {"seed": 0}, "settings": head.settings},
                "state_dict": {"head." + k: v for k, v in head.state_dict().items()},
            }, path)
            loaded, _ = load_scorer(path, "cpu", {"seed": 0})
            with torch.no_grad():
                torch.testing.assert_close(head(context)["scores"], loaded(context)["scores"])
            with self.assertRaises(ValueError):
                load_scorer(path, "cpu", {"seed": 1})


class PCSProvenanceTests(unittest.TestCase):
    def test_navtest_cache_is_rejected_for_training(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "manifest.json").write_text(json.dumps({"dataset": "navtest"}))
            with self.assertRaisesRegex(ValueError, "never navtest"):
                CandidateDataset(temp, "train")

    def test_conflicting_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            write_new_json(path, {"seed": 0})
            write_new_json(path, {"seed": 0})
            with self.assertRaises(ValueError):
                write_new_json(path, {"seed": 1})
            self.assertEqual(json.loads(path.read_text()), {"seed": 0})

    def test_sampling_seed_depends_on_token_not_order(self):
        tokens = ["scenario-a", "scenario-b", "scenario-c"]
        forward = {t: token_seed(t, 0) for t in tokens}
        backward = {t: token_seed(t, 0) for t in reversed(tokens)}
        self.assertEqual(forward, backward)
        self.assertNotEqual(token_seed(tokens[0], 0), token_seed(tokens[0], 1))


if __name__ == "__main__":
    unittest.main()

