"""Small resumable cache of the decisions made by the frozen PCS proposer."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from .common import load_scorer, sha256
from .data import CandidateDataset


SCHEMA = "pcs_decision_pairs_v1"


def _array_path(root, split, name):
    return Path(root) / f"{split}_{name}.npy"


def _open_arrays(root, split, size, create):
    specs = {
        "pcs_mode": (np.uint8, (size,)),
        "base_mode": (np.uint8, (size,)),
        "pcs_score": (np.float32, (size,)),
        "base_score": (np.float32, (size,)),
        "pcs_labels": (np.float16, (size, 5)),
        "base_labels": (np.float16, (size, 5)),
        "completed": (np.bool_, (size,)),
    }
    arrays = {}
    for name, (dtype, shape) in specs.items():
        path = _array_path(root, split, name)
        arrays[name] = (
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            if create else np.load(path, mmap_mode="r+")
        )
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Unexpected pair cache array: {path}")
    if create:
        arrays["completed"][:] = False
        arrays["completed"].flush()
    return arrays


def prepare_pair_cache(cache, scorer_path, output, workers=4, batch_size=32, smoke=False):
    from navsim.planning.script.run_pcs import loader

    root = Path(output)
    datasets = {
        split: CandidateDataset(cache, split, allow_partial=smoke)
        for split in ("train", "val")
    }
    provenance = datasets["train"].provenance
    if datasets["val"].provenance != provenance:
        raise ValueError("Train and validation candidate provenance differs")
    scorer_hash = sha256(scorer_path)
    manifest = {
        "schema": SCHEMA,
        "candidate_provenance": provenance,
        "candidate_root": str(Path(cache).resolve()),
        "pcs_scorer_sha256": scorer_hash,
        "splits": {split: len(dataset) for split, dataset in datasets.items()},
    }
    create = not root.exists()
    if create:
        root.mkdir(parents=True, exist_ok=False)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        for split, dataset in datasets.items():
            (root / f"{split}_tokens.json").write_text(
                json.dumps([record["token"] for record in dataset.records]), encoding="utf-8",
            )
    else:
        existing = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing decision-pair cache has different provenance")

    if not torch.cuda.is_available():
        raise RuntimeError("Decision-pair preparation requires one visible GPU")
    scorer, _ = load_scorer(scorer_path, "cuda:0", provenance)
    all_summary = {}
    for split, dataset in datasets.items():
        tokens = json.loads((root / f"{split}_tokens.json").read_text(encoding="utf-8"))
        if tokens != [record["token"] for record in dataset.records]:
            raise ValueError(f"{split} pair-cache token order differs")
        split_create = not _array_path(root, split, "pcs_mode").exists()
        arrays = _open_arrays(root, split, len(dataset), split_create)
        pending = np.flatnonzero(~np.asarray(arrays["completed"]))
        print(f"{split}: {len(pending)} pending of {len(dataset)}", flush=True)
        pending_dataset = Subset(dataset, pending.tolist())
        for batch in tqdm(
            loader(pending_dataset, workers, batch_size=batch_size, shuffle=False, pin_memory=True),
            desc=f"PCS decision pairs/{split}",
        ):
            context = {key: value.cuda(non_blocking=True) for key, value in batch["context"].items()}
            with torch.no_grad():
                scorer_output = scorer(context)
                pcs_mode = scorer_output["scores"].argmax(-1)
                base_mode = context["base_logits"].argmax(-1)
            rows = torch.arange(len(pcs_mode))
            original_index = batch["index"].long()
            numpy_index = original_index.numpy()
            scores = batch["scores"]
            labels = batch["labels"]
            arrays["pcs_mode"][numpy_index] = pcs_mode.cpu().numpy().astype(np.uint8)
            arrays["base_mode"][numpy_index] = base_mode.cpu().numpy().astype(np.uint8)
            arrays["pcs_score"][numpy_index] = scores[rows, pcs_mode.cpu()].numpy()
            arrays["base_score"][numpy_index] = scores[rows, base_mode.cpu()].numpy()
            arrays["pcs_labels"][numpy_index] = labels[rows, pcs_mode.cpu()].numpy()
            arrays["base_labels"][numpy_index] = labels[rows, base_mode.cpu()].numpy()
            for name in arrays:
                if name != "completed":
                    arrays[name].flush()
            arrays["completed"][numpy_index] = True
            arrays["completed"].flush()
        if not np.asarray(arrays["completed"]).all():
            raise ValueError(f"Incomplete {split} decision-pair cache")
        gain = np.asarray(arrays["pcs_score"]) - np.asarray(arrays["base_score"])
        pcs_labels = np.asarray(arrays["pcs_labels"])
        base_labels = np.asarray(arrays["base_labels"])
        degradation = base_labels[:, (0, 1, 3)] - pcs_labels[:, (0, 1, 3)]
        changed = np.asarray(arrays["pcs_mode"]) != np.asarray(arrays["base_mode"])
        summary = {
            "scenes": len(dataset),
            "pcs_pdm": float(np.asarray(arrays["pcs_score"]).mean()),
            "base_pdm": float(np.asarray(arrays["base_score"]).mean()),
            "mean_gain": float(gain.mean()),
            "changed_count": int(changed.sum()),
            "hard_negative_count": int((changed & (gain < 0)).sum()),
            "severe_hard_negative_count": int((changed & (gain <= -0.25)).sum()),
            "new_zero_count": int(((arrays["base_score"] > 0) & (arrays["pcs_score"] == 0)).sum()),
            "relative_risk_counts": {
                name: int((degradation[:, index] > 0).sum())
                for index, name in enumerate(("nc", "dac", "ttc"))
            },
        }
        all_summary[split] = summary
        print(json.dumps({split: summary}, indent=2), flush=True)
    (root / "summary.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )


class DecisionPairDataset(Dataset):
    def __init__(self, cache, pair_cache, split, allow_partial=False):
        self.base = CandidateDataset(cache, split, allow_partial=allow_partial)
        self.root = Path(pair_cache)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported decision-pair cache")
        if manifest["candidate_provenance"] != self.base.provenance:
            raise ValueError("Decision-pair and candidate-cache provenance differ")
        tokens = json.loads((self.root / f"{split}_tokens.json").read_text(encoding="utf-8"))
        if tokens != [record["token"] for record in self.base.records]:
            raise ValueError("Decision-pair token order differs")
        self.pcs_mode = np.load(_array_path(self.root, split, "pcs_mode"), mmap_mode="r")
        self.base_mode = np.load(_array_path(self.root, split, "base_mode"), mmap_mode="r")
        completed = np.load(_array_path(self.root, split, "completed"), mmap_mode="r")
        if len(self.pcs_mode) != len(self.base) or not np.asarray(completed).all():
            raise ValueError("Decision-pair cache is incomplete")
        self.manifest = manifest
        self.provenance = self.base.provenance

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        pcs_mode = int(self.pcs_mode[index])
        base_mode = int(self.base_mode[index])
        if base_mode != int(item["context"]["base_logits"].argmax()):
            raise ValueError("Cached base mode differs from candidate entry")
        return {**item, "pcs_mode": pcs_mode, "base_mode": base_mode}
