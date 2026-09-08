"""Compact, deterministic GTRS augmentation for PCS candidate caches."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .common import implementation_hashes, load_torch, sha256, token_seed
from .data import entry_path
from .model import METRIC_NAMES, combine_subscores


GTRS_KEYS = {
    "no_at_fault_collisions": "no_at_fault_collisions",
    "drivable_area_compliance": "drivable_area_compliance",
    "ego_progress": "ego_progress",
    "time_to_collision_within_bound": "time_to_collision_within_bound",
    "comfort": "history_comfort",
}


def load_vocabulary(path, num_poses=8):
    vocabulary = np.load(path, mmap_mode="r")
    if vocabulary.ndim != 3 or vocabulary.shape[-1] != 3:
        raise ValueError(f"Expected GTRS [N,T,3], got {vocabulary.shape}")
    if vocabulary.shape[1] == num_poses * 5:
        vocabulary = vocabulary[:, ::5]
    if vocabulary.shape[1:] != (num_poses, 3):
        raise ValueError(f"Expected {num_poses} GTRS poses, got {vocabulary.shape}")
    if not np.isfinite(vocabulary[: min(len(vocabulary), 32)]).all():
        raise ValueError("Nonfinite GTRS vocabulary sample")
    return vocabulary


def extract_labels(record, expected_count):
    columns = []
    for name in METRIC_NAMES:
        key = GTRS_KEYS[name]
        if key not in record:
            raise ValueError(f"GTRS PDM record lacks {key}")
        value = np.asarray(record[key], dtype=np.float32)
        if value.size != expected_count:
            raise ValueError(f"Unexpected {key} shape: {value.shape}")
        columns.append(value.reshape(expected_count))
    labels = np.stack(columns, axis=-1)
    if not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
        raise ValueError("GTRS PDM labels must be finite and in [0, 1]")
    return labels


def stratified_indices(labels, scores, base_score, count, seed):
    """Prefer catastrophic, near-base, and positive examples; fill without duplicates."""
    if count <= 0 or count > len(scores):
        raise ValueError("sample count must be in [1, vocabulary size]")
    delta = scores - base_score
    catastrophic = (
        (labels[:, 0] <= 0) | (labels[:, 1] <= 0) | (labels[:, 3] <= 0)
        | (delta <= -0.25)
    )
    buckets = [
        (np.flatnonzero(catastrophic), count // 2),
        (np.flatnonzero((~catastrophic) & (np.abs(delta) <= 0.05)), count // 4),
        (np.flatnonzero((~catastrophic) & (delta > 0.05)), count // 8),
    ]
    rng = np.random.default_rng(seed)
    selected = []
    used = np.zeros(len(scores), dtype=bool)
    for candidates, requested in buckets:
        candidates = candidates.copy()
        rng.shuffle(candidates)
        chosen = candidates[:requested]
        selected.extend(chosen.tolist())
        used[chosen] = True
    remaining = np.flatnonzero(~used)
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)].tolist())
    result = np.asarray(selected, dtype=np.uint16)
    if len(result) != count or len(np.unique(result)) != count:
        raise ValueError("GTRS sampling failed to produce unique fixed-size indices")
    rng.shuffle(result)
    return result


def prepare_compact_cache(candidate_root, pdm_path, vocabulary_path, output, count=32, seed=0):
    candidate_root = Path(candidate_root)
    pdm_path = Path(pdm_path)
    vocabulary_path = Path(vocabulary_path)
    output = Path(output)
    candidate_manifest = json.loads(
        (candidate_root / "manifest.json").read_text(encoding="utf-8")
    )
    if candidate_manifest["provenance"]["implementation_sha256"] != implementation_hashes():
        raise ValueError("Current frozen-generator implementation differs from candidate cache")
    records = [
        record for record in json.loads((candidate_root / "records.json").read_text(encoding="utf-8"))
        if record["split"] == "train"
    ]
    vocabulary = load_vocabulary(vocabulary_path, candidate_manifest["settings"]["num_poses"])
    if len(vocabulary) > np.iinfo(np.uint16).max:
        raise ValueError("GTRS vocabulary is too large for compact uint16 indices")
    if not 0 < count <= len(vocabulary):
        raise ValueError("GTRS sample count must be in [1, vocabulary size]")

    compact_manifest = {
        "schema": "pcs_gtrs_stratified_v1",
        "candidate_provenance": candidate_manifest["provenance"],
        "candidate_root": str(candidate_root.resolve()),
        "pdm_path": str(pdm_path.resolve()),
        "pdm_size": pdm_path.stat().st_size,
        "vocabulary_path": str(vocabulary_path.resolve()),
        "vocabulary_sha256": sha256(vocabulary_path),
        "vocabulary_shape": list(vocabulary.shape),
        "sample_count": count,
        "seed": seed,
        "training_scenes": len(records),
    }
    token_list = [record["token"] for record in records]
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing != compact_manifest:
            raise ValueError("Existing GTRS cache configuration differs; use a new directory")
        if json.loads((output / "tokens.json").read_text(encoding="utf-8")) != token_list:
            raise ValueError("Existing GTRS cache token order differs")
        indices = np.load(output / "indices.npy", mmap_mode="r+")
        labels_out = np.load(output / "labels.npy", mmap_mode="r+")
        scores_out = np.load(output / "scores.npy", mmap_mode="r+")
        completed = np.load(output / "completed.npy", mmap_mode="r+")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "tokens.json").write_text(json.dumps(token_list), encoding="utf-8")
        (output / "manifest.json").write_text(
            json.dumps(compact_manifest, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        indices = np.lib.format.open_memmap(
            output / "indices.npy", mode="w+", dtype=np.uint16, shape=(len(records), count),
        )
        labels_out = np.lib.format.open_memmap(
            output / "labels.npy", mode="w+", dtype=np.float16,
            shape=(len(records), count, len(METRIC_NAMES)),
        )
        scores_out = np.lib.format.open_memmap(
            output / "scores.npy", mode="w+", dtype=np.float16, shape=(len(records), count),
        )
        completed = np.lib.format.open_memmap(
            output / "completed.npy", mode="w+", dtype=np.bool_, shape=(len(records),),
        )
        completed[:] = False

    expected_shapes = (
        (indices.shape, (len(records), count)),
        (labels_out.shape, (len(records), count, len(METRIC_NAMES))),
        (scores_out.shape, (len(records), count)),
        (completed.shape, (len(records),)),
    )
    if any(actual != expected for actual, expected in expected_shapes):
        raise ValueError("Existing GTRS compact arrays have unexpected shapes")
    pending = np.flatnonzero(~np.asarray(completed))
    print(f"Compact GTRS cache: {len(pending)} pending of {len(records)}", flush=True)
    if not len(pending):
        print(json.dumps(compact_manifest, indent=2), flush=True)
        return

    print(f"Loading GTRS PDM labels once from {pdm_path}", flush=True)
    pdm = joblib.load(pdm_path, mmap_mode="r")
    missing = [records[row]["token"] for row in pending if records[row]["token"] not in pdm]
    if missing:
        raise ValueError(f"GTRS labels miss {len(missing)} training tokens; example {missing[:3]}")

    completed_batch = []
    for position, row in enumerate(tqdm(pending, desc="Compact GTRS cache"), 1):
        record = records[row]
        labels = extract_labels(pdm[record["token"]], len(vocabulary))
        scores = combine_subscores(torch.from_numpy(labels)).numpy()
        candidate = load_torch(entry_path(candidate_root, record))
        if (
            candidate["token"] != record["token"]
            or candidate["provenance"] != candidate_manifest["provenance"]
        ):
            raise ValueError("Candidate entry identity/provenance mismatch")
        base_mode = int(candidate["context"]["base_logits"].argmax())
        base_score = float(candidate["scores"][base_mode])
        chosen = stratified_indices(
            labels, scores, base_score, count, token_seed(record["token"], seed),
        )
        indices[row] = chosen
        labels_out[row] = labels[chosen].astype(np.float16)
        scores_out[row] = scores[chosen].astype(np.float16)
        completed_batch.append(row)
        if position % 512 == 0:
            indices.flush()
            labels_out.flush()
            scores_out.flush()
            completed[completed_batch] = True
            completed.flush()
            completed_batch = []
    indices.flush()
    labels_out.flush()
    scores_out.flush()
    completed[completed_batch] = True
    completed.flush()
    print(json.dumps(compact_manifest, indent=2), flush=True)


class GTRSAugmentedDataset(Dataset):
    """Append compact GTRS samples while preserving the original selector base."""

    def __init__(self, base, root):
        self.base = base
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] != "pcs_gtrs_stratified_v1":
            raise ValueError("Unsupported compact GTRS cache")
        if manifest["candidate_provenance"] != base.provenance:
            raise ValueError("GTRS cache and candidate cache provenance differ")
        tokens = json.loads((self.root / "tokens.json").read_text(encoding="utf-8"))
        if tokens != [record["token"] for record in base.records]:
            raise ValueError("GTRS cache token order differs from training dataset")
        self.indices = np.load(self.root / "indices.npy", mmap_mode="r")
        self.labels = np.load(self.root / "labels.npy", mmap_mode="r")
        self.scores = np.load(self.root / "scores.npy", mmap_mode="r")
        completed = np.load(self.root / "completed.npy", mmap_mode="r")
        if completed.shape != (len(base),) or not np.asarray(completed).all():
            raise ValueError("GTRS compact cache is incomplete")
        self.vocabulary = load_vocabulary(
            manifest["vocabulary_path"], base.manifest["settings"]["num_poses"],
        )
        if self.indices.shape[0] != len(base):
            raise ValueError("GTRS compact cache length differs from training dataset")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        chosen = np.asarray(self.indices[index], dtype=np.int64)
        proposals = torch.tensor(np.asarray(self.vocabulary[chosen]), dtype=torch.float32)
        labels = torch.tensor(np.asarray(self.labels[index]), dtype=torch.float32)
        scores = torch.tensor(np.asarray(self.scores[index]), dtype=torch.float32)
        context = dict(item["context"])
        context["proposals"] = torch.cat([context["proposals"], proposals], dim=0)
        extra_logits = torch.full(
            (len(chosen),), torch.finfo(context["base_logits"].dtype).min,
            dtype=context["base_logits"].dtype,
        )
        context["base_logits"] = torch.cat([context["base_logits"], extra_logits], dim=0)
        return {
            **item,
            "context": context,
            "labels": torch.cat([item["labels"].float(), labels], dim=0),
            "scores": torch.cat([item["scores"].float(), scores], dim=0),
        }
