"""Join immutable compact PCS features with scored timing-action labels."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from navsim.agents.diffusiondrive.cost_rank.data import CompactDataset
from navsim.agents.diffusiondrive.pcs.common import load_torch
from navsim.agents.diffusiondrive.timing_oracle.geometry import ACTIONS

BLOCK = 128


class TimingSelectorDataset(Dataset):
    def __init__(self, features, oracle, split):
        self.compact = CompactDataset(features, split)
        self.split = split
        root = Path(oracle)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] != "bidirectional_timing_oracle_v1":
            raise ValueError("Unexpected timing Oracle schema")
        if manifest["split"] != split or manifest["actions"] != [list(x) for x in ACTIONS]:
            raise ValueError("Timing Oracle split/action mismatch")
        records = json.loads((root / f"{split}_records.json").read_text(encoding="utf-8"))
        if [r["token"] for r in records] != [r["token"] for r in self.compact.records]:
            raise ValueError("Feature and timing Oracle record mismatch")
        pieces = []
        for start in range(0, len(records), BLOCK):
            path = root / split / f"block_{start // BLOCK:05d}.pt"
            if not path.is_file():
                raise ValueError(f"Missing timing Oracle block: {path}")
            item = load_torch(path)
            if item["tokens"] != [r["token"] for r in records[start:start + BLOCK]]:
                raise ValueError(f"Timing Oracle token mismatch: {path}")
            pieces.append(item)
        self.oracle = {key: torch.cat([piece[key] for piece in pieces])
                       for key in ("mode", "labels", "scores", "direction")}
        if len(self.oracle["mode"]) != len(self.compact):
            raise ValueError("Incomplete timing Oracle cache")

    def __len__(self):
        return len(self.compact)

    def __getitem__(self, index):
        source = self.compact[index]
        mode = int(self.oracle["mode"][index])
        base_probability = source["base_logits"].float().softmax(-1)[mode]
        return {
            "index": torch.tensor(index),
            "feature": source["features"][mode],
            "subscores": source["subscores"][mode],
            "pcs_score": source["pcs_scores"][mode],
            "base_probability": base_probability,
            "proposal": source["proposals"][mode],
            "labels": self.oracle["labels"][index].float(),
            "scores": self.oracle["scores"][index].float(),
            "direction": self.oracle["direction"][index].float(),
        }
