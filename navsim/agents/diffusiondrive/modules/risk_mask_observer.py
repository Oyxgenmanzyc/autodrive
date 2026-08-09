import atexit
import json
import os
from pathlib import Path

import torch


class RiskMaskObserver:
    def __init__(self):
        self.path = Path(
            os.environ.get(
                "RISK_MASK_OBSERVER_PATH",
                "/tmp/risk_mask_observer.jsonl",
            )
        )
        self.print_every = int(
            os.environ.get("RISK_MASK_OBSERVER_PRINT_EVERY", "100")
        )
        self.decel_threshold = float(
            os.environ.get("RISK_MASK_GT_DECEL_THRESHOLD", "-0.5")
        )

        self.calls = 0
        self.samples = 0
        self.gt_brake = 0
        self.gt_early_brake = 0
        self.scene_active = 0
        self.tp_brake = 0
        self.tp_early_brake = 0

        self.pair_scenes = 0
        self.pair_count = 0

        self.zero_pair_calls = 0
        self.zero_ranking_loss_calls = 0

        atexit.register(self.flush_final)

    @torch.no_grad()
    def update(
        self,
        targets,
        scene_active,
        pair_mask,
        ranking_loss,
    ):
        trajectory = targets.get("trajectory")

        if not torch.is_tensor(trajectory):
            return

        trajectory = trajectory.detach()
        scene_active = scene_active.detach().bool()
        pair_mask = pair_mask.detach().bool()

        xy = trajectory[..., :2]

        start = torch.zeros_like(xy[:, :1])
        xy = torch.cat([start, xy], dim=1)

        dt = 0.5

        speed = torch.linalg.norm(
            xy[:, 1:] - xy[:, :-1],
            dim=-1,
        ) / dt

        acceleration = (
            speed[:, 1:] - speed[:, :-1]
        ) / dt

        gt_brake = (
            acceleration <= self.decel_threshold
        ).any(dim=1)

        early_steps = min(2, acceleration.shape[1])

        gt_early_brake = (
            acceleration[:, :early_steps]
            <= self.decel_threshold
        ).any(dim=1)

        pair_count_per_scene = pair_mask.sum(dim=(1, 2))
        pair_scene = pair_count_per_scene > 0

        batch_size = trajectory.shape[0]

        self.calls += 1
        self.samples += batch_size

        self.gt_brake += int(gt_brake.sum())
        self.gt_early_brake += int(gt_early_brake.sum())

        self.scene_active += int(scene_active.sum())

        self.tp_brake += int(
            (scene_active & gt_brake).sum()
        )
        self.tp_early_brake += int(
            (scene_active & gt_early_brake).sum()
        )

        self.pair_scenes += int(pair_scene.sum())
        self.pair_count += int(pair_mask.sum())

        if not pair_mask.any():
            self.zero_pair_calls += 1

        if float(ranking_loss.detach().abs().mean()) <= 1e-12:
            self.zero_ranking_loss_calls += 1

        if (
            self.print_every > 0
            and self.calls % self.print_every == 0
        ):
            self.flush("periodic")

    @staticmethod
    def ratio(a, b):
        return None if b == 0 else a / b

    def snapshot(self, tag):
        return {
            "tag": tag,
            "calls": self.calls,
            "samples": self.samples,

            "gt_brake_ratio":
                self.ratio(self.gt_brake, self.samples),

            "gt_early_brake_ratio":
                self.ratio(self.gt_early_brake, self.samples),

            "scene_active_ratio":
                self.ratio(self.scene_active, self.samples),

            "scene_active_precision_gt_brake":
                self.ratio(self.tp_brake, self.scene_active),

            "scene_active_recall_gt_brake":
                self.ratio(self.tp_brake, self.gt_brake),

            "scene_active_precision_gt_early_brake":
                self.ratio(self.tp_early_brake, self.scene_active),

            "scene_active_recall_gt_early_brake":
                self.ratio(self.tp_early_brake, self.gt_early_brake),

            "valid_pair_scene_ratio":
                self.ratio(self.pair_scenes, self.samples),

            "pair_count": self.pair_count,

            "pairs_per_active_scene":
                self.ratio(self.pair_count, self.scene_active),

            "zero_pair_call_ratio":
                self.ratio(self.zero_pair_calls, self.calls),

            "zero_ranking_loss_call_ratio":
                self.ratio(
                    self.zero_ranking_loss_calls,
                    self.calls,
                ),
        }

    def flush(self, tag):
        if self.calls == 0:
            return

        data = self.snapshot(tag)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(data, ensure_ascii=False) + "\n"
            )

        print(
            "[RISK_MASK_OBSERVER]",
            json.dumps(data, ensure_ascii=False),
            flush=True,
        )

    def flush_final(self):
        self.flush("final")


RISK_MASK_OBSERVER = RiskMaskObserver()
