"""3.1.05_3: physical timing queries cross-attend to candidate trajectories; fixed PCS/TRV."""
import argparse
import csv
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

from navsim.agents.diffusiondrive.pcs.common import (
    build_generator, generate_context, load_torch, provenance, sha256, write_new_json,
)
from navsim.agents.diffusiondrive.pcs.veto import load_veto
from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.agents.diffusiondrive.timing_attention.data import prepare_inputs, AttentionDataset, AttentionFeatureSource
from navsim.agents.diffusiondrive.timing_attention.model import install_attention, mutable_parameter, bind_timing_inputs
from navsim.planning.script.run_pcs import loader, new_run, prepare_source


def experiment_sources():
    root = Path(__file__).resolve().parents[2] / "agents/diffusiondrive/timing_attention"
    files = list(root.glob("*.py")) + list((root.parent / "generator_timing").glob("*.py")) + [Path(__file__)]
    return {str(p.relative_to(Path(__file__).resolve().parents[3])): sha256(p) for p in files}


def checked_selector(args, device):
    identity = provenance(args.baseline, args.anchor, args.seed)
    veto, meta = load_veto(args.veto, device, identity)
    if meta["pcs_scorer_sha256"] != sha256(args.pcs_scorer):
        raise ValueError("PCS checkpoint hash differs from the fixed TRV proposer")
    veto.requires_grad_(False)
    return veto, meta, identity


def check_initialization(args):
    """Server audit: same K67 checkpoint, cached input, seed, and generated output."""
    from navsim.agents.diffusiondrive.timing_attention.model import history_validity
    dataset = AttentionDataset(args.targets, args.candidate_cache, args.timing_inputs, 'train', True)
    sample = next((i for i in range(len(dataset))
                   if history_validity(dataset.history[dataset.indices[i]][None])[1].item()), 0)
    context, _ = dataset[sample]
    device = 'cuda:0'
    model, _ = build_generator(args.baseline, args.backbone, args.anchor, device)
    context = {k:v[None].to(device) for k,v in context.items()}
    head = model._trajectory_head
    bev, ego, agents = (context[k].float() for k in ('bev','ego','agents'))
    arguments = (ego, agents, bev, bev.shape[-2:], ego.new_zeros((1,1,ego.shape[-1])))
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(args.seed)
        original = head.forward_test(*arguments, global_img=None)
        install_attention(model,args.residual_cap)
        model.to(device).eval()
        torch.manual_seed(args.seed)
        with bind_timing_inputs(model,context['timing_history'],context['timing_ego_speed']):
            current = head.forward_test(*arguments,global_img=None)
        for key in ('proposal_trajectory','proposal_logits'):
            torch.testing.assert_close(current[key],original[key],rtol=0,atol=0)
        # Disabling valid inputs must restore the original even after adapter edits.
        for layer in head.diff_decoder.layers:
            layer.timing_adapter.output.bias.fill_(1.)
        disabled = context['timing_history'].clone()
        disabled[..., -1] = 0
        torch.manual_seed(args.seed)
        with bind_timing_inputs(model,disabled,context['timing_ego_speed']):
            fallback = head.forward_test(*arguments,global_img=None)
        torch.testing.assert_close(fallback['proposal_trajectory'],original['proposal_trajectory'],rtol=0,atol=0)
    print('PASS: actual K67 zero-initialized trajectories/logits are exactly equal; invalid inputs restore original output.')


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from navsim.agents.diffusiondrive.timing_attention.training import TimingAttentionModule

    pl.seed_everything(args.seed, workers=True)
    # Verify selector lineage now, but do not put selector parameters in training.
    fixed, selector_meta, identity = checked_selector(args, "cpu")
    del fixed
    generator, _ = build_generator(args.baseline, args.backbone, args.anchor, "cpu")
    training = AttentionDataset(args.targets, args.candidate_cache, args.timing_inputs, "train", args.smoke)
    validation = AttentionDataset(args.targets, args.candidate_cache, args.timing_inputs, "val", args.smoke)
    if training.candidate_manifest["provenance"] != identity:
        raise ValueError("Frozen training contexts do not match the baseline/anchor/seed")
    if validation.candidate_manifest != training.candidate_manifest:
        raise ValueError("Train/val candidate cache manifests differ")
    from navsim.agents.diffusiondrive.timing_attention.model import history_validity
    from navsim.agents.diffusiondrive.generator_timing.risk_brake_timing import _brake_timing_observations
    from navsim.agents.diffusiondrive.generator_timing.data import timing_config
    ids = training.indices
    observation = _brake_timing_observations(training.trajectories[ids], training.trajectories[ids],
                                             training.contexts[ids], timing_config())
    _, enabled = history_validity(training.history[ids])
    supervised_count = int((observation['active'] & enabled).sum())
    print(f'Training scenes with both valid timing input and braking supervision: {supervised_count}/{len(training)}')
    if not args.smoke and supervised_count == 0:
        raise ValueError('No valid timing/strength supervision; inspect input cache before training')
    install_attention(generator, args.residual_cap)
    metadata = {
        "schema": "timing_query_cross_attention_v1", "arm": "timing_attention",
        "baseline_provenance": identity, "veto_sha256": sha256(args.veto),
        "pcs_sha256": sha256(args.pcs_scorer), "thresholds": selector_meta["thresholds"],
        "target_sha256": sha256(args.targets), "sources": experiment_sources(),
        "timing_weight": args.timing_weight, "strength_weight": args.strength_weight,
        "jerk_weight": args.jerk_weight, "residual_cap": args.residual_cap,
        "timing_inputs_sha256": sha256(Path(args.timing_inputs) / 'manifest.json'), "epochs": args.epochs, "lr": args.lr,
        "seed": args.seed, "global_batch_size": args.batch_size * args.devices,
        "precision": args.precision, "smoke": args.smoke,
        "candidate_cache_provenance": training.candidate_manifest["provenance"],
        "scope": "only_timing_query_and_attention; all_original_parameters_frozen",
        "timing_supervised_scenes": supervised_count,
    }
    if args.resume and load_torch(args.resume)["timing_attention_metadata"] != metadata:
        raise ValueError("Resume settings/source/selector/targets differ from the saved run")
    if "TIMING_ATTENTION_RUN_DIR" not in os.environ:
        os.environ["TIMING_ATTENTION_RUN_DIR"] = str(new_run(Path(args.output)))
    run = Path(os.environ["TIMING_ATTENTION_RUN_DIR"])
    module = TimingAttentionModule(generator, metadata)
    callback = ModelCheckpoint(
        dirpath=str(run / "checkpoints"), filename="epoch={epoch:02d}", auto_insert_metric_name=False,
        monitor="val/oracle_ade", mode="min", save_top_k=1, save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=args.epochs, precision=args.precision, gradient_clip_val=1.0,
        num_sanity_val_steps=0, log_every_n_steps=20,
        limit_train_batches=2 if args.smoke else 1.0, limit_val_batches=2 if args.smoke else 1.0,
        logger=CSVLogger(str(run), name="csv"), callbacks=[callback], default_root_dir=str(run),
    )
    if trainer.is_global_zero:
        write_new_json(run / "run.json", {
            "arguments": vars(args), "metadata": metadata,
            "train_scenes": len(training), "val_scenes": len(validation),
            "trainable_names": [n for n, p in generator.named_parameters() if p.requires_grad],
            "checkpoint_rule": "Use last.ckpt for the predeclared 10-epoch run; oracle-ADE best is diagnostic only",
        })
    trainer.fit(module,
                loader(training, args.workers, batch_size=args.batch_size, shuffle=True, pin_memory=True),
                loader(validation, args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True),
                ckpt_path=args.resume)
    if trainer.is_global_zero:
        print(f"Final timing-attention checkpoint: {callback.last_model_path}")
        print(f"Best oracle-ADE checkpoint (not best PDM): {callback.best_model_path}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def load_trained_generator(args, reference, identity, device):
    saved = load_torch(args.generator)
    meta = saved["timing_attention_metadata"]
    if (meta["schema"] != "timing_query_cross_attention_v1"
            or meta["baseline_provenance"] != identity
            or meta["veto_sha256"] != sha256(args.veto)
            or meta["pcs_sha256"] != sha256(args.pcs_scorer)
            or meta["sources"] != experiment_sources()):
        raise ValueError("Generator/selector/source lineage mismatch")
    if meta["smoke"] and not args.max_scenes:
        raise ValueError("Smoke-trained checkpoint cannot be evaluated as a full experiment")
    model, _ = build_generator(args.baseline, args.backbone, args.anchor, device)
    install_attention(model, meta["residual_cap"])
    model.to(device)
    state = {}
    for key, value in saved["state_dict"].items():
        if not key.startswith("generator."):
            raise ValueError(f"Unexpected generator checkpoint key: {key}")
        state[key[len("generator."):]] = value
    # Only generation parameters may differ, including zero changes for smoke.
    original = reference.state_dict()
    for name, value in state.items():
        mutable = mutable_parameter(name)
        if not mutable and (name not in original or not torch.equal(value.cpu(), original[name].cpu())):
            raise ValueError(f"Frozen perception/classifier/anchor changed: {name}")
    if not set(original).issubset(state):
        raise ValueError("Original generator tensors missing from checkpoint")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, meta


def evaluate(args):
    args.split, args.feature_cache = "navtest", None
    reference, config, source, paths, device = prepare_source(args)
    source = AttentionFeatureSource(config, args.data_root, args.max_scenes, args.seed)
    veto, selector_meta, identity = checked_selector(args, device)
    changed, training_meta = load_trained_generator(args, reference, identity, device)
    run = new_run(args.output)
    write_new_json(run / "run.json", {
        "arguments": vars(args), "training_metadata": training_meta,
        "baseline_provenance": identity, "generator_sha256": sha256(args.generator),
        "selector_metadata": selector_meta,
        "protocol": "same raw scene, token seed, anchors, fixed selector; all candidates regenerated for both generators",
    })
    rows, pending, saved_labels, saved_scores, saved_proposals = [], [], [], [], []
    stream = (run / "paired_results.csv").open("x", encoding="utf-8", newline="")
    writer = None

    def finish(job):
        nonlocal writer
        futures, record, decisions, proposals, timing_diagnostics = job
        row = {"token": record["token"], "log_name": record["log_name"], "valid": True}
        row.update(timing_diagnostics)
        all_labels, all_scores = [], []
        for prefix, future, decision in zip(("original", "new"), futures, decisions):
            labels, scores, direction = future.result()
            all_labels.append(labels)
            all_scores.append(scores)
            for policy in ("base", "pcs", "final"):
                idx = decision[policy + "_mode"]
                row[f"{prefix}_{policy}_mode"] = idx
                row[f"{prefix}_{policy}_pdm"] = float(scores[idx])
                row[f"{prefix}_{policy}_direction"] = float(direction[idx])
                for j, metric in enumerate(METRIC_NAMES):
                    row[f"{prefix}_{policy}_{metric}"] = float(labels[idx, j])
            row[f"{prefix}_vetoed"] = decision["vetoed"]
            for j, factor in enumerate(("nc", "dac", "ttc")):
                row[f"{prefix}_predicted_{factor}_risk"] = decision["risks"][j]
            row[f"{prefix}_oracle_pdm"] = float(scores.max())
            row[f"{prefix}_candidate_mean_pdm"] = float(scores.mean())
            row[f"{prefix}_selection_regret"] = float(scores.max() - scores[decision["final_mode"]])
            for j, metric in enumerate(METRIC_NAMES):
                row[f"{prefix}_candidate_mean_{metric}"] = float(labels[:, j].mean())
            row[f"{prefix}_ttc_pass_fraction"] = float((labels[:, 3] == 1).mean())
            row[f"{prefix}_safe_candidate_fraction"] = float((labels[:, [0, 1, 3]] == 1).all(-1).mean())
            row[f"{prefix}_has_safe_candidate"] = bool((labels[:, [0, 1, 3]] == 1).all(-1).any())
        row["pdm_delta"] = row["new_final_pdm"] - row["original_final_pdm"]
        delta_xy = proposals[1][..., :2] - proposals[0][..., :2]
        row['candidate_displacement_change_mean_m'] = float(np.linalg.norm(delta_xy, axis=-1).mean())
        row['candidate_displacement_change_max_m'] = float(np.linalg.norm(delta_xy, axis=-1).max())
        # Same-mode changes distinguish generation effects from selector changes.
        dynamics = []
        for poses in proposals:
            displacement = np.diff(np.concatenate([np.zeros_like(poses[:, :1, :2]), poses[..., :2]],axis=1),axis=1)
            speed = np.linalg.norm(displacement,axis=-1)/.5
            previous = np.concatenate([np.full_like(speed[:,:1],row['timing_ego_speed']),speed[:,:-1]],axis=-1)
            dynamics.append((speed,(speed-previous)/.5))
        row['candidate_speed_change_mae_mps'] = float(np.abs(dynamics[1][0]-dynamics[0][0]).mean())
        row['candidate_acceleration_change_mae_mps2'] = float(np.abs(dynamics[1][1]-dynamics[0][1]).mean())
        saved_labels.append(np.stack(all_labels))
        saved_scores.append(np.stack(all_scores))
        saved_proposals.append(np.stack(proposals))
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context("spawn")) as pool:
            for index, (record, features) in enumerate(tqdm(loader(source, args.workers, batch_size=None), desc="Fixed selector: original vs new generator")):
                futures, decisions, proposals = [], [], []
                for model in (reference, changed):
                    raw_features = {k: v for k, v in features.items() if not k.startswith('timing_')}
                    if model is reference:
                        context = generate_context(model, raw_features, record['token'], args.seed, device)
                    else:
                        with torch.no_grad(), bind_timing_inputs(model, features['timing_history'][None].to(device), features['timing_ego_speed'][None].to(device)):
                            context = generate_context(model, raw_features, record['token'], args.seed, device)
                    with torch.no_grad():
                        output = veto({k: v.unsqueeze(0).to(device) for k, v in context.items()})
                    if not torch.isfinite(output["risks"]).all():
                        raise ValueError("Nonfinite fixed-selector risk")
                    decisions.append({**{k: int(output[k][0]) for k in ("base_mode", "pcs_mode", "final_mode")},
                                      "vetoed": bool(output["vetoed"][0]), "risks": output["risks"][0].cpu().tolist()})
                    poses = context["proposals"].numpy()
                    proposals.append(poses)
                    futures.append(pool.submit(score_candidates, paths[record["token"]], poses, index == 0))
                from navsim.agents.diffusiondrive.timing_attention.model import history_validity
                enabled = bool(history_validity(features['timing_history'][None])[1][0])
                if not enabled and not np.array_equal(proposals[0],proposals[1]):
                    raise ValueError('Disabled timing branch changed generated trajectories')
                ratios = [float(layer.timing_adapter.last_ratio.max()) for layer in changed._trajectory_head.diff_decoder.layers]
                timing_diagnostics = dict(timing_enabled=enabled, timing_ego_speed=float(features['timing_ego_speed']),
                                          timing_last_step_residual_ratio_max=max(ratios))
                pending.append((futures, record, decisions, proposals, timing_diagnostics))
                if index == 0 or len(pending) >= args.score_workers:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
    finally:
        stream.close()
    if len(rows) != len(source) or not rows:
        raise ValueError("Incomplete paired evaluation")
    summary = {"scenes": len(rows), "completed": True, "arm": training_meta["arm"]}
    for key, value in rows[0].items():
        if isinstance(value, (float, bool)) and key != "valid":
            summary[key] = float(np.mean([r[key] for r in rows]))
    for prefix in ("original", "new"):
        summary[prefix + "_vetoed_count"] = sum(r[prefix + "_vetoed"] for r in rows)
    # Keep all candidates and official labels for generation/selection diagnosis.
    np.savez_compressed(run / "candidate_diagnostics.npz", tokens=np.array([r["token"] for r in rows]),
                        labels=np.stack(saved_labels), scores=np.stack(saved_scores), proposals=np.stack(saved_proposals))
    write_new_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Paired evaluation: {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("targets", "data-root", "output"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--seed", type=int, default=0)
    for command in ("train", "evaluate", "check-init"):
        sub = commands.add_parser(command)
        for name in ("baseline", "backbone", "anchor", "veto", "pcs-scorer", "output"):
            sub.add_argument("--" + name, required=True)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--workers", type=int, default=0)
        if command in ("train", "check-init"):
            sub.add_argument("--targets", required=True)
            sub.add_argument("--candidate-cache", required=True)
            sub.add_argument("--timing-inputs", required=True)
            sub.add_argument("--strength-weight", type=float, default=.1)
            sub.add_argument("--jerk-weight", type=float, default=.01)
            sub.add_argument("--residual-cap", type=float, default=.1)
            sub.add_argument("--timing-weight", type=float, default=0.1)
            sub.add_argument("--epochs", type=int, default=10)
            sub.add_argument("--devices", type=int, default=4)
            sub.add_argument("--batch-size", type=int, default=32)
            sub.add_argument("--lr", type=float, default=1e-4)
            sub.add_argument("--precision", choices=("16-mixed", "bf16-mixed", "32-true"), default="32-true")
            sub.add_argument("--smoke", action="store_true")
            sub.add_argument("--resume")
        else:
            sub.add_argument("--generator", required=True)
            sub.add_argument("--data-root", required=True)
            sub.add_argument("--metric-cache", required=True)
            sub.add_argument("--max-scenes", type=int, default=0)
            sub.add_argument("--score-workers", type=int, default=4)
    args = parser.parse_args()
    for name in ("epochs", "devices", "batch_size", "score_workers"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(name + " must be positive")
    for name in ("workers", "limit", "max_scenes", "timing_weight", "strength_weight", "jerk_weight"):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(name + " cannot be negative")
    {"prepare": prepare_inputs, "train": train, "evaluate": evaluate, "check-init": check_initialization}[args.command](args)


if __name__ == "__main__":
    main()
