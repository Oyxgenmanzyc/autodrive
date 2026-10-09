"""3.1.05_11: train timing *inside* K67 generation, then audit independently."""
import argparse
import json
import os
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from navsim.agents.diffusiondrive.generator_timing import data as teachers
from navsim.agents.diffusiondrive.generator_timing.model import TimingModeGenerator, timing_loss
from navsim.agents.diffusiondrive.pcs.common import load_torch, sha256


def load_head(baseline, backbone, anchor, device):
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead

    config = TransfuserConfig(bkb_path=str(backbone), plan_anchor_path=str(anchor))
    head = TrajectoryHead(config.trajectory_sampling.num_poses, config.tf_d_ffn,
                          config.tf_d_model, str(anchor), config)
    raw = load_torch(baseline)['state_dict']
    prefix = '_transfuser_model._trajectory_head.'
    state = {}
    for name, value in raw.items():
        if name.startswith('agent.'):
            name = name[len('agent.'):]
        if name.startswith(prefix):
            state[name[len(prefix):]] = value
    if not torch.equal(state['plan_anchor'].float(), head.plan_anchor.detach()):
        raise ValueError('Baseline checkpoint and K67 anchor bank differ')
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False).to(device)


def load_model(args, device, checkpoint=None):
    model = TimingModeGenerator(load_head(args.baseline, args.backbone, args.anchor, device)).to(device)
    if checkpoint:
        saved = load_torch(checkpoint)
        for key, current in (('baseline_sha256', sha256(args.baseline)),
                             ('anchor_sha256', sha256(args.anchor)),
                             ('teacher_manifest_sha256', sha256(Path(args.teachers)/'manifest.json'))):
            if saved[key] != current:
                raise ValueError(f'Timing checkpoint {key} mismatch')
        model.load_adapter_state(saved['adapter'])
    return model


def train(args):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    manifest = json.loads((Path(args.teachers)/'manifest.json').read_text())
    teachers.check_cache(args.candidates, manifest['metric_root'], args.baseline,
                         args.anchor, args.teachers, 'train')
    dataset = teachers.TimingTeacherDataset(args.candidates, args.teachers, 'train', active_only=True)
    if len(dataset) == 0:
        raise ValueError('No safe timing-mode teachers; stop before training')
    if args.max_train_scenes:
        dataset = Subset(dataset, list(range(min(args.max_train_scenes, len(dataset)))))
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.workers, pin_memory=True)
    model = load_model(args, device, args.resume)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    start = 0
    if args.resume:
        saved = load_torch(args.resume)
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['epoch'] + 1
    wrapped = DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    run = Path(args.output)
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        (run/'checkpoints').mkdir(exist_ok=True)
        (run/'run.json').write_text(json.dumps(dict(arguments=vars(args),
                                                     active_scenes=len(dataset), world_size=world), indent=2))
    try:
        for epoch in range(start, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            wrapped.train()
            total = torch.zeros(2, device=device)
            for batch in loader:
                context = {key: value.to(device, non_blocking=True) for key, value in batch['context'].items()}
                target = batch['targets'].to(device=device, dtype=torch.float32)
                valid = batch['valid'].to(device)
                weight = batch['weight'].to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                predicted = wrapped(context).reshape(len(target), 67, 2, 8, 3)
                loss = timing_loss(predicted, target, valid, weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                total += torch.stack((loss.detach(), torch.ones_like(loss)))
            if world > 1:
                dist.all_reduce(total)
            if rank == 0:
                report = dict(epoch=epoch, mean_loss=float(total[0]/total[1]),
                              active_scenes=len(dataset), batch_size=args.batch_size, world_size=world)
                print(json.dumps(report), flush=True)
                (run/f'epoch_{epoch:02d}_train.json').write_text(json.dumps(report, indent=2))
                state = dict(epoch=epoch, adapter=model.adapter_state(), optimizer=optimizer.state_dict(),
                             baseline_sha256=sha256(args.baseline), anchor_sha256=sha256(args.anchor),
                             teacher_manifest_sha256=sha256(Path(args.teachers)/'manifest.json'))
                target_file = run/'checkpoints'/f'epoch={epoch:02d}.pt'
                temporary = target_file.with_suffix('.tmp')
                torch.save(state, temporary)
                os.replace(temporary, target_file)
                # last.pt is a separate checkpoint for exact optimizer resume.
                last = run/'checkpoints'/'last.pt'
                temporary = last.with_suffix('.tmp')
                torch.save(state, temporary)
                os.replace(temporary, last)
            if world > 1:
                dist.barrier()
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('init-cache', 'prepare-shard', 'complete-cache', 'train', 'validate', 'navtest'):
        command = sub.add_parser(name)
        command.add_argument('--candidates', required=True)
        command.add_argument('--teachers', required=True)
        command.add_argument('--metric-cache', required=True)
        command.add_argument('--baseline', required=True)
        command.add_argument('--backbone', required=True)
        command.add_argument('--anchor', required=True)
        command.add_argument('--output', required=True)
        if name == 'prepare-shard':
            command.add_argument('--split', choices=('train', 'val'), required=True)
            command.add_argument('--shard-index', type=int, required=True)
            command.add_argument('--num-shards', type=int, required=True)
        if name == 'train':
            command.add_argument('--epochs', type=int, default=20)
            command.add_argument('--batch-size', type=int, default=1)
            command.add_argument('--workers', type=int, default=0)
            command.add_argument('--lr', type=float, default=1e-4)
            command.add_argument('--resume')
            command.add_argument('--max-train-scenes', type=int, default=0)
        if name in ('validate', 'navtest'):
            command.add_argument('--generator-checkpoint', required=True)
            command.add_argument('--scorer', required=True)
            command.add_argument('--ranker', required=True)
            command.add_argument('--max-scenes', type=int, default=0)
            command.add_argument('--data-root')
            command.add_argument('--decision')
            command.add_argument('--score-workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'init-cache':
        teachers.init_cache(args.candidates, args.metric_cache, args.baseline, args.anchor, args.output)
    elif args.command == 'prepare-shard':
        teachers.prepare_shard(args.candidates, args.metric_cache, args.baseline, args.anchor,
                               args.output, args.split, args.shard_index, args.num_shards)
    elif args.command == 'complete-cache':
        teachers.complete_cache(args.candidates, args.metric_cache, args.baseline, args.anchor, args.output)
    elif args.command == 'train':
        train(args)
    else:
        from navsim.planning.script.generator_timing_eval import evaluate
        evaluate(args)


if __name__ == '__main__':
    main()
