# 新 3.1.05 PCS 服务器指令

环境：navhigh。物理 GPU 0–3。正式训练每卡 BS32，全局 BS128。
服务器代码可以通过 Git clone 或从本地上传；脚本会从自身位置定位 CODE_ROOT，
无需对上传目录执行 git pull。

## 1. 进入新代码目录

```bash
cd /home/hndx/zyc/training_code/3.1.05/autodrive-change-20260907-pdm-candidate-scoring-3.1.05
```

代码路径若不同，只修改 cd。默认基础 checkpoint：

```text
/home/hndx/navsim_workspace/exp/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt
```

如存放位置改变，在同一终端指定真实路径；必须是 3.1.02 原 K67 模型，
不能使用 SPR 或 matched selector control 的 checkpoint：

```bash
export BASELINE_CKPT="/实际路径/原3.1.02的epoch99.ckpt"
```

以上 export 仅在默认地址不正确时执行。脚本会严格检查模型权重和 anchor 内容。

默认原特征 cache：
`/home/hndx/navsim_workspace/exp/training_cache_3_1_02_k67`。

## 2. 单元测试、训练集 metric cache 和 pilot

```bash
bash scripts/pcs/run_3_1_05.sh test
bash scripts/pcs/run_3_1_05.sh metric
bash scripts/pcs/run_3_1_05.sh pilot
bash scripts/pcs/run_3_1_05.sh smoke
```

metric 模式为 navtrain 生成 PDM 仿真缓存，路径：
`/home/hndx/navsim_workspace/exp/metric_cache_navtrain_pcs_3_1_05`。
不强制重算已存在文件；原 navtest metric cache 不改动。
如果已经有完整的 navtrain metric cache，可直接设置：

```bash
export TRAIN_METRIC_CACHE="/实际路径/已有navtrain_metric_cache"
```

确认该 cache 确属 navtrain 后跳过 metric 模式，不要设置为 navtest 路径。

pilot 默认各取 128 个 train 和 val 场景，保存在独立 pilot 目录；
包括批量评分与官方逐条评分的一致性检查，并打印首条缓存体积。
smoke 仅单卡执行 2 个训练 batch、2 个验证 batch，验证数据链路和 checkpoint 保存。
pilot 太小，不用于判断模型最终增益；需要更大样本时使用新的 pilot 目录：

```bash
PILOT_SCENES=1024 PILOT_CACHE=/home/hndx/navsim_workspace/exp/pcs_pilot_1024     bash scripts/pcs/run_3_1_05.sh pilot
```

改变 pilot 规模时不要复用不同 manifest 的旧目录。
pilot summary.json 中 base/oracle/top5/top10 都是 0–1 分数；
显示为 0–100 时乘 100。oracle 仅是使用真实评分后挑选的诊断上限。

## 3. 正式候选缓存

```bash
bash scripts/pcs/run_3_1_05.sh cache
```

同时使用物理 GPU 0–3，各生成一个 shard。需要额外磁盘空间保存候选及 BEV 特征，
根据 pilot 单场景大小和实际样本数估算后启动。
同样的代码、checkpoint、anchor、seed 和 shard 数重跑会续存已完成场景。
如果修改这些条件，使用新 PCS_CACHE 目录。

完成后检查 `pcs_candidates_k67_3_1_05/diagnostics/<时间>/summary.json`。
若 val 候选改选空间不足，则停止，不自动启动训练。

## 4. 正式训练

```bash
bash scripts/pcs/run_3_1_05.sh train
```

固定每卡 BS32、GPU 0–3、20 epochs、LR=3e-4、16-mixed。
训练只更新评分器，原生成器不在 optimizer 中。
每次运行保存在：

```text
/home/hndx/navsim_workspace/exp/pdm_candidate_scoring_k67_3_1_05/train/<时间>/
```

保留：
- checkpoints/epoch=XX.ckpt：按 val/pdm 选出的 best。
- checkpoints/last.ckpt：最后一次及续训状态。
- run.json：基础模型/anchor 哈希、cache 信息和训练参数。
- csv/ 和 tensorboard/：训练与配对 val 指标。

训练完成会打印 best checkpoint 的实际地址。不要默认使用 epoch19 或 last。
PCS checkpoint 仅包含评分器；基础模型 checkpoint 和 anchor 必须同时保留。

若需断点续训，可在已激活 navhigh、已设置环境变量的终端直接使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m navsim.planning.script.run_pcs train \
    --cache /home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05 \
    --output /home/hndx/navsim_workspace/exp/pdm_candidate_scoring_k67_3_1_05/resume \
    --devices 4 --batch-size 32 --epochs 20 --workers 8 --lr 3e-4 \
    --resume "/实际路径/checkpoints/last.ckpt"
```

这会恢复 optimizer 和 epoch；epochs 是总目标 epoch 数，不是额外训练轮数。

## 5. 验证

设置训练打印的 best checkpoint：

```bash
export SCORER_CKPT="/实际路径/checkpoints/epoch=XX.ckpt"
bash scripts/pcs/run_3_1_05.sh eval-smoke
bash scripts/pcs/run_3_1_05.sh eval
```

默认物理 GPU0；若需要改为另一张空闲卡，设置 `EVAL_GPU`。
复用 `/home/hndx/navsim_workspace/exp/metric_cache` 的 navtest metric cache。
如原成功验证使用其他路径，设置 `TEST_METRIC_CACHE` 为该路径。

正式验证生成：
- pcs.csv：新评分器结果，标准 NAVSIM 指标列及 average 行。
- base_selector.csv：同一批候选的原 selector 结果。
- paired_results.csv：token、log_name、原/新指标、候选编号、换选及 PDM 差值。
- summary.json：完成状态、总分、净收益、救回零分和新增零分。
- run.json：本次参数和模型来源。

任何场景失败直接退出，不默默按剩余成功场景报总分。
不要用 eval-smoke 的 8 场景分数作为正式结果。

后续分析请提供上述 CSV、summary.json、run.json、
训练 run.json、训练 csv 日志或 TensorBoard 事件及终端日志。
不需要传送大体积基础 checkpoint 或候选缓存。

## 常见问题

- 新入口为 argparse，不使用 `+trainer.params.devices` 等 Hydra 覆盖。
  只有复用的 metric 缓存入口仍使用 Hydra，脚本先执行配置解析再缓存。
- 出现 checkpoint/anchor 不匹配，不要改成 strict=False 绕过。
- 出现 cache manifest 冲突，选择新的输出目录，避免混用旧候选标签。
- 源码采用纯文本哈希校验。同一实验的 cache/训练/验证应使用同一份代码。
- `exit` 只出现在独立 bash 脚本中；执行脚本不会关闭用户父终端。

