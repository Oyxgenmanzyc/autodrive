# 服务器运行：生成器时机监督＋固定筛选器

代码目录用新包解压后的路径。不要继续使用旧 `timing-weighted-gain-selector` 包。
无需 git pull；支持直接上传 ZIP。以下按步骤执行，前一步非零退出时先处理错误。

## 1. 环境、文件检查和单元测试

```bash
source /home/hndx/miniconda3/etc/profile.d/conda.sh
conda activate navhigh
CODE_ROOT="/home/hndx/zyc/training_code/3.1.05/autodrive-change-20260910-generator-brake-timing-fixed-selector-3.1.05_3"
cd "$CODE_ROOT"
bash scripts/pcs/run_3_1_05_3_generator.sh check
bash scripts/pcs/run_3_1_05_3_generator.sh test
echo "测试退出码：$?"
```

脚本已按 UUID 锁定物理 0–3 卡；不依赖 CUDA 数字编号。使用 navhigh。
若 `FEATURE_CACHE` 已不存在，但原始数据仍在，执行一次：

```bash
bash scripts/pcs/run_3_1_05_3_generator.sh prepare-features
```

若只是原缓存搬家，先 `export FEATURE_CACHE="实际缓存目录"`，不必重建。
原记录默认读取 `$EXP_ROOT/pcs_candidates_k67_3_1_05/records.json`，也可通过 `RECORDS` 指定备份。
时机 sidecar 不含图像；不能用它替代原始特征文件。

## 2. 小缓存及 smoke

```bash
bash scripts/pcs/run_3_1_05_3_generator.sh prepare-pilot
bash scripts/pcs/run_3_1_05_3_generator.sh smoke
echo "生成器Smoke退出码：$?"
```

单卡 smoke 成功后，用同一小缓存检查四卡梯度同步：

```bash
bash scripts/pcs/run_3_1_05_3_generator.sh smoke-ddp
echo "四卡Smoke退出码：$?"
```

pilot 每组128场景，smoke仅2个训练和验证batch，用于运行链路检查。
正式数据生成会打印激活时机监督的场景数量；它决定原3.22监督实际覆盖率。

## 3. 完整监督缓存

```bash
bash scripts/pcs/run_3_1_05_3_generator.sh prepare
```

只需一次；若同一来源的完整文件已存在则复用。源代码/scene记录变更时拒绝混用。
不需生成新的 navtrain metric cache、PCS205GB缓存或 hard-negative pair cache。

## 4. 两组四卡训练

同一 tmux 会话中依次执行，各组从同一原始 K67 checkpoint 开始。
每组每卡 BS32、10 epochs，默认 workers=0 避免复用此前触发的 DataLoader 进程问题。

```bash
bash scripts/pcs/run_3_1_05_3_generator.sh train-control
echo "Control退出码：$?"
```

control 成功后：

```bash
bash scripts/pcs/run_3_1_05_3_generator.sh train-timing
echo "Timing退出码：$?"
```

输出分别位于：

```text
/home/hndx/navsim_workspace/exp/generator_brake_timing_3_1_05_3/train/control/<时间>/checkpoints/last.ckpt
/home/hndx/navsim_workspace/exp/generator_brake_timing_3_1_05_3/train/timing/<时间>/checkpoints/last.ckpt
```

`EPOCHS`、`LR`、`PRECISION`、`BATCH_SIZE`、`WORKERS` 可覆盖默认值，但两组必须一致。
若因显存改成每卡16，两组都用16重新比较，不能混成不同训练预算。
续训保持原超参，仅设置 `RESUME_CKPT`：

```bash
RESUME_CKPT="具体中断运行的checkpoints/last.ckpt" \
bash scripts/pcs/run_3_1_05_3_generator.sh train-timing
```

已完成10轮就无需续训。与旧筛选器的几分钟训练不同，本版需要运行原始感知前向及生成头反向，
实际时间以服务器进度为准；感知被冻结，但原始特征读取和前向仍有开销。

## 5. 两组各自验证

设置具体 control `last.ckpt`，先 smoke，再完整验证：

```bash
export GENERATOR_CKPT="/home/hndx/navsim_workspace/exp/generator_brake_timing_3_1_05_3/train/control/<实际时间>/checkpoints/last.ckpt"
bash scripts/pcs/run_3_1_05_3_generator.sh eval-smoke
bash scripts/pcs/run_3_1_05_3_generator.sh eval
```

对 timing 的 `last.ckpt` 重复相同步骤。脚本固定旧 PCS epoch05 和 TRV epoch02，
验证新生成器不会被误判为旧候选cache，但原来源严格校验仍保留。
复用原 navtest metric cache；全部候选重新生成、重新评分。
每次打印独立输出目录，记录哪个目录对应control、哪个对应timing。

```bash
python -m navsim.planning.script.compare_generator_timing \
  --control "control完整验证输出目录" \
  --timing "timing完整验证输出目录"
```

该程序检查数据、训练设置和筛选器来源一致，并要求相同token集合及可复现的原始参考分。
正式报告比较相同训练步数的两组；`epoch=xx.ckpt` 是oracle ADE最优，不是PDM最优。

## 6. 提供用于分析的文件

- 两组训练的 `run.json`、`csv/version_0/metrics.csv`、训练日志。
- 两组完整验证的 `run.json`、`summary.json`、`paired_results.csv`。
- 两组 `candidate_diagnostics.npz`，用于判断全部候选是否变好、哪些好候选仍未选中。
- `prepare` 输出中的 train/val 激活数量与比例。

不需上传大模型checkpoint或205GB候选缓存来做第一轮结果分析。
