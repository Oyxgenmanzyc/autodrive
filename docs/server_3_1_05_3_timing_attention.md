# 服务器：Timing Query Cross-Attention＋时机/力度监督

新代码入口：`scripts/pcs/run_3_1_05_3_attention.sh`。旧 generator.sh 属于仅loss实验，不要混用。
代码需安装到独立目录。所有示例用函数中的return保留当前终端，不粘贴`exit 1`。

## 1. 解压和环境

上传本分支ZIP后，按实际路径填写ZIP_PATH。仓库ZIP根目录名可能随打包方式变化，用脚本文件定位。

```bash
setup_timing_attention() {
    source /home/hndx/miniconda3/etc/profile.d/conda.sh || return 1
    conda activate navhigh || return 1
    ZIP_PATH="/home/hndx/zyc/training_code/3.1.05/autodrive-change-20260911-timing-cross-attention-generator-3.1.05_3.zip"
    INSTALL_DIR="/home/hndx/zyc/training_code/3.1.05/timing_attention_$(date +%Y%m%d_%H%M%S)"
    test -f "$ZIP_PATH" || { echo "Missing: $ZIP_PATH"; return 1; }
    mkdir -p "$INSTALL_DIR" || return 1
    unzip -q -n "$ZIP_PATH" -d "$INSTALL_DIR" || return 1
    ENTRY=$(find "$INSTALL_DIR" -maxdepth 5 -type f -path '*/scripts/pcs/run_3_1_05_3_attention.sh' -print -quit)
    test -n "$ENTRY" || { echo 'Cannot find new attention entry'; return 1; }
    export CODE_ROOT=$(cd "$(dirname "$ENTRY")/../.." && pwd)
    export DATA_ROOT=/home/hndx/navsim_workspace/dataset
    export EXP_ROOT=/home/hndx/navsim_workspace/exp
    # Avoid settings inherited from a previous experiment in this terminal.
    export GEN_EXP="$EXP_ROOT/timing_cross_attention_3_1_05_3"
    export TIMING_INPUTS="$GEN_EXP/inputs/navtrain"
    export PILOT_INPUTS="$GEN_EXP/inputs/pilot"
    export TIMING_TARGETS="$EXP_ROOT/generator_brake_timing_3_1_05_3/targets/navtrain.pt"
    export PILOT_TARGETS="$EXP_ROOT/generator_brake_timing_3_1_05_3/targets/pilot.pt"
    unset RESUME_CKPT TIMING_ATTENTION_RUN_DIR
    cd "$CODE_ROOT" || return 1
    echo "$CODE_ROOT"
    bash scripts/pcs/run_3_1_05_3_attention.sh check
}
setup_timing_attention
echo "Setup return code: $?"
```

脚本默认使用用户指定的四张Blackwell UUID、navhigh；不会把逻辑序号误映射到5880卡。
先确认所有来源文件/缓存存在。模型、anchor、PCS、TRV使用历史已确认路径。

## 2. 测试、pilot与真实生成器初始化审计

```bash
cd "$CODE_ROOT"
bash scripts/pcs/run_3_1_05_3_attention.sh test
echo "Tests return code: $?"
```

测试通过后：

```bash
bash scripts/pcs/run_3_1_05_3_attention.sh prepare-pilot
echo "Pilot input cache return code: $?"
```

成功后：

```bash
bash scripts/pcs/run_3_1_05_3_attention.sh check-init
echo "Zero-init audit return code: $?"
```

必须看到实际K67轨迹与logits相同的PASS。接着逐步运行：

```bash
bash scripts/pcs/run_3_1_05_3_attention.sh smoke
echo "Single GPU smoke return code: $?"
```

```bash
bash scripts/pcs/run_3_1_05_3_attention.sh smoke-ddp
echo "DDP smoke return code: $?"
```

Smoke验证优化器、反向传播和DDP链路；只有2个batch，不能用于评价PDM。

## 3. 完整小缓存

```bash
CACHE_WORKERS=4 bash scripts/pcs/run_3_1_05_3_attention.sh prepare
echo "Full timing input cache return code: $?"
cat "$TIMING_INPUTS/summary.json"
du -sh "$TIMING_INPUTS"
df -h "$EXP_ROOT"
```

旧GT监督sidecar自动复用；新缓存读取历史LiDAR，不读取相机。可重跑同一命令续建，已完成512场景块会跳过。
检查train/val的enabled和timing_supervised计数，避免有效制动监督覆盖太低而误以为网络学到了时机。
这一步不需要新的metric cache，也不重建205GB PCS候选或94GB原始feature cache。

## 4. 四卡正式训练

```bash
cd "$CODE_ROOT"
unset RESUME_CKPT TIMING_ATTENTION_RUN_DIR
EPOCHS=10 BATCH_SIZE=32 WORKERS=0 PRECISION=32-true LR=1e-4 \
TIMING_WEIGHT=0.1 STRENGTH_WEIGHT=0.1 JERK_WEIGHT=0.01 RESIDUAL_CAP=0.1 \
bash scripts/pcs/run_3_1_05_3_attention.sh train
echo "Timing attention training return code: $?"
```

默认WORKERS=0规避此服务器此前DataLoader退出问题；确认稳定后可自行改为2。
FP32优先保证数值一致性，注意力与动力学始终使用FP32。不要从上一版loss-only checkpoint续训。

只在同一新分支训练中断时续训，所有训练参数保持原值：

```bash
export RESUME_CKPT="填写本次timing_cross_attention训练目录/checkpoints/last.ckpt"
EPOCHS=10 BATCH_SIZE=32 WORKERS=0 PRECISION=32-true LR=1e-4 \
TIMING_WEIGHT=0.1 STRENGTH_WEIGHT=0.1 JERK_WEIGHT=0.01 RESIDUAL_CAP=0.1 \
bash scripts/pcs/run_3_1_05_3_attention.sh train
echo "Resume return code: $?"
unset RESUME_CKPT
```

## 5. 固定PCS/TRV评估

```bash
export GENERATOR_CKPT="填写本次正式训练目录/checkpoints/last.ckpt"
bash scripts/pcs/run_3_1_05_3_attention.sh eval-smoke
echo "Evaluation smoke return code: $?"
```

成功后：

```bash
bash scripts/pcs/run_3_1_05_3_attention.sh eval
echo "Full evaluation return code: $?"
```

默认复用 `/home/hndx/navsim_workspace/exp/metric_cache`。
重新生成/评分原与新候选，输出相同场景的paired_results.csv、summary.json、candidate_diagnostics.npz、run.json。
同时记录timing输入是否启用、注入幅度和候选位置/速度/加速度改变量。
提供整个eval目录与训练目录的run.json、csv/*/metrics.csv，以分析TTC、PDM和刹车效果。
新输出目录为`$EXP_ROOT/timing_cross_attention_3_1_05_3`，旧实验目录保留。
