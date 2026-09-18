# 服务器执行顺序

先使用 `scripts/pcs/install_3_1_05_5.sh <40位commit> [本地zip]` 安装到独立目录，再进入输出的 CODE_ROOT。若服务器不能访问 GitHub，可把同一 commit 导出的 zip 和安装脚本上传后传入本地路径。压缩包要带一个顶层目录。

```bash
source /home/hndx/miniconda3/etc/profile.d/conda.sh
conda activate navhigh
# cd 到安装脚本输出的 CODE_ROOT
bash scripts/pcs/run_3_1_05_5.sh check
bash scripts/pcs/run_3_1_05_5.sh test
bash scripts/pcs/run_3_1_05_5.sh prepare
bash scripts/pcs/run_3_1_05_5.sh smoke
bash scripts/pcs/run_3_1_05_5.sh smoke-ddp
```

按次序确认退出码0再进入下一步。`prepare` 是 CPU 读取98MB旧标签，生成新标签和相似配对；不会重新生成205GB PCS或35GB metric cache。此版 smoke 从完整新标签抽样，因为旧 pilot 的128个train场景全是负例，不能测试正例学习。

物理GPU默认 UUID 绑定 RTX5880 的 PCI 9B/9C/9D/9E 四张卡；每次训练/验证会核对 UUID、PCI及型号并打印占用。默认每卡 batch32，全局128，WORKERS=0，FP32。检查映射不等同于预留GPU，启动前阅读当前占用。

```bash
LOG_ROOT=/home/hndx/navsim_workspace/exp/gated_paired_timing_3_1_05_5/logs
mkdir -p "$LOG_ROOT"
PROBE_LOG="$LOG_ROOT/probe_$(date +%Y%m%d_%H%M%S).log"
nohup env EPOCHS=20 BATCH_SIZE=32 WORKERS=0 \
    bash scripts/pcs/run_3_1_05_5.sh probe > "$PROBE_LOG" 2>&1 &
echo "Gate PID=$!; LOG=$PROBE_LOG"
tail -f "$PROBE_LOG"
```

Ctrl+C只退出tail。probe 完成后用其输出的 result.json 中 `best_checkpoint` 赋值 PROBE_CKPT；先看对应 `epoch_XX_report.json` 的 calibration 和 audit AUPRC、自然比例、预算召回。Gate没找到信号时不应声称拆分已经有效。

```bash
export PROBE_CKPT="/这里替换为probe最佳checkpoint绝对路径/epoch=XX.ckpt"
TRAIN_LOG="$LOG_ROOT/joint_$(date +%Y%m%d_%H%M%S).log"
nohup env EPOCHS=20 BATCH_SIZE=32 WORKERS=0 \
    bash scripts/pcs/run_3_1_05_5.sh train > "$TRAIN_LOG" 2>&1 &
echo "Joint PID=$!; LOG=$TRAIN_LOG"
tail -f "$TRAIN_LOG"
```

joint 完成后：

```bash
export REFINER_CKPT="/这里替换为joint最佳checkpoint绝对路径/epoch=XX.ckpt"
bash scripts/pcs/run_3_1_05_5.sh eval-smoke
# Smoke退出码0后
bash scripts/pcs/run_3_1_05_5.sh eval
```

如果中断，在原阶段用 `RESUME_CKPT=/.../last.ckpt bash scripts/pcs/run_3_1_05_5.sh probe` 或 `train` 续训。EPOCHS表示目标总轮数，LR等设置必须与原训练一致。probe和joint不可互相resume；只能joint通过PROBE_CKPT初始化probe权重。

提交分析文件：gate_cache/summary.json、train_labels.csv/val_labels.csv、run.json、result.json、各轮report/thresholds.csv、csv/version_0/metrics.csv；最终在线验证summary.json及paired_results.csv。原始predictions.pt保留服务器上，后续诊断再取。
