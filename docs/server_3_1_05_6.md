# 3.1.05_6 服务器步骤

使用Git发布zip直接解压，或者 `bash scripts/pcs/install_3_1_05_6.sh FULL_COMMIT /path/to/release.zip`。安装器会打印**实际**CODE_ROOT，后续必须使用这个目录，不要自行推测名称。新目录保留Git LF字节，避免旧缓存provenance因CRLF不一致而失效。以下命令均不含会退出交互终端的顶层`exit`。

## 1. 环境与检查

```bash
source /home/hndx/miniconda3/etc/profile.d/conda.sh
conda activate navhigh
# 设为安装器打印的目录或zip解压目录
export CODE_ROOT="/实际安装目录"
export EXP_ROOT=/home/hndx/navsim_workspace/exp
export RANK_EXP="$EXP_ROOT/cost_ranked_pcs_3_1_05_6"
export TRAIN_GPUS="GPU-ca2805ee-6ed8-b153-e487-087872e087ac,GPU-12dd0e4c-f7aa-3523-a136-63f0782786f6,GPU-fa24445c-8297-4539-4b52-095a9ad39eea,GPU-d9166e6d-dfbc-3785-a88e-d171f0c79c78"
export EVAL_GPU="GPU-ca2805ee-6ed8-b153-e487-087872e087ac"
unset RESUME_CKPT RANK_RESUME TEACHER_RESUME COST_RANK_RUN_DIR
cd "$CODE_ROOT" &&
bash scripts/pcs/run_3_1_05_6.sh check &&
bash scripts/pcs/run_3_1_05_6.sh test
echo "检查及单元测试退出码：$?"
```

默认物理卡为5880 Ada的PCI **9B/9C/9D/9E**。每个GPU任务均核对完整UUID与PCI，拒绝数字序号和重复UUID。不会停止其他用户或实验进程；运行前查看显示的显存/利用率。

## 2. 小缓存与单卡/四卡Smoke

```bash
cd "$CODE_ROOT" &&
bash scripts/pcs/run_3_1_05_6.sh prepare-pilot &&
bash scripts/pcs/run_3_1_05_6.sh smoke &&
bash scripts/pcs/run_3_1_05_6.sh smoke-ddp
echo "Pilot及Smoke退出码：$?"
```

Pilot默认train/val各128。Smoke用原PCS分数代替OOF仅检查链路，不能用于宣称效果；正式train强制完整OOF缓存。初始分数残差和argmax应与原PCS精确一致。Smoke只训练1轮；完整训练每轮覆盖85109场景，4卡每卡32时约665步，和旧Gate每轮32步不同。

## 3. 完整特征缓存

```bash
cd "$CODE_ROOT" &&
bash scripts/pcs/run_3_1_05_6.sh prepare
echo "排序特征缓存退出码：$?"
du -sh "$RANK_EXP/features/navtrain"
```

只生成约8GB量级的附加特征，不重跑metric或205GB候选cache。命令失败时先读Traceback，修复后同命令按done标记继续。不要并发写同一features目录。

## 4. 折外教师与分数缓存（一次性，耗时阶段）

```bash
cd "$CODE_ROOT" && (
    mkdir -p "$RANK_EXP/logs"
    LOG_FILE="$RANK_EXP/logs/oof_background_$(date +%Y%m%d_%H%M%S).log"
    nohup env TEACHER_EPOCHS=20 BATCH_SIZE=32 WORKERS=0 \
        bash scripts/pcs/run_3_1_05_6.sh oof > "$LOG_FILE" 2>&1 &
    echo "OOF PID：$!"
    echo "日志：$LOG_FILE"
)
```

3个教师依次使用指定4张物理卡，每个只训练自己的2/3日志，各20轮，然后预测未训练的一折。原PCS权重不受影响。这一步的BCE属于临时教师，**没有和排序模块的loss混合**。

`tail -f /上面打印的日志路径` 实时查看；Ctrl+C退出查看，不停止nohup任务。

OOF全部完成会打印summary，并产生`$RANK_EXP/oof/complete.json`。再次运行oof会检查并复用完成的teacher与预测；失败teacher若无resume会新建训练run，不会假装续训。

只有明确知道中断的fold和对应last.ckpt时这样续训：

```bash
FOLD_INDEX=1 TEACHER_RESUME="/该fold的/checkpoints/last.ckpt" \
bash scripts/pcs/run_3_1_05_6.sh oof-fold
```

随后运行oof处理剩余fold并合并。不要把Probe/Gate/Rank checkpoint用于teacher。

## 5. 单独训练排序模块

```bash
cd "$CODE_ROOT" && (
    if [ ! -f "$RANK_EXP/oof/complete.json" ]; then
        echo "折外缓存尚未完成，暂不训练"
    else
        LOG_FILE="$RANK_EXP/logs/rank_background_$(date +%Y%m%d_%H%M%S).log"
        nohup env EPOCHS=20 BATCH_SIZE=32 WORKERS=0 LR=1e-4 \
            bash scripts/pcs/run_3_1_05_6.sh train > "$LOG_FILE" 2>&1 &
        echo "排序训练PID：$!"
        echo "日志：$LOG_FILE"
    fi
)
```

仅优化新模块的代价加权排序loss，K67/PCS均不在训练optimizer中。每轮真实验证PDM选checkpoint，不以loss最低选checkpoint。

续训显式使用 `RANK_RESUME=/本次ranker的/checkpoints/last.ckpt`；EPOCHS仍表示总轮数。旧`RESUME_CKPT`不被本版读取，避免上次probe/joint环境变量串用。

## 6. 查看结果，决定是否进入navtest

训练打印best checkpoint和result.json。查看对应run的`epoch_XX_report.json`；不要只看calibration，audit的收益/有害切换/新增安全失败一起看。alpha=0表示没验证出收益，不强行改为1。

```bash
export RANK_CKPT="/训练打印的/checkpoints/epoch=XX.ckpt"
bash scripts/pcs/run_3_1_05_6.sh validate
```

缓存验证采用checkpoint内固定alpha，同时输出4个alpha结果供解释；不利用audit或navtest自动重新选择策略。

经审阅满足收益与安全要求后，执行：

```bash
bash scripts/pcs/run_3_1_05_6.sh eval-smoke &&
bash scripts/pcs/run_3_1_05_6.sh eval
```

navtest才需要原生成器/图像backbone/anchor、navtest原始数据与metric_cache及旧TRV checkpoint。评估输出`rank.csv`、`pcs.csv`、`trv.csv`、`base.csv`和逐场景paired_results.csv。新ranker不接旧TRV，TRV只是固定对照；真实PDM标签只在选轨迹之后用于统计。

旧实验缓存不自动删除；这次任务无需清理文件。
