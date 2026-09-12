# 3.1.05_4 服务器分步执行

使用新ZIP解压目录作为CODE_ROOT。不要source脚本，不使用交互shell的set -e或exit，防止关闭终端。
若上一阶段报错，停止后续步骤。物理0–3卡通过UUID定位。

```bash
source /home/hndx/miniconda3/etc/profile.d/conda.sh
conda activate navhigh
export CODE_ROOT=/home/hndx/zyc/training_code/3.1.05/autodrive-change-20260912-post-selection-timing-refiner-3.1.05_4
cd "$CODE_ROOT"
bash scripts/pcs/run_3_1_05_4.sh check
bash scripts/pcs/run_3_1_05_4.sh test
```

check必须有navtrain metric cache；缺失时先恢复原35G目录或按照原navtrain配置重新生成。
已有metric cache不因本版改变而失效，不能用test的metric_cache代替navtrain。

先准备train/val各128场景官方动作标签（复用完整PCS缓存的前128场景）：

```bash
bash scripts/pcs/run_3_1_05_4.sh prepare-pilot
echo "Pilot退出码：$?"
```

确认controlled Oracle，再测试单卡和四卡：

```bash
bash scripts/pcs/run_3_1_05_4.sh smoke
echo "单卡Smoke退出码：$?"
bash scripts/pcs/run_3_1_05_4.sh smoke-ddp
echo "四卡Smoke退出码：$?"
```

完整标签默认一张GPU完成固定TRV决策、4个CPU评分进程；不用重跑神经生成器。

```bash
bash scripts/pcs/run_3_1_05_4.sh prepare
echo "完整标签退出码：$?"
```

需要四卡加速完整标签时，可以**代替**上面的单卡prepare执行分片，不与其同时运行：

```bash
(
  GPUs=(GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04 GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f GPU-e9db5634-a044-a8fb-c294-6f372876e0e3 GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd)
  jobs=()
  for shard in 0 1 2 3; do
    EVAL_GPU="${GPUs[$shard]}" NUM_SHARDS=4 SHARD_INDEX="$shard" SCORE_WORKERS=4 \
      bash scripts/pcs/run_3_1_05_4.sh prepare-shard &
    jobs+=("$!")
  done
  failed=0
  for job in "${jobs[@]}"; do wait "$job" || failed=1; done
  if [ "$failed" -eq 0 ]; then
    bash scripts/pcs/run_3_1_05_4.sh diagnose
  else
    echo "有分片失败，保留已完成block，修复后重跑相同命令"
  fi
)
```

完整train/val Oracle输出位于 `/home/hndx/navsim_workspace/exp/post_selection_timing_3_1_05_4/edit_cache/navtrain/summary.json`。
查看val中增益和救援场景再启动正式训练。若无正增益，训练入口拒绝运行。

```bash
EPOCHS=20 BATCH_SIZE=32 WORKERS=0 LR=1e-4 bash scripts/pcs/run_3_1_05_4.sh train
echo "训练退出码：$?"
```

断点恢复通过 `RESUME_CKPT=/绝对路径/last.ckpt` 和相同cache/lr设置运行train。
EPOCHS为总轮数，不是额外轮数。各轮policy.json保存是否启用编辑及实际val PDMS。

```bash
export REFINER_CKPT=/实际训练目录/checkpoints/epoch=XX.ckpt
bash scripts/pcs/run_3_1_05_4.sh eval-smoke
echo "验证Smoke退出码：$?"
# Smoke成功后执行：
bash scripts/pcs/run_3_1_05_4.sh eval
echo "完整验证退出码：$?"
```

评估通过原传感器输入重建K67候选，固定PCS/TRV，refiner预测动作后才调用官方评分。
输出 `paired_results.csv`、`summary.json`、`run.json`；训练再提供csv指标和epoch_*_policy.json。
无旧注意力权重参与，不需要旧失败实验的checkpoint。
