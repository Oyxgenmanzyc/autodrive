# 3.1.05_2 服务器运行指令

环境为 `navhigh`，使用物理 GPU 0–3，正式训练每卡 BS32、全局 BS128。
脚本从自身路径确定代码根目录，不要求上传目录包含 `.git`。

```bash
cd /home/hndx/zyc/training_code/3.1.05/autodrive-change-20260908-gtrs-conservative-advantage-selector-3.1.05_2
```

默认直接使用以下已有文件：

- K67 候选 cache：`/home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05`
- 原 PCS 最佳权重：`/home/hndx/navsim_workspace/exp/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt`
- GTRS PDM：`/home/hndx/zyc/training_code/V2/DiffusionDriveV2/navtrain_16384.pkl`
- GTRS 词表：`/home/hndx/zyc/training_code/V2/DiffusionDriveV2/gtrs_traj/16384.npy`

先核对文件并运行测试：

```bash
test -d /home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05
test -f /home/hndx/navsim_workspace/exp/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt
test -f /home/hndx/zyc/training_code/V2/DiffusionDriveV2/navtrain_16384.pkl
test -f /home/hndx/zyc/training_code/V2/DiffusionDriveV2/gtrs_traj/16384.npy
bash scripts/pcs/run_3_1_05_2.sh test
```

先用 pilot 验证数据格式和训练链路：

```bash
bash scripts/pcs/run_3_1_05_2.sh prepare-gtrs-pilot
bash scripts/pcs/run_3_1_05_2.sh smoke
```

pilot 仍需加载一次 29GB pkl，但只写很小的 compact cache。正式预处理再加载一次
pkl，并顺序读取完整 K67 train cache 取得 base 分数；这一步只执行一次：

```bash
bash scripts/pcs/run_3_1_05_2.sh prepare-gtrs
du -sh /home/hndx/navsim_workspace/exp/pcs_gtrs_stratified_k67_3_1_05_2
df -h /home/hndx/navsim_workspace/exp
```

随后在物理 0–3 卡训练：

```bash
bash scripts/pcs/run_3_1_05_2.sh train
```

训练输出位于：

`/home/hndx/navsim_workspace/exp/conservative_advantage_selector_k67_3_1_05_2/train/<时间>/`

程序按 `val/pdm` 保存一个 best checkpoint，同时保存 `last.ckpt`。训练每个 epoch
只读取 K67 cache 和小型 compact GTRS cache，不再执行 GTRS 预处理或 PDM 仿真。

设置程序打印的 best checkpoint 后验证：

```bash
export SCORER_CKPT="/实际路径/checkpoints/epoch=XX.ckpt"
bash scripts/pcs/run_3_1_05_2.sh eval-smoke
bash scripts/pcs/run_3_1_05_2.sh eval
```

如果需要在独立验证方案中测试更保守的门控，可在调用前设置：

```bash
export MIN_DELTA=0.02
export MIN_WIN_PROBABILITY=0.65
export MAX_CATASTROPHIC_RISK=0.05
export RISK_PENALTY=0.75
bash scripts/pcs/run_3_1_05_2.sh eval-smoke
```

未设置的变量使用 checkpoint 中的训练默认值。正式 navtest 只运行预先确定的一组阈值，
避免根据测试集反复选择参数。

正式验证输出 `conservative_pcs.csv`、`base_selector.csv`、
`paired_results.csv`、`summary.json` 和 `run.json`。其中 paired 文件额外包含每个场景的
预测优势、胜率、风险、是否提出换选及最终是否换选，供后续定位高代价误选。

若默认路径变化，可在执行脚本前覆盖 `PCS_CACHE`、`INIT_SCORER_CKPT`、
`GTRS_PDM`、`GTRS_VOCAB`、`TEST_METRIC_CACHE` 或 `EVAL_GPU`。
