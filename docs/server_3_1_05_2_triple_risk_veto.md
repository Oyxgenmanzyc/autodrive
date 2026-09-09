# 3.1.05_2 TRV 服务器指令

环境：`navhigh`。正式训练使用 PCI 34–37 对应的四张 RTX PRO 5000，
每卡 BS32、全局 BS128。脚本以 GPU UUID 固定物理卡。

代码进入服务器后执行：

```bash
cd /home/hndx/zyc/training_code/3.1.05/你的TRV代码目录
bash scripts/pcs/run_3_1_05_2_veto.sh test
bash scripts/pcs/run_3_1_05_2_veto.sh prepare-pairs-pilot
bash scripts/pcs/run_3_1_05_2_veto.sh smoke
bash scripts/pcs/run_3_1_05_2_veto.sh prepare-pairs
bash scripts/pcs/run_3_1_05_2_veto.sh train
```

默认复用：

```text
/home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05
/home/hndx/navsim_workspace/exp/pcs_candidates_pilot_k67_3_1_05
/home/hndx/navsim_workspace/exp/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt
```

pair cache 输出：

```text
/home/hndx/navsim_workspace/exp/pcs_hard_negative_pairs_k67_3_1_05_2
```

它只遍历现有 candidate cache 并运行冻结 PCS，不重算 PDM。完成后检查
`summary.json` 中 train/val 的 PCS PDM、loss、严重损失、三个相对风险数量。

训练输出：

```text
/home/hndx/navsim_workspace/exp/triple_risk_veto_k67_3_1_05_2/train/<时间>/
```

训练完成后使用程序打印的 best checkpoint：

```bash
export VETO_CKPT="/实际路径/checkpoints/epoch=XX.ckpt"
bash scripts/pcs/run_3_1_05_2_veto.sh eval-smoke
bash scripts/pcs/run_3_1_05_2_veto.sh eval
```

正式验证输出：

- `trv.csv`：风险否决后的最终结果；
- `pcs_proposer.csv`：同次原 PCS 结果；
- `base_selector.csv`：同次原 selector 结果；
- `paired_results.csv`：逐场景三个风险、否决决定和三个轨迹的真实评分；
- `summary.json`：相对 PCS/base 的净增益和新增零分变化。

以同次 PCS 为唯一主对照。若 TRV 的 `delta_vs_pcs <= 0`，该方案直接作废。
