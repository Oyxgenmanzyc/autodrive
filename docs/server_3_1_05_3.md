# 3.1.05_3 服务器运行

进入解压后的3.1.05_3代码根目录；脚本自动激活navhigh，使用物理0–3卡的完整UUID。
训练默认每卡BS32、4卡、10轮、bf16；脚本使用argparse，不再手写Hydra覆盖项。

```bash
bash scripts/pcs/run_3_1_05_3.sh test
```

通过后运行单卡完整pilot验证的smoke（训练只跑2个batch）。

```bash
bash scripts/pcs/run_3_1_05_3.sh smoke
```

使用已经准备完成的两类cache：

```text
/home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05
/home/hndx/navsim_workspace/exp/pcs_hard_negative_pairs_k67_3_1_05_2
```

pilot使用原同名pilot目录。只有pilot决策对不在时才运行
`bash scripts/pcs/run_3_1_05_3.sh prepare-pairs-pilot`；完整cache不在时才运行
`prepare-pairs`。它们不生成新轨迹，不跑新的PDM标签；不要常规重建已有缓存。

正式四卡训练：

```bash
bash scripts/pcs/run_3_1_05_3.sh train
```

结果位于`/home/hndx/navsim_workspace/exp/timing_weighted_gain_k67_3_1_05_3/train/<时间>/`。
每轮的`diagnostics/epoch_XX.json`会记录校准策略与本轮留出的val子集效果：
关注`val/result/gain_vs_reference`、`policy.mode`以及各风险分布。reference模式表示
加权策略未通过校准收益/安全条件，不应宣称新模型有效。

时序特征消融（主训练完成后单独运行，不同时占用这四张卡）：

```bash
bash scripts/pcs/run_3_1_05_3.sh train-no-timing
```

中断续训（填真实last.ckpt路径，总轮数仍为10，不是额外10轮）：

```bash
export RESUME_CKPT='/实际运行目录/checkpoints/last.ckpt'
bash scripts/pcs/run_3_1_05_3.sh train
unset RESUME_CKPT
```

验证使用训练打印的Best timing-gain checkpoint：

```bash
export TWG_CKPT='/实际运行目录/checkpoints/epoch=XX.ckpt'
bash scripts/pcs/run_3_1_05_3.sh eval-smoke
# 确认退出码为0后再执行
bash scripts/pcs/run_3_1_05_3.sh eval
```

验证目录包含`timing_gain.csv`、`reference_trv.csv`、`pcs_proposer.csv`、
`base_selector.csv`、`paired_results.csv`、`summary.json`、`run.json`。
逐场景对比需要排除标准CSV的`token=average`行。

默认引用的原TRV为：

```text
/home/hndx/navsim_workspace/exp/triple_risk_veto_k67_3_1_05_2/train/2026.09.09.14.53.12.850967/checkpoints/epoch=02.ckpt
```

`REFERENCE_TRV`、`PCS_SCORER`、`PCS_CACHE`、`PAIR_CACHE`、`PILOT_CACHE`、
`PILOT_PAIR_CACHE`、`TWG_EXP`、`EVAL_GPU`、`TRAIN_GPUS`都可在运行前export覆盖。
DataLoader workers默认0；需要加速且服务器资源允许时设置`WORKERS=2`。
代码使用冻结reference头，新的checkpoint包含它的参数，推理不另行读取原TRV文件。
原PCS/生成器/anchor仍执行provenance校验。下载GitHub ZIP或git archive保留LF；
若再次报告源码hash不符，先核对具体文件和换行，不能跳过校验继续训练。
