# 3.1.05_10：空间候选 × 时机模态

## 要回答的问题

已定稿的 K67 + PCS + Cost Ranker 在完整 navtest 的 PDM 为 **0.8945933668**。K67 的每条轨迹隐含了一套时间安排，但同一空间路线没有显式的提前、延后等时机选项。旧的选后 Timing Selector 只修改最终一条轨迹，完整 navtest 增益很小。本实验保留 Ranker 评分前五的**不同空间路径**，每条扩成四种连续时机轨迹，再从 20 条候选选一条。基线固定为 Ranker 选择的原始轨迹，所有收益都相对它计算。

这一步发生在**原始 K67 扩散生成之后、最终轨迹选择之前**。因此它能检验“多个空间候选同时有时机模态”是否有用，但不能证明端到端重训扩散生成器必然有效。原 K67、PCS、Ranker 的代码和权重保持不变；只有新的 `TimingModeValue` 接受梯度。它只有一个 pairwise 排序损失。

四种动作：`identity` 保留原始 FP32 轨迹；`frontload` 在前半段更早沿原路径前进；`backload` 在前半段更晚前进；`early_brake` 在 0.5 秒开始平滑降低纵向进度。后两种时间扭曲与原轨迹有相同四秒终点；制动可能提前停下。各动作从完整连续弧长函数生成八个采样点，不逐帧独立移动。固定路径也意味着不能绕开道路障碍，不能推断四秒终点之后。

冻结 PCS 会对**新生成的 20 条时机候选重新评分**；模型读取每条新候选的 PCS decoder 特征和五项预测子分数、原空间路径的锁定 Ranker 分数、原始分类概率、原/新轨迹几何和新轨迹逐步速度/加速度。PCS 在 20 候选集合上的 self-attention 与原 K67 集合不同，因此重新评分后的 identity PCS 预测不要求等于原 K67 PCS 预测；但它的**官方 PDM 真值**必须与原缓存一致。PCS 的 TTC 等子分数是**模型预测的部署代理值**；官方未来 TTC/NC/DAC 只用于缓存训练标签和离线评估，不进入推理模型。现有 K67 输入不包含足够的车辆历史速度，不能凭空声称使用了准确 TTC、ΔTTC、THW 或 DRAC。

## 数据与决策规则

1. 用 `cost_ranked_pcs_3_1_05_6/features/navtrain` 的 85,109 train 和 18,179 val 记录、锁定 `epoch=14.ckpt` 排出每场景前五条空间候选。
2. 为每条生成四种时机轨迹，用 navtrain metric cache 官方 PDM 对 20 条轨迹评分。每一条 identity 的分项及总分必须与原 compact cache 在 1e-5 内相等，否则停止；不把评分漂移当收益。
   缓存完成后先报告安全约束下的 hindsight Oracle：完整 20 候选与仅五条原样路径的增益差。如果这项差值很小，扩充时机动作没有足够上限，应先停止，不把训练分数作为补救。
3. 只在 train 上训练新的价值模型。val 按完整 recording log 固定拆成 calibration/audit。每轮 checkpoint 和阈值只看 calibration PDM；audit 不调 epoch 或阈值。
4. 同时计算 `spatial_only_control`：用同一个已训练模型和阈值，只允许五条原样空间路径竞争，禁止三种新时机动作。完整策略超过 Ranker 却不超过这一对照时，只能说明空间重选有效，不能归因于新时机模态。
5. 若 calibration 无正增益、安全均值下降或有严重损失，策略回退为 Ranker 原样。独立 audit 必须正增益、相对 `spatial_only_control` 也有正增益、无严重损失、NC/DAC/TTC 新失败为零且安全均值不下降，才视为可进 navtest 的候选。这个门槛比历史 3.1.05_7 更严格，是为了约束稀疏收益与重灾损失。
6. navtest 固定 checkpoint 和阈值，输出同场景 timing/spatial-only/Ranker/PCS/base 五列配对结果。不得用 navtest 重新选模型或阈值。结果可能不涨分；Oracle 上限与训练后可部署增益是不同问题。

训练准备需要原始 `pcs_candidates_k67_3_1_05`（读取场景 BEV/agent/ego context，对新轨迹重跑冻结 PCS）、compact feature cache、navtrain metric cache 和锁定 Ranker。旧服务器磁盘曾只剩约 4–5 GiB；新缓存包含 20 条轨迹的 FP16 PCS 特征，估计约 2 GiB，仍要检查 `df -h`、日志和 checkpoint 增长，避免同时生成其它大型缓存。`prepare-all` 的每个 shard 原子写入独立 block，失败后原命令可续跑，既有完整块不会覆盖。脚本不删除任何旧文件。

## 服务器运行

从本分支安装或同步代码后，在旧服务器执行。GitHub 推送不会自动更新服务器工作目录。脚本用物理 GPU UUID 校验 PCI，默认使用旧服务器八卡；如有占用，可设置 `TRAIN_GPUS` 和 `EVAL_GPU` 为实际空闲的完整 UUID。`NUM_SHARDS` 由 `prepare-all` 自动按卡数确定，准备标签阶段可使用八卡，训练阶段可改四卡或单卡。

```bash
source /home/hndx/miniconda3/etc/profile.d/conda.sh
conda activate navhigh
export CODE_ROOT=/home/hndx/zyc/training_code/3.1.05/autodrive-3.1.05_10
cd "$CODE_ROOT"
bash scripts/pcs/run_3_1_05_10.sh check
bash scripts/pcs/run_3_1_05_10.sh test
bash scripts/pcs/run_3_1_05_10.sh prepare-all
bash scripts/pcs/run_3_1_05_10.sh diagnose
bash scripts/pcs/run_3_1_05_10.sh smoke
EPOCHS=20 BATCH_SIZE=32 WORKERS=0 bash scripts/pcs/run_3_1_05_10.sh train
```

训练结束后从 `$MODE_EXP/train/<run>/result.json` 取得 best checkpoint，再执行：

```bash
export MODE_CKPT=/home/hndx/navsim_workspace/exp/risk_timing_modes_3_1_05_10/train/<run>/checkpoints/epoch=XX.ckpt
bash scripts/pcs/run_3_1_05_10.sh validate
```

人工检查 validation 的 `report.json`、`pass_for_navtest` 及严重损失明细后，只有通过预定门槛才继续：

```bash
bash scripts/pcs/run_3_1_05_10.sh eval-smoke
bash scripts/pcs/run_3_1_05_10.sh eval
```

若某个 prepare shard 失败，先看 `$MODE_EXP/logs/prepare_<split>_shard_<i>.log`，修复根因后仅重跑该 shard：

```bash
SPLIT=train SHARD_INDEX=0 NUM_SHARDS=8 EVAL_GPU=GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04 bash scripts/pcs/run_3_1_05_10.sh prepare-shard
bash scripts/pcs/run_3_1_05_10.sh complete-cache
```

## 状态

代码、单元测试和重现入口已写入分支；尚无 3.1.05_10 的训练或 PDM 结果。本地按实验工作流未运行测试，服务器 `check`、`test`、完整标签、Smoke、训练和独立 audit 均待执行。未经这些结果不能声称第一瓶颈已被解决。
