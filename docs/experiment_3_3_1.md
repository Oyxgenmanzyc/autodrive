# 3.3.1：LAST-BEV-Gate 第一版

## 基线与改动范围

- 发布分支：`change/20260921-last-bev-gate-3.3.1`。
- 基线：本仓库 `main`，提交 `3cf6b7b2750a9bf8ba6d6e4728eafdad44e68e3a`（原版 DiffusionDrive）。不叠加 3.1.01 时序训练或后续 proposal/selector 实验。
- 依据：用户提供的《感知架构修改.md》最终实施方案，优先落地 Stage B 的 B2 / C1。
- 上游参考：[LAST-ViT](https://github.com/ChengShiest/LAST-ViT/blob/main/cls_pretrain/conf.py)、[DiffusionDrive](https://github.com/hustvl/DiffusionDrive/blob/main/navsim/agents/diffusiondrive/transfuser_model_v2.py)。

这是 LAST token-selection 思想的适配，不是替换为 ViT backbone，也不是原 LAST 分类模型的完整复现。

```text
原 CNN / TransFuser fusion → 512×8×8 → 1×1 projection → 64×256 BEV tokens
                                                       ├─ LAST gate → + position → decoder queries
                                                       └─ 原特征 → + position → cross-BEV
```

默认 `diffusiondrive_agent` 和 Python config 的 `last_enable=False`。新增 `diffusiondrive_last_agent` 开启 3.3.1：K=16/64、alpha=0.25、sigma=sqrt(C)、仅 gate decoder memory；所有 64 tokens 及 ego-status token 保留。

ResNet、四层 GPT fusion、20 anchors、diffusion decoder、agent head、BEV semantic head、损失、优化器、100 epoch 调度均沿用原版。语义分支本身不经过 gate；联合训练时仍通过共享 backbone 产生间接影响。

## 适配细节

1. 在 positional embedding 之前计算分数；score 可选无参数 LayerNorm，默认开启。
2. 通道维 FFT、Gaussian low-pass、signed stability=`x/(abs(lowpass-x)+eps)`；每个通道沿空间维选 Top-K。
3. vote 为每个空间 token 被选中的通道比例；gate=`1+alpha*tanh(zscore(vote))`，默认范围 [0.75,1.25]。不删 token，不把分数直接当作前景标签。
4. 原始值参与 gate 与 global gather，不 detach。Top-K 排名本身不可微，gate 是离散选票产生的有界权重，不应宣称训练了可微选择器。
5. FFT 和统计固定 FP32；恢复输入 dtype。方差使用 population std，兼容单 token。Gaussian 核中心对齐 fftshift 的 DC 位置，修正参考实现偶数 C 下的一格偏移，支持任意 C。
6. decoder/cross-BEV 使用独立开关；位置编码采用与原版相同的 in-place dtype 语义，避免混合精度下额外改变 cross-BEV。
7. 模块不包含参数、persistent buffer 或额外 RNG 消耗，不增加 state_dict 键。保留原来的 strict evaluation loader。

`last_query_injection=True` 可把 selected global feature 注入第一个 trajectory 输入 query，其他 agent 输入 queries 不直接注入；经过 decoder 后，agent 输出仍可能受 query self-attention 间接影响。

## 服务器准备

在已有可运行 DiffusionDrive 的 Linux / NAVSIM 环境中执行。以下配置已按当前服务器目录和物理 GPU 4–7 固化；新 checkout 需重新设置 `NAVSIM_DEVKIT_ROOT`，避免运行旧版本代码。

```bash
conda activate navsim
git clone --branch change/20260921-last-bev-gate-3.3.1 --single-branch \
  https://github.com/Oyxgenmanzyc/autodrive.git autodrive-3.3.1
cd autodrive-3.3.1
export NAVSIM_DEVKIT_ROOT="$PWD"
export NAVSIM_EXP_ROOT=/home/hndx/navsim_workspace/exp
export OPENSCENE_DATA_ROOT=/home/hndx/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=/home/hndx/navsim_workspace/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BKB_PATH=/home/hndx/navsim_workspace/dataset/pytorch_model.bin
export PLAN_ANCHOR_PATH=/home/hndx/navsim_workspace/dataset/kmeans_navsim_traj_20.npy
export TRAIN_CACHE_PATH=/home/hndx/navsim_workspace/exp/training_cache
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DEVICES=4
export BATCH_SIZE=32
export NUM_WORKERS=4
export LR=1e-4
```

依赖沿用已有环境，无需安装 LAST-ViT 或下载其分类 checkpoint。ResNet 权重和 20×8×2 anchors 使用原 DiffusionDrive 资源，下载地址见 [train_eval.md](train_eval.md)。原 backbone 先尝试 timm 预训练权重，离线时使用 `BKB_PATH` fallback。

`CUDA_VISIBLE_DEVICES=4,5,6,7` 指定物理 GPU 4–7；Lightning 进程内会把它们映射为逻辑 GPU 0–3，因此 `DEVICES=4` 表示使用全部四张可见卡。`BATCH_SIZE=32` 是每个 DDP 进程、即每张 GPU 的 batch size，未使用梯度累积时全局 batch size 为 `4 × 32 = 128`。传给 agent/optimizer/scheduler 的基础学习率固定为 `1e-4`，不按全局 batch size再次线性放大；其余优化器和 warmup-cosine 调度逻辑保持原版。正式 baseline 和 LAST 必须保持 GPU 数、batch size、学习率、初始化、seed、100 epoch 和数据完全相同。

## 缓存与检查

已有相同特征构建方式的原版 DiffusionDrive cache 可复用：LAST 在网络内部运算，不改变输入/target cache schema。如果没有缓存：

```bash
python navsim/planning/script/run_dataset_caching.py \
  agent=diffusiondrive_last_agent \
  experiment_name=cache_diffusiondrive_3_3_1 \
  train_test_split=navtrain \
  cache_path="$TRAIN_CACHE_PATH" \
  force_cache_computation=false \
  +agent.config.bkb_path="$BKB_PATH" \
  +agent.config.plan_anchor_path="$PLAN_ANCHOR_PATH"
```

服务器先运行单元测试和真实数据 smoke test：

```bash
python -m unittest discover -s tests -p 'test_last_*.py' -v
CUDA_VISIBLE_DEVICES=4 DEVICES=1 BATCH_SIZE=2 bash scripts/training/run_last_3_3_1.sh \
  experiment_name=smoke_diffusiondrive_3_3_1 \
  trainer.params.fast_dev_run=true \
  trainer.params.strategy=auto
```

单元测试覆盖数值/梯度/奇偶通道/FP16、BF16，以及使用冻结原版 V2 class 的关闭等价性、旧 state_dict strict load、decoder/cross-BEV 隔离和 global 注入。路由测试仅替换昂贵的 backbone 和 task heads；不能据此声称真实模型 checkpoint、真实 loss 或 PDMS 已验证。CUDA 可用时同时执行 CUDA autocast 测试。

## 正式训练

```bash
# B2 / C1：3.3.1 主实验。GPU 4–7、每卡 BS=32、全局 BS=128、LR=1e-4。
# 新模型从 ResNet-34 预训练权重开始训练。
unset INIT_CKPT
bash scripts/training/run_last_3_3_1.sh

# B0：同设置重训 baseline；独立输出目录。
bash scripts/training/run_last_3_3_1.sh \
  experiment_name=training_diffusiondrive_3_3_1_baseline \
  agent.config.last_enable=false
```

如需由已有**原版 DiffusionDrive** checkpoint 初始化，先设置 `export INIT_CKPT=/实际路径/diffusiondrive_baseline.ckpt`，baseline 和 LAST 使用同一 checkpoint。它只初始化模型权重，不恢复 optimizer / epoch；本版没有新增 resume 功能。含额外实验模块的 3.1.x checkpoint 不属于本版兼容性承诺。

原调度器固定 100 epochs；正式对照保留 100 epochs，不能仅把 `max_epochs` 改短就称为完整短程微调方案。

可选消融命令：

```bash
# B1：LAST-Global
bash scripts/training/run_last_3_3_1.sh \
  experiment_name=training_3_3_1_global \
  agent.config.last_apply_decoder=false agent.config.last_query_injection=true

# B3：Global + Gate
bash scripts/training/run_last_3_3_1.sh \
  experiment_name=training_3_3_1_global_gate agent.config.last_query_injection=true

# K=8；其它 K 比例：1/64=0.015625、16/64=0.25、32/64=0.5
bash scripts/training/run_last_3_3_1.sh \
  experiment_name=training_3_3_1_k8 agent.config.last_topk_ratio=0.125

# Alpha 消融
bash scripts/training/run_last_3_3_1.sh \
  experiment_name=training_3_3_1_alpha010 agent.config.last_gate_alpha=0.10
```

`last_apply_cross_bev` 已接线并有隔离测试，但默认关闭；backbone gate、causal masking、attention bias、前景辅助损失留待后续实验，不混入第一版。

## 评估

使用训练时相同的 LAST 开关和超参数。checkpoint 只含权重，不自动恢复本版 Hydra 配置；实际配置记录在各输出目录的 `code/hydra` 下。评估 baseline 时另加 `agent.config.last_enable=false`。

```bash
# 已有 navtest metric cache 可跳过。
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache"

export CKPT=/home/hndx/navsim_workspace/exp/training_diffusiondrive_3_3_1_last_gate/实际时间目录/lightning_logs/version_0/checkpoints/实际文件.ckpt
python navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=diffusiondrive_last_agent \
  worker=ray_distributed \
  agent.checkpoint_path="$CKPT" \
  +agent.config.bkb_path="$BKB_PATH" \
  +agent.config.plan_anchor_path="$PLAN_ANCHOR_PATH" \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  experiment_name=eval_diffusiondrive_3_3_1_last_gate
```

`last_debug=true` 在模型输出字典中返回 `last_plan_similarity`、`last_vote`、`last_gate`（B×8×8）；仅供读取，不自动写入日志，不改变损失。相似度不是 attention weight。第一版未实现 mask 因果诊断和语义命中率统计，尚不能证明 DiffusionDrive 存在 lazy aggregation，也不能保证 PDMS 提升。

## 验证记录

本地未运行单元测试、smoke test 或训练；待服务器执行上述命令。已进行代码路径审阅，尚无训练/评估指标。工作流约定将运行验证放在服务器，本地 Windows 不代跑；本仓库为 Python 项目，没有 `mvnw`，Java/Maven 检查不适用。
