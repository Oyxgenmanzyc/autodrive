# 3.1.05_3：制动需求 Query 与候选轨迹 Cross-Attention

本分支响应用户的机制澄清：timing query 不等同于历史 query。
唯一默认训练组是 Cross-Attention＋时机与力度监督，不运行普通微调对照。
初始化使用原 K67 epoch99，而不是已退化的 loss-only last.ckpt。
PCS epoch05 与 TRV epoch02 使用原权重、阈值与决策逻辑。

## 论文与代码依据

- BridgeAD, CVPR 2025，用户提供 PDF 的 §3.2、§3.4、Fig.2：借鉴逐未来时刻的 query 表达。
  https://arxiv.org/abs/2503.14182 。本文并不复现其历史 planning-query 队列、历史帧预测对齐、Mot2Plan 或整个网络。
- MTR, NeurIPS 2022，§3.2：借鉴按候选几何构造并随解码更新特征的思路。
  https://papers.nips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf 。没有迁入 intention query pair 或 Waymo 预测模型。
- ControlNet, ICCV 2023，§3.1：借鉴冻结原网络、零初始化新增条件输出的原则。
  https://arxiv.org/abs/2302.05543 。这里是线性残差条件分支，不是 ControlNet 结构复现。
- `history.py` 前方走廊风险量计算来自本仓库 `233457e` 的 `modules/risk_utils.py` 的历史输入部分。
- 软刹车起点沿用本仓库原 3.22 `risk_brake_timing.py`。连续力度损失、物理 timing query 和幅度约束是本实验新增设计。

## Q、K、V 的明确方向

Q 是未来 0.5…4 秒的制动需求，由当前可观测间距、接近速度、自车速度/加速度、
若等待到该时刻的估计剩余间距、THW、TTC 余量、所需减速度、TTC 变化构造。
恒定接近速度外推只是带误差的输入先验；不是 GT 未来、不是强制刹车规则。

K/V 是同一候选的 8 个时刻轨迹特征：原 decoder feature＋位置/速度/加速度编码＋未来时间编码。
每个候选独立做 8×8 attention，不混合不同候选，不增加候选数。
几何来自本层先计算的临时去噪预测，避免直接差分扩散噪声；每层、每次 DDIM 步都会更新。
历史 LiDAR 的唯一职责是估计当前风险量，不缓存、不编码历史 planning queries。

Q 的 8 个输出按时间顺序拼接，经零初始化线性层映射回 256 维，注入回归头输入。
原分类头使用未直接注入的本层特征；随前一层/去噪步骤的轨迹变化，最终 logits 仍可能间接改变。

## 尺度与冻结边界

- Query 物理单位：位置用 32m 风险距离尺度，速度用 15/30m/s，加速度用 6m/s²，THW/TTC 用 10s，变化用 5s；归一化后限幅到 [-2,2]。
- 候选 K/V：x/60m、y/30m、速度/30m/s、加速度/6m/s²，限幅后编码。
- 物理未来时间独立于原 DDIM denoising timestep。保留原 norm_odo/denorm_odo 坐标转换。
- 默认 FP32 训练；注意力和动力学差分始终 FP32，LayerNorm 稳定 feature 尺度。
- `delta = 0.1 × RMS(original_feature) × tanh(zero_initialized_projection)`。
  每层单次注入的 RMS 相对幅度 <=0.1。该界限不保证最终轨迹米制位移 <=10%，也不保证 PDM 不下降。
- 所有原 K67 参数与 BN 状态冻结，只优化 `_timing_query.*` 和 `*.timing_adapter.*`。
- 最近两帧必须都有有效风险观测；缺失或非有限输入不注入。没有全掩码 softmax。
- 独立 `check-init` 在真实 K67 上比较同输入同噪声的零初始化结果和关闭分支结果，要求轨迹、logits 精确一致。

## 监督和训练

主 loss 保留原生成器轨迹损失，通过固定回归头反传到新增模块。
额外监督只作用于匹配 GT anchor 的候选，且需原 3.22 的 pre-risk、持续制动条件及有效输入同时满足。
时机监督权重 0.1；力度监督权重 0.1（加速度 SmoothL1/3m/s² 与速度 SmoothL1/10m/s 各一半）；
jerk 权重 0.01，只惩罚超过 GT 幅值+2m/s³ 的部分，再以 5m/s³ 归一化。
连续力度损失避免 soft-brake sigmoid 对强减速饱和。不把所有候选拉向同一专家曲线。
DDP 使用跨卡 active-count 归一化，支持某卡没有刹车场景。

默认 navhigh、UUID 对应物理0–3卡、每卡BS32、global BS128、10epochs、AdamW LR1e-4、梯度裁剪1。
仅新分支学习，学习率与上一版解冻原轨迹头的2e-5不直接等价。
last.ckpt 是预先指定训练时长的主要评估点；oracle-ADE best 只作诊断，不叫 best PDM。
训练日志观察有效输入比例、有效制动样本数、起点误差、力度误差、残差幅度是否学起来。

## 缓存与风险边界

复用205GB PCS cache的 bev/agents/ego，不读取其旧PDM标签来监督新轨迹。
复用已有GT sidecar `generator_brake_timing_3_1_05_3/targets/{pilot,navtrain}.pt`；缺失时脚本生成。
新增 timing input sidecar 只保存4×12风险量和当前速度，预计约20–30MB张量，另有manifest；按512场景存块，可断点继续。
构建它需要读取历史LiDAR，文件输出小不意味着构建立即完成。不重建94GB feature cache。
metric cache刻画场景/评估环境，可以复用；原候选PDM标签不适用于改变后的轨迹，navtest会重新生成并评分全部候选。

走廊LiDAR并非真实前车跟踪，可能包含路侧物、切入及转弯误差。有效标记只说明测量存在，不能证明目标一致或安全。
低覆盖率、几何差分噪声、归一化饱和和新候选分布对固定PCS的影响，都必须通过输出诊断检查。
BridgeAD论文只是机制参考，不能据此承诺本数据集收益。

## 上一版结果的准确记录

12146场景：候选均值0.73071045→0.73309638；oracle0.97238042→0.97147136；
固定PCS/TRV最终0.89181861→0.89090669，下降0.09119个百分制点。
这证明那次运行无最终收益，但尚不能证明“缺少attention导致下降”或“轨迹被平均化”。
没有等时长对照时，亦不能将变化完全归因于时机loss。

## 验证状态

新增单元测试、实际checkpoint零初始化审计、单卡及DDP smoke命令已提供。
未在本地运行；待服务器navhigh测试。尚无本版训练或PDM提升结果。
