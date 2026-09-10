# 3.1.05_3：时序特征增强的加权相对增益选择器

日期：2026-09-10。分支：`change/20260910-timing-weighted-gain-selector-3.1.05_3`。
基于已完成 navtest 的 TRV 分支 `611e87b`，原 PCS 与 TRV 的入口仍保留。

## 实验依据与阈值解释

重新分析 2026.09.09.17.41.31.736847 的 `paired_results.csv`，剔除 `token=average`
汇总行并按 token 验证唯一性。实际是 12,146 个场景；标准 CSV 中的第 12,147 行是
平均值，并不是场景数错误。上次把该行计入逐场景改善数的表述应更正。

原 TRV 否决 448 次：正确 107 次、误否决 271 次、PDM 不变 70 次。
以下均为 0–1 制 PDM 的真实收益绝对值：

| 分组 | 均值 | P10 | P25 | 中位数 | P75 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正确否决挽回的损失 | 0.369468 | 0.015952 | 0.045018 | 0.377215 | 0.583333 | 0.903187 |
| 误否决放弃的收益 | 0.098240 | 0.021847 | 0.032433 | 0.043128 | 0.070770 | 0.115094 |

正确否决挽回 39.5330，误否决损失 26.6230，总量净增 12.9100，平均 +0.001063。
错误否决组的预测 TTC 风险中位数 0.52039，正确否决组为 0.35195，说明仅提高/降低
一个风险阈值不能可靠地区分收益与代价。上述统计只能解释改动动机：真实 PDM 在
部署时不可见，不能直接把 0.0431、0.0982 或 0.3695 写成推理阈值。

复现旧结果诊断：

```bash
python scripts/pcs/analyze_veto_rescues.py /path/to/paired_results.csv
```

## 新决策

继续让冻结 epoch5 PCS 选 candidate，让原 selector 选 base。额外学习：

```text
delta = PDM(candidate) - PDM(base)                       # 仅训练标签
gain_target = max(delta, 0)
loss_target = max(-delta, 0)
utility = predicted_gain - lambda * predicted_loss
          - w_nc * risk_nc - w_dac * risk_dac - w_ttc * risk_ttc
utility >= threshold：接受 PCS；否则退回 base。
```

gain/loss 用不额外按严重度加权的 MSE，目标是估计正收益/负损失的条件期望，包含
胜负发生频率，不等同于“已知会赢时的平均收益”。三个风险头预测相对 base 的
NC/DAC/TTC 退化，输入包含 candidate、base 和特征差。基础相对风险标签仍为
`clamp(base_metric-candidate_metric, 0, 1)`。风险惩罚是额外安全偏好，不声称 utility
等于精确的期望 PDM；校准网格含全零风险权重，避免必须重复扣除已学到的损失。

风险训练使用仅由 train 决策对估计的逐类 `pos_weight`：
`clip(sqrt(negative_mass / max(positive_mass, 1)), 1, 30)`。这是为了减少大量正常
决策淹没 NC/DAC/TTC 退化样本；不采用原来每个因子完全相同的严重样本加权。
加权 BCE 的输出被明确当作风险分数，不能未经校准解释成真实概率。稀少NC事件
的泛化仍可能失败，诊断文件会输出每个风险头在真失败/其他场景的分数分布。

## 从 best_3.22 迁移的内容

来源文件：`best_3.22/navsim/agents/diffusiondrive/modules/risk_brake_timing.py`。
用户提供的 zip SHA256：`dbc5857d0eedc3d71c2cf4d7538643c019547dae2be0a94394e7b26ab276881f`。
原项目 MIT 许可证保留于仓库根目录。3.22 是用户已有实验，本文不将它描述成新论文。

保留并适配的机制：

- FP32平滑位移范数；按0.5秒推导速度、加速度、jerk。
- `sigmoid((-0.5-acceleration)/0.35)` 连续刹车特征。
- 两个相邻刹车概率的乘积表示持续刹车；累计最大值记录开始时间。
- 每个时刻保留位置、速度、加速度、jerk、刹车、持续刹车、累计起点、航向和时间。
- 分别提取PCS候选/base及其差，接在已有场景条件特征之后输入风险和收益网络。

这使网络能够观察“何时减速、是否持续、在哪条路径上减速”，不把早刹车直接等同
于安全。DAC可以利用路径横向变化和已有BEV特征，但刹车特征不能保证改善DAC。
本次仍在PCS候选与base之间二选一；如果两条轨迹都减速过晚，本模块无法凭空生成
及时刹车的轨迹，也不会重新搜索其余65条候选。这是当前实验明确保留的能力边界。

原3.22需要GT前车连续跟踪和GT轨迹的监督激活条件，用于训练生成器。现有K67缓存
只包含proposals/BEV/agent/ego latent，没有原始当前车速或GT前车上下文。因此本次
不移植这部分生成器损失或历史LiDAR模块；也不从GT构造推理输入。首段加速度被
标为未知，速度只由候选位置间隔推导，能辨识的最早持续刹车始于后续区间。这是
3.22时序表示在选择器上的适配，不声称完整复现3.22或新增了直接TTC时间标签。

## 校准、对照与实验边界

训练数据仍只使用navtrain train决策对。navtrain val按整个log的SHA256分两组：
一组搜索lambda、风险权重、阈值；另一组只用于本轮checkpoint选择。
同一log不会跨这两组，DDP padding按原始index去重。

阈值候选包含固定PDM单位网格，以及校准组中“旧TRV正确救回/错误拦截”的
**预测utility**分位数（P10/P25/P50/P75/P90）。用真实PDM判断这些阈值是否真的
减少误拦，并要求校准组PDM改善、NC/DAC/TTC均值不低于旧TRV、新零分不增加。
这些是校准样本上的全局约束，不是推理时任一风险越线就拒绝。

搜索保留原TRV和原PCS作为回退选项。如果没有更好的加权策略，checkpoint记录
`policy.mode=reference`，准确报告本次尚未超过旧方法；不能把回退复现当作新模块
提升。这些保证只适用于校准组，不能保证新场景上没有安全回归。
已有TRV本身曾使用过完整val选择，因此本轮划分不将其包装成从未见过的新测试集。
navtest不参与训练、阈值搜索，也不提供命令行阈值调优接口。

训练会生成逐epoch预测CSV/诊断JSON，包含真实/预测相对收益、权重阈值、三类风险、
原TRV正确/误否决的分布及放回场景表现。正式navtest同次评分四种输出：新方法、
旧TRV、原PCS、base；先核对旧TRV能否复现89.1819，再判定新方法的增益。

对照：`train-no-timing`保留完全相同损失、初始化规则和校准过程，只去掉3.22时序
特征；与`train`比较才能判断时序表示是否贡献TTC收益。不能仅因最终总分上升就
声称刹车时机或风险召回已改善。成功标准包括独立验证PDM高于旧TRV、TTC/NC/DAC
无实质回归，以及减少原TRV误否决而未释放更多严重损失。

## 兼容与验证状态

复用K67 candidate cache、train/val decision pair cache、navtest metric cache。
不需GTRS或新的feature/metric cache；生成器与旧cache的hash文件未修改。
新模型从旧TRV风险头初始化，新增时序输入权重为0；冻结旧风险头作为精确对照，
冻结PCS编码器/解码器。独立保存新checkpoint，训练默认物理0–3卡UUID、每卡32、
navhigh、bf16、10轮，worker默认0避开旧版数据加载子进程崩溃。

已增加服务端单元测试覆盖持续刹车与单次尖峰、时序先后、未知首段、收益补偿、
校准回退、冻结梯度、与旧TRV初始等价、checkpoint保存恢复。按本地实验约定，
未运行本地测试/训练，待服务器test和smoke。3.1.05_3尚无训练完成结果或PDM结论。
