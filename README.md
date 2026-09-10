# 3.1.05_3：时序特征增强的加权相对增益选择器

在原PCS + TRV上比较预计收益、预计损失与加权NC/DAC/TTC风险，加入用户3.22的持续刹车与起点时序表示。
旧89.18分是3.1.05_2的验证结果；本分支尚未完成服务器训练，不能沿用为新结果。

- [实验设计、真实救援收益与迁移边界](docs/experiment_3_1_05_3_timing_gain.md)
- [服务器test、smoke、四卡训练与验证](docs/server_3_1_05_3.md)
- 入口：`bash scripts/pcs/run_3_1_05_3.sh test`
- 复用K67候选、决策对及metric cache；物理0–3卡UUID，每卡BS32，navhigh。
- `train-no-timing`用于时序特征消融；阈值仅在navtrain val指定log子集上选择。

以下保留前一版本说明。

# 新 3.1.05_2：三因子风险否决器（TRV）

本分支冻结原 3.1.05 PCS 作为候选 proposer，使用其真实 hard-negative 决策训练
独立 NC、DAC、TTC 相对风险头；任一风险超过 val 校准阈值时回退 base。

完整 navtest（12,146 场景）达到 **89.181861 PDM**：相对同次原 PCS 的
89.075571 提升 0.106290 点，相对 base 的 88.488936 提升 0.692925 点；
仅否决 448 次 PCS 换选，新增零分由 231 降至 199。

- [实验机制、边界与止损](docs/experiment_3_1_05_2_triple_risk_veto.md)
- [navhigh / 物理 0–3 卡 / 每卡 BS32 指令](docs/server_3_1_05_2_triple_risk_veto.md)
- 统一入口：`bash scripts/pcs/run_3_1_05_2_veto.sh help`

## 原 3.1.05：PDM Candidate Scoring（PCS）

本分支以原 3.1.02 K67 为基础，冻结候选生成器，迁移 DiffusionDriveV2 的单阶段 PDM 子指标评分器。
原 SPR 分支保留；原 PCS 已完成服务器验证并达到 89.075571 PDM。

- [实验设计、来源与取舍](docs/experiment_3_1_05_pcs.md)
- [navhigh / 物理 0–3 卡 / 每卡 BS32 服务器指令](docs/server_3_1_05_pcs.md)
- 统一入口：`bash scripts/pcs/run_3_1_05.sh help`

---

<div align="center">
<img src="assets/logo.png" width="80">
<h1>DiffusionDrive</h1>
<h3>Truncated Diffusion Model for End-to-End Autonomous Driving</h3>

[Bencheng Liao](https://github.com/LegendBC)<sup>1,2</sup>, [Shaoyu Chen](https://scholar.google.com/citations?user=PIeNN2gAAAAJ&hl=en&oi=sra)<sup>2,3</sup>, Haoran Yin<sup>3</sup>, [Bo Jiang](https://scholar.google.com/citations?user=UlDxGP0AAAAJ&hl=en)<sup>2</sup>, [Cheng Wang](https://scholar.google.com/citations?user=PdJIyPIAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Sixu Yan](https://sixu-yan.github.io/)<sup>2</sup>, Xinbang Zhang<sup>3</sup>, Xiangyu Li<sup>3</sup>, Ying Zhang<sup>3</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>3</sup>, [Xinggang Wang](https://xwcv.github.io)<sup>2 :email:</sup>
 
<sup>1</sup> Institute of Artificial Intelligence, HUST, <sup>2</sup> School of EIC, HUST, <sup>3</sup> Horizon Robotics

(<sup>:email:</sup>) corresponding author, xgwang@hust.edu.cn

Accepted to CVPR 2025 as Highlight!

[![DiffusionDrive](https://img.shields.io/badge/Paper-DiffusionDrive-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2411.15139)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-DiffusionDrive-yellow)](https://huggingface.co/hustvl/DiffusionDrive)&nbsp;



</div>

## 新 3.1.02：原算法 + Anchor 诊断

本分支基于原 3.1.02 (`0c18ceb`)，保留原版 adaptive NPY anchor、静态 ADE 匹配、loss 和推理逻辑，迁回 3.1.02_2 的只读统计与离线可视化能力。训练诊断默认关闭；使用方法、迁移边界及服务器测试命令见 [新 3.1.02 说明](docs/anchor_diagnostics_3_1_02.md)。

## News
* **` Apr. 4th, 2025`:** DiffusionDrive is awarded as CVPR 2025 Highlight!
* **` Feb. 27th, 2025`:** DiffusionDrive is accepted to CVPR 2025!
* **` Jan. 18th, 2025`:** We release the initial version of code and weight on nuScenes, along with documentation and training/evaluation scripts. Please run `git checkout nusc` to use it.
* **` Dec. 16th, 2024`:** We release the initial version of code and weight on NAVSIM, along with documentation and training/evaluation scripts.
* **` Nov. 25th, 2024`:** We released our paper on [Arxiv](https://arxiv.org/abs/2411.15139). Code/Models are coming soon. Please stay tuned! ☕️


## Table of Contents
- [Introduction](#introduction)
- [Qualitative Results on NAVSIM Navtest Split](#qualitative-results-on-navsim-navtest-split)
- [Video Demo on Real-world Application](#video-demo-on-real-world-application)
- [Getting Started](#getting-started)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction
Diffusion policy exhibits promising multimodal property and distributional expressivity in robotic field, while not ready for real-time end-to-end autonomous driving in more dynamic and open-world traffic scenes. To bridge this gap, we propose a novel truncated diffusion model, DiffusionDrive, for real-time end-to-end autonomous driving, which is much faster (10x reduction in diffusion denoising steps), more accurate (3.5 higher PDMS on NAVSIM), and more diverse (64% higher mode diversity score) than the vanilla diffusion policy. Without bells and whistles, DiffusionDrive achieves record-breaking 88.1 PDMS on NAVSIM benchmark with the same ResNet-34 backbone by directly learning from human demonstrations, while running at a real-time speed of 45 FPS.

<div align="center"><b>Truncated Diffusion Policy.</b>
<img src="assets/truncated_diffusion_policy.png" />
<b>Pipeline of DiffusionDrive. DiffusionDrive is highly flexible to integrate with onboard sensor data and existing perception modules.</b>
<img src="assets/pipeline.png" />
</div>

## Qualitative Results on NAVSIM Navtest Split
<div align="center">
<b>Going straight with car-following and lane-changing behaviors.</b>
<img src="assets/straight_0.png" />
<b>Going straight with diverse lane-changing behavior, which interacts with traffic light and stops at the stop line.</b>
<img src="assets/straight_1.png" />
<b>Turning left with diverse lane-changing behavior, which interacts with surrounding agents.</b>
<img src="assets/left_0.png" />
<b>Turning right with car-following and overtaking behaviors.</b>
<img src="assets/right_0.png" />
</div>

## Video Demo on Real-world Application


https://github.com/user-attachments/assets/bd2364f3-73fd-4c29-b8b2-ead11f78926d





## Getting Started

- [Getting started from NAVSIM environment preparation](https://github.com/autonomousvision/navsim?tab=readme-ov-file#getting-started-)
- [Preparation of DiffusionDrive environment](docs/install.md)
- [Training and Evaluation](docs/train_eval.md)


## Checkpoint

> Results on NAVSIM


| Method | Model Size | Backbone | PDMS | Weight Download |
| :---: | :---: | :---: | :---:  | :---: |
| DiffusionDrive | 60M | [ResNet-34](https://huggingface.co/timm/resnet34.a1_in1k) | [88.1](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/diffusiondrive_88p1_PDMS.csv) | [Hugging Face](https://huggingface.co/hustvl/DiffusionDrive) |

> Results on nuScenes


| Method | Backbone | Weight | Log | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: | :---: |
| DiffusionDrive | ResNet-50 | [HF](https://huggingface.co/hustvl/DiffusionDrive) | [Github](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_nuScenes/diffusiondrive_stage2.log.log) |  0.27 | 0.54  | 0.90 |0.57 | 0.03  | 0.05 | 0.16 | 0.08  |



## Contact
If you have any questions, please contact [Bencheng Liao](https://github.com/LegendBC) via email (bcliao@hust.edu.cn).

## Acknowledgement
DiffusionDrive is greatly inspired by the following outstanding contributions to the open-source community: [NAVSIM](https://github.com/autonomousvision/navsim), [Transfuser](https://github.com/autonomousvision/transfuser), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [MapTR](https://github.com/hustvl/MapTR), [VAD](https://github.com/hustvl/VAD), [SparseDrive](https://github.com/swc-17/SparseDrive).

## Citation
If you find DiffusionDrive is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @article{diffusiondrive,
  title={DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving},
  author={Bencheng Liao and Shaoyu Chen and Haoran Yin and Bo Jiang and Cheng Wang and Sixu Yan and Xinbang Zhang and Xiangyu Li and Ying Zhang and Qian Zhang and Xinggang Wang},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2025, Nashville, TN, USA, June 11-15, 2025},
  pages        = {12037--12047},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2025},
  url          = {https://openaccess.thecvf.com/content/CVPR2025/html/Liao\_DiffusionDrive\_Truncated\_Diffusion\_Model\_for\_End-to-End\_Autonomous\_Driving\_CVPR\_2025\_paper.html},
  doi          = {10.1109/CVPR52734.2025.01124}
}
```
