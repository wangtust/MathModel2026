# MathModel_restructured

## 2026年中国研究生数学建模竞赛 E 题  
### 复杂场景下多模态情感预测的数学建模与算法设计

本项目实现赛题三个问题的完整建模、实验、分析与可视化流程：

```text
Question 1
原始视频
→ 文本 / 语音 / 视觉特征提取
→ 多模态时序对齐
→ 质量审计与可复现材料

Question 2
aligned_50 多模态特征
→ 模态局部连续缺失模拟
→ 鲁棒情感预测
→ 基线 / 消融 / 鲁棒性 / 三随机种子实验
→ 附件3预测

Question 3
冻结 Question 2 模型
→ Integrated Gradients
→ 删除实验
→ 模态贡献与关键证据
→ 原视频时间定位
→ Fidelity / Sensitivity / Gate Audit
→ 附件4预测与解释
```

项目提供统一根入口：

```bash
python run.py q1
python run.py q2
python run.py q3
```

并支持：

```bash
--test
--force
--force-figures
```

用于 smoke test、强制重跑和仅重建论文图表。

---

# 1. 项目目标

本项目围绕真实复杂场景下多模态情感预测中的三个核心问题展开：

1. **不同模态时间尺度不一致**  
   从原始视频中提取文本、语音、视觉特征，并建立可核验的跨模态时序映射。

2. **局部模态信息缺失**  
   模拟文本、语音、视觉在局部连续时段失效，训练并验证鲁棒多模态情感预测模型。

3. **模型决策依据难以追溯**  
   对冻结预测模型建立局部解释、模态贡献、删除验证和原视频证据回查机制。

---

# 2. 三个问题之间的关系

本项目三个问题既可以独立复现，也构成一个连续的建模体系：

```text
┌─────────────────────────────────────────────┐
│ Question 1                                  │
│ 原始视频 → 三模态特征 → 时间对齐 → 质量审计 │
└───────────────────┬─────────────────────────┘
                    │
                    │ 提供特征提取与时序对齐方法论
                    ▼
┌─────────────────────────────────────────────┐
│ Question 2                                  │
│ aligned_50 → 缺失模拟 → MRRF → 鲁棒预测     │
│                ↓                            │
│        附件3无标签专项预测                   │
└───────────────────┬─────────────────────────┘
                    │
                    │ 冻结 Q2 checkpoint
                    ▼
┌─────────────────────────────────────────────┐
│ Question 3                                  │
│ IG → 删除实验 → 模态贡献 → 时间证据回查      │
│                ↓                            │
│        附件4预测与解释                       │
└─────────────────────────────────────────────┘
```

需要注意：

- Question 1 使用附件1原始视频；
- Question 2 当前正式实验统一使用附件2官方 `aligned_50.pkl`；
- Question 3 同样使用附件2 `aligned_50.pkl`，并冻结 Question 2 已训练模型；
- Question 2 当前尚未把 Question 1 的自主对齐算法直接接入 `unaligned_50.pkl` 后重新训练；
- 附件3和附件4均为无标签专项测试集，不用于训练、调参或模型选择。

---

# 3. 推荐项目结构

```text
MathModel_restructured/
│
├── run.py                     # 项目统一入口
│
├── common/
│   └── paths.py               # 数据与输出路径统一管理
│
├── question1/
│   ├── alignment/
│   ├── analysis/
│   ├── extraction/
│   ├── visualization/
│   ├── assets/
│   ├── config.py
│   ├── pipeline.py
│   ├── run.py
│   └── README.md
│
├── question2/
│   ├── analysis/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── submission/
│   ├── training/
│   ├── visualization/
│   ├── config.py
│   ├── run.py
│   └── README.md
│
├── question3/
│   ├── analysis/
│   ├── attribution/
│   ├── evaluation/
│   ├── evidence/
│   ├── visualization/
│   ├── config.py
│   ├── pipeline.py
│   ├── run.py
│   └── README.md
│
├── vendor/
│   ├── MMSA-FET/
│   ├── whisperX/
│   └── silero-vad/
│
├── cache/
│   └── models/
│
├── data/
│   ├── 附件1-数据集原始多模态样本/
│   ├── 附件2-数据集特征文件/
│   ├── 附件3-模态缺失特征样本/
│   ├── 附件4-可解释专项视频样本与特征文件/
│   ├── processed/
│   └── features/
│
└── outputs/
    ├── q1/
    ├── q2/
    ├── q3/
    └── _smoke/
```

> 实际数据根目录由 `common/paths.py` 统一解析。  
> 不建议在各问题模块中硬编码绝对路径。

---

# 4. 数据说明

## 4.1 附件1

Question 1 输入。

```text
附件1-数据集原始多模态样本/
└── MOSEI数据集部分原始视频-100条/
    ├── label-100.xlsx
    └── <video_id>/
        └── <clip_id>.mp4
```

共：

```text
100 条原始视频
37 个 video_id 目录
```

用于：

```text
文本特征
语音特征
视觉特征
真人语音检测
词级时间对齐
典型样本验证
```

---

## 4.2 附件2

Question 2 / Question 3 的主建模数据。

当前项目统一使用：

```text
aligned_50.pkl
```

输入维度：

| 模态 | Shape |
|---|---|
| Text | `(N, 50, 768)` |
| Audio | `(N, 50, 74)` |
| Vision | `(N, 50, 35)` |

固定划分：

```text
Train = 3395
Valid = 728
Test  = 727
```

Question 2 / 3 均保持：

```text
同一 aligned 特征版本
同一标准化规则
同一输入接口
```

---

## 4.3 附件3

Question 2 无标签专项测试集。

```text
附件3-模态缺失特征样本/
└── 对齐版本/
```

共：

```text
30 条
```

用于：

```text
最终鲁棒情感预测
模型预测分歧分析
submission生成
```

不能用于：

```text
训练
调参
模型选择
Accuracy/F1/MAE评价
```

---

## 4.4 附件4

Question 3 无标签专项测试集。

```text
附件4-可解释专项视频样本与特征文件/
└── 对齐版本/
    ├── 01.pkl
    ├── ...
    ├── 20.pkl
    └── videos/
```

共：

```text
20 条
```

用于：

```text
情感极性预测
情感强度预测
模态贡献分析
局部关键证据定位
原视频时间回查
逐样本解释卡
```

---

# 5. 环境

当前项目主要在：

```text
Linux
Python 3.11
conda environment: mathmodel
CUDA GPU
```

环境中运行。

示例：

```bash
conda activate mathmodel
```

---

# 6. 主要Python依赖

项目核心依赖包括：

```text
numpy
pandas
scipy
scikit-learn

torch
transformers

opencv-python
mediapipe

matplotlib

openpyxl
nltk

faster-whisper
whisperx / vendor whisperX

MMSA-FET
Silero VAD

optuna
```

Question 1 还使用：

```text
openSMILE
ffmpeg
ffprobe
```

Question 3 还使用：

```text
pdfunite
pdfinfo
```

用于生成与检查附件4解释卡合订 PDF。

---

# 7. GPU与CUDA

Question 1 的 WhisperX 默认：

```text
device = cuda
compute_type = float16
```

根目录 smoke test 会先检查：

```text
torch.cuda.is_available()
CUDA version
cuDNN version
```

Question 1 入口还会为当前 conda 环境补充：

```text
nvidia/cudnn/lib
nvidia/cublas/lib
```

到：

```text
LD_LIBRARY_PATH
```

因此推荐始终通过：

```bash
python run.py q1
```

或：

```bash
python question1/run.py
```

运行 Question 1。

---

# 8. 根目录统一入口

项目统一使用：

```bash
python run.py <q1|q2|q3|test>
```

支持参数：

```text
--test
--force
--force-figures
```

---

# 9. Smoke Test

在正式长实验之前，建议先执行：

```bash
python run.py q1 --test
python run.py q2 --test
python run.py q3 --test
```

测试产物统一写入：

```text
outputs/_smoke/
```

不会污染：

```text
outputs/q1
outputs/q2
outputs/q3
```

---

## 9.1 Q1 Smoke Test

```bash
python run.py q1 --test
```

实际执行：

```text
Python编译检查
↓
Q1关键模块导入
↓
CUDA/cuDNN检查
↓
真实附件1单样本
↓
BERT
↓
openSMILE
↓
Silero VAD
↓
WhisperX
↓
MediaPipe
↓
词级时序对齐
↓
NPZ维度与mask检查
```

因此不是简单 import test，而是真正跑一条原始视频。

---

## 9.2 Q2 Smoke Test

```bash
python run.py q2 --test
```

实际检查：

```text
aligned_50.pkl读取
Train / Valid / Test数量
连续局部缺失模拟
CUDA

Concat-MLP
BiGRU
MulT no augmentation
TFR Feature Adapter

8个MRRF variant

forward
backward

附件3读取
```

已验证的 MRRF variant：

```text
full
no_recon
no_reliability
no_consistency
plain_fusion
plain_no_aug
recon_only
dual_recon
```

---

## 9.3 Q3 Smoke Test

```bash
python run.py q3 --test
```

实际执行：

```text
Q3模块导入
↓
检查Q2冻结checkpoint
↓
读取附件4 20条
↓
真实附件4单样本预测
↓
2-step Integrated Gradients
↓
局部删除实验
↓
三模态evidence
```

注意：

```text
Q3 smoke test 依赖已经存在的 Q2 no_recon seed2026 checkpoint。
```

因此在一个完全空的新项目中，不应期待：

```bash
python run.py test
```

在未训练 Question 2 的情况下直接跑完 Q3 smoke test。

---

# 10. 一次执行全部Smoke Test

如果 Question 2 所需冻结 checkpoint 已存在：

```bash
python run.py test
```

会顺序运行：

```text
Q1 smoke
→
Q2 smoke
→
Q3 smoke
```

---

# 11. Question 1

详细说明：

```text
question1/README.md
```

---

## 11.1 任务

Question 1 从附件1原始视频建立：

```text
文本
语音
视觉
```

三模态特征与时序对齐。

---

## 11.2 文本

当前文本特征：

```text
bert-base-uncased
```

输出：

```text
768维
```

核心原则：

```text
官方文本 = 文本模态内容
ASR = 用于寻找官方词在原视频中的真实发音时间
```

ASR不会替换官方文本。

---

## 11.3 语音

使用：

```text
openSMILE
eGeMAPSv02
LowLevelDescriptors
```

当前输出：

```text
25维声学特征
```

并使用：

```text
Silero VAD
```

判断真人语音。

---

## 11.4 视觉

使用：

```text
MediaPipe Face Detector
MediaPipe Face Landmarker
```

当前输出：

```text
52维 Face Blendshape
```

默认：

```text
2 FPS
```

多人脸场景不会强行选择目标脸。

---

## 11.5 时序对齐

使用：

```text
官方文本
↔
WhisperX观测词
```

建立词级时间映射。

只有文本-ASR匹配覆盖：

```text
>= 0.55
```

才保留可信词级时间。

固定最终词序列：

```text
MAX_WORDS = 80
```

同时保存：

```text
feature
mask
coverage
timestamp
```

---

## 11.6 跨语言

非英语语音不强制产生：

```text
英语逐词发音时间
```

而使用：

```text
source-segment / clip-level
```

跨语言支持。

---

## 11.7 Q1完整运行

```bash
python run.py q1
```

内部完成：

```text
100条原视频
↓
三模态提取
↓
VAD / ASR
↓
词级时序对齐
↓
跨语言升级
↓
质量审计
↓
典型样本
↓
图表
↓
复现报告
```

---

## 11.8 Q1断点续跑

默认支持 resume。

如果某条样本已有完整日志/特征：

```text
自动跳过
```

中途失败后：

```bash
python run.py q1
```

即可继续。

---

## 11.9 Q1强制重跑

```bash
python run.py q1 --force
```

等价于关闭 Q1 sample resume。

---

## 11.10 Q1只重建报告

```bash
python run.py q1 --force-figures
```

不会重新执行昂贵的：

```text
BERT
WhisperX
MediaPipe
```

只重新生成：

```text
审计
表格
图表
典型案例
论文结果摘要
```

---

# 12. Question 2

详细说明：

```text
question2/README.md
```

---

## 12.1 任务

针对：

```text
文本 / 语音 / 视觉
局部连续时段缺失
```

建立鲁棒情感预测模型。

输出：

```text
情感极性
情感强度
```

评价：

```text
Accuracy
Macro-F1
MAE
Pearson
```

---

# 13. Question 2主模型

当前主模型：

```text
MRRF
Missing-aware Reconstruction and Reliability Fusion
```

主要组件包括：

```text
缺失增强
重建
可靠性建模
一致性约束
多模态融合
```

---

# 14. Q2缺失模拟

支持：

```text
Text
Audio
Vision
Text+Audio
Text+Vision
Audio+Vision
```

缺失位置：

```text
beginning
middle
end
random
```

缺失布局：

```text
shared
independent
mixed
```

---

# 15. Q2正式训练规模

根目录：

```bash
python run.py q2
```

确保：

```text
29 个正式固定 checkpoint
+
至少 30 个 Optuna trial
```

---

## 15.1 5类独立模型 × 3 seeds

```text
MMSA MulT
Concat-MLP
BiGRU
MulT no augmentation
TFR Feature Adapter
```

种子：

```text
2026
2027
2028
```

共：

```text
15 checkpoints
```

---

## 15.2 MRRF主要模型 × 3 seeds

```text
full
no_recon
dual_recon
```

共：

```text
9 checkpoints
```

---

## 15.3 单种子消融

```text
no_reliability
no_consistency
plain_no_aug
plain_fusion
recon_only
```

共：

```text
5 checkpoints
```

因此：

```text
15 + 9 + 5 = 29
```

---

# 16. Q2完整实验

`python run.py q2` 会自动执行：

```text
Optuna超参数搜索
↓
29个正式checkpoint
↓
附件3关键模型预测
↓
MMSA MulT专项鲁棒性
↓
Optuna结果汇总
↓
三随机种子稳定性
↓
累计消融
↓
Dual Reconstruction
↓
缺失持续时长
↓
缺失比例 × 缺失位置
↓
shared / independent gap
↓
fig01–fig10
↓
七模型 × 三随机种子综合比较
↓
附件3模型分歧
↓
MRRF结构图
↓
错误分析
↓
最终模型选择
↓
附件3submission
↓
Q2 Final Audit
```

---

# 17. Q2论文主图

当前编号：

```text
fig01–fig19
```

覆盖：

```text
学习曲线
缺失时长 F1
缺失时长 MAE
主要消融
缺失位置
强度校准
混淆矩阵
Reliability Gate
附件3缺失模式
附件3预测
三随机种子
附件3模型分歧
MRRF结构
连续mask示意
基线比较
多模型缺失鲁棒性
累计消融
双路重建
位置×缺失率交互
```

---

# 18. Q2最终审计

只有：

```text
29个正式checkpoint完整
>=30 Optuna trial
fig01–fig19完整
附件3预测完整
selected_model.json存在
attachment3_submission.csv存在
```

才会生成：

```text
outputs/q2/full_pipeline_complete.json
```

并打印：

```text
Q2 FINAL AUDIT | PASS
```

---

# 19. Q2断点续跑

所有关键阶段使用产物检查。

例如已有：

```text
checkpoint.pt
history.csv
metrics.json
```

则自动：

```text
[SKIP]
```

中途失败：

```bash
python run.py q2
```

重新运行即可。

---

# 20. Q2只重新画图

```bash
python run.py q2 --force-figures
```

不会重新训练正式模型。

---

# 21. Question 3

详细说明：

```text
question3/README.md
```

---

# 22. Question 3冻结模型

当前主解释模型：

```text
MRRF no_recon
seed = 2026
```

路径：

```text
outputs/q2/runs/mrrf/no_recon/seed_2026/
```

Fidelity / Sensitivity 对照模型：

```text
MRRF full
seed = 2026
```

Question 3 不重新训练预测模型。

---

# 23. Q3 Integrated Gradients

默认：

```text
IG steps = 48
baseline = 0.1 × input
window fraction = 20%
random repetitions = 8
```

解释目标：

```text
当前预测类别 logit
-
最强竞争类别 logit
```

即：

```text
classification margin
```

---

# 24. Q3局部解释

每个模态选择：

```text
Top IG连续窗口
Low IG匹配窗口
Random匹配窗口
```

然后真实删除对应窗口：

```text
feature → 0
availability → False
```

重新运行模型。

比较：

```text
delta margin
delta probability
delta strength
```

因此不是只画 attention / gradient heatmap。

---

# 25. Q3模态贡献

每模态贡献基于：

```text
正向 signed deletion response
```

得到：

```text
Text contribution
Audio contribution
Vision contribution
```

主参考模态允许：

```text
unclear
```

不会强制所有样本选一个模态。

---

# 26. Q3证据回查

证据链：

```text
BERT token
↓
原始字符区间
↓
原始英文单词
↓
WhisperX Forced Alignment
↓
真实音频时间
↓
原视频秒数
↓
视频关键帧
```

若时间对齐不可信：

```text
不输出伪造秒数
```

---

# 27. Q3 Fidelity

比较：

```text
Top
vs
Random
vs
Low
```

并对：

```text
MRRF no_recon
vs
MRRF full
```

执行相同解释流程。

使用：

```text
2000 bootstrap
95% CI
```

---

# 28. Q3 Sensitivity

附件4上比较：

```text
no_recon
vs
full
```

包括：

```text
极性一致性
情感强度差异
概率差异
```

附件4无标签，因此这里只表示：

```text
模型选择敏感性
```

不能判断哪个模型更准确。

---

# 29. Q3 Gate Audit

比较：

```text
模型内部 fusion gate 排序
```

和：

```text
实际删除响应排序
```

计算：

```text
Spearman rho
top-1 agreement
marginal baseline
Cohen's kappa
```

---

# 30. Q3正确正式顺序

根目录 `run.py` 当前执行：

```text
pipeline
↓
ig-check
↓
evidence
↓
fidelity
↓
sensitivity
↓
alignment
↓
figures
↓
verify
```

必须：

```text
figures 在 verify 前
```

因为最终 verification 会检查所有图和附件4解释卡。

---

# 31. Q3图表

Figures阶段生成：

```text
9张主图
+
20张附件4逐样本解释卡
```

每张同时保存：

```text
PDF
PNG
```

因此：

```text
29 PDF
29 PNG
```

还会生成：

```text
paper/attachment4_explanation_cards.pdf
```

必须：

```text
20 pages
```

---

# 32. Q3主图

```text
fig01_performance
fig02_fidelity
fig03_modality_contributions
fig04_local_importance
fig05_alignment_quality
fig06_timed_evidence
fig07_model_sensitivity
fig08_validation_variant_fidelity
fig09_gate_response_rank
```

---

# 33. Q3最终验证

最后：

```text
verification.json
```

必须：

```text
passed = true
issues = []
```

Verification检查：

```text
附件4 01–20覆盖
每样本三模态evidence
概率和为1
情感极性与强度一致
证据时间合法
关键帧存在
token范围合法
BERT truncation不越界
728 validation解释完整
IG completeness
full模型解释完整
paired fidelity完整
Gate audit完整
29 PDF
29 PNG
20页解释卡
环境版本
核心代码SHA256
```

---

# 34. Q3断点续跑

正式：

```bash
python run.py q3
```

已有完整阶段自动：

```text
[SKIP]
```

如果：

```text
verification.json
```

存在但：

```text
passed != true
```

或：

```text
issues 非空
```

根 runner 不会认为 Q3 已完成。

---

# 35. Q3只重新生成图表

```bash
python run.py q3 --force-figures
```

执行：

```text
强制重建所有图
↓
20张case
↓
20页合订PDF
↓
重新verify
```

---

# 36. 推荐完整复现流程

在一个已有正确数据和环境的项目中：

```bash
conda activate mathmodel
```

首先检查 Question 1：

```bash
python run.py q1 --test
```

然后正式：

```bash
python run.py q1
```

检查 Question 2：

```bash
python run.py q2 --test
```

正式 Question 2：

```bash
python run.py q2
```

Question 2 完成以后再检查 Question 3：

```bash
python run.py q3 --test
```

最后：

```bash
python run.py q3
```

---

# 37. 最推荐的一套命令

```bash
conda activate mathmodel

python run.py q1 --test
python run.py q1

python run.py q2 --test
python run.py q2

python run.py q3 --test
python run.py q3
```

---

# 38. 不建议在全新项目直接这样运行

```bash
python run.py test
```

如果：

```text
Q2 no_recon seed2026 checkpoint
```

还不存在，Q3 smoke test 会失败。

因此：

```text
run.py test
```

更适合：

```text
已经至少完成关键Q2训练
```

的项目环境。

---

# 39. 强制运行参数

## `--force`

例如：

```bash
python run.py q2 --force
```

忽略普通阶段已有结果并重新执行。

---

## `--force-figures`

例如：

```bash
python run.py q3 --force-figures
```

只重建后处理 / 图表相关阶段。

---

# 40. 输出根目录

```text
outputs/
├── q1/
├── q2/
├── q3/
└── _smoke/
```

---

# 41. Question 1输出

主要：

```text
outputs/q1/
├── features/
├── logs/
├── tables/
├── figures/
├── case_assets/
├── paper/
└── run_manifest.json
```

重点：

```text
100条NPZ
100条JSON日志
样本汇总
质量审计
典型样本
跨语言分析
论文图表
复现manifest
```

---

# 42. Question 2输出

主要：

```text
outputs/q2/
├── runs/
├── tuning/
├── experiments/
├── analysis/
├── figures/
└── full_pipeline_complete.json
```

重点：

```text
29个正式checkpoint
>=30 Optuna trial
三种子结果
鲁棒性实验
消融实验
fig01–fig19
附件3submission
```

---

# 43. Question 3输出

主要：

```text
outputs/q3/
├── tables/
├── figures/
├── paper/
├── config.json
├── attachment4_metadata.json
├── full_validation_comparison.json
├── model_sensitivity.json
└── verification.json
```

重点：

```text
728验证集解释
20条附件4解释
Fidelity
IG completeness
Evidence mapping
Model sensitivity
Gate audit
9张主图
20张解释卡
20页合订PDF
最终verification
```

---

# 44. 子模块README

更详细的方法、参数和输出字段请查看：

```text
question1/README.md
question2/README.md
question3/README.md
```

根 README 主要负责：

```text
项目整体结构
模块依赖
统一运行入口
完整复现顺序
```

三个子 README 负责：

```text
各问题算法细节
单独命令
详细输出字段
实验解释
```

---

# 45. 断点续跑策略

根 `run.py` 使用：

```text
关键输出文件是否完整存在
```

判断某阶段是否已经完成。

不是简单检查：

```text
目录是否存在
```

例如 Question 2 的训练会检查：

```text
checkpoint.pt
history.csv
metrics.json
```

MRRF还包括：

```text
training_config.json
train_scaler.npz
```

Question 3 figures 会检查：

```text
9主图 PDF/PNG
20 case PDF/PNG
q3_results.md
figure_captions.md
20页合订PDF
```

因此：

```text
只跑一半的目录
```

不会被误判为完成。

---

# 46. 中断恢复

如果任一阶段中途报错：

```text
已经成功生成的产物不会删除
```

修复问题后重复：

```bash
python run.py q1
```

或：

```bash
python run.py q2
```

或：

```bash
python run.py q3
```

即可从未完成阶段继续。

---

# 47. 项目日志约定

统一 runner 输出：

```text
[SKIP]
```

表示该阶段关键产物完整，自动跳过。

```text
PASS
```

表示测试/审计通过。

```text
FAILED
```

表示当前阶段失败。

如果子进程异常，根 runner 会保留：

```text
checkpoint
CSV
PDF
JSON
```

等已完成结果。

---

# 48. 数据泄漏原则

本项目严格区分：

```text
Train
Validation
Held-out Test
附件3
附件4
```

模型参数：

```text
只在Train学习
```

模型结构 / 超参数 / checkpoint：

```text
只依赖Validation选择
```

附件3：

```text
只用于Q2最终无标签预测
```

附件4：

```text
只用于Q3最终无标签预测和解释
```

---

# 49. 可复现性原则

项目尽可能保存：

```text
随机种子
训练配置
scaler
checkpoint
数据划分
模型版本
软件版本
SHA256
原视频映射
时间位置
实验表格
图表
```

使论文结果可以从：

```text
输入数据
```

逐步追溯至：

```text
最终图表
最终预测
最终解释
```

---

# 50. 推荐提交前检查

## Question 1

确认：

```text
100条样本全部处理
审计无未解决硬错误
典型样本可回查
图表完整
run_manifest.json存在
```

---

## Question 2

确认：

```text
Q2 FINAL AUDIT | PASS
```

并存在：

```text
outputs/q2/full_pipeline_complete.json
outputs/q2/analysis/attachment3_submission.csv
outputs/q2/analysis/selected_model.json
```

---

## Question 3

确认：

```text
outputs/q3/verification.json
```

满足：

```text
passed = true
issues = []
```

并确认：

```text
20条附件4预测
20条附件4解释
29 PDF
29 PNG
20页解释卡合订PDF
```

---

# 51. 最终完整运行概览

```text
                    MathModel_restructured
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
     Question 1         Question 2         Question 3
          │                  │                  │
    原始100视频          aligned_50          冻结Q2模型
          │                  │                  │
    三模态提取            缺失模拟               IG
          │                  │                  │
      时间对齐              MRRF              删除验证
          │                  │                  │
      质量审计             基线消融           模态贡献
          │                  │                  │
     典型样本验证          鲁棒性分析          时间证据
          │                  │                  │
       Q1图表             附件3预测            Fidelity
                             │                  │
                         Q2 Audit          Sensitivity
                                                │
                                           Gate Audit
                                                │
                                           附件4解释
                                                │
                                           Q3 Verify
```

---

# 52. 快速开始

```bash
cd /root/MathModel_restructured
conda activate mathmodel
```

然后：

```bash
python run.py q1 --test
python run.py q1

python run.py q2 --test
python run.py q2

python run.py q3 --test
python run.py q3
```

---

# 53. 最终状态判定

整个项目可以认为完整复现成功，当：

```text
Question 1
100条样本完整生成并通过质量审计

Question 2
Q2 FINAL AUDIT | PASS

Question 3
verification.json:
    passed = true
    issues = []
```

三项同时满足。

---

## 子模块文档

- `question1/README.md`：三模态特征提取与时序对齐
- `question2/README.md`：模态缺失鲁棒情感预测
- `question3/README.md`：可解释性情感预测与证据回查

---

## 统一入口

```bash
python run.py q1
python run.py q2
python run.py q3
```

完整项目按照：

```text
Q1 → Q2 → Q3
```

顺序复现。
