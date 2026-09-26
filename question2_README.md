# Question 2：模态信息缺失条件下的鲁棒情感预测

本目录实现赛题 **问题2：模态信息缺失条件下的情感预测建模与验证**。  
核心目标是在文本、语音、视觉模态发生**局部连续缺失**时，稳定输出：

- 情感极性：Negative / Neutral / Positive
- 情感强度：连续值
- 缺失条件下的鲁棒性评价结果
- 附件3无标签专项测试集预测结果

当前实现以 **MRRF（Missing-aware Reconstruction and Reliability Fusion）** 为主模型，并配置多种独立基线、消融模型、三随机种子实验、缺失比例/位置/时长实验以及完整可视化分析。

---

## 1. 当前数据接口

问题2当前统一使用赛题附件2中的：

```text
附件2-数据集特征文件/
└── aligned_50.pkl
```

对应配置：

```python
# question2/config.py
DATA = ATTACHMENT2_DIR / "aligned_50.pkl"
```

三模态输入形状：

| 模态 | 输入形状 |
|---|---|
| Text | `(N, 50, 768)` |
| Audio | `(N, 50, 74)` |
| Vision | `(N, 50, 35)` |

固定数据划分：

| Split | 样本数 |
|---|---:|
| Train | 3395 |
| Validation | 728 |
| Test | 727 |

附件3使用：

```text
附件3-模态缺失特征样本/
└── 对齐版本/
```

附件3共 30 条无标签样本，仅用于最终推理，不参与训练、调参或模型选择。

> **当前边界**
>
> 本模块目前直接使用官方 `aligned_50.pkl`。  
> 尚未把问题1的自主时序对齐算法接入 `unaligned_50.pkl → Q1 alignment → Q2` 的实验流程；如后续增加该实验，建议作为独立 alignment-ablation 模块加入，而不要直接替换当前正式数据接口。

---

## 2. 目录结构

```text
question2/
├── analysis/
│   ├── attachment3_comparison.py   # 附件3不同模型预测分歧
│   ├── error_analysis.py           # 验证集错误归因
│   ├── model_comparison.py         # 七类配置 × 三随机种子综合比较
│   ├── model_selection.py          # 最终验证集选模 + 附件3提交
│   └── tuning_summary.py           # Optuna调参结果汇总
│
├── data/
│   ├── dataset.py                  # aligned_50读取、标准化、有效位置判断
│   └── missingness.py              # 连续局部缺失构造
│
├── evaluation/
│   ├── evaluator.py                # clean / missing场景统一评估
│   ├── metrics.py                  # Accuracy / Macro-F1 / MAE / Pearson
│   └── predictor.py
│
├── experiments/
│   ├── ablation.py                 # 累计消融
│   ├── baselines.py                # Concat-MLP / BiGRU / MulT-no-aug / TFR
│   ├── dual_reconstruction.py      # full vs dual_recon
│   ├── missing_duration.py         # 缺失持续比例实验
│   ├── repeats.py                  # 三随机种子稳定性
│   ├── robustness.py               # 缺失比例 × 位置 × 模态组合
│   └── tuning.py                   # Optuna超参数搜索
│
├── models/
│   ├── baselines.py                # 简单独立基线
│   ├── factory.py
│   ├── mrrf.py                     # MRRF主体
│   ├── mult.py                     # MulT
│   └── tfr.py                      # TFR特征适配器
│
├── submission/
│   └── export.py
│
├── training/
│   ├── checkpoint.py
│   ├── losses.py
│   ├── mult_trainer.py             # MMSA MulT训练/预测/专项分析
│   └── trainer.py                  # MRRF训练/预测
│
├── visualization/
│   ├── architecture.py             # MRRF结构图与连续缺失示意图
│   ├── report.py                   # 主报告图 fig01–fig10
│   └── ...
│
├── config.py
├── run.py                          # Q2模块内部统一入口
└── README.md
```

---

## 3. 数据预处理与有效位置

`data/dataset.py` 对输入做以下处理：

1. 读取固定 train / valid / test 划分；
2. 根据 `text_bert` 的 attention mask 去除 `[CLS]`、`[SEP]` 与 padding；
3. 区分：
   - 正常有效位置；
   - padding；
   - 原生全零模态位置；
4. Audio / Vision 仅使用训练集有效位置统计均值和标准差；
5. 标准化后截断到 `[-5, 5]`；
6. 原生无效位置保持为 0。

这样可避免把 padding 或原生模态缺失误当成正常观测。

---

## 4. 连续局部缺失模拟

`data/missingness.py` 在有效正文时间位置内构造**连续区间缺失**。

训练阶段可出现：

```text
clean
text
audio
vision
text+audio
text+vision
audio+vision
```

默认训练采样权重：

```text
clean         0.150
audio         0.200
vision        0.200
audio+vision  0.200
text          0.100
text+audio    0.075
text+vision   0.075
```

支持：

### 缺失比例

```text
10%
20%
30%
40%
50%
```

### 缺失位置

```text
beginning
middle
end
random
```

### 多模态缺失布局

```text
shared       # 多个模态共享同一连续缺失区间
independent  # 各模态独立生成缺失区间
mixed        # 训练时随机选择 shared / independent
```

---

## 5. 评价指标

统一使用：

```text
Accuracy
Macro-F1
MAE
Pearson correlation
```

其中：

- Accuracy / Macro-F1：情感极性三分类；
- MAE / Pearson：情感强度回归。

模型输出的情感强度还会经过极性一致性校正：

```text
Negative → 强度 < 0
Neutral  → 强度 = 0
Positive → 强度 > 0
```

---

## 6. MRRF模型变体

`training/trainer.py` 当前支持：

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

主要用途：

| Variant | 用途 |
|---|---|
| `full` | 完整 MRRF |
| `no_recon` | 去除重建模块 |
| `no_reliability` | 去除可靠性建模 |
| `no_consistency` | 去除一致性约束 |
| `plain_no_aug` | 不使用训练时缺失增强 |
| `plain_fusion` | 基础融合对照 |
| `recon_only` | 重建增强对照 |
| `dual_recon` | 双路重建扩展模型 |

训练总损失由以下部分构成：

```text
classification loss
+ regression_weight × regression loss
+ reconstruction_weight × reconstruction loss
+ consistency_weight × consistency loss
```

默认：

```text
regression_weight     = 0.5
reconstruction_weight = 0.2
consistency_weight    = 0.1
```

优化器：

```text
AdamW
weight_decay = 1e-4
gradient clipping = 1.0
```

---

## 7. 模型选择规则

正式 MRRF checkpoint 只使用**固定验证集**选择，不使用测试集或附件3。

默认 `selection_metric=official`：

```text
score =
mean(
    clean Macro-F1 - clean MAE / 3,
    A+V 30% missing Macro-F1 - A+V 30% missing MAE / 3
)
```

最终 `submit` 阶段再对三个随机种子：

```text
2026
2027
2028
```

的验证集选择分数取均值，比较完整候选配置。

附件3没有标签，因此**不得用于模型选择或评价 Accuracy / F1 / MAE**。

---

# 8. 推荐运行方式

## 8.1 先做最小化 smoke test

在项目根目录MathModel_restructured运行：

```bash
python run.py q2 --test
```

该测试不会启动正式训练，而会最小化检查：

- Q2所有关键模块能否导入；
- `aligned_50.pkl` 能否正常读取；
- train / valid / test 数量和维度；
- 连续缺失模拟；
- Concat-MLP；
- BiGRU；
- MulT-no-augmentation；
- TFR Feature Adapter；
- 全部8个MRRF variant；
- forward + backward；
- CUDA；
- 附件3读取。

测试输出位于：

```text
outputs/_smoke/q2/
```

测试通过后再执行正式实验。

---

## 8.2 一键执行完整问题2

推荐直接在项目根目录运行：

```bash
python run.py q2
```

根目录统一 runner 会自动：

1. 检查已有实验结果；
2. 已完成 checkpoint / CSV / PDF 自动跳过；
3. 从缺失步骤继续执行；
4. 完成所有训练、实验、分析、图表与附件3提交；
5. 最后执行完整性审计。

如果中途报错，修复后重新执行：

```bash
python run.py q2
```

即可断点续跑。

---

## 8.3 强制重新生成图表

如果只修改了画图代码：

```bash
python run.py q2 --force-figures
```

不会重新训练已有正式模型。

---

## 8.4 强制重新执行普通正式流程

```bash
python run.py q2 --force
```

会忽略已有普通实验产物重新执行。

Optuna 完整 study 不会因为 `--force` 被无条件重复追加 30 个 trial。

---

# 9. 完整正式训练规模

根目录 `python run.py q2` 会确保以下正式 checkpoint 完整存在。

## 9.1 独立模型：15个

每个模型训练：

```text
seed 2026
seed 2027
seed 2028
```

包括：

```text
MMSA MulT
Concat-MLP
BiGRU
MulT no augmentation
TFR Feature Adapter
```

共：

```text
5 × 3 = 15
```

## 9.2 MRRF主配置：9个

```text
full
no_recon
dual_recon
```

每种三个种子：

```text
3 × 3 = 9
```

## 9.3 单种子消融：5个

```text
no_reliability
no_consistency
plain_no_aug
plain_fusion
recon_only
```

固定：

```text
seed = 2026
```

共：

```text
5
```

因此正式固定 checkpoint 总数：

```text
15 + 9 + 5 = 29
```

此外 Optuna 默认至少完成：

```text
30 trials
```

---

# 10. 问题2完整实验流程

`python run.py q2` 的总体实验顺序为：

```text
aligned_50.pkl
        │
        ▼
Optuna超参数搜索
        │
        ▼
5类独立模型 × 3 seeds
        │
        ▼
MRRF主要模型与消融模型
        │
        ▼
29个正式checkpoint
        │
        ├──────────────► 三随机种子稳定性
        │
        ├──────────────► 累计消融
        │
        ├──────────────► Dual Reconstruction
        │
        ├──────────────► 缺失持续时长
        │
        ├──────────────► 缺失比例 × 缺失位置
        │
        ├──────────────► shared / independent gap
        │
        ├──────────────► 七模型综合比较
        │
        └──────────────► 错误归因
        │
        ▼
附件3全量推理
        │
        ▼
验证集三种子选模
        │
        ▼
附件3最终submission
        │
        ▼
fig01–fig19 + 补充图表
```

---

# 11. 主要实验说明

## 11.1 三随机种子稳定性

模块：

```text
experiments/repeats.py
```

种子：

```text
2026 / 2027 / 2028
```

用于验证结果是否依赖单次随机初始化。

主要输出：

```text
outputs/q2/experiments/repeats/
├── three_seed_metrics.csv
├── three_seed_summary.csv
├── three_seed_results.md
└── fig11_three_seeds.pdf
```

---

## 11.2 累计消融

模块：

```text
experiments/ablation.py
```

顺序：

```text
plain_no_aug
    ↓
plain_fusion
    ↓
recon_only
    ↓
no_consistency
    ↓
full
```

主要输出：

```text
outputs/q2/experiments/ablation/
├── cumulative_ablation_seed2026.csv
├── cumulative_ablation_results.md
└── fig17_cumulative_ablation.pdf
```

---

## 11.3 双路重建实验

模块：

```text
experiments/dual_reconstruction.py
```

比较：

```text
MRRF full
vs
MRRF dual_recon
```

固定验证条件共 55 条/模型，并汇总三个随机种子。

主要输出：

```text
dual_reconstruction_grid.csv
dual_reconstruction_three_seed.csv
dual_reconstruction_results.md
fig18_dual_reconstruction.pdf
```

---

## 11.4 缺失持续时长实验

模块：

```text
experiments/missing_duration.py
```

分别测试：

```text
Text
Audio
Vision
```

持续缺失比例：

```text
5%
10%
20%
30%
50%
```

主要输出：

```text
outputs/q2/experiments/missing_duration/
└── duration_results.csv
```

---

## 11.5 缺失比例 × 缺失位置

模块：

```text
experiments/robustness.py
```

模态组合：

```text
T
A
V
T+A
T+V
A+V
```

缺失率：

```text
10%
20%
30%
40%
50%
```

位置：

```text
beginning
middle
end
```

主网格共：

```text
6 × 5 × 3 = 90 conditions
```

此外单独测试：

```text
shared 30%
independent 30%
independent + per-modality 10%–50%
```

主要输出：

```text
outputs/q2/experiments/robustness/
├── position_rate_grid.csv
├── gap_layout_validation.csv
├── position_rate_results.md
└── fig19_position_rate_interaction.pdf
```

---

## 11.6 七模型 × 三种子综合比较

模块：

```text
analysis/model_comparison.py
```

模型：

```text
Concat-MLP
BiGRU
MulT no augmentation
MMSA MulT
TFR feature adapter
MRRF full
MRRF no reconstruction
```

比较：

```text
validation / held-out test
clean / A+V 30% missing
Accuracy / Macro-F1 / MAE / Pearson
```

主要输出：

```text
outputs/q2/analysis/
├── all_baselines_three_seed_metrics.csv
├── all_baselines_three_seed_summary.csv
├── all_baselines_validation_robustness.csv
├── all_baselines_results.md
├── fig15_baseline_comparison.pdf
└── fig16_all_baselines_missingness.pdf
```

---

## 11.7 验证集错误分析

模块：

```text
analysis/error_analysis.py
```

分析：

```text
错误样本
置信度
不同缺失模式
分类与回归异常
```

输出：

```text
outputs/q2/analysis/error_analysis/
├── error_cases.csv
├── confidence_analysis.csv
├── error_summary.json
└── missing_pattern.json
```

---

## 11.8 附件3模型分歧

模块：

```text
analysis/attachment3_comparison.py
```

主要比较：

```text
MRRF full
vs
MRRF no_recon
```

附件3没有标签，因此只分析模型预测差异，不计算真实性能。

输出：

```text
outputs/q2/analysis/
├── attachment3_model_disagreement.csv
├── attachment3_disagreement_summary.md
└── fig12_attachment3_disagreement.pdf
```

---

# 12. 主要论文图

主可视化由：

```text
visualization/report.py
```

生成。

| 编号 | 文件 | 内容 |
|---|---|---|
| Fig01 | `fig01_learning_curves.pdf` | 训练/验证学习曲线 |
| Fig02 | `fig02_duration_f1.pdf` | 缺失持续比例与 F1 |
| Fig03 | `fig03_duration_mae.pdf` | 缺失持续比例与 MAE |
| Fig04 | `fig04_ablation.pdf` | 主要MRRF消融 |
| Fig05 | `fig05_position_mae.pdf` | 缺失位置与 MAE |
| Fig06 | `fig06_strength_calibration.pdf` | 情感强度预测/校准 |
| Fig07 | `fig07_confusion.pdf` | 分类混淆矩阵 |
| Fig08 | `fig08_reliability_gate.pdf` | Reliability Gate行为 |
| Fig09 | `fig09_attachment3_missingness.pdf` | 附件3真实缺失模式 |
| Fig10 | `fig10_attachment3_predictions.pdf` | 附件3预测结果 |
| Fig11 | `fig11_three_seeds.pdf` | 三随机种子稳定性 |
| Fig12 | `fig12_attachment3_disagreement.pdf` | 附件3模型分歧 |
| Fig13 | `fig13_mrrf_architecture.pdf` | MRRF结构图 |
| Fig14 | `fig14_contiguous_mask_example.pdf` | 连续缺失mask示例 |
| Fig15 | `fig15_baseline_comparison.pdf` | 七模型三种子比较 |
| Fig16 | `fig16_all_baselines_missingness.pdf` | 多模型缺失鲁棒性 |
| Fig17 | `fig17_cumulative_ablation.pdf` | 累计消融 |
| Fig18 | `fig18_dual_reconstruction.pdf` | 双路重建实验 |
| Fig19 | `fig19_position_rate_interaction.pdf` | 缺失比例 × 位置 |

不同编号图分布在：

```text
outputs/q2/figures/
outputs/q2/analysis/
outputs/q2/experiments/
```

而不是全部集中在同一个目录。

---

# 13. MMSA MulT专项输出

MMSA MulT除了作为baseline参与统一比较，还会单独生成：

```text
outputs/q2/runs/mult/seed_2026/
├── validation_robustness_grid.csv
├── missing_rate_curves.pdf
├── missing_position_heatmap.pdf
├── validation_confusion.pdf
└── baseline_results.md
```

---

# 14. Optuna调参输出

```text
outputs/q2/tuning/
├── study.db
├── trials.json
├── best_config.json
└── comparison/
    ├── run_summary.csv
    ├── paired_accuracy.csv
    ├── seed_summary.csv
    ├── validation_tuning_results.md
    └── figures/
        ├── fig01_validation_accuracy_tradeoff.pdf
        ├── fig02_validation_metrics.pdf
        ├── fig03_clean_confusion.pdf
        └── fig04_paired_accuracy.pdf
```

注意这些是**调参辅助图**，不属于论文主编号 `fig01–fig19`。

---

# 15. 附件3最终提交

最终执行：

```bash
python question2/run.py submit
```

或由：

```bash
python run.py q2
```

自动执行。

主要输出：

```text
outputs/q2/analysis/
├── selected_model.json
└── attachment3_submission.csv
```

`attachment3_submission.csv` 保留：

```text
sample_id
annotation
classification_label
sentiment_strength
```

并检查：

- 样本 01–30 是否齐全；
- 极性文字与类别编号是否一致；
- 情感强度是否位于 `[-3, 3]`；
- 情感强度符号是否与极性一致。

---

# 16. 单独运行Q2模块

除了根目录统一 runner，也可以直接：

```bash
python question2/run.py <command>
```

支持：

```text
train
predict
experiment
error-analysis
analyze
visualize
submit
```

例如：

### MRRF full

```bash
python question2/run.py train \
  --model mrrf \
  --variant full \
  --seed 2026 \
  --epochs 30 \
  --patience 5
```

### MRRF no_recon

```bash
python question2/run.py train \
  --model mrrf \
  --variant no_recon \
  --seed 2026 \
  --epochs 30 \
  --patience 5
```

### MMSA MulT

```bash
python question2/run.py train \
  --model mult \
  --seed 2026 \
  --epochs 30 \
  --patience 5
```

### 独立基线

```bash
python question2/run.py train --model concat_mlp --seed 2026
python question2/run.py train --model gru --seed 2026
python question2/run.py train --model mult_no_aug --seed 2026
python question2/run.py train --model tfr_feature --seed 2026
```

### 实验

```bash
python question2/run.py experiment --name tuning
python question2/run.py experiment --name repeats
python question2/run.py experiment --name ablation --summary-only
python question2/run.py experiment --name dual_reconstruction
python question2/run.py experiment --name missing_duration
python question2/run.py experiment --name robustness
```

### 分析与作图

```bash
python question2/run.py analyze --summary-only
python question2/run.py error-analysis
python question2/run.py visualize
python question2/run.py submit
```

---

# 17. 输出目录概览

完整运行后主要结构类似：

```text
outputs/q2/
├── runs/
│   ├── mult/
│   ├── concat_mlp/
│   ├── gru/
│   ├── mult_no_aug/
│   ├── tfr_feature/
│   └── mrrf/
│       ├── full/
│       ├── no_recon/
│       ├── dual_recon/
│       ├── no_reliability/
│       ├── no_consistency/
│       ├── plain_no_aug/
│       ├── plain_fusion/
│       ├── recon_only/
│       └── optuna_trial_*/
│
├── tuning/
├── experiments/
│   ├── repeats/
│   ├── ablation/
│   ├── missing_duration/
│   └── robustness/
│
├── analysis/
│   └── error_analysis/
│
├── figures/
│
└── full_pipeline_complete.json
```

---

# 18. 训练目录中的典型文件

MRRF正式训练目录通常包含：

```text
checkpoint.pt
history.csv
metrics.json
training_config.json
train_scaler.npz
valid_predictions_clean.csv
valid_predictions_audio_vision.csv
attachment3_predictions_mrrf.csv   # 若执行predict
```

其中：

- `checkpoint.pt`：最佳验证epoch参数；
- `history.csv`：逐epoch训练/验证指标；
- `metrics.json`：最终验证/测试结果；
- `training_config.json`：完整训练参数；
- `train_scaler.npz`：Audio/Vision标准化参数。

---

# 19. 断点续跑

推荐使用根目录：

```bash
python run.py q2
```

统一 runner 会检查：

```text
checkpoint.pt
history.csv
metrics.json
training_config.json
CSV
PDF
submission
```

已完整存在的步骤自动跳过。

因此长实验中途失败后，不需要从头重跑：

```bash
python run.py q2
```

即可继续。

完整流程最终必须通过：

```text
Q2 FINAL AUDIT | PASS
```

并生成：

```text
outputs/q2/full_pipeline_complete.json
```

该审计会检查至少：

```text
29个正式训练checkpoint
>=30个Optuna trial
fig01–fig19
附件3关键预测文件
selected_model.json
attachment3_submission.csv
```

---

# 20. 推荐复现顺序

第一次在新环境运行时：

```bash
# 1. 进入环境
conda activate mathmodel

# 2. 最小化流程测试
python run.py q2 --test

# 3. 完整正式实验
python run.py q2
```

若仅修改可视化：

```bash
python run.py q2 --force-figures
```

若正式流程中途报错：

```bash
# 修复问题后直接重新运行
python run.py q2
```

---

## 21. 注意事项

1. **不要使用附件3进行调参或选模。**  
   附件3无标签，只用于最终专项推理。

2. **不要混用 aligned / unaligned 特征接口。**  
   当前正式实现统一使用 `aligned_50.pkl` 与附件3对齐版本。

3. **测试集不是超参数选择集。**  
   checkpoint选择只使用 validation。

4. **三随机种子实验不能被单次最佳结果替代。**  
   正式比较读取 2026 / 2027 / 2028 三个独立训练结果。

5. **图表不都位于 `outputs/q2/figures/`。**  
   `fig11–fig19` 部分位于 `experiments/` 或 `analysis/`。

6. **`--test` 只是 smoke test。**  
   它验证数据链路和模型代码能否运行，不代表正式30 epoch实验已经完成。

---

## 22. 一句话运行

```bash
python run.py q2 --test && python run.py q2
```

前者检查流程，后者完成问题2全部正式实验、分析、图表和附件3提交。
