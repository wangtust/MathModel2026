# Question 3：可解释性多模态情感预测建模与验证

本目录实现赛题 **问题3：可解释性情感预测建模与验证**。

问题3不重新训练一个新的预测模型，而是在问题2已经完成模型选择之后，**冻结问题2的情感预测模型**，围绕其预测结果构建可量化、可复核、可回查原始视频的解释体系。

当前主解释模型固定为：

```text
MRRF no_recon
seed = 2026
```

对应问题2 checkpoint：

```text
outputs/q2/runs/mrrf/no_recon/seed_2026/checkpoint.pt
```

问题3的主要目标是：

- 输出情感极性；
- 输出情感强度；
- 给出文本、语音、视觉三模态的作用差异；
- 判断当前样本的主要参考模态；
- 找到各模态中的关键局部片段；
- 用独立删除实验验证局部解释是否真的影响模型预测；
- 将关键 token 映射回原始文本、语音时间和视频帧；
- 对解释结果进行数值完备性、模型敏感性和门控一致性核验；
- 对附件4全部 20 条样本生成预测、解释和逐样本解释卡。

---

# 1. 当前数据接口

问题3使用两类数据：

## 1.1 附件2

配置：

```python
# question3/config.py
DATA = ATTACHMENT2_DIR / "aligned_50.pkl"
```

用途：

```text
valid
test
```

主要用于：

- 冻结模型基础性能评价；
- 全验证集解释；
- Integrated Gradients 数值完备性检查；
- Fidelity 删除实验；
- no_recon 与 full 冻结模型解释对照；
- gate 与删除响应排序一致性分析。

问题3不会使用附件2重新训练模型。

---

## 1.2 附件4

当前使用：

```text
附件4-可解释专项视频样本与特征文件/
└── 对齐版本/
    ├── 01.pkl
    ├── 02.pkl
    ├── ...
    ├── 20.pkl
    └── videos/
        ├── 01.mp4
        ├── ...
        └── 20.mp4
```

代码要求恰好：

```text
20 个 [00-99].pkl
```

并检查：

```text
pkl中的id == 文件名
```

每条特征必须满足：

```text
text   = (50, 768)
audio  = (50, 74)
vision = (50, 35)
text_bert = (3, 50)
```

同时必须存在：

```text
videos/<sample_id>.mp4
```

附件4为**无标签专项测试集**：

```text
只用于最终推理和解释
不用于训练
不用于调参
不用于模型选择
```

---

# 2. Question 3目录结构

```text
question3/
├── analysis/
│   ├── cases.py
│   └── verification.py
│
├── attribution/
│   ├── integrated_gradients.py
│   └── modality_importance.py
│
├── evaluation/
│   ├── alignment_quality.py
│   ├── fidelity.py
│   ├── ig_check.py
│   └── sensitivity.py
│
├── evidence/
│   ├── aligner.py
│   ├── mapper.py
│   └── temporal_evidence.py
│
├── visualization/
│   ├── contributions.py
│   ├── evidence.py
│   ├── report.py
│   └── sensitivity.py
│
├── config.py
├── pipeline.py
├── run.py
└── README.md
```

核心实现主要集中在：

```text
pipeline.py
evaluation/ig_check.py
evidence/mapper.py
evaluation/fidelity.py
evaluation/sensitivity.py
evaluation/alignment_quality.py
analysis/verification.py
visualization/report.py
```

---

# 3. 问题3整体流程

完整流程建议理解为：

```text
问题2冻结模型
MRRF no_recon seed2026
          │
          ▼
附件2 valid/test + 附件4
          │
          ▼
基础性能评价
          │
          ▼
Integrated Gradients
          │
          ▼
20%连续关键窗口选择
          │
          ▼
局部删除实验
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
   Top   Low   Random
          │
          ▼
模态贡献 / 主参考模态
          │
          ▼
IG数值完备性检查
          │
          ▼
token → 原词 → WhisperX时间
          │
          ▼
原文本 / 语音秒段 / 视频关键帧
          │
          ▼
full vs no_recon Fidelity
          │
          ▼
冻结模型敏感性
          │
          ▼
Gate vs 删除响应一致性
          │
          ▼
9张主图 + 20张解释卡
          │
          ▼
20页附件4解释卡合订PDF
          │
          ▼
最终 Verification
```

---

# 4. 固定随机种子

```python
SEED = 20260924
```

用于：

- 验证集解释样本选择；
- 随机对照窗口；
- bootstrap；
- 解释复现。

---

# 5. 冻结模型

主模型路径：

```text
outputs/q2/runs/mrrf/no_recon/seed_2026/
```

问题3不会更新该 checkpoint。

`pipeline.py` 中：

```text
CKPT =
outputs/q2/runs/mrrf/no_recon/seed_2026
```

会读取：

```text
checkpoint.pt
train_scaler.npz
```

并使用问题2训练阶段得到的 Audio / Vision scaler。

---

# 6. 为什么问题3使用冻结模型

问题3解释的是：

```text
问题2已经训练并选择好的预测系统
```

因此当前实现不允许：

```text
因为附件4解释结果不好
→ 回头重新训练
→ 再挑一个附件4上更好看的模型
```

这样避免附件4信息泄漏。

---

# 7. Tokenizer一致性检查

附件4每条样本都会使用：

```text
bert-base-uncased
```

重新 tokenization。

优先使用：

```text
cache/models/bert-base-uncased
```

否则使用：

```text
bert-base-uncased
```

代码要求重新编码结果必须和附件4提供的：

```text
text_bert
```

完全一致。

具体检查：

```text
input_ids
attention_mask
```

否则直接：

```text
ValueError: Tokenizer/feature mismatch
```

该设计保证解释时的：

```text
BERT token index
```

和原始附件特征位置一一对应。

---

# 8. 附件4标准化

Audio / Vision 不重新估计 scaler。

直接使用问题2冻结模型对应训练阶段的：

```text
train_scaler.npz
```

执行：

```text
z = (x - train_mean) / train_std
```

再：

```text
clip[-5, 5]
```

原生无效位置继续保持：

```text
0
```

因此附件4预处理接口与问题2模型一致。

---

# 9. 情感预测输出

每条样本输出：

```text
prediction_class
annotation
raw_strength
sentiment_strength

prob_negative
prob_neutral
prob_positive

margin
predicted_probability
```

三分类：

```text
0 = Negative
1 = Neutral
2 = Positive
```

情感强度最终通过问题2的：

```text
coherent_strength()
```

进行极性一致化。

---

# 10. 模型基础性能评价

问题3首先在附件2固定划分上评价被冻结的：

```text
MRRF no_recon seed2026
```

包括：

```text
validation
held-out test
```

指标：

```text
Accuracy
Macro-F1
MAE
Pearson
```

输出：

```text
outputs/q3/tables/model_performance.csv
```

附件4没有标签，因此不会计算：

```text
Accuracy
F1
MAE
Pearson
```

---

# 11. Integrated Gradients解释目标

当前 IG 不是针对单个回归值，而是针对模型当前预测类别的：

```text
logit margin
```

定义为：

```text
预测类别 logit
-
其它类别中的最大 logit
```

即：

```python
margin = own_logit - max(other_logits)
```

这样 IG 解释的是：

> 哪些输入位置支持“当前预测类别相对于最强竞争类别”的决策优势。

---

# 12. IG基线

默认：

```text
IG steps = 48
baseline_factor = 0.1
```

使用：

```text
0.1 × 原输入
```

作为衰减参考点。

不是：

```text
全零输入
```

原因是当前冻结网络在精确零特征附近存在数值变化较陡的问题。

当前实现使用：

```text
midpoint Integrated Gradients
```

并保持：

```text
availability masks
```

固定。

---

# 13. IG默认参数

`pipeline.py`：

```bash
--valid-n 728
--ig-steps 48
--window-fraction 0.2
--random-reps 8
```

默认即：

```text
解释全部728条验证样本
+
附件4全部20条
```

---

# 14. 局部关键窗口选择

每个样本、每个模态独立选择关键窗口。

默认窗口长度：

```text
20% BERT content positions
```

即：

```text
window_fraction = 0.2
```

候选窗口均为：

```text
固定长度连续窗口
```

---

## 14.1 Top window

优先按：

```text
窗口内 positive signed IG 之和
```

选择最大窗口。

如果所有候选窗口：

```text
positive IG sum = 0
```

则退化为：

```text
absolute IG sum
```

选择。

---

## 14.2 Low window

在与 Top window：

```text
相同内容宽度
+
相同原生有效位置数
```

的候选中选择最低 IG 窗口。

---

## 14.3 Random window

同样要求：

```text
相同内容宽度
+
相同原生有效位置数
```

随机抽取：

```text
8次
```

这样可以避免 Audio / Vision 原生缺失位置不同导致随机窗口天然“更弱”。

---

# 15. 删除实验

选出窗口后，不只看 IG。

代码会真的修改输入：

```text
对应模态窗口特征 → 0
对应 available mask → False
```

该操作与问题2缺失机制保持一致。

然后重新前向推理。

记录：

```text
delta_margin
delta_probability
absolute_delta_strength
```

---

# 16. 为什么必须做删除实验

Integrated Gradients本身是：

```text
梯度归因
```

并不自动证明：

```text
删除这个片段
→ 模型预测真的明显变化
```

因此当前问题3使用：

```text
IG提出候选片段
+
Deletion response验证
```

形成两级解释。

---

# 17. 三种局部对照

每个模态比较：

```text
Top IG window
Low IG window
Random matched window
```

重点统计：

```text
Top - Random
Top - Low
```

如果 Top 删除造成的预测 margin 下降更大，说明该局部解释具有更好的模型忠实度。

---

# 18. 有符号删除响应

当前代码保留：

```text
signed delta margin
```

而不是全部取绝对值。

因此：

```text
delta_margin > 0
```

表示：

```text
删除该窗口
→ 原预测类别margin下降
```

说明该片段正向支持当前预测。

而：

```text
delta_margin < 0
```

表示：

```text
删除后反而增强原预测
```

因此不能把所有 IG 高值都解释为：

```text
正向因果贡献
```

---

# 19. 模态贡献

对每个模态取：

```text
max(top_delta_margin, 0)
```

然后在三模态之间归一化：

```text
deletion_contribution_text
deletion_contribution_audio
deletion_contribution_vision
```

这表示：

```text
基于局部有符号删除响应得到的相对正向贡献
```

并不是直接把模型 gate 当作贡献百分比。

---

# 20. 主要参考模态

当前规则：

```text
total positive effect > 0.05
and
max modality effect > 0.02
```

才给出：

```text
text
audio
vision
```

否则：

```text
unclear
```

因此：

```text
dominant_modality
```

允许不确定。

不会强制每条样本都选一个主模态。

---

# 21. Gate权重

模型内部同时保存：

```text
gate_text
gate_audio
gate_vision
```

以及：

```text
availability_text
availability_audio
availability_vision
```

但当前代码明确把：

```text
Gate
```

视为：

```text
模型内部融合机制
```

不是：

```text
因果贡献比例
```

主要参考模态仍以删除响应为主要依据。

---

# 22. Validation Explanation

默认：

```text
valid_n = 728
```

即解释附件2：

```text
全部728条validation
```

选择函数：

```text
choose_validation_indices()
```

按真实类别比例进行：

```text
固定种子
分层抽样
```

默认覆盖全验证集时即：

```text
728/728
```

---

# 23. Pipeline阶段主要输出

执行：

```bash
python question3/run.py pipeline
```

生成：

```text
outputs/q3/
├── config.json
├── attachment4_metadata.json
└── tables/
    ├── model_performance.csv
    ├── validation_predictions_explained.csv
    ├── validation_evidence.csv
    ├── validation_position_ig.csv
    ├── validation_explanation_metrics.csv
    ├── attachment4_predictions.csv
    ├── attachment4_evidence_raw.csv
    └── attachment4_position_ig.csv
```

---

# 24. `config.json`

记录：

```text
model
checkpoint
checkpoint_sha256
checkpoint_epoch

source_data
attachment4
tokenizer
device

ig_steps
ig_baseline
ig_baseline_factor

window_fraction
random_reps

validation_explanation_subset_requested
seed

selection_rule
dominant_rule
```

用于保证解释实验可复现。

---

# 25. Attachment 4预测表

核心文件：

```text
tables/attachment4_predictions.csv
```

至少包含：

```text
sample_id

prediction_class
annotation

raw_strength
sentiment_strength

margin
predicted_probability

prob_negative
prob_neutral
prob_positive

dominant_modality

gate_text
gate_audio
gate_vision

availability_text
availability_audio
availability_vision

deletion_contribution_text
deletion_contribution_audio
deletion_contribution_vision

top_delta_margin_text
top_delta_margin_audio
top_delta_margin_vision
```

后续：

```text
evidence
sensitivity
```

阶段还会继续向该表追加质量字段和 full 模型敏感性信息。

---

# 26. 每模态证据

原始证据：

```text
attachment4_evidence_raw.csv
```

每条样本最多：

```text
3 rows
```

对应：

```text
text
audio
vision
```

记录：

```text
token_start
token_end
window_width

native_valid_positions_in_window

ig_signed_sum
ig_abs_sum
ig_positive_sum

top_delta_margin
random_delta_margin
low_delta_margin

top_delta_prob
random_delta_prob
low_delta_prob

top_abs_delta_strength
random_abs_delta_strength
low_abs_delta_strength
```

---

# 27. 位置级IG

输出：

```text
attachment4_position_ig.csv
validation_position_ig.csv
```

每个内容 token 记录：

```text
sample_id
modality
token_index
native_available
ig_signed
ig_absolute
selected
```

用于：

- 局部重要性热图；
- 数值完备性检查；
- 逐样本解释卡。

---

# 28. IG数值完备性检查

执行：

```bash
python question3/run.py ig-check
```

输出：

```text
tables/attachment4_ig_completeness.csv
tables/validation_ig_completeness.csv
```

检查：

```text
Σ IG
```

是否接近：

```text
margin(x) - margin(0.1x)
```

记录：

```text
margin_difference
sum_ig
absolute_completeness_error
```

最终 verification 要求：

```text
附件4 20条全部覆盖
验证集728条全部覆盖
最大 absolute completeness error < 0.1
```

---

# 29. 原始证据时间映射

执行：

```bash
python question3/run.py evidence
```

主要模块：

```text
evidence/mapper.py
evidence/aligner.py
```

目标：

```text
BERT token
→ 原始文本字符
→ 原始单词
→ WhisperX词时间
→ 原视频秒数
→ 视频关键帧
```

---

# 30. 原始文本词切分

使用规则：

```text
[A-Za-z0-9]+
+
内部撇号
+
可选标点
```

得到：

```text
word_index
word
char_start
char_end
```

---

# 31. WhisperX Forced Alignment

`ForcedAligner` 使用项目内：

```text
vendor/whisperX
```

加载：

```text
English align model
```

输入：

```text
附件4原视频抽出的16kHz音频
+
附件4官方raw_text
```

输出只保留具有：

```text
word
start
end
score
```

且时间合法的词。

---

# 32. 原始词与WhisperX词匹配

第一阶段使用：

```text
SequenceMatcher
```

做严格词序列匹配。

对 replace block 中未匹配的词，再允许：

```text
string similarity >= 0.8
```

的保守补充匹配。

---

# 33. 独立ASR一致性检查

除了 WhisperX forced alignment，还使用：

```text
faster-whisper tiny.en
CPU
int8
beam_size = 3
vad_filter = True
```

重新听一次原始音频。

它只作为：

```text
独立 mismatch screen
```

不是 ground truth。

计算：

```text
ASR transcript
vs
官方 raw_text
```

序列相似度。

---

# 34. 时间映射可信规则

样本级时间定位只有同时满足：

```text
word coverage >= 0.85
mean WhisperX alignment score >= 0.30
independent ASR similarity >= 0.35
无alignment error
```

才：

```text
alignment_status = trusted
```

否则可能是：

```text
asr_unavailable
asr_mismatch
alignment_failed
low_coverage_or_score
```

---

# 35. 词级时间可信规则

即使样本整体：

```text
trusted
```

单个词还必须：

```text
word alignment score >= 0.30
```

才：

```text
timing_trusted = True
```

因此：

```text
样本整体可信
≠
所有局部证据都有可信秒数
```

---

# 36. 不可信时不伪造时间

如果某个解释片段没有可信词时钟：

```text
保留：
token index
source text
IG
deletion response
```

但：

```text
start_sec
end_sec
keyframe_sec
```

保持空。

不会用：

```text
token位置 × 视频时长 / 50
```

这类简单线性映射伪造时间。

---

# 37. Evidence阶段输出

```text
tables/
├── attachment4_alignment_audit.csv
├── attachment4_word_map.csv
├── attachment4_token_map.csv
└── attachment4_evidence.csv
```

其中：

### `attachment4_alignment_audit.csv`

样本级：

```text
duration_sec
source_word_count
aligned_word_count
word_coverage
mean_alignment_score
asr_similarity
asr_transcript
alignment_status
text_truncated
retained_char_end
```

### `attachment4_word_map.csv`

词级：

```text
word_index
word
char_start
char_end
start_sec
end_sec
score
whisperx_word
timing_trusted
```

### `attachment4_token_map.csv`

token级：

```text
token_index
char_start
char_end
token_text
word_indices
alignment_status
```

### `attachment4_evidence.csv`

最终可回查证据：

```text
token范围
原文片段
秒数范围
关键帧时间
关键帧路径
定位状态
质量信息
```

---

# 38. Text truncation

BERT最大：

```text
50 positions
```

因此部分长文本可能被截断。

代码显式记录：

```text
text_truncated
retained_char_end
```

解释不能引用：

```text
模型实际未看到的截断后文本
```

Verification 会检查证据字符范围不能越过：

```text
retained_char_end
```

---

# 39. Fidelity实验

执行：

```bash
python question3/run.py fidelity
```

该阶段做两件事：

1. 对主模型 `no_recon` 已有解释结果做对照；
2. 在**完全相同的728条validation** 上，用 `MRRF full seed2026` 重新做一遍完整解释。

full checkpoint：

```text
outputs/q2/runs/mrrf/full/seed_2026/
```

---

# 40. Full模型解释

生成：

```text
tables/
├── full_validation_predictions_explained.csv
├── full_validation_evidence.csv
├── full_validation_position_ig.csv
├── full_validation_explanation_metrics.csv
├── full_validation_ig_completeness.csv
└── full_validation_ig_quadrature_check.csv
```

使用和主模型完全一致的：

```text
validation IDs
IG steps
window fraction
random seed
random reps
```

---

# 41. no_recon vs full配对比较

生成：

```text
validation_fidelity_model_paired.csv
validation_fidelity_model_summary.csv
```

每个可比较：

```text
sample × modality
```

对比：

```text
no_recon
vs
full
```

主要指标：

```text
Top - Random margin advantage
Top - Low margin advantage

Top - Random probability advantage
Top - Low probability advantage
```

同时记录：

```text
same_predicted_class
```

用于区分：

```text
两个模型解释的是同一个预测类别
```

还是：

```text
两个模型本身预测类别不同
```

---

# 42. Fidelity Bootstrap

统计使用：

```text
2000次 bootstrap
```

输出：

```text
mean
95% CI
```

用于判断：

```text
Top IG窗口是否稳定优于随机窗口
```

---

# 43. Fidelity分层

`fidelity_tables()` 会分：

```text
all
correct
incorrect

short_window_1to4
medium_window_5to8
long_window_9plus
```

分别统计。

因此不只看总体均值。

---

# 44. Full模型最差IG数值复核

对 full 模型：

```text
48步IG
```

中 completeness error 最大的样本，自动继续使用：

```text
48
96
192
```

步重新计算。

输出：

```text
full_validation_ig_quadrature_check.csv
```

最终要求：

```text
192步后的 error < 0.1
```

---

# 45. Fidelity记录文件

```text
full_validation_comparison.json
```

保存：

```text
full checkpoint
checkpoint SHA256
checkpoint epoch

validation sample count
paired modality row count

IG steps
baseline factor
window fraction
random reps
bootstrap reps

comparison scope
selection policy
```

并明确：

```text
这是选模后的诊断
不会反向改变问题3主模型
```

---

# 46. 冻结模型敏感性

执行：

```bash
python question3/run.py sensitivity
```

比较：

```text
MRRF no_recon seed2026
vs
MRRF full seed2026
```

仅对：

```text
附件4 20条
```

比较预测变化。

---

# 47. Sensitivity输出

```text
tables/attachment4_model_sensitivity.csv
model_sensitivity.json
```

记录：

```text
primary_annotation
primary_strength

full_annotation
full_strength
full_raw_strength

full_prob_negative
full_prob_neutral
full_prob_positive

class_agreement
absolute_strength_difference
```

由于附件4无标签：

```text
该实验只描述模型选择敏感性
不能判断full还是no_recon更准确
```

---

# 48. Gate vs删除响应

执行：

```bash
python question3/run.py alignment
```

这里的 `alignment` 指：

```text
解释机制之间的一致性
```

不是：

```text
多模态时间对齐
```

主要比较：

```text
模型内部 gate 权重排序
```

和：

```text
实际删除响应排序
```

是否一致。

---

# 49. 两种删除响应

分别比较：

```text
top_window
```

即：

```text
局部20%关键窗口删除
```

以及：

```text
whole_modality
```

即：

```text
整模态清零
+
availability mask置False
```

注意：

```text
whole_modality
```

比问题2主要训练缺失场景更强，因此只作为额外敏感性诊断。

---

# 50. 排序评价

每条样本计算：

```text
Spearman rho
top-1 agreement
```

并分别汇总：

```text
validation
attachment4
```

×：

```text
top_window
whole_modality
```

共四组。

---

# 51. Gate一致性输出

```text
tables/
├── whole_modality_deletion.csv
├── gate_response_rank.csv
└── gate_response_rank_summary.csv
```

`gate_response_rank_summary.csv` 还包括：

```text
top1 agreement
95% CI

independent marginal expected top1
Cohen's kappa

mean Spearman rho
95% CI

各模态成为top的比例
```

---

# 52. Figures阶段

执行：

```bash
python question3/run.py figures
```

生成：

```text
9张主图
+
20张逐样本解释卡
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

---

# 53. 主图说明

## Fig01

```text
fig01_performance.pdf
```

内容：

```text
冻结模型基础性能
+
解释忠实度摘要
```

---

## Fig02

```text
fig02_fidelity.pdf
```

内容：

```text
Top IG窗口
vs
Random
vs
Low IG窗口
```

的验证集配对差异及：

```text
95% bootstrap CI
```

---

## Fig03

```text
fig03_modality_contributions.pdf
```

附件4全部20条：

左：

```text
Text / Audio / Vision
signed deletion response
```

右：

```text
模型内部fusion gate
```

用于强调：

```text
门控权重 ≠ 因果贡献
```

---

## Fig04

```text
fig04_local_importance.pdf
```

展示附件4：

```text
Text / Audio / Vision
×
20 samples
×
BERT positions
```

的：

```text
signed IG
```

并用黑框标记选中20%窗口。

---

## Fig05

```text
fig05_alignment_quality.pdf
```

展示：

```text
WhisperX word coverage
Independent ASR similarity
```

用于审核 token → 原视频时间映射质量。

---

## Fig06

```text
fig06_timed_evidence.pdf
```

当前代码固定展示：

```text
sample 19
```

包括：

```text
原始音轨波形
三模态候选秒段
对应原文
视频关键帧
```

---

## Fig07

```text
fig07_model_sensitivity.pdf
```

比较：

```text
no_recon
vs
full
```

在附件4上的：

```text
情感强度
极性一致性
```

---

## Fig08

```text
fig08_validation_variant_fidelity.pdf
```

比较：

```text
full - no_recon
```

的局部解释 Fidelity 差异。

同时报告：

```text
margin尺度
probability尺度
```

以及：

```text
all paired samples
same predicted class subset
```

---

## Fig09

```text
fig09_gate_response_rank.pdf
```

比较：

```text
gate最高模态
vs
删除响应最高模态
```

同时展示：

```text
top-1 agreement
marginal baseline
Spearman rho
```

---

# 54. 逐样本解释卡

附件4：

```text
01
02
...
20
```

全部生成：

```text
figures/case_01.pdf
figures/case_02.pdf
...
figures/case_20.pdf
```

以及对应 PNG。

每张解释卡包括：

```text
a. 三模态局部 signed IG
b. token / 原文 / 可信秒段
c. 删除后的预测margin变化
d. 可用时的原视频关键帧
```

---

# 55. 解释卡合订PDF

Figures阶段会调用：

```text
pdfunite
```

将：

```text
case_01.pdf
...
case_20.pdf
```

合并为：

```text
paper/attachment4_explanation_cards.pdf
```

最终必须：

```text
20 pages
```

---

# 56. 论文结果说明

生成：

```text
paper/q3_results.md
```

包含：

- 冻结模型说明；
- validation/test基础性能；
- Fidelity结果；
- IG完备性；
- no_recon vs full；
- Gate vs deletion；
- 附件4定位与异常；
- 全部20条预测摘要；
- 论文主图链接。

---

# 57. 图注文件

生成：

```text
paper/figure_captions.md
```

包含：

```text
Fig01–Fig09
附件4 20条解释卡
```

的正式图注。

适合直接作为论文图注基础。

---

# 58. 推荐正文典型样本

当前 report 中特别指出：

```text
01
07
13
14
15
19
```

可作为正文典型样本。

其余：

```text
14条
```

放补充材料。

这些样本分别覆盖：

```text
普通案例
文本截断
视觉原生缺失
局部时间不可信
ASR/文本对齐拒绝
可可信回查的视觉时间证据
```

等复杂情况。

---

# 59. Verification阶段

执行：

```bash
python question3/run.py verify
```

输出：

```text
outputs/q3/verification.json
```

如果任一硬条件失败：

```text
exit code = 1
```

---

# 60. Verification检查范围

包括但不限于：

### 附件4覆盖

```text
01–20全部存在
```

### Evidence

```text
每条样本3个模态
```

### 概率

```text
Negative + Neutral + Positive = 1
```

### 极性/强度一致性

```text
Negative → strength <= 0
Neutral  → strength = 0
Positive → strength >= 0
```

### 时间

```text
0 <= start < end <= 视频时长
```

### 时间可信性

有秒数必须：

```text
localization_status = trusted_word_alignment
```

### Vision frame

视觉关键帧路径必须存在。

### Token范围

有效解释必须位于：

```text
1 <= token <= 48
```

### BERT truncation

不能引用：

```text
模型未看到的截断后文本
```

---

# 61. 特殊样本硬检查

Verification还显式检查：

### Sample 15

必须：

```text
alignment_status != trusted
```

且所有解释：

```text
start_sec = NaN
```

### Sample 13

必须保留：

```text
vision available = False
```

### Sample 07 / 18

必须：

```text
text_truncated = True
```

---

# 62. Validation解释覆盖

必须：

```text
728 validation samples
```

全部获得解释。

---

# 63. IG完整性

必须：

```text
attachment4 completeness rows = 20
validation completeness rows = 728
```

并：

```text
max absolute completeness error < 0.1
```

---

# 64. Full模型解释覆盖

必须：

```text
full_validation_predictions = 728
full_validation_evidence = 728 × 3
full_validation_IG = 728
```

---

# 65. Full最差IG复核

必须：

```text
48
96
192
```

三档步数。

而且：

```text
192-step error < 0.1
```

---

# 66. Paired Fidelity

最终至少要求：

```text
paired rows >= 2000
```

且：

```text
sample_id + modality
```

不能重复。

汇总：

```text
12 rows
```

并包含：

```text
margin
prob
```

两种 metric。

---

# 67. Gate审核覆盖

Whole-modality删除：

```text
(728 + 20) × 3
```

行。

Gate rank：

```text
(728 + 20) × 2
```

行。

summary：

```text
4 rows
```

---

# 68. 图表完整性

Verification要求：

```text
29 PDFs
```

每个 PDF 必须有对应：

```text
PNG
```

其中：

```text
9张主图
+
20张case卡
```

---

# 69. 20页解释卡

必须存在：

```text
paper/attachment4_explanation_cards.pdf
```

并通过：

```text
pdfinfo
```

确认：

```text
Pages = 20
```

---

# 70. Verification复现信息

`verification.json` 还保存：

```text
package_versions
```

包括：

```text
torch
numpy
transformers
whisperx
faster-whisper
matplotlib
pandas
scikit-learn
scipy
```

以及主要问题3代码文件的：

```text
SHA256
```

---

# 71. 正确的完整运行顺序

直接运行问题3模块时，推荐：

```bash
python question3/run.py pipeline
python question3/run.py ig-check
python question3/run.py evidence
python question3/run.py fidelity
python question3/run.py sensitivity
python question3/run.py alignment
python question3/run.py figures
python question3/run.py verify
```

注意：

```text
verify 必须放在 figures 后面
```

因为 verification 会检查：

```text
29个PDF
29个PNG
20张case
20页合订PDF
```

---

# 72. 各阶段依赖关系

```text
pipeline
   │
   ├────► ig-check
   │
   ├────► evidence
   │
   ├────► fidelity
   │
   └────► sensitivity
             │
             ▼
alignment
   │
   ▼
figures
   │
   ▼
verify
```

更严格地说：

```text
figures
```

依赖：

```text
pipeline
ig-check
evidence
fidelity
sensitivity
alignment
```

的全部结果。

---

# 73. Smoke Test

项目根目录支持：

```bash
python run.py q3 --test
```

当前 smoke test 会：

1. 编译问题3代码；
2. 导入 pipeline / IG / evidence / fidelity / sensitivity / alignment / figures；
3. 检查问题2冻结 checkpoint；
4. 读取附件4全部20条；
5. 取1条真实样本；
6. 实际执行预测；
7. 使用极小：
   ```text
   IG steps = 2
   random reps = 1
   ```
8. 实际执行 deletion explanation；
9. 检查三模态 evidence。

测试结果：

```text
outputs/_smoke/q3/
```

不会写入：

```text
outputs/q3/
```

---

# 74. Smoke Test边界

`q3 --test` 的目的：

```text
快速发现数据路径
checkpoint
CUDA
模型前向
IG
deletion
模块import
```

问题。

它不会完整执行：

```text
728 validation × 48-step IG
20 attachment4 × full mapping
全部fidelity
全部figures
```

因此：

```text
smoke PASS
≠
正式实验已经全部完成
```

---

# 75. Pipeline可调参数

可以直接：

```bash
python question3/run.py pipeline \
  --valid-n 728 \
  --ig-steps 48 \
  --window-fraction 0.2 \
  --random-reps 8
```

正式论文结果建议保持当前默认配置。

---

# 76. 调试用小规模Pipeline

若仅排查代码：

```bash
python question3/run.py pipeline \
  --valid-n 30 \
  --ig-steps 4 \
  --random-reps 2 \
  --output outputs/_q3_debug
```

注意：

```text
这种结果不能通过最终verification
```

因为 verification 要求：

```text
728 validation解释
```

完整覆盖。

---

# 77. 自定义输出目录

除 pipeline 外，Q3阶段统一支持：

```bash
--output <path>
```

例如：

```bash
python question3/run.py ig-check \
  --output outputs/_q3_debug
```

这样可以避免污染正式：

```text
outputs/q3/
```

---

# 78. 输出目录概览

完整运行后：

```text
outputs/q3/
├── config.json
├── attachment4_metadata.json
├── full_validation_comparison.json
├── model_sensitivity.json
├── verification.json
│
├── tables/
│   ├── model_performance.csv
│   │
│   ├── validation_predictions_explained.csv
│   ├── validation_evidence.csv
│   ├── validation_position_ig.csv
│   ├── validation_explanation_metrics.csv
│   ├── validation_ig_completeness.csv
│   │
│   ├── attachment4_predictions.csv
│   ├── attachment4_evidence_raw.csv
│   ├── attachment4_position_ig.csv
│   ├── attachment4_ig_completeness.csv
│   │
│   ├── attachment4_alignment_audit.csv
│   ├── attachment4_word_map.csv
│   ├── attachment4_token_map.csv
│   ├── attachment4_evidence.csv
│   │
│   ├── full_validation_predictions_explained.csv
│   ├── full_validation_evidence.csv
│   ├── full_validation_position_ig.csv
│   ├── full_validation_explanation_metrics.csv
│   ├── full_validation_ig_completeness.csv
│   ├── full_validation_ig_quadrature_check.csv
│   │
│   ├── validation_fidelity_model_paired.csv
│   ├── validation_fidelity_model_summary.csv
│   │
│   ├── attachment4_model_sensitivity.csv
│   │
│   ├── whole_modality_deletion.csv
│   ├── gate_response_rank.csv
│   └── gate_response_rank_summary.csv
│
├── figures/
│   ├── fig01_performance.pdf/.png
│   ├── fig02_fidelity.pdf/.png
│   ├── fig03_modality_contributions.pdf/.png
│   ├── fig04_local_importance.pdf/.png
│   ├── fig05_alignment_quality.pdf/.png
│   ├── fig06_timed_evidence.pdf/.png
│   ├── fig07_model_sensitivity.pdf/.png
│   ├── fig08_validation_variant_fidelity.pdf/.png
│   ├── fig09_gate_response_rank.pdf/.png
│   │
│   ├── case_01.pdf/.png
│   ├── ...
│   └── case_20.pdf/.png
│
├── paper/
│   ├── q3_results.md
│   ├── figure_captions.md
│   └── attachment4_explanation_cards.pdf
│
└── ...
```

---

# 79. Q3主要论文图

| 图 | 文件 | 主要信息 |
|---|---|---|
| Fig01 | `fig01_performance` | 冻结模型性能 + 解释摘要 |
| Fig02 | `fig02_fidelity` | Top vs Random/Low |
| Fig03 | `fig03_modality_contributions` | 附件4三模态删除贡献 + gate |
| Fig04 | `fig04_local_importance` | 20样本局部IG |
| Fig05 | `fig05_alignment_quality` | WhisperX / ASR定位质量 |
| Fig06 | `fig06_timed_evidence` | 样本19原视频可回查证据 |
| Fig07 | `fig07_model_sensitivity` | no_recon vs full |
| Fig08 | `fig08_validation_variant_fidelity` | 两冻结模型解释忠实度 |
| Fig09 | `fig09_gate_response_rank` | gate vs deletion排序 |

---

# 80. 问题3核心结果文件

如果只挑最重要的交付文件：

```text
tables/model_performance.csv

tables/attachment4_predictions.csv
tables/attachment4_evidence.csv

tables/attachment4_alignment_audit.csv
tables/attachment4_word_map.csv

tables/validation_explanation_metrics.csv
tables/validation_fidelity_model_summary.csv
tables/gate_response_rank_summary.csv

paper/q3_results.md
paper/attachment4_explanation_cards.pdf

verification.json
```

---

# 81. 问题3解释体系总结

当前问题3不是只生成：

```text
一张attention heatmap
```

而是建立：

```text
预测
→ IG候选
→ 删除验证
→ 模态贡献
→ 主参考模态
→ token定位
→ 原词定位
→ 原视频秒数
→ 关键帧
→ IG完备性
→ Fidelity
→ 冻结模型敏感性
→ Gate一致性
→ Verification
```

完整证据链。

---

# 82. 关键方法边界

## 1. IG不是绝对因果贡献

它解释的是：

```text
相对0.1×input参考点
```

的梯度积分。

---

## 2. Gate不是贡献百分比

模型内部 gate：

```text
不能直接当作因果解释
```

因此代码单独用删除响应验证。

---

## 3. 删除响应保留正负号

如果：

```text
delta_margin < 0
```

必须如实报告：

```text
删除后模型反而更坚定
```

不能强行解释为正贡献。

---

## 4. 时间映射是候选证据映射

Audio / Vision 特征本身没有逐位置秒数。

当前使用：

```text
BERT token
→ 原词
→ WhisperX forced alignment
→ 原视频秒数
```

建立可审核映射。

对齐不可信时：

```text
不输出秒数
```

---

## 5. 关键帧只是时间定位

关键帧表示：

```text
该视觉token对应原视频的大致时刻
```

并不意味着：

```text
模型关注了画面中的具体物体
```

---

## 6. 附件4无标签

因此附件4上的：

```text
模型差异
解释差异
gate差异
```

只能做：

```text
敏感性和一致性分析
```

不能判断：

```text
哪个模型更准确
```

---

# 83. 运行前依赖

问题3至少要求：

```text
问题2 no_recon seed2026 checkpoint
问题2 full seed2026 checkpoint

aligned_50.pkl

附件4对齐版本
附件4 20条原视频

bert-base-uncased tokenizer

WhisperX
faster-whisper
ffmpeg
ffprobe
pdfunite
pdfinfo
```

因此正式运行问题3前，应先完成：

```bash
python run.py q2
```

至少确保：

```text
outputs/q2/runs/mrrf/no_recon/seed_2026/
outputs/q2/runs/mrrf/full/seed_2026/
```

存在。

---

# 84. 推荐复现顺序

第一次运行：

```bash
conda activate mathmodel

# 1. 最小真实链路测试
python run.py q3 --test

# 2. 正式问题3
python question3/run.py pipeline
python question3/run.py ig-check
python question3/run.py evidence
python question3/run.py fidelity
python question3/run.py sensitivity
python question3/run.py alignment
python question3/run.py figures
python question3/run.py verify
```

---

# 85. 关于项目根目录 `run.py q3`

问题3最终验证逻辑要求：

```text
figures
```

先于：

```text
verify
```

因为：

```text
verification.py
```

会检查已经生成的：

```text
29个PDF
29个PNG
20张case卡
20页合订PDF
```

因此根目录统一 runner 的正确顺序应为：

```text
pipeline
ig-check
evidence
fidelity
sensitivity
alignment
figures
verify
```

如果根目录 `run.py` 当前仍是：

```text
...
alignment
verify
figures
```

请先调整这两个阶段的顺序，再使用：

```bash
python run.py q3
```

---

# 86. Figures重建

若前面所有表格已经完成，只修改可视化：

```bash
python question3/run.py figures
```

即可重建：

```text
9张主图
20张case卡
q3_results.md
figure_captions.md
20页合订PDF
```

之后建议重新：

```bash
python question3/run.py verify
```

---

# 87. 验证成功标准

最终：

```bash
python question3/run.py verify
```

应该：

```text
passed = true
issues = []
```

并生成：

```text
outputs/q3/verification.json
```

只有在：

```text
预测
解释
时间映射
IG
Fidelity
Sensitivity
Gate audit
Figures
Case cards
```

全部齐全时，问题3才算完整完成。

---

# 88. 一句话流程

```text
冻结Q2模型
→ 全验证集与附件4预测
→ Integrated Gradients
→ 连续关键窗口
→ 删除忠实度验证
→ 模态贡献
→ 主参考模态
→ token/原词/秒数/关键帧
→ full模型解释对照
→ 模型敏感性
→ gate排序核验
→ 9张主图
→ 20张解释卡
→ 最终Verification
```

---

# 89. 最推荐的运行命令

```bash
python run.py q3 --test
```

确认最小链路后，按正确阶段顺序执行正式实验：

```bash
python question3/run.py pipeline && \
python question3/run.py ig-check && \
python question3/run.py evidence && \
python question3/run.py fidelity && \
python question3/run.py sensitivity && \
python question3/run.py alignment && \
python question3/run.py figures && \
python question3/run.py verify
```

最终：

```text
outputs/q3/verification.json
```

显示：

```text
passed: true
```

即表示问题3完整复现实验通过。
