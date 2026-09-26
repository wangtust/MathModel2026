# Question 1：多模态情感特征提取与时序对齐

本目录实现赛题 **问题1：多模态情感特征提取与时序对齐的数学建模**。

核心任务是从附件1的 100 条原始视频中提取：

- 文本特征
- 语音特征
- 视觉特征
- 原生时间戳
- 词级跨模态时序对齐结果
- 秒级时间网格特征
- 模态有效性 mask
- 异常/待复核状态
- 典型样本验证图与可追溯证据

最终每条样本保存为独立 `.npz` 特征文件，同时保存 JSON 日志、全量汇总表、质量审计结果和论文级可视化。

---

## 1. 当前数据接口

问题1读取赛题附件1：

```text
附件1-数据集原始多模态样本/
└── MOSEI数据集部分原始视频-100条/
    ├── label-100.xlsx
    ├── <video_id>/
    │   ├── <clip_id>.mp4
    │   └── ...
    └── ...
```

配置入口：

```python
# question1/config.py
DATA = ATTACHMENT1_DIR / "MOSEI数据集部分原始视频-100条"
OUT = Q1_OUTPUT_DIR
```

`label-100.xlsx` 必须包含：

```text
video_id
clip_id
text
label
annotation
```

代码会检查：

- 样本总数是否为 100；
- `video_id + clip_id` 是否唯一；
- 每条样本是否能够映射到对应 MP4；
- 官方文本、标签与输出日志是否保持一致。

样本 ID 统一写成：

```text
<video_id>_<clip_id>
```

例如：

```text
-3g5yACwYnA_13
```

---

# 2. 问题1整体流程

完整流程：

```text
附件1原始视频 + label-100.xlsx
              │
              ▼
        样本清单与媒体检查
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
     文本    语音      视觉
       │      │        │
      BERT  openSMILE  MediaPipe
       │      │        │
       │   Silero VAD  Face Detector
       │      │        Face Landmarker
       │   WhisperX       │
       │      │           │
       └──────┼───────────┘
              ▼
        官方文本—ASR匹配
              │
              ▼
       可信词级时间区间
              │
       ┌──────┴──────┐
       ▼             ▼
   词级对齐       1秒时间网格
       │
       ▼
固定80词序列 + mask
       │
       ▼
跨语言语音检测与句段级升级
       │
       ▼
质量审计 / 异常样本复核
       │
       ▼
论文图表 / 典型样本 / 映射表
```

---

# 3. 目录结构

```text
question1/
├── alignment/
│   ├── temporal.py
│   └── crosslingual.py
│
├── analysis/
│   ├── cases.py
│   └── quality.py
│
├── extraction/
│   ├── core.py
│   ├── text.py
│   ├── audio.py
│   └── vision.py
│
├── visualization/
│   ├── cases.py
│   ├── crosslingual.py
│   └── journal_cases.py
│
├── assets/
│   ├── blaze_face_full_range.tflite
│   ├── face_landmarker.task
│   └── punkt_tab.zip
│
├── config.py
├── pipeline.py
├── run.py
└── README.md
```

实际主要实现集中在：

```text
extraction/core.py
pipeline.py
alignment/temporal.py
alignment/crosslingual.py
analysis/quality.py
```

`text.py`、`audio.py`、`vision.py` 当前主要作为 extraction facade，底层统一调用 `core.Extractors`。

---

# 4. 关键配置

`question1/config.py`：

```python
FPS = 2.0
SECOND_STEP = 1.0
MAX_WORDS = 80

ASR_MODEL = "base.en"
ASR_DEVICE = "cuda"
ASR_COMPUTE = "float16"

MATCH_THRESHOLD = 0.55
VAD_THRESHOLD = 0.5
```

含义：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `FPS` | 2.0 | 视觉采样频率 |
| `SECOND_STEP` | 1.0 s | 秒级统一时间网格 |
| `MAX_WORDS` | 80 | 固定长度词序列 |
| `ASR_MODEL` | `base.en` | 英语 WhisperX |
| `ASR_DEVICE` | CUDA | ASR设备 |
| `ASR_COMPUTE` | float16 | ASR计算精度 |
| `MATCH_THRESHOLD` | 0.55 | 官方文本与ASR词形覆盖接受阈值 |
| `VAD_THRESHOLD` | 0.5 | Silero VAD阈值 |

---

# 5. 文本特征

文本以 `label-100.xlsx` 中的**官方文本**为主轴。

使用：

```text
bert-base-uncased
```

通过 MMSA-FET 提取 WordPiece 表示。

原始 token 特征：

```text
text_token
shape ≈ [token_count, 768]
```

再根据 BERT `word_ids` 将同一个官方词对应的 WordPiece 向量求均值，得到：

```text
text_word_native
shape = [official_word_count, 768]
```

最后填充到固定长度：

```text
word_text
shape = [80, 768]
```

同时保存：

```text
text_token_count
word_valid
```

用于记录每个官方词的 token 数以及固定80位置中的真实词/padding。

---

## 5.1 官方文本不是由ASR替代

本模块的设计原则是：

```text
官方文本 = 文本模态内容
ASR      = 用于寻找原视频中官方词的真实发音时间
```

因此即使：

```text
官方文本 ≠ ASR结果
```

也不会直接把 ASR 文本替换官方文本。

这类样本会进入：

```text
mismatch_candidate
```

供后续人工复核。

---

# 6. 语音特征

语音通过 MMSA-FET / openSMILE 提取：

```text
feature_set   = eGeMAPSv02
feature_level = LowLevelDescriptors
sample_rate   = 16000
```

当前实际输出声学维度：

```text
25
```

保存：

```text
audio_native
audio_time
```

其中：

```text
audio_native : [audio_frames, 25]
audio_time   : [audio_frames]
```

保留原生语音特征时间轴，不只保存池化结果。

---

# 7. 真人语音检测

使用：

```text
Silero VAD
```

在 16 kHz 波形上检测真人语音区间：

```text
speech_segments
```

并生成：

```text
audio_speech_mask
```

只有处于真人语音区间内的声学帧才被视为有效语音证据。

如果视频有音轨但没有检测到真人语音：

```text
text_audio_status = no_speech
```

不会因为“存在音频流”就认为存在语音模态。

---

# 8. WhisperX转写与词级时间

对于存在真人语音的样本：

```text
WhisperX base.en
        ↓
英语ASR
        ↓
WhisperX align
        ↓
word_segments
```

每个观测词包含：

```text
word
start
end
score
```

这些时间是**ASR观测词的时间**，不能直接认为就是官方文本的词时间。

因此还需要下一步：

```text
官方词序列 ↔ ASR观测词序列
```

匹配。

---

# 9. 官方文本与ASR匹配

实现：

```text
alignment/temporal.py
```

处理流程：

1. 官方文本按空格分词；
2. 官方词和ASR词统一小写；
3. 去除非字母/数字/撇号字符；
4. 使用 `SequenceMatcher` 找相同词形块；
5. 仅保留：
   - 官方词与ASR词词形相同；
   - ASR包含合法 start/end；
   - ASR word score >= 0.1；
6. 统计官方词覆盖率。

覆盖率：

```text
matched official words / lexical official words
```

只有：

```text
coverage >= 0.55
```

才把整条样本视为可信词级匹配：

```text
text_audio_status = matched
```

否则：

```text
text_audio_status = mismatch_candidate
```

并且官方词时间全部保持：

```text
NaN
```

不进行强制拉伸或猜测。

---

# 10. ASR重复幻觉处理

代码额外计算连续 4-gram 的重复比例。

如果：

```text
observed words >= 12
and
repetition_ratio >= 0.25
```

则标记：

```text
asr_unreliable = True
text_audio_status = asr_unreliable
```

并清空所有官方词级时间：

```text
word_start = NaN
word_end   = NaN
```

防止 Whisper 重复幻觉产生伪对齐。

---

# 11. 词级语音对齐

设第 `i` 个可信官方词真实发音区间为：

```text
[s_i, e_i)
```

从 `audio_native` 中选取：

```text
audio_time ∈ [s_i, e_i)
```

且：

```text
audio_speech_mask = 1
```

的帧进行平均池化。

得到：

```text
word_audio
shape = [official_word_count, 25]
```

同时保存：

```text
word_audio_mask
word_audio_coverage
word_speech_fraction
```

另外要求词区间真人语音覆盖：

```text
word_speech_fraction >= 0.25
```

否则：

```text
word_audio_mask = 0
word_audio = 0
```

---

# 12. 视觉特征

视觉通过：

```text
MediaPipe Face Detector
+
MediaPipe Face Landmarker
```

处理。

当前视觉采样：

```text
2 fps
```

对于较宽视频：

```text
width > 640
```

会按比例缩放到：

```text
width = 640
```

再进行人脸处理。

---

## 12.1 视觉输出

每个采样位置保存：

```text
vision_time
vision_native
vision_face_mask
vision_valid_mask
vision_multiface_mask
```

视觉特征：

```text
52维 Face Blendshape
```

因此：

```text
vision_native
shape = [vision_frames, 52]
```

---

## 12.2 人脸有效性规则

代码最多允许检测：

```text
num_faces = 5
```

但只有：

```text
恰好 1 张脸
+
恰好 1 组 Face Blendshapes
```

才认为这一帧：

```text
vision_valid_mask = 1
```

如果：

```text
n_faces > 1
```

则：

```text
vision_multiface_mask = 1
```

不会自动猜哪一张脸是说话人。

---

## 12.3 孤立误检抑制

人脸存在标记还经过：

```text
temporal_face_presence()
```

抑制只有一次 detector 命中、前后没有时间支持的孤立假阳性。

但如果该帧有有效 landmark/blendshape，则允许保留短暂真实人脸。

---

# 13. 词级视觉对齐

视觉以：

```text
FPS = 2
```

采样，因此每一视觉采样点视为具有：

```text
± 0.5 / FPS
```

的时间支持范围。

对于词区间：

```text
[s_i, e_i)
```

根据视觉采样支持区间与词区间的**时间交叠长度加权平均**。

得到：

```text
word_vision
shape = [official_word_count, 52]
```

同时保存：

```text
word_vision_mask
word_vision_coverage
```

如果该词区间中出现：

```text
multi-face
```

则这一词位置视觉特征直接作废：

```text
word_vision_mask = 0
word_vision = 0
```

不会强行选择目标人脸。

---

# 14. 固定长度词级序列

所有样本最终保存固定：

```text
MAX_WORDS = 80
```

的词级表示：

```text
word_text        [80, 768]
word_audio       [80, 25]
word_vision      [80, 52]

word_valid       [80]
word_audio_mask  [80]
word_vision_mask [80]
```

真实词数：

```text
word_count
```

也保存在 JSON 日志。

padding 部分：

```text
word_valid = 0
```

并要求所有模态 mask 同样为 0。

---

# 15. 填零和缺失mask的区别

本模块始终区分：

```text
特征值 = 0
```

和：

```text
该模态是否真实有效
```

例如：

```text
word_audio = 0
word_audio_mask = 0
```

表示该位置没有可信语音证据。

不能根据：

```text
特征是不是全0
```

反推模态一定存在或缺失。

后续建模应优先使用显式 mask。

---

# 16. 秒级时间网格

除了词级表示，还额外生成统一：

```text
1 second
```

时间网格：

```text
second_start
second_end

second_audio
second_audio_mask

second_vision
second_vision_mask

second_speech_fraction
```

用途包括：

- 非词级分析；
- 异常视频分析；
- 模态时间覆盖统计；
- 后续可扩展模型接口。

---

# 17. 单条NPZ内容

每个样本输出：

```text
outputs/q1/features/<sample_id>.npz
```

主要字段：

## 原生文本

```text
text_token
text_word_native
text_token_count
```

## 原生语音

```text
audio_native
audio_time
audio_speech_mask
```

## 原生视觉

```text
vision_native
vision_time
vision_face_mask
vision_valid_mask
vision_multiface_mask
```

## 词级时间

```text
word_start
word_end
```

## 固定词级三模态

```text
word_text
word_audio
word_vision
```

## 词级mask

```text
word_valid
word_audio_mask
word_vision_mask
```

## 覆盖率

```text
word_speech_fraction
word_audio_coverage
word_vision_coverage
```

## 秒级特征

```text
second_start
second_end
second_audio
second_audio_mask
second_vision
second_vision_mask
second_speech_fraction
```

对于跨语言样本，还会增加：

```text
crosslingual_*
```

相关字段。

---

# 18. 每条样本JSON日志

输出：

```text
outputs/q1/logs/samples/<sample_id>.json
```

保存可追溯信息，例如：

```text
sample_id
video_id
clip_id
video_path

official_text
label
annotation

duration
video_codec
audio_codec

sha256_video

speech_present
face_present
face_feature_valid
target_face_uncertain

text_audio_status
text_audio_match
matched_word_fraction

asr_text
asr_words
speech_segments

source_fps
decoded_frames

feature_file
failures
```

因此每条 NPZ 都可以回溯到：

```text
原始MP4
官方文本
标签
ASR
时间区间
模型状态
```

---

# 19. 异常样本状态

Q1不会为了追求100%“完整三模态”而伪造结果。

主要状态：

```text
matched
no_speech
asr_unavailable
asr_unreliable
mismatch_candidate
crosslingual_translation_candidate
crosslingual_processing_failed
```

典型质量事件：

```text
no_human_speech
no_detected_face
target_face_uncertain
text_audio_mismatch_review
asr_repetition_review
language_id_uncertain
crosslingual_translation_review
```

这些样本仍保留在100条总结果中，但相应模态 mask 为无效。

---

# 20. 跨语言处理

部分视频的原声可能并非英语，而官方文本是英语译文。

这种情况下：

```text
官方英语单词
```

不能直接强制对应：

```text
外语发音单词
```

所以本模块单独实现：

```text
alignment/crosslingual.py
```

---

## 20.1 语言识别

所有存在真人语音的样本先使用多语言：

```text
Whisper base
```

进行语言筛查。

输出：

```text
outputs/q1/tables/07_multilingual_language_screen.csv
```

只有语言概率：

```text
>= 0.80
```

才认为语言识别具有较高置信度。

否则：

```text
language_review_required = True
```

进入人工核听。

---

## 20.2 非英语语音处理

对于：

```text
confidence >= 0.80
and
language != en
```

使用：

```text
multilingual Whisper small
```

分别执行：

```text
task = transcribe
task = translate
```

得到：

```text
源语言ASR
英语机器翻译候选
```

---

## 20.3 跨语言不生成假词级时间

核心原则：

```text
官方英语译文没有真实“英语单词发音时间”
```

因此：

```text
word_start
word_end
```

继续保持：

```text
NaN
```

且：

```text
word_audio_mask = 0
word_vision_mask = 0
```

不会把英文单词强制贴到外语发音。

跨语言样本使用：

```text
clip / source-segment level
```

对齐粒度。

---

## 20.4 跨语言候选区间

机器翻译片段与官方英语文本提取内容词词干，寻找词汇锚点。

如果发现锚点：

```text
crosslingual_semantic_status =
lexical_anchor_candidate
```

如果没有锚点：

```text
no_lexical_anchor_review
```

此时整段作为候选支持区间，但明确标记需人工复核。

---

## 20.5 跨语言三模态特征

在候选源视频时间片段上聚合：

```text
crosslingual_clip_text   [768]
crosslingual_clip_audio  [25]
crosslingual_clip_vision [52]
```

以及：

```text
crosslingual_clip_*_mask
```

并保留分段级：

```text
crosslingual_start
crosslingual_end
crosslingual_audio
crosslingual_vision
...
```

---

# 21. 跨语言升级不会重跑全部特征

函数：

```python
upgrade_existing_outputs()
```

是在已经完成的：

```text
logs/
features/
```

基础上升级。

因此它不会重新执行：

```text
BERT
openSMILE
MediaPipe
```

只补充语言识别与跨语言信息，并原地更新：

```text
NPZ
JSON
02_sample_summary.csv
```

---

# 22. 质量审计

`analysis/quality.py` 会对所有100条结果执行硬检查。

包括：

```text
是否缺少样本输出
NPZ/JSON是否存在
原始视频路径是否一致
原视频SHA256是否变化
官方文本是否一致
标签是否一致
annotation是否一致
词数是否一致
固定特征维度是否正确
padding mask是否合法
时间戳是否单调
词时间是否合法
词时间是否越界
无语音时是否错误存在audio mask
无人脸时是否错误存在vision mask
ASR不可靠时是否仍错误保留词时间
跨语言样本是否错误生成英语词级时间
跨语言特征维度是否合法
```

审计结果：

```text
outputs/q1/tables/03_audit_issues.csv
```

理想完整运行应：

```text
n_audit_errors = 0
```

---

# 23. 主要表格输出

完整运行后：

```text
outputs/q1/tables/
```

主要包含：

| 文件 | 内容 |
|---|---|
| `01_manifest.csv` | 100条原始样本清单 |
| `02_sample_summary.csv` | 每条样本的完整处理状态汇总 |
| `03_audit_issues.csv` | 可复现性/完整性硬审计问题 |
| `04_review_candidates.csv` | 建议人工复核样本 |
| `05_typical_case_validation.csv` | 三个典型词级对齐案例验证 |
| `06_status_matrix_order.csv` | 状态矩阵图中的样本排序 |
| `07_multilingual_language_screen.csv` | 多语言识别结果 |
| `08_crosslingual_alignment.csv` | 跨语言处理汇总 |
| `09_crosslingual_case_validation.csv` | 跨语言典型案例验证 |
| `10_journal_case_manifest.csv` | 论文案例图对应样本清单 |

---

# 24. 主质量图

输出：

```text
outputs/q1/figures/
```

## Fig01：模态可用性

```text
fig01_modality_availability.pdf
fig01_modality_availability.png
```

展示：

```text
可解码音轨
真人语音
检测到人脸
有效人脸表情
英语词级对齐
```

在100条样本中的覆盖数量。

---

## Fig02：100条样本状态矩阵

```text
fig02_sample_status_matrix.pdf
```

质量事件包括：

```text
无人声
无人脸
无有效表情
无可信英语词对齐
跨语言译文候选
目标人脸不确定
```

对应样本排序保存：

```text
tables/06_status_matrix_order.csv
```

---

## Fig03：词级覆盖率

```text
fig03_word_level_coverage.pdf
```

横轴：

```text
有效语音词 / 官方词
```

纵轴：

```text
有效人脸词 / 官方词
```

用于观察三模态词级对齐覆盖质量。

跨语言样本的片段级特征不计入英语词级覆盖。

---

# 25. 典型样本验证

赛题要求至少展示一个典型样本。

当前代码固定维护三个经过检查的案例：

```text
Negative  -UacrmKiTn4_4
Neutral   -s9qJ7ATP7w_6
Positive  -THoVjtIkeU_2
```

要求它们：

```text
speech_present = True
face_feature_valid = True
target_face_uncertain = False
text_audio_status = matched
所有官方词均有可信时间
所有官方词均有有效语音/视觉特征
```

---

## 25.1 三个典型案例图

输出：

```text
outputs/q1/figures/alignment/
├── fig01a_<sample_id>.pdf
├── fig01b_<sample_id>.pdf
├── fig01c_<sample_id>.pdf
├── fig01_three_typical_cases.pdf
└── figure_manifest.json
```

每个案例在同一时间轴上展示：

```text
原始视频帧
语音波形
官方词区间
文本特征
语音特征
视觉特征
整段情感标签
```

---

## 25.2 逐词源素材映射

每个典型案例还输出：

```text
outputs/q1/case_assets/<sample_id>/
├── audio_16k.wav
└── word_mapping.csv
```

`word_mapping.csv` 可追溯每个官方词对应：

```text
词序号
官方词
开始时间
结束时间
ASR匹配信息
原始视频帧
音频采样位置
语音/视觉特征行号
```

用于证明图中的词级对齐不是手工拼接。

---

## 25.3 交互核验页面

输出：

```text
outputs/q1/paper/alignment_review.html
```

可以：

- 播放原视频音频；
- 点击官方词；
- 跳转对应真实发音时段；
- 对照图与映射表人工复核。

---

# 26. 异常案例图

`visualization/journal_cases.py` 自动选择典型异常样本。

包括：

```text
mismatch
no_face
multiple_faces
no_speech
intermittent_face
```

主要生成：

```text
fig04_mismatch_alignment_<sample_id>.pdf
fig05_visual_evidence.pdf
fig07_quality_exception_timeline.pdf
fig_case_typical_<sample_id>.pdf
```

主要用途：

- 说明官方文本与原声不符时如何处理；
- 说明无人脸时不会伪造视觉特征；
- 说明多人脸时为什么不任意选脸；
- 说明没有真人语音时 audio mask 如何失效；
- 说明人脸间歇出现时模态 mask 的时序行为。

---

# 27. 跨语言论文图

当前跨语言验证代码针对三个固定源语案例：

```text
-yRb-Jum7EQ_1
-yRb-Jum7EQ_5
-yRb-Jum7EQ_6
```

当三条均被判定：

```text
crosslingual_translation_candidate
```

时生成：

```text
fig06_crosslingual_hebrew_translation.pdf
fig06b_crosslingual_text_evidence.pdf
```

并生成：

```text
paper/crosslingual_review.html
tables/09_crosslingual_case_validation.csv
```

注意：

```text
机器翻译仅用于定位候选源语片段
```

不能作为“官方英语文本与外语语音完全语义等价”的证明。

---

# 28. 最终论文摘要

完整报告生成：

```text
outputs/q1/paper/q1_results.md
```

内容包括：

- 已处理样本数；
- 审计错误数；
- 真人语音覆盖；
- 人脸覆盖；
- 有效视觉特征覆盖；
- 文本语音匹配；
- ASR异常；
- 跨语言案例；
- 官方词总量；
- 有真实时间的词数量；
- 有效语音/视觉词数量；
- 论文方法表述；
- 当前生成的全部图表说明。

---

# 29. 可复现性清单

输出：

```text
outputs/q1/run_manifest.json
```

自动记录：

```text
data_root
n_manifest
n_outputs
n_audit_errors

Git仓库信息
软件版本
模型/asset SHA256
模型名称
关键参数

vision_fps
vad_threshold
match_threshold
max_words
foreign_language_probability_threshold

典型案例ID
跨语言案例ID
```

用于论文和附件复现。

---

# 30. 输出目录概览

完整运行后主要结构：

```text
outputs/q1/
├── features/
│   ├── <sample_001>.npz
│   ├── ...
│   └── <sample_100>.npz
│
├── logs/
│   ├── samples/
│   │   ├── <sample_id>.json
│   │   └── ...
│   └── FAILED_<sample_id>.txt
│
├── tables/
│   ├── 01_manifest.csv
│   ├── 02_sample_summary.csv
│   ├── 03_audit_issues.csv
│   ├── 04_review_candidates.csv
│   ├── 05_typical_case_validation.csv
│   ├── 06_status_matrix_order.csv
│   ├── 07_multilingual_language_screen.csv
│   ├── 08_crosslingual_alignment.csv
│   ├── 09_crosslingual_case_validation.csv
│   └── 10_journal_case_manifest.csv
│
├── figures/
│   ├── fig01_modality_availability.pdf
│   ├── fig02_sample_status_matrix.pdf
│   ├── fig03_word_level_coverage.pdf
│   ├── fig04_*.pdf
│   ├── fig05_visual_evidence.pdf
│   ├── fig06_*.pdf
│   ├── fig07_quality_exception_timeline.pdf
│   ├── fig_case_*.pdf
│   └── alignment/
│       ├── fig01a_*.pdf
│       ├── fig01b_*.pdf
│       ├── fig01c_*.pdf
│       ├── fig01_three_typical_cases.pdf
│       └── figure_manifest.json
│
├── case_assets/
│   └── <sample_id>/
│       ├── audio_16k.wav
│       └── word_mapping.csv
│
├── paper/
│   ├── q1_results.md
│   ├── alignment_review.html
│   └── crosslingual_review.html
│
├── tmp/
├── run_manifest.json
└── ...
```

---

# 31. 推荐运行方式

## 31.1 先做smoke test

在项目根目录：

```bash
python run.py q1 --test
```

会：

1. 编译 Q1 Python 模块；
2. 导入关键模块；
3. 检查 CUDA / cuDNN；
4. 选择一条真实附件1视频；
5. 实际执行：
   - BERT；
   - openSMILE；
   - Silero VAD；
   - WhisperX；
   - MediaPipe；
   - 词级对齐；
6. 检查生成 NPZ 与关键维度。

测试输出：

```text
outputs/_smoke/q1/
```

不会覆盖正式：

```text
outputs/q1/
```

测试通过后再运行正式流程。

---

# 32. 一键运行完整问题1

项目根目录：

```bash
python run.py q1
```

根统一 runner 会检查是否已经完整存在：

```text
>=100个NPZ
01_manifest.csv
02_sample_summary.csv
03_audit_issues.csv
run_manifest.json
q1_results.md
fig01
fig02
fig03
```

如果完整：

```text
[SKIP]
```

不会重复执行。

否则进入 Q1 自己的断点续跑逻辑。

---

# 33. Q1内部完整运行

也可以直接：

```bash
python question1/run.py
```

默认顺序：

```text
pipeline.run()
      ↓
100条特征提取/词级对齐
      ↓
upgrade_existing_outputs()
      ↓
多语言筛查与跨语言升级
      ↓
create_reports()
      ↓
审计 + 表格 + 图表 + 论文摘要
```

---

# 34. 按样本运行

例如：

```bash
python question1/run.py \
  --ids=-3g5yACwYnA_13,-THoVjtIkeU_2
```

只处理指定样本。

注意：

```text
--ids
```

模式不会自动执行全量：

```text
upgrade_existing_outputs()
```

但仍会尝试生成当前已有结果对应的报告。

如果只是做单样本调试，推荐配合 smoke test 或独立测试输出目录，避免破坏正式汇总。

---

# 35. 断点续跑

Q1原生支持：

```bash
python question1/run.py
```

默认：

```text
resume = True
```

如果：

```text
outputs/q1/logs/samples/<sample_id>.json
```

已经存在，则该样本不会重新执行昂贵的：

```text
BERT
openSMILE
WhisperX
MediaPipe
```

而是读取日志加入当前汇总。

因此长时间运行中断后，重新：

```bash
python run.py q1
```

或：

```bash
python question1/run.py
```

即可继续。

---

# 36. 强制重跑

项目根目录：

```bash
python run.py q1 --force
```

根 runner 会转换为：

```bash
python question1/run.py --no-resume
```

即100条全部重新处理。

Q1内部也可以：

```bash
python question1/run.py --no-resume
```

---

# 37. 只重建报告和图表

如果只修改：

```text
analysis/
visualization/
```

不需要重新跑特征提取。

使用：

```bash
python run.py q1 --force-figures
```

等价于调用：

```bash
python question1/run.py --reports-only
```

它会使用已有：

```text
features/
logs/
tables/
```

重新生成：

```text
审计
review candidates
图表
典型案例
q1_results.md
run_manifest.json
```

---

# 38. 只执行跨语言升级

已有100条基础 NPZ / JSON 后，可以：

```bash
python question1/run.py --crosslingual-only
```

执行：

```text
语言识别
跨语言候选处理
更新NPZ/JSON
更新02_sample_summary.csv
生成报告
```

不会重新跑基础：

```text
BERT
openSMILE
MediaPipe
```

---

# 39. CUDA运行环境

`question1/run.py` 会自动把当前 conda 环境中的：

```text
nvidia/cudnn/lib
nvidia/cublas/lib
```

加入：

```text
LD_LIBRARY_PATH
```

并重新启动 Python。

因此正式运行推荐使用：

```bash
python question1/run.py
```

或根目录：

```bash
python run.py q1
```

而不是直接绕过入口调用：

```python
question1.pipeline.run(...)
```

否则 WhisperX 可能无法找到 cuDNN 动态库。

---

# 40. 项目级依赖

当前实现依赖项目根目录中的第三方代码/资源。

代码实际引用：

```text
vendor/MMSA-FET/src
vendor/whisperX
vendor/silero-vad/src

assets/blaze_face_full_range.tflite
assets/face_landmarker.task
```

还需要系统命令：

```text
ffmpeg
ffprobe
```

主要 Python 依赖：

```text
numpy
pandas
torch
opencv-python
mediapipe
matplotlib
scipy
nltk
faster-whisper
openpyxl
```

以及对应 CUDA / cuDNN 环境。

---

# 41. Q1关键设计原则

为了保证问题1结果可核验，当前实现遵循以下原则：

### 1. 官方文本优先

```text
官方文本决定文本模态内容；
ASR只负责寻找原视频中的真实发音时间。
```

### 2. 不强行对齐

如果官方文本与ASR覆盖不足：

```text
word time = NaN
```

而不是插值硬贴。

### 3. 不伪造无语音样本

视频存在音轨：

```text
≠
存在真人语音
```

以 VAD 为准。

### 4. 不在多人脸中任意挑脸

如果存在目标歧义：

```text
vision mask = 0
```

### 5. 跨语言只做句段级映射

不把：

```text
英语翻译词
```

强制贴到：

```text
外语发音词
```

### 6. 原生时间轴始终保留

除固定80词序列外，还保存：

```text
audio_time
vision_time
second_start/end
word_start/end
```

### 7. 特征和有效性分开

始终同时保存：

```text
feature
mask
coverage
```

### 8. 每条结果可追溯至原视频

使用：

```text
video_path
SHA256
PTS
JSON日志
word_mapping.csv
```

构成可复核证据链。

---

# 42. 推荐复现顺序

第一次在新环境运行：

```bash
conda activate mathmodel

# 1. 最小真实运行检查
python run.py q1 --test

# 2. 完整100样本实验
python run.py q1
```

如果中途失败：

```bash
# 修复环境或代码后继续
python run.py q1
```

如果只修改图表：

```bash
python run.py q1 --force-figures
```

如果需要全部从头重新提取：

```bash
python run.py q1 --force
```

---

# 43. 一句话运行

```bash
python run.py q1 --test && python run.py q1
```

前者验证最小真实链路，后者完成：

```text
100条原始视频
→ 三模态特征
→ 真人语音检测
→ ASR
→ 词级时间
→ 跨模态对齐
→ 跨语言升级
→ 质量审计
→ 典型样本验证
→ 论文图表
→ 可复现材料
```
