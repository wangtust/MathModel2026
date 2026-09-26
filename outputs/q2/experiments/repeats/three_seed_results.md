# 三训练种子重复实验

每个模型使用 2026/2027/2028 三个训练种子；固定数据划分和模型配置。每次只用验证集选择训练轮次，留出测试仅评价。

| 模型 | 数据 | 场景 | Macro-F1 均值±SD | MAE 均值±SD |
|:--|:--|:--|--:|--:|
| Baseline MulT | validation | clean | 0.584±0.013 | 0.608±0.013 |
| Baseline MulT | validation | audio+vision | 0.596±0.007 | 0.596±0.009 |
| Baseline MulT | heldout_test | clean | 0.574±0.013 | 0.654±0.013 |
| Baseline MulT | heldout_test | audio+vision | 0.594±0.016 | 0.643±0.010 |
| MRRF-Net | validation | clean | 0.599±0.008 | 0.593±0.005 |
| MRRF-Net | validation | audio+vision | 0.605±0.004 | 0.589±0.010 |
| MRRF-Net | heldout_test | clean | 0.588±0.029 | 0.643±0.013 |
| MRRF-Net | heldout_test | audio+vision | 0.603±0.027 | 0.638±0.007 |
| No reconstruction | validation | clean | 0.602±0.011 | 0.590±0.005 |
| No reconstruction | validation | audio+vision | 0.610±0.003 | 0.588±0.003 |
| No reconstruction | heldout_test | clean | 0.591±0.007 | 0.637±0.005 |
| No reconstruction | heldout_test | audio+vision | 0.599±0.006 | 0.634±0.010 |

三次重复可展示训练随机性，但样本量仍小；不要将细微差异写成显著优势。
附件三 30 条无标签，只输出模型预测，不能用它验证性能。

## 验证选模分数

分数定义为 clean 与 A+V 30% 两种验证条件下 `(Macro-F1 − MAE/3)` 的平均。

| 模型 | 三种子均值±SD |
|:--|--:|
| Baseline MulT | 0.389±0.013 |
| MRRF-Net | 0.405±0.008 |
| No reconstruction | 0.410±0.008 |
