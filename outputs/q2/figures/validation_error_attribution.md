# 验证集错误归因审计

模型：No reconstruction；同一批 728 条验证样本，比较 clean 与音频+视觉共享区间 30% 遮挡。
以下是错误分布与遮挡的关联，不是单个模态的因果贡献。

## 遮挡前后分类状态

| 状态 | 样本数 |
|:--|--:|
| correct_to_wrong | 14 |
| wrong_to_correct | 15 |
| wrong_to_wrong | 248 |
| correct_to_correct | 451 |

## clean 条件下的类别混淆

| 真实类别 → 预测类别 | 样本数 |
|:--|--:|
| Neutral → Positive | 73 |
| Positive → Neutral | 49 |
| Positive → Negative | 42 |
| Neutral → Negative | 39 |
| Negative → Positive | 36 |
| Negative → Neutral | 24 |

## 按真实情感强度分层

| 绝对强度 | 样本数 | clean MAE | 遮挡 MAE |
|:--|--:|--:|--:|
| neutral | 184 | 0.344 | 0.338 |
| weak (0,1] | 345 | 0.503 | 0.511 |
| moderate (1,2] | 163 | 0.853 | 0.848 |
| strong (2,3] | 36 | 1.437 | 1.374 |

## 最大强度误差样本

| 样本 ID | 真实类别 | clean 预测 | 遮挡预测 | clean 绝对误差 | 遮挡绝对误差 |
|:--|:--|:--|:--|--:|--:|
| f_ZJ7L14oYQ$_$21 | Positive | Negative | Negative | 3.799 | 3.765 |
| 266791$_$23 | Negative | Positive | Positive | 3.361 | 3.440 |
| 130456$_$11 | Positive | Negative | Negative | 3.007 | 3.133 |
| H1DpVwktfDQ$_$0 | Positive | Negative | Negative | 2.986 | 3.060 |
| 102408$_$5 | Negative | Positive | Positive | 2.670 | 2.451 |
