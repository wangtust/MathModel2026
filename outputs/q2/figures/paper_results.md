# 问题二：缺失模态实验结果

下表使用固定验证集、固定缺失掩码（种子 20260923）。模型选择仅依据验证集 clean 与 Audio+Vision 30% 场景；表内其余场景不参与选模。

| 模型 | 场景 | Accuracy | Macro-F1 | MAE | Pearson |
|:--|:--|--:|--:|--:|--:|
| Baseline MulT | Clean | 0.615 | 0.588 | 0.601 | 0.618 |
| Baseline MulT | Text 30% | 0.611 | 0.567 | 0.621 | 0.588 |
| Baseline MulT | Audio 30% | 0.621 | 0.596 | 0.597 | 0.625 |
| Baseline MulT | Vision 30% | 0.613 | 0.588 | 0.600 | 0.620 |
| Baseline MulT | Text+Audio 30% | 0.615 | 0.575 | 0.613 | 0.599 |
| Baseline MulT | Text+Vision 30% | 0.604 | 0.563 | 0.616 | 0.592 |
| Baseline MulT | Audio+Vision 30% | 0.617 | 0.592 | 0.596 | 0.626 |
| MRRF-Net | Clean | 0.646 | 0.604 | 0.594 | 0.625 |
| MRRF-Net | Text 30% | 0.620 | 0.571 | 0.609 | 0.613 |
| MRRF-Net | Audio 30% | 0.643 | 0.602 | 0.593 | 0.629 |
| MRRF-Net | Vision 30% | 0.633 | 0.594 | 0.596 | 0.624 |
| MRRF-Net | Text+Audio 30% | 0.613 | 0.567 | 0.606 | 0.617 |
| MRRF-Net | Text+Vision 30% | 0.614 | 0.565 | 0.610 | 0.612 |
| MRRF-Net | Audio+Vision 30% | 0.641 | 0.602 | 0.591 | 0.632 |
| No reconstruction | Clean | 0.639 | 0.608 | 0.587 | 0.646 |
| No reconstruction | Text 30% | 0.609 | 0.565 | 0.614 | 0.609 |
| No reconstruction | Audio 30% | 0.640 | 0.609 | 0.586 | 0.648 |
| No reconstruction | Vision 30% | 0.637 | 0.607 | 0.588 | 0.648 |
| No reconstruction | Text+Audio 30% | 0.610 | 0.565 | 0.611 | 0.615 |
| No reconstruction | Text+Vision 30% | 0.610 | 0.566 | 0.612 | 0.617 |
| No reconstruction | Audio+Vision 30% | 0.640 | 0.610 | 0.585 | 0.651 |
| No reliability | Clean | 0.630 | 0.585 | 0.616 | 0.605 |
| No reliability | Text 30% | 0.610 | 0.564 | 0.622 | 0.594 |
| No reliability | Audio 30% | 0.630 | 0.591 | 0.609 | 0.613 |
| No reliability | Vision 30% | 0.635 | 0.599 | 0.599 | 0.623 |
| No reliability | Text+Audio 30% | 0.618 | 0.584 | 0.613 | 0.603 |
| No reliability | Text+Vision 30% | 0.617 | 0.584 | 0.604 | 0.608 |
| No reliability | Audio+Vision 30% | 0.632 | 0.600 | 0.597 | 0.625 |
| No consistency | Clean | 0.646 | 0.601 | 0.603 | 0.624 |
| No consistency | Text 30% | 0.617 | 0.559 | 0.639 | 0.594 |
| No consistency | Audio 30% | 0.646 | 0.603 | 0.603 | 0.627 |
| No consistency | Vision 30% | 0.637 | 0.594 | 0.607 | 0.625 |
| No consistency | Text+Audio 30% | 0.609 | 0.554 | 0.642 | 0.594 |
| No consistency | Text+Vision 30% | 0.614 | 0.555 | 0.644 | 0.597 |
| No consistency | Audio+Vision 30% | 0.650 | 0.607 | 0.602 | 0.632 |

## 留出测试集（只展示预设 clean 和 A+V 30% 条件）

| 模型 | 场景 | Accuracy | Macro-F1 | MAE | Pearson |
|:--|:--|--:|--:|--:|--:|
| Baseline MulT | clean | 0.626 | 0.566 | 0.659 | 0.639 |
| Baseline MulT | audio+vision | 0.634 | 0.580 | 0.647 | 0.656 |
| MRRF-Net | clean | 0.645 | 0.563 | 0.653 | 0.650 |
| MRRF-Net | audio+vision | 0.651 | 0.575 | 0.644 | 0.660 |
| No reconstruction | clean | 0.649 | 0.586 | 0.632 | 0.667 |
| No reconstruction | audio+vision | 0.659 | 0.597 | 0.622 | 0.677 |
| No reliability | clean | 0.653 | 0.571 | 0.668 | 0.648 |
| No reliability | audio+vision | 0.675 | 0.610 | 0.639 | 0.664 |
| No consistency | clean | 0.644 | 0.549 | 0.654 | 0.654 |
| No consistency | audio+vision | 0.644 | 0.559 | 0.647 | 0.660 |

按预先设定的验证选择分数（three_seed_validation_mean），当前候选模型是 `no_recon`，其附件三预测文件为 `/root/MathModel_restructured/outputs/q2/runs/mrrf/no_recon/seed_2026/attachment3_predictions_mrrf.csv`；配置分数为 baseline=0.389, full=0.405, no_recon=0.410。
图 1–10 为种子 2026 的结果；三种子均值与标准差另见 `three_seed_results.md` 和图 11。细微差异不可写成显著优势。
图 1–10 分别展示训练曲线、缺失时长、消融、缺失位置、强度预测、混淆矩阵、融合权重、附件三实际零值位置及无标签预测。强度 MAE/Pearson 使用与分类极性一致化的最终输出。
附件三不含标签，不能计算其预测准确率；附件三预测图仅展示模型输出。

消融提示重建存在场景权衡：验证集文本连续缺失 60% 时，完整模型 Macro-F1 0.521、MAE 0.661；去重建版本为 0.475、0.710。单次训练中，去重建版在 clean 与 A+V 30% 的预设选模场景中略优；三种子验证均值须单独查看。附件三的文本 token 有效，不能据此断言重建在所有缺失机制下无用。

附件三 30 条样本的音频全零占正文比例为 0%–50%，视觉为 0%–50%；文本 token 均有有效位置。零值比例仅描述输入可用性，不等于标注误差。

## 配对 bootstrap：候选模型减去基线

仅反映固定验证样本和固定训练种子下的抽样不确定性；不覆盖训练随机性。Macro-F1 差值越高越好，MAE 差值越低越好。

| 模型 | 场景 | 指标 | 差值 | 95% 区间 |
|:--|:--|:--|--:|:--|
| MRRF-Net | clean 0% | macro_f1 | +0.016 | [-0.011, +0.042] |
| MRRF-Net | clean 0% | mae | -0.007 | [-0.025, +0.012] |
| No reconstruction | clean 0% | macro_f1 | +0.019 | [-0.004, +0.042] |
| No reconstruction | clean 0% | mae | -0.013 | [-0.031, +0.005] |
| MRRF-Net | text 30% | macro_f1 | +0.004 | [-0.028, +0.035] |
| MRRF-Net | text 30% | mae | -0.012 | [-0.031, +0.008] |
| No reconstruction | text 30% | macro_f1 | -0.002 | [-0.032, +0.029] |
| No reconstruction | text 30% | mae | -0.006 | [-0.025, +0.014] |
| MRRF-Net | audio+vision 30% | macro_f1 | +0.010 | [-0.016, +0.035] |
| MRRF-Net | audio+vision 30% | mae | -0.005 | [-0.022, +0.014] |
| No reconstruction | audio+vision 30% | macro_f1 | +0.018 | [-0.009, +0.044] |
| No reconstruction | audio+vision 30% | mae | -0.010 | [-0.029, +0.008] |
| MRRF-Net | text 60% | macro_f1 | +0.015 | [-0.024, +0.054] |
| MRRF-Net | text 60% | mae | -0.022 | [-0.047, +0.003] |
| No reconstruction | text 60% | macro_f1 | -0.031 | [-0.068, +0.008] |
| No reconstruction | text 60% | mae | +0.026 | [+0.001, +0.050] |
| MRRF-Net | audio+vision 60% | macro_f1 | -0.006 | [-0.033, +0.021] |
| MRRF-Net | audio+vision 60% | mae | -0.007 | [-0.025, +0.012] |
| No reconstruction | audio+vision 60% | macro_f1 | -0.001 | [-0.025, +0.025] |
| No reconstruction | audio+vision 60% | mae | -0.012 | [-0.035, +0.006] |
