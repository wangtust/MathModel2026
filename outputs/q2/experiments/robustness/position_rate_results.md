# 缺失比例×位置补充实验

对 MRRF full（种子 2026）使用附件二固定验证集，分别在 20%、30%、40% 缺失比例下，测试六类单/双模态缺失在开头、中间、结尾的位置。
共 90 条条件，clean Macro-F1=0.604、MAE=0.594。
文本缺失中 MAE 增幅最大的一格为 50%、beginning，ΔMAE=+0.052；不同模态组合应分别读取 `position_rate_grid.csv`。
图 `fig19_position_rate_interaction.pdf` 展示 ΔMAE；同一表还含 Macro-F1、Accuracy、Pearson 及其 clean 差值。
本补充图为一个已选检查点上的验证扰动分析，不是三个训练种子的误差区间；不据此更换模型。
