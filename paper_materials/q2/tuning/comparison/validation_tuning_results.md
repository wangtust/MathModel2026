# 问题二验证集调参实验

这是一组基于附件二固定验证集的探索实验。排序分数是 clean 与 A+V 30% 两种验证条件 Accuracy 的平均值；论文既有正式模型仍按原先预设的 Macro-F1−MAE/3 规则选择。historical_full 使用旧规则选轮，供参照，不作为同规则消融。调参期间未评估留出 test，也未使用附件三无标签预测选模。

|运行|选轮规则|最佳轮次|clean Acc|clean Macro-F1|clean MAE|clean Pearson|A+V30 Acc|A+V30 Macro-F1|A+V30 MAE|A+V30 Pearson|中性召回|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|historical_full|official|18|0.646|0.604|0.594|0.625|0.641|0.602|0.591|0.632|0.337|
|optuna_trial_0|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_1|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_10|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_11|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_12|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_13|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_14|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_15|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_16|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_17|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_18|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_19|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_2|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_20|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_21|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_22|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_23|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_24|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_25|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_26|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_27|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_28|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_3|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_4|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_5|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_9|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_6|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_7|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|optuna_trial_8|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|tuning_2|official|4|0.639|0.605|0.589|0.636|0.644|0.613|0.581|0.647|0.391|
|tuning_1|official|12|0.640|0.598|0.596|0.626|0.635|0.593|0.594|0.636|0.332|
|tuning_0|official|12|0.632|0.599|0.595|0.630|0.640|0.613|0.587|0.637|0.380|
|optuna_trial_29|official|7|0.635|0.606|0.587|0.625|0.633|0.607|0.581|0.637|0.424|

图 1 比较完整输入与 A+V 局部缺失的分类准确率；图 2 同时保留 Accuracy、Macro-F1、MAE、Pearson；图 3 的每一行按真类归一化，括号是该类召回率。中性类样本较少，应结合其召回率及总体指标阅读。

- [图1：验证集 Accuracy 权衡](figures/fig01_validation_accuracy_tradeoff.pdf)
- [图2：分类与强度指标](figures/fig02_validation_metrics.pdf)
- [图3：旧正式模型与表现最高的修改配置的 clean 验证集混淆矩阵](figures/fig03_clean_confusion.pdf)
- [图4：相同验证样本上的 Accuracy 配对差值与 95% bootstrap 区间](figures/fig04_paired_accuracy.pdf)

单个运行的全部四项指标与数值配置见 `run_summary.csv`、各运行目录的 `training_config.json`、`metrics.json` 和 `history.csv`。单种子筛选不能证明稳定提升。


配对差值以旧正式 full 检查点为参照，在相同 728 条验证样本上计算；区间只反映验证样本抽样，不反映训练种子变动。

|运行|条件|Accuracy 差值|95% 区间|
|---|---|---:|---:|
|optuna_trial_0|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_1|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_10|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_11|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_12|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_13|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_14|clean|-0.007|[-0.025, +0.010]|
|optuna_trial_15|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_16|clean|-0.007|[-0.022, +0.010]|
|optuna_trial_17|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_18|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_19|clean|-0.007|[-0.022, +0.010]|
|optuna_trial_2|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_20|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_21|clean|-0.007|[-0.025, +0.010]|
|optuna_trial_22|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_23|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_24|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_25|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_26|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_27|clean|-0.007|[-0.022, +0.010]|
|optuna_trial_28|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_29|clean|-0.011|[-0.030, +0.008]|
|optuna_trial_3|clean|-0.007|[-0.022, +0.010]|
|optuna_trial_4|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_5|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_6|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_7|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_8|clean|-0.007|[-0.023, +0.010]|
|optuna_trial_9|clean|-0.007|[-0.025, +0.008]|
|tuning_0|clean|-0.014|[-0.034, +0.007]|
|tuning_1|clean|-0.005|[-0.018, +0.007]|
|tuning_2|clean|-0.007|[-0.023, +0.008]|
|optuna_trial_0|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_1|audio+vision|+0.003|[-0.014, +0.018]|
|optuna_trial_10|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_11|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_12|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_13|audio+vision|+0.003|[-0.014, +0.018]|
|optuna_trial_14|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_15|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_16|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_17|audio+vision|+0.003|[-0.014, +0.018]|
|optuna_trial_18|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_19|audio+vision|+0.003|[-0.014, +0.018]|
|optuna_trial_2|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_20|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_21|audio+vision|+0.003|[-0.014, +0.019]|
|optuna_trial_22|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_23|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_24|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_25|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_26|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_27|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_28|audio+vision|+0.003|[-0.012, +0.019]|
|optuna_trial_29|audio+vision|-0.008|[-0.029, +0.011]|
|optuna_trial_3|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_4|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_5|audio+vision|+0.003|[-0.014, +0.019]|
|optuna_trial_6|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_7|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_8|audio+vision|+0.003|[-0.012, +0.018]|
|optuna_trial_9|audio+vision|+0.003|[-0.012, +0.018]|
|tuning_0|audio+vision|-0.001|[-0.023, +0.021]|
|tuning_1|audio+vision|-0.007|[-0.019, +0.005]|
|tuning_2|audio+vision|+0.003|[-0.014, +0.018]|