# Question2：缺失模态鲁棒融合与情感预测

本模块针对复杂场景下多模态情感预测中的**局部模态缺失问题**，构建鲁棒多模态融合模型，实现：

- 情感极性分类预测
- 情感强度回归预测

主要研究内容：

- MRRF（Masked Robust Representation Fusion）鲁棒融合模型
- 多模态 Baseline 对比
- 局部模态缺失模拟与鲁棒性验证
- 缺失类型、缺失比例、缺失位置、缺失时长影响分析
- 消融实验
- Optuna自动超参数优化
- 错误案例分析与模型归因


---

# 模块结构

```text
question2/

├── data/                         # 数据读取与缺失模态模拟
│   ├── dataset.py
│   └── missingness.py             # 连续缺失mask生成
│
├── models/                        # 模型定义
│   ├── mrrf.py                    # MRRF鲁棒融合模型
│   ├── mult.py                    # MulT模型
│   └── baselines.py               # 基线模型
│
├── training/                      # 模型训练
│   ├── trainer.py                 # MRRF训练入口
│   └── mult_trainer.py            # MulT训练
│
├── experiments/                   # 实验模块
│   ├── tuning.py                  # Optuna超参数优化
│   ├── robustness.py              # 缺失类型/比例/位置实验
│   ├── missing_duration.py        # 缺失时长实验
│   ├── ablation.py                # 消融实验
│   ├── repeats.py                 # 多随机种子实验
│   └── dual_reconstruction.py     # 双重重建实验
│
├── evaluation/                    # 模型评价
│   ├── evaluator.py               # 统一评估接口
│   └── metrics.py                 # 指标计算
│
├── analysis/                      # 结果分析
│   ├── model_comparison.py        # 模型性能比较
│   └── error_analysis.py          # 错误案例与归因分析
│
├── visualization/                 # 论文图表生成
│
├── submission/                    # 测试结果导出
│
└── run.py                         # 统一入口
```

---

# 实验流程

完整实验流程：

```text
环境与数据检查
    ↓
Optuna超参数优化
    ↓
seed 2026 主模型与全部基线训练
    ↓
缺失模态鲁棒性实验
    ↓
缺失时长实验
    ↓
消融实验
    ↓
seed 2027、2028 主模型与全部基线训练
    ↓
三随机种子稳定性汇总
    ↓
错误案例分析
    ↓
结果可视化
    ↓
七种模型三随机种子结果分析
    ↓
模型选择与附件3提交文件导出
```

---

# 1. 模型训练

## MRRF主模型

默认训练：

```bash
python question2/run.py train
```

等价：

```bash
python question2/run.py train --model mrrf
```

输出：

```text
outputs/q2/runs/mrrf/
```

保存：

```text
checkpoint.pt        # 模型参数
metrics.json         # 性能指标
training_config.json # 参数配置
history.csv          # 训练过程
```

---

## Baseline模型

用于验证MRRF模型有效性。

训练：

```bash
python question2/run.py train --model concat_mlp

python question2/run.py train --model gru

python question2/run.py train --model tfr

python question2/run.py train --model mult

python question2/run.py train --model mult_no_aug
```

包含：

* Concat MLP
* GRU
* TFR
* MulT
* MulT（无缺失增强）

---

# 2. Optuna超参数优化

采用 Optuna 自动超参数搜索。

运行：

```bash
python question2/run.py experiment --name tuning
```

优化目标：

* 自动搜索模型参数组合
* 基于验证集综合指标选择最佳参数
* 保存最优配置

搜索流程：

```text
Trial 1
  ↓
参数采样
  ↓
模型训练
  ↓
验证集评价

Trial 2
  ↓
参数更新

...

Best Trial
  ↓
best_config.json
```

输出：

```text
outputs/q2/tuning/

├── study.db
├── trials.json
├── best_config.json
└── optuna.log
```

重新训练：

```bash
python question2/run.py train
```

自动加载最佳参数。

---

# 3. 缺失模态鲁棒性实验

运行：

```bash
python question2/run.py experiment --name robustness
```

分析：

## 缺失类型

包括：

* Text缺失
* Audio缺失
* Vision缺失
* 双模态缺失

## 缺失比例

测试不同缺失比例：

* 20%
* 30%
* 40%

## 缺失位置

测试：

* Beginning
* Middle
* End

输出：

```text
outputs/q2/experiments/robustness/
```

---

# 4. 缺失时长实验

运行：

```bash
python question2/run.py experiment --name missing_duration
```

用于分析：

> 连续缺失时间长度对模型性能的影响。

实验：

```text
duration:

5%
10%
20%
30%
50%
```

测试：

* 不同模态
* 不同缺失持续时间

输出：

```text
outputs/q2/experiments/missing_duration/

└── duration_results.csv
```

---

# 5. 消融实验

运行：

```bash
python question2/run.py experiment --name ablation
```

该命令会复用已有结果，并自动训练缺失的消融变体，最后生成汇总表和图。若只想汇总已有结果并检查缺失项：

```bash
python question2/run.py experiment --name ablation --summary-only
```

分析MRRF各模块贡献：

* Reconstruction模块
* Reliability Fusion模块
* Consistency Constraint模块

用于验证：

不同组件对鲁棒预测性能的提升作用。

---

# 6. 多随机种子实验

`repeats` 负责汇总已有结果。运行汇总前，需要为 Baseline MulT、MRRF 完整模型和 MRRF 无重建变体准备 2026、2027、2028 三个随机种子的结果。若 seed 2026 已完成，只需补跑：

```bash
# seed 2027
python question2/run.py train --model mult --seed 2027 --skip-predict
python question2/run.py train --model mrrf --variant full --seed 2027 --skip-predict
python question2/run.py train --model mrrf --variant no_recon --seed 2027 --skip-predict

# seed 2028
python question2/run.py train --model mult --seed 2028 --skip-predict
python question2/run.py train --model mrrf --variant full --seed 2028 --skip-predict
python question2/run.py train --model mrrf --variant no_recon --seed 2028 --skip-predict
```

然后运行汇总：

```bash
python question2/run.py experiment --name repeats
```

目的：

* 验证模型稳定性
* 减少随机因素影响
* 输出均值和方差

---

# 7. 错误案例分析

运行：

```bash
python question2/run.py error-analysis
```

分析内容：

* 错误样本统计
* 不同缺失模式错误率
* 分类错误情况
* 情感强度预测偏差

输出：

```text
outputs/q2/analysis/error_analysis/

├── error_cases.csv
├── error_summary.json
└── missing_pattern.json
```

用于论文中的：

* 错误归因分析
* 模型局限性讨论

---

# 8. 可视化

运行：

```bash
python question2/run.py visualize
```

该命令至少需要 seed 2026 的 MulT、MRRF full 和相关消融检查点。如果最终选中模型缺少附件3预测文件，程序会使用对应检查点自动生成。

生成论文图表：

* 模型性能对比图
* 缺失率影响曲线
* 缺失位置影响图
* 缺失时长影响图
* 消融实验图
* 鲁棒性分析图

输出：

```text
outputs/q2/figures/
```

---

# 9. 结果分析

完整模型对比使用以下七种配置在 2026、2027、2028 三个随机种子上的结果：

* Concat-MLP
* BiGRU
* MulT（无缺失增强）
* Baseline MulT
* TFR feature adapter
* MRRF full
* MRRF no reconstruction

可视化步骤会生成结果分析所需的 `validation_comparison_grid.csv`。完成可视化后运行；命令会复用已有训练结果并自动补跑缺失的模型与随机种子：

```bash
python question2/run.py analyze
```

只检查缺失项而不启动训练：

```bash
python question2/run.py analyze --summary-only
```

分析：

* MRRF与Baseline性能比较
* 实验结果汇总
* 指标统计

输出：

```text
outputs/q2/analysis/
```

---

# 10. 模型选择与附件3提交文件

在三随机种子训练、可视化和结果分析完成后运行：

```bash
python question2/run.py submit
```

程序只根据 2026、2027、2028 三个随机种子的验证集平均选模分数选择配置，固定使用所选配置的 seed 2026 检查点生成附件3结果。附件3无标签，不参与模型选择。

输出：

```text
outputs/q2/analysis/

├── selected_model.json
└── attachment3_submission.csv
```

---

# 输出目录

```text
outputs/q2/

├── runs/
│   ├── mrrf/
│   ├── concat_mlp/
│   ├── gru/
│   ├── tfr_feature/
│   ├── mult_no_aug/
│   └── mult/
│
├── tuning/
│
├── experiments/
│   ├── robustness/
│   ├── missing_duration/
│   ├── ablation/
│   └── repeats/
│
├── analysis/
│   └── error_analysis/
│
└── figures/
```

---

# 评价指标

## 情感极性分类

采用：

* Accuracy
* Macro-F1

## 情感强度回归

采用：

* MAE
* Pearson相关系数

评价方式符合赛题要求：

分类任务评价情感极性；

回归任务评价连续情感强度。

---

# 推荐运行顺序

下面是从环境检查到附件3提交文件的完整依赖顺序。所有命令均在项目根目录执行；已经存在完整 `metrics.json` 和 `checkpoint.pt` 的训练项可以跳过。

```bash
# 0. 进入项目并激活环境

cd /root/MathModel_restructured
conda activate mathmodel

# 首次创建环境时安装依赖；已经安装完成可跳过这一条
python -m pip install -r requirements.txt

# 检查问题二输入数据
ls -lh "../data/附件2-数据集特征文件/aligned_50.pkl"
ls -ld "../data/附件3-模态缺失特征样本/对齐版本"


# 1. Optuna 超参数搜索，生成 outputs/q2/tuning/best_config.json

python question2/run.py experiment --name tuning


# 2. 使用最优参数训练 seed 2026 的 MRRF 主模型

python question2/run.py train --model mrrf --variant full --seed 2026

# 训练 seed 2026 的 MRRF 无重建对照
python question2/run.py train --model mrrf --variant no_recon --seed 2026 --skip-predict

# 训练 seed 2026 的五种独立基线
python question2/run.py train --model concat_mlp --seed 2026
python question2/run.py train --model gru --seed 2026
python question2/run.py train --model mult_no_aug --seed 2026
python question2/run.py train --model mult --seed 2026
python question2/run.py train --model tfr --seed 2026


# 3. 使用 seed 2026 的 MRRF full 执行缺失类型、比例和位置鲁棒性实验

python question2/run.py experiment --name robustness


# 4. 执行缺失时长实验

python question2/run.py experiment --name missing_duration


# 5. 执行累加消融实验
# 自动补齐 plain_no_aug、plain_fusion、recon_only、no_consistency、full 中缺失的变体

python question2/run.py experiment --name ablation


# 6. 训练 seed 2027 的全部三随机种子对比模型

python question2/run.py train --model mrrf --variant full --seed 2027 --skip-predict
python question2/run.py train --model mrrf --variant no_recon --seed 2027 --skip-predict
python question2/run.py train --model concat_mlp --seed 2027
python question2/run.py train --model gru --seed 2027
python question2/run.py train --model mult_no_aug --seed 2027
python question2/run.py train --model mult --seed 2027 --skip-predict
python question2/run.py train --model tfr --seed 2027


# 7. 训练 seed 2028 的全部三随机种子对比模型

python question2/run.py train --model mrrf --variant full --seed 2028 --skip-predict
python question2/run.py train --model mrrf --variant no_recon --seed 2028 --skip-predict
python question2/run.py train --model concat_mlp --seed 2028
python question2/run.py train --model gru --seed 2028
python question2/run.py train --model mult_no_aug --seed 2028
python question2/run.py train --model mult --seed 2028 --skip-predict
python question2/run.py train --model tfr --seed 2028


# 8. 汇总 MulT、MRRF full、MRRF no_recon 的三随机种子稳定性

python question2/run.py experiment --name repeats


# 9. 使用 seed 2026 的 MRRF full 执行错误案例分析

python question2/run.py error-analysis


# 10. 生成论文图表和 analyze 所需的 validation_comparison_grid.csv
# 最终选中模型缺少附件3预测时会自动生成

python question2/run.py visualize


# 11. 汇总七种模型的三随机种子结果并生成模型对比图表
# 若前面的某个训练结果缺失，该命令会自动补跑

python question2/run.py analyze


# 12. 按三随机种子验证集均值选择最终模型并导出附件3提交文件

python question2/run.py submit
```

## 可选扩展：双路重建实验

双路重建不属于上述主流程。如需生成该扩展实验，三个随机种子都必须训练：

```bash
python question2/run.py train --model mrrf --variant dual_recon --seed 2026 --skip-predict
python question2/run.py train --model mrrf --variant dual_recon --seed 2027 --skip-predict
python question2/run.py train --model mrrf --variant dual_recon --seed 2028 --skip-predict
python question2/run.py experiment --name dual_reconstruction
```

---

# 注意事项

1. 所有实验默认使用固定随机种子，保证结果可复现。

2. `visualize` 需要 seed 2026 的模型检查点；`analyze` 还需要七种配置的三个随机种子结果，并应在 `visualize` 之后运行。

3. 参数搜索完成后生成：

```text
outputs/q2/tuning/best_config.json
```

最终训练会自动读取最佳参数。

4. 问题2模型需要在附件2训练集上训练，在验证集选择模型和参数，附件3仅用于最终预测。

5. 缺失模态模拟采用连续时间段mask方式，符合题目中“局部时段模态信息不可用”的定义。

6. `repeats` 只汇总已有的 MulT、MRRF full 和 MRRF no_recon 三随机种子结果，不负责启动训练。

7. `ablation` 和 `analyze` 会自动检测并补跑缺失结果；可分别使用 `--summary-only` 只检查缺失项。

8. seed 2027、2028 使用 `--skip-predict`，因为附件3最终只使用验证规则选定配置的 seed 2026 检查点生成预测。

9. `submit` 必须放在三随机种子训练和可视化之后，输出文件为 `outputs/q2/analysis/attachment3_submission.csv`。


# 问题三
python question3/run.py pipeline

python question3/run.py ig-check

python question3/run.py evidence

python question3/run.py fidelity

python question3/run.py sensitivity

python question3/run.py alignment

python question3/run.py verify

python question3/run.py figures