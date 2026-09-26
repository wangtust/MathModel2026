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


# 推荐运行顺序


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
python question2/run.py train --model mrrf --variant no_recon --seed 2026 

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

python question2/run.py train --model mrrf --variant full --seed 2027 
python question2/run.py train --model mrrf --variant no_recon --seed 2027 
python question2/run.py train --model concat_mlp --seed 2027
python question2/run.py train --model gru --seed 2027
python question2/run.py train --model mult_no_aug --seed 2027
python question2/run.py train --model mult --seed 2027 
python question2/run.py train --model tfr --seed 2027


# 7. 训练 seed 2028 的全部三随机种子对比模型

python question2/run.py train --model mrrf --variant full --seed 2028 
python question2/run.py train --model mrrf --variant no_recon --seed 2028 
python question2/run.py train --model concat_mlp --seed 2028
python question2/run.py train --model gru --seed 2028
python question2/run.py train --model mult_no_aug --seed 2028
python question2/run.py train --model mult --seed 2028 
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
python question2/run.py train --model mrrf --variant dual_recon --seed 2026 
python question2/run.py train --model mrrf --variant dual_recon --seed 2027 
python question2/run.py train --model mrrf --variant dual_recon --seed 2028 
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

8. seed 2027、2028 使用 ``，因为附件3最终只使用验证规则选定配置的 seed 2026 检查点生成预测。

9. `submit` 必须放在三随机种子训练和可视化之后，输出文件为 `outputs/q2/analysis/attachment3_submission.csv`。


# Question 3 可解释多模态情感预测

## 功能

问题3在问题2模型基础上，实现：

不仅预测情感，同时解释预测依据。

输出：

-   情感类别
-   情感强度
-   主要贡献模态
-   关键文本片段
-   关键语音时间段
-   关键视觉区域

## 方法

### Integrated Gradients

计算不同输入特征对预测结果的贡献：

    Text importance

    Audio importance

    Vision importance

### 模态贡献分析

输出不同模态贡献比例，确定主要参考模态。

### 关键证据定位

建立：

    模型贡献
        |
        v
    时间位置
        |
        v
    原始文本/语音/视觉证据

## 运行流程

``` bash
python question3/run.py pipeline

python question3/run.py ig-check

python question3/run.py evidence

python question3/run.py fidelity

python question3/run.py sensitivity

python question3/run.py alignment

python question3/run.py verify

python question3/run.py figures
```

## 输出

    outputs/q3/

    ├── predictions/
    ├── explanations/
    ├── evidence/
    ├── figures/
    └── reports/

