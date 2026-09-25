"""Aggregate matched-seed baselines and visualize the same missingness grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from question2.experiments.baselines import construct
from question2.training.trainer import make_loader
from question2.config import DATA
from question2.data.dataset import FeatureStore
from question2.evaluation.evaluator import evaluate


ROOT = Path(__file__).resolve().parents[2]
from question2.config import run_dir
from common.paths import Q2_OUTPUT_DIR
OUT = Q2_OUTPUT_DIR / "analysis"
MODELS = (
    ("Concat-MLP", "concat_mlp"),
    ("BiGRU", "gru"),
    ("MulT no augmentation", "mult_no_aug"),
    ("MMSA MulT", "mmsa_baseline"),
    ("TFR feature adapter", "tfr_feature"),
    ("MRRF full", "mrrf_full"),
    ("MRRF no reconstruction", "mrrf_no_recon"),
)
SCENARIOS = ("text", "audio", "vision", "text+audio", "text+vision", "audio+vision")
SEEDS = (2026, 2027, 2028)
REPORT_SCENARIOS = ("clean", "audio+vision")
REPORT_SPLITS = ("validation", "heldout_test")
REPORT_METRICS = ("accuracy", "macro_f1", "mae", "pearson")


def metric_path(model, seed):
    if model in ("concat_mlp", "gru", "mult_no_aug", "tfr_feature"):
        return run_dir(model, seed) / "metrics.json"
    if model == "mmsa_baseline":
        return run_dir("mult", seed) / "metrics.json"
    return run_dir("mrrf", seed, model.removeprefix("mrrf_")) / "metrics.json"


def complete_metrics(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        score = data.get("selection_score", data.get("best_validation_score"))
        float(score)
        for split in REPORT_SPLITS:
            for scenario in REPORT_SCENARIOS:
                for metric in REPORT_METRICS:
                    float(data[split][scenario][metric])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return True


def train_command(model: str, seed: int, epochs: int, patience: int) -> list[str]:
    command = [sys.executable, str(ROOT / "question2" / "run.py"), "train"]
    if model == "mmsa_baseline":
        command += ["--model", "mult", "--skip-predict"]
    elif model.startswith("mrrf_"):
        command += ["--model", "mrrf", "--variant",
                    model.removeprefix("mrrf_"), "--skip-predict"]
    else:
        command += ["--model", model]
    command += ["--seed", str(seed), "--epochs", str(epochs),
                "--patience", str(patience)]
    return command


def ensure_repetitions(epochs: int, patience: int, summary_only: bool) -> None:
    missing = [(label, model, seed, metric_path(model, seed))
               for label, model in MODELS for seed in SEEDS
               if not complete_metrics(metric_path(model, seed))]
    if not missing:
        return
    details = "\n".join(f"  - {label}, seed {seed}: {path}"
                        for label, _, seed, path in missing)
    if summary_only:
        raise SystemExit(
            "Model-comparison metrics are missing or incomplete:\n"
            f"{details}\n"
            "Run without --summary-only to train them automatically."
        )
    print("Missing model-comparison runs:\n" + details, flush=True)
    for label, model, seed, path in missing:
        print(f"[analyze] Training {label}, seed {seed}", flush=True)
        subprocess.run(train_command(model, seed, epochs, patience),
                       cwd=ROOT, check=True)
        if not complete_metrics(path):
            raise RuntimeError(
                f"Training finished but required metrics were not written: {path}"
            )


def read_repetitions():
    rows = []
    for label, model in MODELS:
        for seed in SEEDS:
            path = metric_path(model, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            for split in ("validation", "heldout_test"):
                for scenario in ("clean", "audio+vision"):
                    result = data[split][scenario]
                    rows.append({"model": label, "seed": seed, "split": split,
                                 "scenario": scenario, "selection_score": data.get(
                                     "selection_score", data.get("best_validation_score")),
                                 **{metric: result[metric] for metric in
                                    ("accuracy", "macro_f1", "mae", "pearson")}})
    with (OUT / "all_baselines_three_seed_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for label, _ in MODELS:
        for split in ("validation", "heldout_test"):
            for scenario in ("clean", "audio+vision"):
                subset = [row for row in rows if row["model"] == label
                          and row["split"] == split and row["scenario"] == scenario]
                item = {"model": label, "split": split, "scenario": scenario,
                        "seeds": 3}
                for metric in ("selection_score", "accuracy", "macro_f1", "mae", "pearson"):
                    values = np.asarray([row[metric] for row in subset], dtype=float)
                    item[f"{metric}_mean"] = float(values.mean())
                    item[f"{metric}_sd"] = float(values.std(ddof=1))
                summary.append(item)
    with (OUT / "all_baselines_three_seed_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return rows, summary


def fig15(summary):
    colors = ("#7597a2", "#4b98a8", "#a6a9ad", "#7d8790", "#c67828",
              "#176b89", "#75558b")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharey=True)
    labels = [label for label, _ in MODELS]
    y = np.arange(len(labels))
    for i, scenario in enumerate(("clean", "audio+vision")):
        for j, metric in enumerate(("macro_f1", "mae")):
            ax = axes[i, j]
            for k, label in enumerate(labels):
                item = next(row for row in summary if row["model"] == label
                            and row["split"] == "heldout_test" and row["scenario"] == scenario)
                ax.errorbar(item[f"{metric}_mean"], k,
                            xerr=item[f"{metric}_sd"], fmt="o", color=colors[k],
                            markersize=5, capsize=3, linewidth=1.2)
            ax.set_yticks(y, labels)
            ax.set_xlabel("Macro-F1 ↑" if metric == "macro_f1" else "MAE ↓")
            ax.set_title("Clean" if scenario == "clean" else "A+V 30% missing",
                         loc="left", fontsize=9)
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Independent baselines: held-out test, three seeds (mean ± SD)",
                 x=.05, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92), w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig15_baseline_comparison.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


@torch.no_grad()
def robustness():
    source_path = Q2_OUTPUT_DIR / "figures" / "validation_comparison_grid.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path}\nRun `python question2/run.py visualize` first."
        )
    source = list(csv.DictReader(source_path.open(
        newline="", encoding="utf-8-sig")))
    existing = {"MMSA MulT": "Baseline MulT", "MRRF full": "MRRF-Net",
                "MRRF no reconstruction": "No reconstruction"}
    rows = []
    for name, source_name in existing.items():
        for row in source:
            if row["model"] != source_name or row["position"] != "random":
                continue
            if row["scenario"] == "clean" or (row["scenario"] in SCENARIOS
                                                   and float(row["missing_rate"]) in (.3, .6)):
                rows.append({"model": name, "scenario": row["scenario"],
                             "missing_rate": float(row["missing_rate"]),
                             **{key: float(row[key]) for key in ("accuracy", "macro_f1", "mae", "pearson")}})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    store = FeatureStore(DATA)
    loader = make_loader(store, "valid", 64, device)
    for label, model_name in MODELS:
        if model_name not in ("concat_mlp", "gru", "mult_no_aug", "tfr_feature"):
            continue
        checkpoint = torch.load(run_dir(model_name, 2026) / "checkpoint.pt", map_location=device, weights_only=True)
        model = construct(model_name).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        conditions = [("clean", 0.)] + [(s, rate) for rate in (.3, .6) for s in SCENARIOS]
        for scenario, rate in conditions:
            result = evaluate(model, loader, device, scenario=scenario, rate=rate)
            rows.append({"model": label, "scenario": scenario, "missing_rate": rate,
                         **{key: result[key] for key in ("accuracy", "macro_f1", "mae", "pearson")}})
        print(f"Robustness evaluated: {label}", flush=True)
    with (OUT / "all_baselines_validation_robustness.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def fig16(rows):
    labels = [label for label, _ in MODELS]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), sharey=True)
    bounds = {}
    for metric in ("macro_f1", "mae"):
        clean = {label: next(row[metric] for row in rows
                             if row["model"] == label and row["scenario"] == "clean")
                 for label in labels}
        bounds[metric] = max(.02, max(abs(row[metric] - clean[row["model"]])
                                       for row in rows if row["scenario"] != "clean"))
    for i, rate in enumerate((.3, .6)):
        for j, metric in enumerate(("macro_f1", "mae")):
            ax = axes[i, j]
            matrix = np.array([[next(row[metric] for row in rows
                                     if row["model"] == label and row["scenario"] == scenario
                                     and row["missing_rate"] == rate)
                                - next(row[metric] for row in rows
                                       if row["model"] == label and row["scenario"] == "clean")
                                for scenario in SCENARIOS] for label in labels])
            bound = bounds[metric]
            image = ax.imshow(matrix, cmap="RdBu_r" if metric == "mae" else "RdBu",
                              vmin=-bound, vmax=bound, aspect="auto")
            ax.set_xticks(range(6), ("T", "A", "V", "T+A", "T+V", "A+V"))
            ax.set_yticks(range(len(labels)), labels)
            ax.set_title(f"{int(rate*100)}% missing | Δ {'Macro-F1' if metric == 'macro_f1' else 'MAE'}",
                         loc="left", fontsize=9)
            for a in range(len(labels)):
                for b in range(6):
                    value = matrix[a, b]
                    ax.text(b, a, "0.00" if abs(value) < .005 else f"{value:+.2f}",
                            ha="center", va="center", fontsize=6.8)
            fig.colorbar(image, ax=ax, shrink=.70)
    fig.suptitle("Fixed-mask validation changes from each model's clean condition",
                 x=.05, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig16_all_baselines_missingness.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(summary):
    selection = {label: next(item["selection_score_mean"] for item in summary
                             if item["model"] == label and item["split"] == "validation"
                             and item["scenario"] == "clean") for label, _ in MODELS}
    winner = max(selection, key=selection.get)
    test_clean = {label: next(item for item in summary if item["model"] == label
                              and item["split"] == "heldout_test"
                              and item["scenario"] == "clean") for label, _ in MODELS}
    test_f1_leader = max(test_clean, key=lambda label: test_clean[label]["macro_f1_mean"])
    test_mae_leader = min(test_clean, key=lambda label: test_clean[label]["mae_mean"])
    lines = ["# 问题二：独立架构基线与增强对照", "",
             "全部模型使用附件二固定 train/valid/test 划分和相同的 30% A+V 连续遮挡验证条件；每个模型 2026/2027/2028 三个训练种子。训练轮次仅由验证集 clean 与 A+V 30% 的平均 `(Macro-F1 − MAE/3)` 选择。",
             "", "| 模型 | 验证选模分数 | 测试条件 | Accuracy | Macro-F1 | MAE | Pearson |",
             "|:--|--:|:--|--:|--:|--:|--:|"]
    for label, _ in MODELS:
        clean = next(item for item in summary if item["model"] == label
                     and item["split"] == "heldout_test" and item["scenario"] == "clean")
        av = next(item for item in summary if item["model"] == label
                  and item["split"] == "heldout_test" and item["scenario"] == "audio+vision")
        for scenario, item in (("clean", clean), ("A+V 30%", av)):
            score = (f"{clean['selection_score_mean']:.3f}±{clean['selection_score_sd']:.3f}"
                     if scenario == "clean" else "—")
            metrics = " | ".join(
                f"{item[f'{metric}_mean']:.3f}±{item[f'{metric}_sd']:.3f}"
                for metric in ("accuracy", "macro_f1", "mae", "pearson"))
            lines.append(f"| {label} | {score} | {scenario} | {metrics} |")
    lines += ["", f"按预设三种子验证选模分数，七个配置中最高的是 **{winner}**（{selection[winner]:.3f}）。这是配置选择，不表示其留出测试表现显著优于其他模型。",
              f"留出测试 clean 的 Macro-F1 均值最高的是 **{test_f1_leader}**（{test_clean[test_f1_leader]['macro_f1_mean']:.3f}），MAE 均值最低的是 **{test_mae_leader}**（{test_clean[test_mae_leader]['mae_mean']:.3f}）。这与验证选模排序不完全一致，说明小幅领先存在划分和指标敏感性；不能用测试排名反过来挑选附件三模型。",
              "`MulT no augmentation` 与 `MMSA MulT` 架构相同，差别是训练时是否注入连续局部缺失。",
              "`TFR feature adapter` 复用官方 MMSA TFR-Net 的对齐、生成器及 C-GATE 组件，但输入是附件二已经给出的 BERT 特征而非再次运行 BERT；并跳过上游会把批次轴置入序列轴的 temporal-final 分支。它不是未经修改的官方 TFR-Net 数值复现。",
              "各模型可有不同优化学习率；选模规则、数据划分和验证掩码相同。留出测试结果只作评价，不用来重新选择配置。",
              "附件三无标签，不能比较预测准确率。单次遮挡鲁棒性见 `all_baselines_validation_robustness.csv` 与图 16。", ""]
    (OUT / "all_baselines_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Complete missing three-seed baselines and build comparison reports."
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs for each missing run")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience for each missing run")
    parser.add_argument("--summary-only", action="store_true",
                        help="Report missing runs without training them")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_repetitions(args.epochs, args.patience, args.summary_only)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    _, summary = read_repetitions()
    fig15(summary)
    rows = robustness()
    fig16(rows)
    write_report(summary)
    print("Wrote all-baseline reports and paper figures", flush=True)


if __name__ == "__main__":
    main()
