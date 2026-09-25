"""Summarize three matched training seeds without using test scores for selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
from question2.config import run_dir
from common.paths import Q2_OUTPUT_DIR
OUT = Q2_OUTPUT_DIR / "experiments" / "repeats"
MODELS = ("Baseline MulT", "MRRF-Net", "No reconstruction")
DIRS = {"Baseline MulT": ("mult", None), "MRRF-Net": ("mrrf", "full"),
        "No reconstruction": ("mrrf", "no_recon")}
COLORS = {"Baseline MulT": "#7d8790", "MRRF-Net": "#176b89",
          "No reconstruction": "#c98b3e"}


def path_for(label, seed):
    model, variant = DIRS[label]
    return run_dir(model, seed, variant) / "metrics.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    rows = []
    for seed in (2026, 2027, 2028):
        for label in MODELS:
            path = path_for(label, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            for split in ("validation", "heldout_test"):
                for scenario in ("clean", "audio+vision"):
                    result = report[split][scenario]
                    rows.append({"model": label, "seed": seed, "split": split,
                                 "scenario": scenario, "best_epoch": report["best_epoch"],
                                 "selection_score": report.get("selection_score", report.get("best_validation_score")),
                                 **{metric: result[metric] for metric in
                                    ("accuracy", "macro_f1", "mae", "pearson")}})
    with (OUT / "three_seed_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for label in MODELS:
        for split in ("validation", "heldout_test"):
            for scenario in ("clean", "audio+vision"):
                subset = [row for row in rows if row["model"] == label
                          and row["split"] == split and row["scenario"] == scenario]
                item = {"model": label, "split": split, "scenario": scenario,
                        "seeds": len(subset)}
                item["selection_score_mean"] = float(np.mean([row["selection_score"] for row in subset]))
                item["selection_score_sd"] = float(np.std([row["selection_score"] for row in subset], ddof=1))
                for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                    values = np.array([row[metric] for row in subset])
                    item[f"{metric}_mean"] = float(values.mean())
                    item[f"{metric}_sd"] = float(values.std(ddof=1))
                summary.append(item)
    with (OUT / "three_seed_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.25))
    for row_index, split in enumerate(("validation", "heldout_test")):
        for col_index, metric in enumerate(("macro_f1", "mae")):
            ax = axes[row_index, col_index]
            for j, scenario in enumerate(("clean", "audio+vision")):
                for i, label in enumerate(MODELS):
                    values = np.array([row[metric] for row in rows
                                       if row["split"] == split and row["scenario"] == scenario
                                       and row["model"] == label])
                    x = j * 1.25 + (i - 1) * .24
                    ax.errorbar(x, values.mean(), yerr=values.std(ddof=1),
                                fmt="o", markersize=5, capsize=3, capthick=1.1,
                                color=COLORS[label], label=label if j == 0 else None)
                    ax.scatter(np.full(3, x) + np.array((-.045, 0, .045)), values,
                               color=COLORS[label], s=10, alpha=.55, zorder=3)
            ax.set_xlim(-.5, 1.75)
            ax.set_xticks((0, 1.25), ("Clean", "A+V 30%"))
            ax.set_title(f"{'Validation' if split == 'validation' else 'Held-out test'} {'Macro-F1' if metric == 'macro_f1' else 'MAE'}",
                         loc="left", fontsize=9)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False,
               fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.015))
    fig.suptitle("Three training seeds (points) and mean ± SD", x=.07,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, .93))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig11_three_seeds.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    lines = ["# 三训练种子重复实验", "",
             "每个模型使用 2026/2027/2028 三个训练种子；固定数据划分和模型配置。每次只用验证集选择训练轮次，留出测试仅评价。",
             "", "| 模型 | 数据 | 场景 | Macro-F1 均值±SD | MAE 均值±SD |",
             "|:--|:--|:--|--:|--:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['split']} | {row['scenario']} | "
                     f"{row['macro_f1_mean']:.3f}±{row['macro_f1_sd']:.3f} | "
                     f"{row['mae_mean']:.3f}±{row['mae_sd']:.3f} |")
    lines += ["", "三次重复可展示训练随机性，但样本量仍小；不要将细微差异写成显著优势。",
              "附件三 30 条无标签，只输出模型预测，不能用它验证性能。", ""]
    lines += ["## 验证选模分数", "",
              "分数定义为 clean 与 A+V 30% 两种验证条件下 `(Macro-F1 − MAE/3)` 的平均。",
              "", "| 模型 | 三种子均值±SD |", "|:--|--:|"]
    for label in MODELS:
        row = next(item for item in summary if item["model"] == label
                   and item["split"] == "validation" and item["scenario"] == "clean")
        lines.append(f"| {label} | {row['selection_score_mean']:.3f}±{row['selection_score_sd']:.3f} |")
    lines.append("")
    (OUT / "three_seed_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote three-seed summary and figure to {OUT}")


if __name__ == "__main__":
    main()
