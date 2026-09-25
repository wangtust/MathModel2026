"""Train missing MRRF ablation stages and build the cumulative report."""

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


ROOT = Path(__file__).resolve().parents[2]
from question2.config import run_dir
from common.paths import Q2_OUTPUT_DIR
OUT = Q2_OUTPUT_DIR / "experiments" / "ablation"
STAGES = (
    ("Plain fusion", "plain_no_aug", (0, 0, 0, 0)),
    ("+ Missing augmentation", "plain_fusion", (1, 0, 0, 0)),
    ("+ Reconstruction", "recon_only", (1, 1, 0, 0)),
    ("+ Reliability gate", "no_consistency", (1, 1, 1, 0)),
    ("+ Consistency (MRRF)", "full", (1, 1, 1, 1)),
)
REQUIRED_SPLITS = ("validation", "heldout_test")
REQUIRED_SCENARIOS = ("clean", "audio+vision")
REQUIRED_METRICS = ("accuracy", "macro_f1", "mae", "pearson")


def _read_complete_metrics(path: Path) -> dict | None:
    """Return metrics only when every value needed by this report exists."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        float(data["selection_score"])
        for split in REQUIRED_SPLITS:
            for scenario in REQUIRED_SCENARIOS:
                result = data[split][scenario]
                for metric in REQUIRED_METRICS:
                    float(result[metric])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return data


def _train_variant(variant: str, seed: int, epochs: int, patience: int) -> None:
    command = [
        sys.executable,
        str(ROOT / "question2" / "run.py"),
        "train",
        "--model", "mrrf",
        "--variant", variant,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--skip-predict",
    ]
    print(f"[ablation] Training missing variant: {variant}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _load_stage_results(seed: int, epochs: int, patience: int,
                        summary_only: bool) -> dict[str, dict]:
    results = {}
    missing = []
    for _, variant, _ in STAGES:
        path = run_dir("mrrf", seed, variant) / "metrics.json"
        data = _read_complete_metrics(path)
        if data is None:
            missing.append((variant, path))
        else:
            results[variant] = data

    if missing and summary_only:
        details = "\n".join(f"  - {variant}: {path}" for variant, path in missing)
        raise SystemExit(
            "Ablation metrics are missing or incomplete:\n"
            f"{details}\n"
            "Run without --summary-only to train these variants automatically."
        )

    if missing:
        variants = ", ".join(variant for variant, _ in missing)
        print(f"[ablation] Missing variants: {variants}", flush=True)
    for variant, path in missing:
        _train_variant(variant, seed, epochs, patience)
        data = _read_complete_metrics(path)
        if data is None:
            raise RuntimeError(
                f"Training finished but required metrics were not written: {path}"
            )
        results[variant] = data
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train missing cumulative-ablation variants and generate the report."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs for each missing variant")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience for each missing variant")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only summarize existing results; report missing inputs without training")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stage_results = _load_stage_results(
        args.seed, args.epochs, args.patience, args.summary_only
    )
    rows = []
    for label, variant, components in STAGES:
        data = stage_results[variant]
        for split in REQUIRED_SPLITS:
            for scenario in REQUIRED_SCENARIOS:
                result = data[split][scenario]
                rows.append({"stage": label, "variant": variant,
                             "missing_augmentation": components[0],
                             "reconstruction": components[1],
                             "reliability": components[2],
                             "consistency": components[3],
                             "seed": args.seed, "split": split, "scenario": scenario,
                             "selection_score": data["selection_score"],
                             **{name: result[name] for name in
                                ("accuracy", "macro_f1", "mae", "pearson")}})
    with (OUT / f"cumulative_ablation_seed{args.seed}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    labels = [stage[0] for stage in STAGES]
    y = np.arange(len(labels))
    for ax, metric, xlabel in zip(axes, ("macro_f1", "mae"),
                                  ("Macro-F1 ↑", "MAE ↓")):
        for scenario, offset, color, marker in (("clean", -.09, "#176b89", "o"),
                                                ("audio+vision", .09, "#bf7131", "s")):
            x = [next(row[metric] for row in rows if row["stage"] == label
                      and row["split"] == "validation" and row["scenario"] == scenario)
                 for label in labels]
            ax.scatter(x, y + offset, color=color, marker=marker, s=32,
                       label="Clean" if scenario == "clean" else "A+V 30% missing",
                       zorder=3)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.suptitle(f"Cumulative MRRF ablation: validation set, seed {args.seed}",
                 x=.03, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig17_cumulative_ablation.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = ["# MRRF-Net 逐步累加消融", "",
             f"同一固定划分、验证遮挡、训练种子 {args.seed} 和验证选模规则。每一行在上一行基础上增加一个组件；因此差值只是单种子的条件比较，不能推断统计显著性。",
             "", "| 阶段 | 验证选模分数 | 验证 clean F1 / MAE | 验证 A+V 30% F1 / MAE | 测试 clean F1 / MAE |",
             "|:--|--:|--:|--:|--:|"]
    for label in labels:
        vals = {(row["split"], row["scenario"]): row for row in rows
                if row["stage"] == label}
        vc, va, tc = vals[("validation", "clean")], vals[("validation", "audio+vision")], vals[("heldout_test", "clean")]
        lines.append(f"| {label} | {vc['selection_score']:.3f} | "
                     f"{vc['macro_f1']:.3f} / {vc['mae']:.3f} | "
                     f"{va['macro_f1']:.3f} / {va['mae']:.3f} | "
                     f"{tc['macro_f1']:.3f} / {tc['mae']:.3f} |")
    lines += ["", "此表按组件有无构造，模型参数数目也会随重建头和门控层改变；它不能单独把性能差异归因于某一个组件。完整模型另有三个种子的稳定性结果。", ""]
    (OUT / "cumulative_ablation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote cumulative ablation table and figure", flush=True)


if __name__ == "__main__":
    main()
