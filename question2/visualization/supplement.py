"""Publication-ready Q2 mechanism/diagnostic figures kept outside the six core figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common.paths import Q2_OUTPUT_DIR
from common.plotting import (
    DOUBLE_COLUMN, GRID, INK, MODALITY_COLORS, MUTED,
    apply_paper_style, panel_title, save_figure,
)


OUT = Q2_OUTPUT_DIR / "paper_figures" / "supplementary"
SOURCE = Q2_OUTPUT_DIR / "figures"
PAPER = Q2_OUTPUT_DIR / "paper_figures"
MODS = ("text", "audio", "vision")
CONFIGS = (
    ("clean", 0.0, "Clean"),
    ("text", 0.3, "T30"),
    ("audio", 0.3, "A30"),
    ("vision", 0.3, "V30"),
    ("audio+vision", 0.3, "A+V30"),
    ("text", 0.6, "T60"),
    ("audio+vision", 0.6, "A+V60"),
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Q2 visualization source is missing: {path}")
    return path


def _one(df: pd.DataFrame, scenario: str, rate: float, modality: str) -> pd.Series:
    rows = df[
        (df.scenario == scenario) &
        np.isclose(df.missing_rate.astype(float), rate) &
        (df.modality == modality)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Reliability table expected one row for {scenario}/{rate}/{modality}; found {len(rows)}"
        )
    return rows.iloc[0]


def reliability_gate() -> None:
    """Redraw availability and gate weights without implying identical normalization."""
    df = pd.read_csv(_require(SOURCE / "validation_reliability_weights.csv"))
    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.2), constrained_layout=True)
    x = np.arange(len(CONFIGS))
    offsets = (-.22, 0.0, .22)

    for ax, field, letter, title in (
        (axes[0], "mean_availability", "a", "Observed fraction（各模态独立 0–1）"),
        (axes[1], "mean_gate_weight", "b", "Learned fusion weight（三模态和为 1）"),
    ):
        for mod, offset in zip(MODS, offsets):
            values = np.array([
                float(_one(df, scenario, rate, mod)[field])
                for scenario, rate, _ in CONFIGS
            ])
            ax.bar(
                x + offset, values, width=.20,
                color=MODALITY_COLORS[mod], label=mod.title(), zorder=3,
            )
        ax.set_xticks(x, [label for _, _, label in CONFIGS], rotation=28, ha="right")
        ax.set_ylim(0, 1.05)
        panel_title(ax, letter, title)
        ax.grid(axis="y", color=GRID, linewidth=.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Fraction / weight")

    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.23), frameon=False)
    fig.text(
        .5, -.035,
        "左右两图刻意使用并列柱而非堆叠柱：availability 是三个独立可用比例，gate weight 才是归一化融合权重；两者数学含义不同。",
        ha="center", fontsize=6.0, color=MUTED,
    )
    save_figure(fig, OUT, "figS_q2_reliability_gate")


def normalized_confusion() -> None:
    """Final selected model: row-normalized clean vs A+V30 classification confusion."""
    df = pd.read_csv(_require(PAPER / "fig05_error_attribution_samples.csv"))
    true = df.true_class.to_numpy(int)
    predictions = (
        ("clean_pred_class", "a", "Clean"),
        ("masked_pred_class", "b", "A+V 30%"),
    )
    labels = ("Negative", "Neutral", "Positive")

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.15), constrained_layout=True)
    for ax, (column, letter, title) in zip(axes, predictions):
        matrix = confusion_matrix(true, df[column].to_numpy(int), labels=[0, 1, 2])
        denom = matrix.sum(axis=1, keepdims=True)
        pct = np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom > 0)
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), labels, rotation=25, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        panel_title(ax, letter, f"最终模型 · {title}")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, f"{pct[i,j]:.1%}\n(n={matrix[i,j]})",
                    ha="center", va="center", fontsize=6.6,
                    color="white" if pct[i,j] > .56 else INK,
                )
    fig.colorbar(im, ax=axes, shrink=.74, pad=.025, label="Row-normalized proportion")
    fig.text(
        .5, -.02,
        "按真实类别逐行归一化，避免类别数量不均衡时 raw counts 造成视觉误判；括号保留原始样本数便于审计。",
        ha="center", fontsize=6.0, color=MUTED,
    )
    save_figure(fig, OUT, "figS_q2_normalized_confusion")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reliability_gate()
    normalized_confusion()
    manifest = {
        "figS_q2_reliability_gate": "grouped observed fractions versus learned normalized fusion weights",
        "figS_q2_normalized_confusion": "row-normalized selected-model confusion, clean versus A+V30",
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote Q2 supplementary paper figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
