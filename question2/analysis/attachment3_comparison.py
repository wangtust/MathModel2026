"""Flag unlabeled samples on which the two validated model variants disagree."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
from question2.config import run_dir
from common.paths import Q2_OUTPUT_DIR
OUT = Q2_OUTPUT_DIR / "analysis"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def main():
    full = read(run_dir("mrrf", 2026, "full") / "attachment3_predictions_mrrf.csv")
    no_recon = read(run_dir("mrrf", 2026, "no_recon") / "attachment3_predictions_mrrf.csv")
    if set(full) != set(no_recon) or len(full) != 30:
        raise ValueError("Expected the same 30 attachment-three IDs")
    rows = []
    for sample_id in sorted(full):
        a, b = full[sample_id], no_recon[sample_id]
        p = np.array([float(a[f"prob_{name}"]) for name in ("negative", "neutral", "positive")])
        q = np.array([float(b[f"prob_{name}"]) for name in ("negative", "neutral", "positive")])
        middle = (p + q) / 2
        js = float(.5 * np.sum(p * np.log(np.clip(p / middle, 1e-12, None)))
                   + .5 * np.sum(q * np.log(np.clip(q / middle, 1e-12, None))))
        strength_full = float(a["sentiment_strength"])
        strength_no_recon = float(b["sentiment_strength"])
        rows.append({"sample_id": sample_id,
                     "class_full": a["annotation"], "class_no_recon": b["annotation"],
                     "class_disagrees": int(a["annotation"] != b["annotation"]),
                     "strength_full": strength_full,
                     "strength_no_recon": strength_no_recon,
                     "absolute_strength_difference": abs(strength_full - strength_no_recon),
                     "probability_js_divergence": js,
                     "confidence_full": float(p.max()),
                     "confidence_no_recon": float(q.max())})
    rows.sort(key=lambda row: (-row["class_disagrees"],
                               -row["probability_js_divergence"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "attachment3_model_disagreement.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    for disagrees, color, label in ((0, "#397d9f", "Same class"),
                                    (1, "#b45955", "Different class")):
        subset = [row for row in rows if row["class_disagrees"] == disagrees]
        axes[0].scatter([row["strength_full"] for row in subset],
                        [row["strength_no_recon"] for row in subset],
                        s=35, color=color, alpha=.82, label=label, zorder=3)
    axes[0].plot((-3, 3), (-3, 3), color="#777777", linewidth=1)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Full model predicted strength")
    axes[0].set_ylabel("No-reconstruction predicted strength")
    axes[0].set_title("Attachment 3: agreement between variants", loc="left", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    top = rows[:10][::-1]
    y = np.arange(len(top))
    axes[1].barh(y, [row["probability_js_divergence"] for row in top],
                 color=["#b45955" if row["class_disagrees"] else "#397d9f" for row in top])
    axes[1].set_yticks(y, [row["sample_id"].replace("附件3_", "") for row in top])
    axes[1].set_xlabel("Probability Jensen–Shannon divergence")
    axes[1].set_title("Largest prediction differences", loc="left", fontsize=9)
    axes[1].grid(axis="x", alpha=.17)
    axes[1].set_axisbelow(True)
    fig.tight_layout(w_pad=2.4)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig12_attachment3_disagreement.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    count = sum(row["class_disagrees"] for row in rows)
    (OUT / "attachment3_disagreement_summary.md").write_text(
        "# 附件三无标签样本的模型分歧\n\n"
        f"完整 MRRF 与去重建版本在 30 条中有 {count} 条类别预测不同；"
        f"强度预测的绝对差中位数为 {np.median([row['absolute_strength_difference'] for row in rows]):.3f}。"
        "分歧不等于模型错误，但适合优先人工核查。\n\n"
        "详表见 `attachment3_model_disagreement.csv`，红色点/条表示类别分歧。\n",
        encoding="utf-8")
    print(f"Wrote attachment-three disagreement audit: {count}/30 class differences")


if __name__ == "__main__":
    main()
