"""Final publication-facing Q2 renderer.

The validated V2 figures are reused for framework, robustness, position-rate,
model/ablation and error-attribution panels.  Attachment 3 is rendered with a
single compact T/A/V missingness matrix so the main-text figure does not spend
three full axes on the same state encoding.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common.paths import Q2_OUTPUT_DIR
from common.plotting import (
    DANGER, DOUBLE_COLUMN, EDGE, LIGHT, MUTED, apply_paper_style, panel_title,
    save_figure,
)
from question2.config import ATTACHMENT3, run_dir
from question2.data.dataset import content_positions
from question2.visualization.main_text import _attachment3_payload
from question2.visualization.paper_v2 import (
    _require,
    _selection,
    baseline_ablation_figure,
    error_attribution_figure,
    framework_figure,
    position_rate_figure,
    robustness_figure,
)


OUT = Q2_OUTPUT_DIR / "paper_figures"


def attachment3_figure() -> None:
    """Compact Attachment-3 main-text summary: one state matrix + two outputs."""
    variant, seed, selected_label = _selection()
    pred = pd.read_csv(
        _require(run_dir("mrrf", seed, variant) / "attachment3_predictions_mrrf.csv")
    )
    paths = sorted(ATTACHMENT3.glob("*.pkl"))
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 Attachment-3 samples, found {len(paths)}")

    modalities = ("text", "audio", "vision")
    patterns = {m: [] for m in modalities}
    missing_rows: list[dict[str, object]] = []
    for path in paths:
        data = _attachment3_payload(path)
        content = np.asarray(content_positions(data["text_bert"])[0], dtype=bool)
        observed = {"text": content.copy()}
        for name in ("audio", "vision"):
            arr = np.asarray(data[name])
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            observed[name] = content & np.any(np.abs(arr) > 1e-12, axis=-1)

        row: dict[str, object] = {
            "sample_id": path.stem,
            "content_length": int(content.sum()),
        }
        for name in modalities:
            state = np.zeros(len(content), np.int8)
            state[content] = 1
            state[content & ~observed[name]] = 2
            patterns[name].append(state)
            missing = int((content & ~observed[name]).sum())
            row[f"{name}_missing_count"] = missing
            row[f"{name}_missing_fraction"] = missing / max(int(content.sum()), 1)
        missing_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing_rows).to_csv(
        OUT / "fig06_attachment3_missingness.csv", index=False, encoding="utf-8-sig"
    )

    pred = pred.sort_values("sample_id").reset_index(drop=True)
    strength = pred.sentiment_strength.to_numpy(float)
    probs = pred[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy(float)

    # Stack three 30x50 matrices into one 90x50 state panel.  Horizontal separators
    # preserve modality identity while eliminating three repeated axes/color keys.
    composite = np.vstack([np.stack(patterns[m]) for m in modalities])
    state_cmap = ListedColormap((LIGHT, "#DCEAE7", DANGER))
    state_norm = BoundaryNorm((-.5, .5, 1.5, 2.5), 3)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.92))
    gs = fig.add_gridspec(
        2, 2, left=.075, right=.985, top=.955, bottom=.11,
        height_ratios=[1.08, .92], hspace=.34, wspace=.24,
    )

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(composite, cmap=state_cmap, norm=state_norm,
              aspect="auto", interpolation="nearest")
    ax.axhline(29.5, color="white", lw=1.5)
    ax.axhline(59.5, color="white", lw=1.5)
    ax.set_xticks([0, 9, 19, 29, 39, 49], ["1", "10", "20", "30", "40", "50"])
    ax.set_yticks([14.5, 44.5, 74.5], ["Text", "Audio", "Vision"])
    ax.set_xlabel("aligned position")
    ax.set_ylabel("modality × 30 samples")
    panel_title(ax, "a", "附件3三模态局部缺失状态总览", pad=3)
    legend = [
        Patch(facecolor=LIGHT, edgecolor="none", label="Padding/special"),
        Patch(facecolor="#DCEAE7", edgecolor="none", label="Observed"),
        Patch(facecolor=DANGER, edgecolor="none", label="Zero-filled missing"),
    ]
    ax.legend(handles=legend, ncol=3, loc="upper right", frameon=False, fontsize=5.7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.arange(30)
    ax = fig.add_subplot(gs[1, 0])
    sentiment_colors = np.where(
        strength < 0, "#B65A55", np.where(strength > 0, "#4E7FA6", "#7A8288")
    )
    ax.bar(x, strength, color=sentiment_colors, width=.78)
    ax.axhline(0, color=EDGE, lw=.65)
    ax.set_xlim(-.7, 29.7)
    ax.set_ylabel("预测强度")
    ax.set_xticks([0, 4, 9, 14, 19, 24, 29], ["01", "05", "10", "15", "20", "25", "30"])
    ax.set_xlabel("sample")
    panel_title(ax, "b", "30 条无标签样本情感强度", pad=4)

    ax = fig.add_subplot(gs[1, 1])
    left = np.zeros(30)
    colors = ("#B65A55", "#8A9197", "#4E7FA6")
    for j, label in enumerate(("Negative", "Neutral", "Positive")):
        ax.bar(x, probs[:, j], bottom=left, color=colors[j], width=.78, label=label)
        left += probs[:, j]
    ax.set_xlim(-.7, 29.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("类别概率")
    ax.set_xticks([0, 4, 9, 14, 19, 24, 29], ["01", "05", "10", "15", "20", "25", "30"])
    ax.set_xlabel("sample")
    panel_title(ax, "c", f"三分类概率 · {selected_label}", pad=4)
    ax.legend(ncol=3, loc="upper center", fontsize=5.5, frameon=False)

    save_figure(fig, OUT, "fig06_q2_attachment3")


def write_manifest() -> None:
    payload = {
        "version": "v2_final_publication_layout",
        "main_text_figures": {
            "fig01_q2_framework": "compact missingness construction + MRRF framework",
            "fig02_q2_missing_type_rate": "shared-scale degradation heatmaps with sparse annotations",
            "fig03_q2_position_rate_interaction": "complete 90-condition position x rate interaction",
            "fig04_q2_models_and_ablation": "three-seed architecture comparison + controlled cumulative ablation",
            "fig05_q2_error_attribution": "classification transition + regression error stratification",
            "fig06_q2_attachment3": "single compact T/A/V missingness matrix + all unlabeled predictions",
        },
        "layout_rules": [
            "No explanatory footer is placed outside the figure canvas.",
            "Heatmaps emphasize pattern; complete numeric values remain in CSV outputs.",
            "Repeated state panels and color keys are merged whenever they encode the same information.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    framework_figure()
    robustness_figure()
    position_rate_figure()
    baseline_ablation_figure()
    error_attribution_figure()
    attachment3_figure()
    write_manifest()
    print(f"Wrote final Q2 publication figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
