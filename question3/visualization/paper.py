"""Publication-layer figures for Question 3.

Uses saved Q3 tables to separate four concepts explicitly:
(1) local IG location, (2) perturbation response magnitude, (3) source-time
localization quality, and (4) internal gate diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common.paths import Q3_OUTPUT_DIR
from common.publication import AUDIO, INK, MUTED, TEXT, VISION, apply_publication_style, light_grid, panel_label, save_figure, symmetric_limit

OUT = Q3_OUTPUT_DIR / "paper_figures"
TABLES = Q3_OUTPUT_DIR / "tables"
MODS = ("text", "audio", "vision")


def plot_explainability_pipeline() -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, text, edge="#71808A", face="#F8FAFB", fs=7.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.06", edgecolor=edge, facecolor=face, linewidth=.9)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#7B8790"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, linewidth=.8, color=color))

    box(.2, 2.55, 1.55, 1.0, "T / A / V\nfeature sequence", "#66757E")
    arrow(1.75, 3.05, 2.25, 3.05)
    box(2.25, 2.45, 1.55, 1.2, "Frozen Q2\npredictor", "#536D7C", face="#F1F5F7", fs=8)
    arrow(3.8, 3.05, 4.25, 4.25); box(4.25, 3.75, 1.65, 1.0, "Polarity +\nstrength", "#536D7C", face="white")
    arrow(3.8, 3.05, 4.25, 2.0); box(4.25, 1.5, 1.65, 1.0, "Predicted-class\nlogit margin", "#536D7C", face="white")
    arrow(5.9, 2.0, 6.35, 2.0); box(6.35, 1.5, 1.55, 1.0, "Integrated\nGradients", TEXT, face="#F7FAFC")
    arrow(7.9, 2.0, 8.35, 2.0); box(8.35, 1.5, 1.6, 1.0, "Top 20%\ncontinuous window", TEXT, face="white")
    arrow(9.15, 2.5, 9.15, 3.1); box(8.35, 3.1, 1.6, 1.0, "Deletion test\nvs random/low", AUDIO, face="#FCF8F2")
    arrow(9.95, 3.6, 10.35, 3.6); box(10.35, 3.1, 1.45, 1.0, "Fidelity\nresponse", AUDIO, face="white")
    arrow(9.95, 2.0, 10.35, 2.0); box(10.35, 1.5, 1.45, 1.0, "Token→word\nclock audit", VISION, face="#F5FAF7")
    arrow(11.05, 1.5, 11.05, .9); box(10.25, .15, 1.65, .75, "Text / audio /\nvideo evidence", VISION, face="white", fs=6.7)

    ax.text(.2, 5.75, "Auditable multimodal explanation pipeline", fontsize=10, fontweight="bold", color=INK)
    ax.text(.2, 5.35, "IG locates candidate positions; deletion quantifies effect size; clock audit decides whether a second-level location is trustworthy.", fontsize=7.0, color=MUTED)
    save_figure(fig, OUT, "q3_fig01_explainability_pipeline")


def plot_fidelity() -> None:
    apply_publication_style()
    df = pd.read_csv(TABLES / "validation_explanation_metrics.csv")
    df = df[(df.stratum == "all") & (df.comparison == "top-minus-random")]
    rows = [df[df.modality == mod].iloc[0] for mod in MODS]
    values = np.array([r.mean_delta_margin_difference for r in rows], float)
    lo = np.array([r.ci95_low for r in rows], float); hi = np.array([r.ci95_high for r in rows], float)
    ns = [int(r.n) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(3)
    axes[0].errorbar(values, y, xerr=[values-lo, hi-values], fmt="o", capsize=2, color="#2F6B8A", ecolor="#8998A2", linewidth=1.2)
    axes[0].axvline(0, color="#606D76", linestyle="--", linewidth=.8)
    axes[0].set_yticks(y, [f"{m.title()} (n={n})" for m, n in zip(MODS, ns)]); axes[0].invert_yaxis()
    axes[0].set_xlabel("Top-window − matched-random margin drop"); light_grid(axes[0], "x")
    panel_label(axes[0], "a", "Validation fidelity (95% bootstrap CI)")

    idx = [1, 2]
    axes[1].errorbar(values[idx], np.arange(2), xerr=[values[idx]-lo[idx], hi[idx]-values[idx]], fmt="o", capsize=2, color="#C9822B", ecolor="#A8B0B6", linewidth=1.2)
    axes[1].axvline(0, color="#606D76", linestyle="--", linewidth=.8)
    axes[1].set_yticks(np.arange(2), ["Audio", "Vision"]); axes[1].invert_yaxis()
    pad = max(.006, float(max(abs(lo[idx].min()), abs(hi[idx].max()))) * .25)
    axes[1].set_xlim(float(lo[idx].min())-pad, float(hi[idx].max())+pad)
    axes[1].set_xlabel("Margin-drop difference"); light_grid(axes[1], "x"); panel_label(axes[1], "b", "Non-text zoom")
    save_figure(fig, OUT, "q3_fig02_fidelity")


def plot_modality_effects() -> None:
    apply_publication_style()
    q = pd.read_csv(TABLES / "attachment4_predictions.csv", dtype={"sample_id": str})
    q.sample_id = q.sample_id.str.zfill(2)
    delta = q[[f"top_delta_margin_{m}" for m in MODS]].to_numpy(float)
    lim = symmetric_limit(delta, .05, quantile=.95)
    dominant_order = {"text": 0, "audio": 1, "vision": 2, "unclear": 3}
    dom = q.dominant_modality.fillna("unclear").str.lower().map(dominant_order).fillna(3).to_numpy(int)
    dom_cmap = ListedColormap([TEXT, AUDIO, VISION, "#C3CAD0"])
    dom_norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.8), sharey=True, gridspec_kw={"width_ratios": [3.1, .55]}, constrained_layout=True)
    im = axes[0].imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(q)):
        for j in range(3):
            value = delta[i, j]
            axes[0].text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.1, color="white" if abs(value) > .60*lim else INK)
    axes[0].set_xticks(range(3), [m.title() for m in MODS]); axes[0].set_yticks(range(len(q)), q.sample_id)
    axes[0].set_xlabel("Predicted-class margin drop after selected-window deletion"); panel_label(axes[0], "a", "Signed modality response")
    fig.colorbar(im, ax=axes[0], shrink=.72, pad=.02)

    axes[1].imshow(dom[:, None], cmap=dom_cmap, norm=dom_norm, aspect="auto")
    axes[1].set_xticks([0], ["Main\nmodality"]); axes[1].set_yticks(range(len(q)), [])
    for i, label in enumerate(q.dominant_modality.fillna("unclear")):
        short = {"text":"T", "audio":"A", "vision":"V", "unclear":"?"}.get(str(label).lower(), "?")
        axes[1].text(0, i, short, ha="center", va="center", fontsize=6.4, color="white" if short != "?" else INK, fontweight="bold")
    panel_label(axes[1], "b", "Dominant reference")
    fig.text(.5, -.025, "Dominant modality is derived from signed deletion response, not from the internal fusion gate.", ha="center", fontsize=6.4, color=MUTED)
    save_figure(fig, OUT, "q3_fig03_modality_effects")


def plot_alignment_audit() -> None:
    apply_publication_style()
    df = pd.read_csv(TABLES / "attachment4_alignment_audit.csv", dtype={"sample_id": str})
    df.sample_id = df.sample_id.str.zfill(2)
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    trusted = df.alignment_status.eq("trusted")
    ax.scatter(df.loc[trusted, "word_coverage"], df.loc[trusted, "asr_similarity"], s=34, color="#5A9674", edgecolor="white", linewidth=.6, label="Trusted")
    ax.scatter(df.loc[~trusted, "word_coverage"], df.loc[~trusted, "asr_similarity"], s=42, facecolor="white", edgecolor="#B75D57", linewidth=1.1, label="Rejected")
    for _, row in df.iterrows():
        if (row.alignment_status != "trusted") or row.sample_id in {"13", "15", "19"}:
            ax.annotate(row.sample_id, (row.word_coverage, row.asr_similarity), xytext=(4, 4), textcoords="offset points", fontsize=6.4)
    ax.axvline(.85, color="#7A8790", linestyle="--", linewidth=.8); ax.axhline(.35, color="#7A8790", linestyle="--", linewidth=.8)
    ax.set_xlim(-.02, 1.03); ax.set_ylim(-.02, 1.03); ax.set_xlabel("WhisperX word coverage"); ax.set_ylabel("Independent ASR agreement")
    light_grid(ax, "both"); ax.legend(frameon=False, loc="lower right"); panel_label(ax, "a", "Source-time alignment audit")
    save_figure(fig, OUT, "q3_fig05_alignment_audit")


def generate_all() -> None:
    required = [TABLES / "validation_explanation_metrics.csv", TABLES / "attachment4_predictions.csv", TABLES / "attachment4_alignment_audit.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing saved Q3 outputs:\n" + "\n".join(missing))
    plot_explainability_pipeline(); plot_fidelity(); plot_modality_effects(); plot_alignment_audit()
    print(f"Q3 paper figures written to {OUT}")


if __name__ == "__main__": generate_all()
