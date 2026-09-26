"""Publication-layer figures for Question 2.

The module consumes saved CSV results only; it does not retrain or re-evaluate
models. Existing audit figures remain available under ``outputs/q2/figures``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.paths import Q2_OUTPUT_DIR
from common.publication import MUTED, add_heatmap_values, apply_publication_style, light_grid, panel_label, save_figure, symmetric_limit

OUT = Q2_OUTPUT_DIR / "paper_figures"
GRID_CSV = Q2_OUTPUT_DIR / "figures" / "validation_comparison_grid.csv"
POSITION_CSV = Q2_OUTPUT_DIR / "experiments" / "robustness" / "position_rate_grid.csv"
BASELINE_CSV = Q2_OUTPUT_DIR / "analysis" / "all_baselines_three_seed_summary.csv"
ABLATION_CSV = Q2_OUTPUT_DIR / "experiments" / "ablation" / "cumulative_ablation_seed2026.csv"

SCENARIOS = ["text", "audio", "vision", "text+audio", "text+vision", "audio+vision"]
SCENARIO_LABELS = ["Text", "Audio", "Vision", "T+A", "T+V", "A+V"]


def _selected_model(df: pd.DataFrame) -> str:
    for name in ("No reconstruction", "MRRF-Net"):
        if name in set(df.model): return name
    return str(df.model.iloc[0])


def plot_missingness_degradation() -> None:
    apply_publication_style()
    df = pd.read_csv(GRID_CSV)
    models = [m for m in ["Baseline MulT", _selected_model(df)] if m in set(df.model)]
    rates = sorted(df[(df.position == "random") & (df.scenario != "clean")].missing_rate.unique())
    fig, axes = plt.subplots(2, len(models), figsize=(7.15, 5.15), constrained_layout=True, squeeze=False)

    all_f1, all_mae, matrices = [], [], {}
    for model in models:
        clean = df[(df.model == model) & (df.scenario == "clean")].iloc[0]
        f1 = np.full((len(SCENARIOS), len(rates)), np.nan)
        mae = np.full_like(f1, np.nan)
        for i, scenario in enumerate(SCENARIOS):
            for j, rate in enumerate(rates):
                row = df[(df.model == model) & (df.scenario == scenario) & np.isclose(df.missing_rate, rate) & (df.position == "random")]
                if len(row):
                    f1[i, j] = float(row.iloc[0].macro_f1 - clean.macro_f1)
                    mae[i, j] = float(row.iloc[0].mae - clean.mae)
        matrices[model] = (f1, mae); all_f1.append(f1); all_mae.append(mae)
    f1_lim = symmetric_limit(np.concatenate([x.ravel() for x in all_f1]), .02)
    mae_lim = symmetric_limit(np.concatenate([x.ravel() for x in all_mae]), .02)

    for col, model in enumerate(models):
        for row_idx, (metric, label, lim) in enumerate(((matrices[model][0], "Δ Macro-F1", f1_lim), (matrices[model][1], "Δ MAE", mae_lim))):
            ax = axes[row_idx, col]
            im = ax.imshow(metric, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            add_heatmap_values(ax, metric, threshold=.62, limit=lim, fontsize=5.9)
            ax.set_xticks(range(len(rates)), [f"{r:.0%}" for r in rates])
            ax.set_yticks(range(len(SCENARIOS)), SCENARIO_LABELS if col == 0 else [])
            ax.set_xlabel("Contiguous missing fraction")
            panel_label(ax, "ab"[row_idx] + str(col+1), f"{model}: {label} vs clean")
            fig.colorbar(im, ax=ax, shrink=.75, pad=.02)
    fig.text(.01, .5, "Missing modality", rotation=90, va="center", fontsize=8)
    save_figure(fig, OUT, "q2_fig02_missingness_degradation")


def plot_position_rate_interaction() -> None:
    apply_publication_style()
    df = pd.read_csv(POSITION_CSV)
    rates = sorted(df.missing_rate.unique())
    positions = ["beginning", "middle", "end"]
    lim = symmetric_limit(df.delta_mae.to_numpy(float), .02)
    fig, axes = plt.subplots(1, len(rates), figsize=(7.15, 3.05), sharey=True, constrained_layout=True, squeeze=False)
    for k, rate in enumerate(rates):
        ax = axes[0, k]
        mat = np.full((len(SCENARIOS), len(positions)), np.nan)
        for i, scenario in enumerate(SCENARIOS):
            for j, pos in enumerate(positions):
                row = df[(df.scenario == scenario) & np.isclose(df.missing_rate, rate) & (df.position == pos)]
                if len(row): mat[i, j] = float(row.iloc[0].delta_mae)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        add_heatmap_values(ax, mat, threshold=.60, limit=lim, fontsize=5.3)
        ax.set_xticks(range(3), ["Beg.", "Mid.", "End"])
        ax.set_yticks(range(len(SCENARIOS)), SCENARIO_LABELS if k == 0 else [])
        ax.set_xlabel("Missing position"); panel_label(ax, chr(ord("a") + k), f"{rate:.0%} missing")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75, pad=.015, label="Δ MAE from clean")
    save_figure(fig, OUT, "q2_fig03_position_rate_interaction")


def plot_baselines_and_ablation() -> None:
    apply_publication_style()
    base = pd.read_csv(BASELINE_CSV)
    sub = base[(base.split == "validation") & (base.scenario == "clean")].copy().sort_values("selection_score_mean", ascending=True)
    abl = pd.read_csv(ABLATION_CSV)
    abl = abl[(abl.split == "validation") & (abl.scenario.isin(["clean", "audio+vision"]))]
    stages = list(dict.fromkeys(abl.stage.tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.6), constrained_layout=True)
    y = np.arange(len(sub))
    axes[0].errorbar(sub.macro_f1_mean, y, xerr=sub.macro_f1_sd, fmt="o", capsize=2, color="#2F6B8A", ecolor="#9AA6AE", linewidth=1.0)
    axes[0].set_yticks(y, sub.model); axes[0].set_xlabel("Validation Macro-F1 (mean ± SD, 3 seeds)")
    light_grid(axes[0], "x"); panel_label(axes[0], "a", "Architecture comparison")

    x = np.arange(len(stages))
    clean = [abl[(abl.stage == s) & (abl.scenario == "clean")].macro_f1.iloc[0] for s in stages]
    av = [abl[(abl.stage == s) & (abl.scenario == "audio+vision")].macro_f1.iloc[0] for s in stages]
    axes[1].plot(x, clean, marker="o", label="Clean", color="#2F6B8A")
    axes[1].plot(x, av, marker="o", label="A+V 30%", color="#C9822B")
    axes[1].set_xticks(x, [s.replace("+ ", "+\n") for s in stages])
    axes[1].set_ylabel("Validation Macro-F1"); axes[1].legend(frameon=False)
    light_grid(axes[1], "y"); panel_label(axes[1], "b", "Controlled cumulative ablation (seed 2026)")
    axes[1].text(0, -.25, "Single-seed controlled ablation; not a 3-seed uncertainty estimate.", transform=axes[1].transAxes, fontsize=6.2, color=MUTED)
    save_figure(fig, OUT, "q2_fig04_baselines_ablation")


def generate_all() -> None:
    required = [GRID_CSV, POSITION_CSV, BASELINE_CSV, ABLATION_CSV]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing saved Q2 outputs:\n" + "\n".join(missing))
    plot_missingness_degradation(); plot_position_rate_interaction(); plot_baselines_and_ablation()
    print(f"Q2 paper figures written to {OUT}")


if __name__ == "__main__": generate_all()
