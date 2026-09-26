"""V2 compact publication figures for Question 2.

This renderer reuses validated experiment tables/checkpoints but removes excess
canvas whitespace, redundant in-cell labels and out-of-canvas captions.  Legacy
figures remain untouched as supplementary/audit material.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import torch

from common.paths import Q2_OUTPUT_DIR
from common.plotting import (
    ACCENT, DANGER, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT, MODALITY_COLORS,
    MUTED, OCHRE, apply_paper_style, panel_title, save_figure,
)
from question2.config import ATTACHMENT3, DATA, run_dir
from question2.data.dataset import FeatureStore, content_positions
from question2.evaluation.evaluator import evaluate
from question2.training.trainer import load_trained, make_loader
from question2.visualization.architecture import _display_id, _draw_mask, _mask_arrays
from question2.visualization.main_text import _attachment3_payload, _one_row, _selection, _strength_group


OUT = Q2_OUTPUT_DIR / "paper_figures"
SCENARIOS = ("text", "audio", "vision", "text+audio", "text+vision", "audio+vision")
SCENARIO_NAMES = ("T", "A", "V", "T+A", "T+V", "A+V")
PURPLE = "#75638A"
MODEL_COLORS = {
    "Baseline MulT": "#7D8790",
    "MRRF full": ACCENT,
    "MRRF no reconstruction": "#75558B",
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Q2 result is missing: {path}")
    return path


def _box(ax, x, y, w, h, text, color, *, dashed=False, fill="white", fontsize=6.8):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.04",
        facecolor=fill, edgecolor=color, linewidth=.85,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, fontweight="medium")


def _arrow(ax, a, b, color=EDGE, *, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=8, linewidth=.75,
        color=color, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2,
    ))


def framework_figure() -> None:
    sid, length, before, after = _mask_arrays()
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.68))
    gs = fig.add_gridspec(2, 1, left=.035, right=.985, top=.965, bottom=.12,
                          height_ratios=[1.40, .78], hspace=.20)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 12.4); ax.set_ylim(.35, 5.25); ax.axis("off")
    tc, ac, vc = MODALITY_COLORS["text"], MODALITY_COLORS["audio"], MODALITY_COLORS["vision"]
    streams = ((4.25, "Text", "50x768", tc), (2.95, "Audio", "50x74", ac),
               (1.65, "Vision", "50x35", vc))
    _box(ax, .20, 2.45, 1.15, 1.00, "Availability\n+ masks", PURPLE, fill="#F8F7FA")
    for y, name, shape, color in streams:
        _box(ax, 1.75, y, 1.32, .60, f"{name}\n{shape}", color)
        _arrow(ax, (1.37, 2.95), (1.70, y+.30), color)
        _box(ax, 3.55, y, 1.48, .60, f"MulT\n{name} stream", color)
        _arrow(ax, (3.10, y+.30), (3.50, y+.30), color)
        _arrow(ax, (5.08, y+.30), (7.10, 2.95), color)

    _box(ax, 5.35, .48, 1.95, .68, "optional reconstruction", OCHRE,
         dashed=True, fill="#FCFAF5", fontsize=6.4)
    for y, _, _, _ in streams:
        _arrow(ax, (4.30, y), (6.25, 1.19), OCHRE, dashed=True)
    _box(ax, 7.15, 2.35, 1.65, 1.20, "Reliability gate\n+ adaptive fusion", PURPLE,
         fill="#F8F7FA", fontsize=7.0)
    _box(ax, 9.40, 3.55, 1.52, .62, "3-class polarity", tc)
    _box(ax, 9.40, 1.72, 1.52, .62, "strength [-3,3]", vc)
    _arrow(ax, (8.84, 2.95), (9.35, 3.86), PURPLE)
    _arrow(ax, (8.84, 2.95), (9.35, 2.03), PURPLE)
    ax.text(11.22, 2.95, "joint output", ha="center", va="center",
            fontsize=6.0, color=MUTED)
    panel_title(ax, "a", "局部连续缺失下的 MRRF 主干；重建为可选辅助支路", pad=1)

    lower = gs[1].subgridspec(1, 2, wspace=.10)
    ax1 = fig.add_subplot(lower[0]); ax2 = fig.add_subplot(lower[1])
    _draw_mask(ax1, before, "(b) 缺失注入前")
    _draw_mask(ax2, after, "(c) A+V 30% 中段连续缺失", show_xlabel=True)
    ax1.set_xlabel("aligned position", fontsize=6.1)
    legend = [Patch(facecolor=color, label=label) for color, label in (
        (LIGHT, "padding/special"), ("#DCEAE7", "observed"),
        (OCHRE, "native zero"), (DANGER, "injected missing"),
    )]
    fig.legend(handles=legend, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, .015), fontsize=5.9)
    fig.text(.985, .955, f"traceable example: {_display_id(sid)} | content={length}",
             ha="right", va="top", fontsize=5.5, color=MUTED)
    save_figure(fig, OUT, "fig01_q2_framework")


def _cell(value: float) -> str:
    return "0.00" if abs(value) < .005 else f"{value:+.2f}"


def robustness_figure() -> None:
    df = pd.read_csv(_require(Q2_OUTPUT_DIR / "figures/validation_comparison_grid.csv"))
    _, _, selected_label = _selection()
    labels = ("Baseline MulT", selected_label)
    rates = (.1, .2, .3, .4, .5, .6)

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for label in labels:
        subset = df[(df.model == label) & (df.position == "random")]
        clean = _one_row(subset, context="robustness clean", scenario="clean", missing_rate=0.0)
        for metric in ("macro_f1", "mae"):
            matrix = np.empty((6, 6), float)
            for i, scenario in enumerate(SCENARIOS):
                for j, rate in enumerate(rates):
                    row = _one_row(subset, context="robustness grid",
                                   scenario=scenario, missing_rate=rate)
                    matrix[i, j] = float(row[metric]) - float(clean[metric])
            matrices[(label, metric)] = matrix

    f1_bound = max(.03, max(np.abs(matrices[(x, "macro_f1")]).max() for x in labels))
    mae_bound = max(.03, max(np.abs(matrices[(x, "mae")]).max() for x in labels))
    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 4.42))
    fig.subplots_adjust(left=.09, right=.94, top=.96, bottom=.10, hspace=.25, wspace=.18)
    images = [None, None]

    for row_i, label in enumerate(labels):
        for col_i, metric in enumerate(("macro_f1", "mae")):
            ax = axes[row_i, col_i]
            matrix = matrices[(label, metric)]
            bound = f1_bound if metric == "macro_f1" else mae_bound
            cmap = "RdBu" if metric == "macro_f1" else "RdBu_r"
            im = ax.imshow(matrix, cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                           aspect="auto")
            images[col_i] = im
            letter = chr(ord("a") + row_i*2 + col_i)
            metric_name = "Delta Macro-F1" if metric == "macro_f1" else "Delta MAE"
            panel_title(ax, letter, f"{label} · {metric_name}", pad=4)
            ax.set_xticks(range(6), [f"{int(r*100)}%" for r in rates])
            ax.set_yticks(range(6), SCENARIO_NAMES)
            if row_i == 1:
                ax.set_xlabel("连续缺失比例")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if col_i == 0:
                ax.set_ylabel("缺失模态")
            else:
                ax.tick_params(axis="y", labelleft=False)
            for i in range(6):
                for j in range(6):
                    value = matrix[i, j]
                    # Main-text heatmaps communicate pattern first.  Only large
                    # deviations are annotated; the complete values remain in CSV.
                    if abs(value) >= .45 * bound:
                        ax.text(j, i, _cell(value), ha="center", va="center", fontsize=5.4,
                                color="white" if abs(value) > .66*bound else INK)
    fig.colorbar(images[0], ax=axes[:, 0], shrink=.78, pad=.015, label="Delta F1")
    fig.colorbar(images[1], ax=axes[:, 1], shrink=.78, pad=.015, label="Delta MAE")
    save_figure(fig, OUT, "fig02_q2_missing_type_rate")


def position_rate_figure() -> None:
    df = pd.read_csv(_require(Q2_OUTPUT_DIR / "experiments/robustness/position_rate_grid.csv"))
    rates = sorted(float(x) for x in df.missing_rate.unique())
    expected = [.1, .2, .3, .4, .5]
    if len(df) != 90 or len(rates) != 5 or not np.allclose(rates, expected):
        raise RuntimeError("Q2 position-rate grid must be the complete 5x6x3=90 design")
    matrices = []
    for rate in rates:
        matrix = np.empty((6, 3), float)
        for i, scenario in enumerate(SCENARIOS):
            for j, position in enumerate(("beginning", "middle", "end")):
                row = _one_row(df, context="position-rate", scenario=scenario,
                               missing_rate=rate, position=position)
                matrix[i, j] = float(row.delta_mae)
        matrices.append(matrix)
    bound = max(.03, max(np.abs(x).max() for x in matrices))

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COLUMN, 2.88), sharey=True)
    fig.subplots_adjust(left=.07, right=.92, top=.91, bottom=.20, wspace=.07)
    for k, (ax, rate, matrix) in enumerate(zip(axes, rates, matrices)):
        im = ax.imshow(matrix, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                       aspect="auto")
        panel_title(ax, chr(ord("a")+k), f"{int(rate*100)}%", pad=3)
        ax.set_xticks(range(3), ("Beg.", "Mid.", "End"))
        ax.set_yticks(range(6), SCENARIO_NAMES)
        ax.set_xlabel("位置", fontsize=6.4)
        if k == 0:
            ax.set_ylabel("缺失模态")
        for i in range(6):
            for j in range(3):
                value = matrix[i, j]
                if abs(value) >= .58 * bound:
                    ax.text(j, i, _cell(value), ha="center", va="center", fontsize=4.9,
                            color="white" if abs(value) > .68*bound else INK)
    fig.colorbar(im, ax=axes, shrink=.72, pad=.018, label="Delta MAE")
    save_figure(fig, OUT, "fig03_q2_position_rate_interaction")


def baseline_ablation_figure() -> None:
    summary = pd.read_csv(_require(Q2_OUTPUT_DIR / "analysis/all_baselines_three_seed_summary.csv"))
    ablation = pd.read_csv(_require(Q2_OUTPUT_DIR / "experiments/ablation/cumulative_ablation_seed2026.csv"))
    models = ["Concat-MLP", "BiGRU", "MulT no augmentation", "MMSA MulT",
              "TFR feature adapter", "MRRF full", "MRRF no reconstruction"]
    short = ["Concat", "BiGRU", "MulT-noaug", "MMSA MulT", "TFR", "MRRF-full", "MRRF-no-recon"]
    test = summary[(summary.split == "heldout_test") & (summary.scenario == "clean") &
                   summary.model.isin(models)].copy()
    test["order"] = test.model.map({m:i for i,m in enumerate(models)})
    test = test.sort_values("order")
    stages = ["Plain fusion", "+ Missing augmentation", "+ Reconstruction",
              "+ Reliability gate", "+ Consistency (MRRF)"]
    val = ablation[(ablation.split == "validation") & ablation.stage.isin(stages)].copy()

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 4.62))
    fig.subplots_adjust(left=.18, right=.985, top=.96, bottom=.10, hspace=.32, wspace=.23)
    y = np.arange(len(models))
    for ax, metric, letter in zip(axes[0], ("macro_f1", "mae"), ("a", "b")):
        means = test[f"{metric}_mean"].to_numpy(float)
        sds = test[f"{metric}_sd"].to_numpy(float)
        for yi, (_, row) in enumerate(test.iterrows()):
            color = MODEL_COLORS.get(row.model, "#6F8792")
            ax.errorbar(float(row[f"{metric}_mean"]), yi, xerr=float(row[f"{metric}_sd"]),
                        fmt="o", ms=4.2, capsize=2.0, lw=.9, color=color)
        ax.set_yticks(y, short); ax.invert_yaxis()
        ax.set_xlabel("Macro-F1" if metric == "macro_f1" else "MAE")
        panel_title(ax, letter, "七架构 · held-out clean · 3 seeds", pad=4)
        lo = float(np.min(means-sds)); hi = float(np.max(means+sds)); span=max(hi-lo,.01)
        ax.set_xlim(lo-.16*span, hi+.16*span)
        ax.grid(axis="x", color=GRID, linewidth=.45); ax.set_axisbelow(True)

    y2 = np.arange(len(stages)); stage_names = ["Plain", "+Aug", "+Recon", "+Reliability", "+Consistency"]
    for ax, metric, letter in zip(axes[1], ("macro_f1", "mae"), ("c", "d")):
        all_values = []
        for scenario, offset, color, marker in (("clean", -.10, "#627E8B", "o"),
                                                ("audio+vision", .10, MODALITY_COLORS["audio"], "s")):
            values = [float(_one_row(val, context="ablation", stage=stage,
                                     scenario=scenario)[metric]) for stage in stages]
            all_values += values
            ax.scatter(values, y2+offset, s=24, marker=marker, color=color,
                       label="Clean" if scenario == "clean" else "A+V 30%", zorder=3)
            ax.plot(values, y2+offset, color=color, alpha=.28, lw=.65)
        ax.set_yticks(y2, stage_names); ax.invert_yaxis()
        ax.set_xlabel("Macro-F1" if metric == "macro_f1" else "MAE")
        panel_title(ax, letter, "控制变量累计消融 · validation", pad=4)
        lo=min(all_values); hi=max(all_values); span=max(hi-lo,.01)
        ax.set_xlim(lo-.18*span, hi+.18*span)
        ax.grid(axis="x", color=GRID, linewidth=.45); ax.set_axisbelow(True)
    axes[1,0].legend(loc="lower left", frameon=False)
    save_figure(fig, OUT, "fig04_q2_models_and_ablation")


def error_attribution_figure() -> None:
    variant, seed, selected_label = _selection()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model, _ = load_trained(run_dir("mrrf", seed, variant), device)
    loader = make_loader(FeatureStore(DATA), "valid", 64, device)
    clean = evaluate(model, loader, device, "clean")
    masked = evaluate(model, loader, device, "audio+vision", rate=.3)
    if clean["ids"] != masked["ids"]:
        raise RuntimeError("Q2 error attribution requires identical validation order")
    true_cls = np.asarray(clean["true_class"], int)
    clean_cls = np.asarray(clean["pred_class"], int)
    masked_cls = np.asarray(masked["pred_class"], int)
    clean_ok = clean_cls == true_cls; masked_ok = masked_cls == true_cls
    transition = np.array([[np.sum(clean_ok & masked_ok), np.sum(clean_ok & ~masked_ok)],
                           [np.sum(~clean_ok & masked_ok), np.sum(~clean_ok & ~masked_ok)]], int)
    true_strength = np.asarray(clean["true_strength"], float)
    clean_pred = np.asarray(clean["pred_strength"], float)
    masked_pred = np.asarray(masked["pred_strength"], float)
    groups = np.asarray([_strength_group(v) for v in true_strength])
    group_order = ("Neutral", "Weak", "Moderate", "Strong")
    clean_mae = np.array([np.mean(np.abs(true_strength[groups==g]-clean_pred[groups==g])) for g in group_order])
    masked_mae = np.array([np.mean(np.abs(true_strength[groups==g]-masked_pred[groups==g])) for g in group_order])
    pd.DataFrame({"sample_id": clean["ids"], "true_class": true_cls,
                  "clean_pred_class": clean_cls, "masked_pred_class": masked_cls,
                  "true_strength": true_strength, "clean_pred_strength": clean_pred,
                  "masked_pred_strength": masked_pred, "strength_group": groups}).to_csv(
        OUT / "fig05_error_attribution_samples.csv", index=False, encoding="utf-8-sig")

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.92))
    fig.subplots_adjust(left=.09, right=.985, top=.90, bottom=.18, wspace=.30)
    ax=axes[0]; ax.imshow(transition, cmap="Blues", aspect="auto")
    ax.set_xticks((0,1),("Masked correct","Masked wrong")); ax.set_yticks((0,1),("Clean correct","Clean wrong"))
    panel_title(ax,"a","Clean -> A+V 30% 分类状态转移",pad=4)
    total=transition.sum()
    for i in range(2):
        for j in range(2):
            n=int(transition[i,j]); ax.text(j,i,f"{n}\n({n/total:.1%})",ha="center",va="center",
                                           fontsize=7.1,color="white" if n>transition.max()*.55 else INK)
    ax=axes[1]; x=np.arange(4)
    ax.plot(x,clean_mae,marker="o",lw=1.25,ms=3.8,color="#627E8B",label="Clean")
    ax.plot(x,masked_mae,marker="s",lw=1.25,ms=3.8,color=MODALITY_COLORS["audio"],label="A+V 30%")
    ax.set_xticks(x,group_order); ax.set_ylabel("MAE")
    panel_title(ax,"b","按真实情感强度分层的回归误差",pad=4)
    ax.grid(axis="y",color=GRID,linewidth=.45); ax.legend(frameon=False); ax.set_axisbelow(True)
    save_figure(fig, OUT, "fig05_q2_error_attribution")


def attachment3_figure() -> None:
    variant, seed, selected_label = _selection()
    pred = pd.read_csv(_require(run_dir("mrrf", seed, variant) / "attachment3_predictions_mrrf.csv"))
    paths = sorted(ATTACHMENT3.glob("*.pkl"))
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 Attachment-3 samples, found {len(paths)}")
    patterns = {m: [] for m in ("text","audio","vision")}; missing_rows=[]
    for path in paths:
        data = _attachment3_payload(path)
        content = np.asarray(content_positions(data["text_bert"])[0], dtype=bool)
        observed={"text":content.copy()}
        for name in ("audio","vision"):
            arr=np.asarray(data[name]); arr=arr[0] if arr.ndim==3 and arr.shape[0]==1 else arr
            observed[name]=content & np.any(np.abs(arr)>1e-12,axis=-1)
        row={"sample_id":path.stem,"content_length":int(content.sum())}
        for name in ("text","audio","vision"):
            state=np.zeros(len(content),np.int8); state[content]=1; state[content & ~observed[name]]=2
            patterns[name].append(state)
            miss=int((content & ~observed[name]).sum()); row[f"{name}_missing_count"]=miss
            row[f"{name}_missing_fraction"]=miss/max(int(content.sum()),1)
        missing_rows.append(row)
    pd.DataFrame(missing_rows).to_csv(OUT/"fig06_attachment3_missingness.csv",index=False,encoding="utf-8-sig")
    pred=pred.sort_values("sample_id").reset_index(drop=True)
    strength=pred.sentiment_strength.to_numpy(float)
    probs=pred[["prob_negative","prob_neutral","prob_positive"]].to_numpy(float)

    apply_paper_style(chinese=True)
    fig=plt.figure(figsize=(DOUBLE_COLUMN,5.18))
    gs=fig.add_gridspec(2,1,left=.075,right=.985,top=.95,bottom=.12,height_ratios=[1.38,1.0],hspace=.34)
    top=gs[0].subgridspec(1,3,wspace=.08); state_cmap=ListedColormap((LIGHT,"#DCEAE7",DANGER)); state_norm=BoundaryNorm((-.5,.5,1.5,2.5),3)
    for i,name in enumerate(("text","audio","vision")):
        ax=fig.add_subplot(top[i]); matrix=np.stack(patterns[name])
        ax.imshow(matrix,cmap=state_cmap,norm=state_norm,aspect="auto",interpolation="nearest")
        panel_title(ax,chr(ord("a")+i),name.title(),pad=3)
        ax.set_xticks([0,9,19,29,39,49],["1","10","20","30","40","50"])
        if i==0:
            ax.set_yticks([0,9,19,29],["01","10","20","30"]); ax.set_ylabel("sample")
        else:
            ax.set_yticks([])
        ax.set_xlabel("aligned position",fontsize=5.9)
    fig.text(.985,.972,"gray=padding · green=observed · red=zero-filled missing",ha="right",va="top",fontsize=5.5,color=MUTED)

    bottom=gs[1].subgridspec(1,2,wspace=.22)
    ax=fig.add_subplot(bottom[0]); x=np.arange(30)
    sentiment_colors=np.where(strength<0,"#B65A55",np.where(strength>0,"#4E7FA6","#7A8288"))
    ax.bar(x,strength,color=sentiment_colors,width=.78); ax.axhline(0,color=EDGE,lw=.65); ax.set_xlim(-.7,29.7)
    ax.set_ylabel("预测强度"); ax.set_xticks([0,4,9,14,19,24,29],["01","05","10","15","20","25","30"])
    panel_title(ax,"d","30 条无标签样本情感强度",pad=4)
    ax=fig.add_subplot(bottom[1]); left=np.zeros(30); colors=("#B65A55","#8A9197","#4E7FA6")
    for j,label in enumerate(("Negative","Neutral","Positive")):
        ax.bar(x,probs[:,j],bottom=left,color=colors[j],width=.78,label=label); left+=probs[:,j]
    ax.set_xlim(-.7,29.7); ax.set_ylim(0,1); ax.set_ylabel("类别概率")
    ax.set_xticks([0,4,9,14,19,24,29],["01","05","10","15","20","25","30"])
    panel_title(ax,"e",f"三分类概率 · {selected_label}",pad=4); ax.legend(ncol=3,loc="upper center",fontsize=5.5,frameon=False)
    save_figure(fig,OUT,"fig06_q2_attachment3")


def write_manifest() -> None:
    payload={
        "version":"v2_compact_journal_layout",
        "main_text_figures":{
            "fig01_q2_framework":"compact missingness construction + MRRF framework",
            "fig02_q2_missing_type_rate":"shared-scale degradation heatmaps with sparse annotations",
            "fig03_q2_position_rate_interaction":"complete 90-condition position x rate interaction",
            "fig04_q2_models_and_ablation":"three-seed architecture comparison + controlled cumulative ablation",
            "fig05_q2_error_attribution":"classification transition + regression error stratification",
            "fig06_q2_attachment3":"compact Attachment-3 missingness + full unlabeled predictions",
        },
        "notes":[
            "Long explanatory prose is kept in manuscript captions, not outside the figure canvas.",
            "Heatmaps emphasize visual pattern; complete numeric values remain in CSV outputs.",
            "No-reconstruction remains a tuned candidate, separate from controlled cumulative ablation.",
        ],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"figure_manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    framework_figure(); robustness_figure(); position_rate_figure(); baseline_ablation_figure()
    error_attribution_figure(); attachment3_figure(); write_manifest()
    print(f"Wrote compact Q2 V2 paper figures to {OUT}",flush=True)


if __name__=="__main__":
    main()
