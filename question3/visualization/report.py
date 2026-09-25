"""Create journal-sized, vector PDF and 300-dpi PNG Q3 figures and report."""
from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from common.paths import Q3_OUTPUT_DIR, Q2_OUTPUT_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from question3.pipeline import ROOT, MODS

COLORS = {"text": "#245D83", "audio": "#D18532", "vision": "#5A9674"}
BG = "#FBFCFD"
INK = "#23313E"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#9EAAB4",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def save(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=10)
    ax.grid(axis="y", color="#E5EAEE", linewidth=.6, zorder=0)


def source_path(value: str) -> Path:
    """Resolve recorded paths made by runs launched from the workspace root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT.parent / path


def performance(out, figs):
    df = pd.read_csv(out / "tables/model_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)
    order = ["accuracy", "macro_f1", "mae", "pearson"]
    labels = ["Accuracy", "Macro-F1", "MAE ↓", "Pearson r"]
    x = np.arange(4)
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].bar(x + (i - .5) * .32, [row[k] for k in order], width=.30,
                    label=f"{row['split']} (n={int(row['n'])})",
                    color=["#245D83", "#D18532"][i], zorder=3)
    panel(axes[0], "a", "Frozen model performance")
    axes[0].set_ylim(0, .8); axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Metric value")
    m = pd.read_csv(out / "tables/validation_explanation_metrics.csv")
    m = m[(m.stratum == "all") & (m.comparison == "top-minus-random")]
    y = np.arange(3)
    vals = [float(m[m.modality == mod].mean_delta_margin_difference.iloc[0]) for mod in MODS]
    axes[1].barh(y, vals, color=[COLORS[z] for z in MODS], height=.55, zorder=3)
    axes[1].set_yticks(y, [z.title() for z in MODS]); axes[1].invert_yaxis()
    panel(axes[1], "b", "Explanation advantage over random")
    axes[1].set_xlabel("Mean paired margin difference")
    for yi, value in zip(y, vals):
        axes[1].text(value + .006, yi, f"{value:.3f}", va="center", fontsize=7)
    axes[1].set_xlim(0, max(vals) * 1.25)
    save(fig, figs, "fig01_performance")


def fidelity(out, figs):
    m = pd.read_csv(out / "tables/validation_explanation_metrics.csv")
    m = m[m.stratum == "all"]
    fig, ax = plt.subplots(figsize=(7.15, 2.75), constrained_layout=True)
    centers = np.array([2, 1, 0], float)
    for j, control in enumerate(("random", "low")):
        rows = [m[(m.modality == mod) & (m.comparison == f"top-minus-{control}")].iloc[0]
                for mod in MODS]
        y = centers + (j - .5) * .23
        value = np.array([r.mean_delta_margin_difference for r in rows])
        lo = np.array([r.ci95_low for r in rows]); hi = np.array([r.ci95_high for r in rows])
        ax.errorbar(value, y, xerr=[value - lo, hi - value], fmt="o", capsize=2,
                    color=["#245D83", "#B57129"][j], label=f"Top − {control}",
                    markersize=4, linewidth=1.3)
    ax.axvline(0, color="#4D5963", lw=.8, linestyle="--")
    ax.set_yticks(centers, [f"{mod.title()}  (n={int(m[(m.modality == mod) & (m.comparison == 'top-minus-random')].n.iloc[0])})"
                            for mod in MODS])
    ax.set_xlabel("Paired difference in predicted-class logit-margin drop (95% bootstrap CI)")
    ax.legend(loc="lower right", frameon=False)
    panel(ax, "", "Validation-set fidelity, fixed 20% content window")
    ax.set_xlim(-.045, max(m.ci95_high) + .08)
    save(fig, figs, "fig02_fidelity")


def contributions(out, figs):
    q = pd.read_csv(out / "tables/attachment4_predictions.csv", dtype={"sample_id": str})
    q["sample_id"] = q.sample_id.str.zfill(2)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 5.45), sharey=True,
                             gridspec_kw={"width_ratios": [1.05, 2.2]}, constrained_layout=True)
    y = np.arange(len(q))
    delta = q[[f"top_delta_margin_{m}" for m in MODS]].to_numpy(float)
    vmax = max(.1, float(np.percentile(np.abs(delta), 95)))
    im = axes[0].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        interpolation="nearest", extent=(-.5, 2.5, len(q)-.5, -.5))
    for i in range(len(q)):
        for j in range(3):
            axes[0].text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center",
                         fontsize=6.3, color="white" if abs(delta[i,j]) > .65*vmax else INK)
    axes[0].set_xticks(range(3), [m.title() for m in MODS])
    axes[0].set_title("Signed deletion response", loc="left", fontweight="bold")
    axes[0].set_xlabel("Predicted-class margin drop")
    fig.colorbar(im, ax=axes[0], location="bottom", shrink=.85, pad=.035,
                 label="Color clipped at 95th percentile; numbers are exact")
    left = np.zeros(len(q))
    for mod in MODS:
        vals = q[f"gate_{mod}"].to_numpy(float)
        axes[1].barh(y, vals, left=left, color=COLORS[mod], height=.72,
                     label=mod.title(), edgecolor="white", linewidth=.2)
        left += vals
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("Internal fusion weight")
    axes[1].set_title("Internal fusion weight", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#E5EAEE", linewidth=.6)
    axes[1].set_axisbelow(True)
    polarity = {"Negative": "−", "Neutral": "0", "Positive": "+"}
    axes[0].set_yticks(y, [f"{sid}  {polarity[ann]}{'*' if dom == 'unclear' else ''}"
                            for sid, ann, dom in zip(q.sample_id, q.annotation, q.dominant_modality)])
    axes[1].legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=3, frameon=False)
    fig.text(.5, -.075, "−/0/+ = Negative/Neutral/Positive; * = no clear main modality. Negative drop means deletion strengthened the original prediction.",
             ha="center", fontsize=6.5)
    save(fig, figs, "fig03_modality_contributions")


def position_overview(out, figs):
    df = pd.read_csv(out / "tables/attachment4_position_ig.csv", dtype={"sample_id": str})
    df.sample_id = df.sample_id.str.zfill(2)
    ids = sorted(df.sample_id.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 5.2), sharey=True, constrained_layout=True)
    for ax, mod in zip(axes, MODS):
        sub = df[df.modality == mod]
        mat = np.full((len(ids), 50), np.nan)
        selected = []
        for yi, sid in enumerate(ids):
            rows = sub[sub.sample_id == sid]
            for _, r in rows.iterrows():
                if bool(r.native_available):
                    mat[yi, int(r.token_index)] = r.ig_signed
            ss = rows[rows.selected == True].token_index.to_numpy(int)
            if len(ss):
                selected.append((yi, ss.min(), ss.max()))
            finite = np.isfinite(mat[yi])
            scale = np.max(np.abs(mat[yi, finite])) if finite.any() else 0
            if scale > 0:
                mat[yi, finite] /= scale
        cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#E6E9EC")
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                       interpolation="nearest", extent=(-.5, 49.5, len(ids)-.5, -.5))
        for yi, start, end in selected:
            ax.add_patch(Rectangle((start - .5, yi - .45), end - start + 1, .9,
                                   fill=False, edgecolor="#26323D", linewidth=.75))
        ax.set_title(mod.title(), loc="left", fontweight="bold")
        ax.set_xlabel("BERT content position")
        ax.set_xticks([0, 10, 20, 30, 40, 49]); ax.set_yticks(range(len(ids)), ids)
    fig.colorbar(im, ax=axes, location="bottom", shrink=.6, aspect=35,
                 label="Signed IG / per-sample modality max |IG|", pad=.08)
    fig.text(.5, -.075, "Outlined cells: selected 20% window. Gray: padding, special token or native absence. 07 and 18 are text-truncated.",
             ha="center", fontsize=6.5)
    save(fig, figs, "fig04_local_importance")


def alignment_quality(out, figs):
    df = pd.read_csv(out / "tables/attachment4_alignment_audit.csv", dtype={"sample_id": str})
    df.sample_id = df.sample_id.str.zfill(2)
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.75), sharey=True, constrained_layout=True)
    for ax, col, threshold, title in [(axes[0], "word_coverage", .85, "WhisperX word coverage"),
                                       (axes[1], "asr_similarity", .35, "Independent ASR agreement")]:
        colors = np.where(df.alignment_status == "trusted", "#5A9674", "#BC6B5E")
        ax.barh(y, df[col], color=colors, height=.65)
        ax.axvline(threshold, color="#5C6370", linestyle="--", linewidth=.9)
        ax.set_xlim(0, 1.04); ax.set_xlabel("Fraction / sequence similarity")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E5EAEE", linewidth=.6); ax.set_axisbelow(True)
    axes[0].set_yticks(y, df.sample_id); axes[0].invert_yaxis()
    fig.text(.5, -.005, "Green: sample-level trusted; red: rejected. Word-level low confidence can still leave individual evidence un-timed.",
             ha="center", fontsize=6.5)
    save(fig, figs, "fig05_alignment_quality")


def timed_evidence(out, figs, sid="19"):
    ev = pd.read_csv(out / "tables/attachment4_evidence.csv", dtype={"sample_id": str})
    ev.sample_id = ev.sample_id.str.zfill(2)
    rows = ev[ev.sample_id == sid]
    video = str(rows.video_path.iloc[0])
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-f", "f32le",
                           "-ac", "1", "-ar", "16000", "pipe:1"], capture_output=True, check=True)
    wav = np.frombuffer(proc.stdout, dtype="<f4")
    block = 320
    levels = np.max(np.abs(wav[:len(wav)//block*block].reshape(-1, block)), axis=1)
    ts = (np.arange(len(levels)) + .5) * block / 16000
    fig = plt.figure(figsize=(7.15, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(ts, levels, color="#687A89", lw=.65)
    for mod in MODS:
        r = rows[rows.modality == mod].iloc[0]
        if pd.notna(r.start_sec):
            ax.axvspan(r.start_sec, r.end_sec, color=COLORS[mod], alpha=.27,
                       label=f"{mod.title()}: {r.start_sec:.1f}–{r.end_sec:.1f} s")
    panel(ax, "a", f"Original audio waveform and model-selected windows | sample {sid}")
    ax.set_xlim(0, len(wav)/16000); ax.set_xlabel("Clip-relative time (s)")
    ax.set_ylabel("Peak amplitude / 20 ms")
    ax.legend(ncol=3, frameon=False, fontsize=6.5, loc="upper right")
    txt = fig.add_subplot(gs[1, 0]); txt.axis("off")
    txt.set_title("b  Source-text evidence at those times", loc="left", fontweight="bold")
    for i, mod in enumerate(MODS):
        r = rows[rows.modality == mod].iloc[0]
        excerpt = str(r.source_excerpt).replace("\n", " ")
        txt.text(0, .88-i*.30, f"{mod.title()}: {textwrap.shorten(excerpt, width=58, placeholder='…')}",
                 color=COLORS[mod], fontsize=7.5, va="top")
    frame = fig.add_subplot(gs[1, 1]); frame.axis("off")
    vision = rows[rows.modality == "vision"].iloc[0]
    frame.imshow(plt.imread(source_path(vision.keyframe_path)))
    frame.set_title(f"c  Video frame at {vision.keyframe_sec:.2f} s", loc="left", fontweight="bold")
    save(fig, figs, "fig06_timed_evidence")


def model_sensitivity(out, figs):
    s = pd.read_csv(out / "tables/attachment4_model_sensitivity.csv", dtype={"sample_id": str})
    s.sample_id = s.sample_id.str.zfill(2)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    colors = np.where(s.class_agreement, "#5A9674", "#BC6B5E")
    axes[0].scatter(s.primary_strength, s.full_strength, c=colors, s=32,
                    edgecolor="white", linewidth=.5, zorder=3)
    axes[0].plot([-3, 3], [-3, 3], linestyle="--", color="#657483", lw=.8)
    for _, r in s[~s.class_agreement].iterrows():
        axes[0].annotate(r.sample_id, (r.primary_strength, r.full_strength),
                         xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_xlabel("Primary no_recon strength")
    axes[0].set_ylabel("Full-variant strength")
    panel(axes[0], "a", "Frozen-variant strength sensitivity")
    order = ["Negative", "Neutral", "Positive"]
    tab = pd.crosstab(s.primary_annotation, s.full_annotation).reindex(index=order, columns=order, fill_value=0)
    axes[1].imshow(tab.to_numpy(), cmap="Blues", vmin=0)
    axes[1].set_xticks(range(3), order, rotation=25, ha="right")
    axes[1].set_yticks(range(3), order)
    axes[1].set_xlabel("Full variant polarity")
    axes[1].set_ylabel("Primary polarity")
    axes[1].set_title("b  Polarity agreement", loc="left", fontweight="bold")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(tab.iat[i,j]), ha="center", va="center",
                         color="white" if tab.iat[i,j] > tab.to_numpy().max()*.55 else INK,
                         fontsize=8, fontweight="bold")
    save(fig, figs, "fig07_model_sensitivity")


def validation_variant_fidelity(out, figs):
    df = pd.read_csv(out / "tables/validation_fidelity_model_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), sharey=True,
                             constrained_layout=True)
    y = np.array([2, 1, 0], float)
    for ax, metric, title in zip(axes, ("margin", "prob"),
                                 ("Predicted-class margin", "Predicted-class probability")):
        for offset, stratum, color, label in (
                (-.12, "all", "#245D83", "All paired samples"),
                (+.12, "same_predicted_class", "#B57129", "Same predicted class")):
            rows = [df[(df.modality == mod) & (df.stratum == stratum) &
                       (df.metric == metric)].iloc[0] for mod in MODS]
            value = np.array([r.full_minus_no_recon for r in rows])
            low = np.array([r.ci95_low for r in rows])
            high = np.array([r.ci95_high for r in rows])
            ax.errorbar(value, y + offset, xerr=[value-low, high-value],
                        color=color, fmt="o", capsize=2, markersize=4,
                        linewidth=1.2, label=label)
        ax.axvline(0, color="#62717D", linestyle="--", linewidth=.8)
        ax.set_yticks(y, [m.title() for m in MODS])
        ax.set_xlabel("Full − no_recon fidelity advantage")
        panel(ax, "a" if metric == "margin" else "b", title)
    axes[0].legend(frameon=False, loc="upper center", fontsize=6.5,
                   bbox_to_anchor=(1.05, -.30), ncol=2)
    fig.suptitle("Frozen-variant explanation fidelity | paired validation samples",
                 x=.03, ha="left", fontsize=10, fontweight="bold")
    save(fig, figs, "fig08_validation_variant_fidelity")


def gate_rank_diagnostic(out, figs):
    df = pd.read_csv(out / "tables/gate_response_rank_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), sharey=True,
                             constrained_layout=True)
    conditions = [("validation", "top_window"), ("validation", "whole_modality"),
                  ("attachment4", "top_window"), ("attachment4", "whole_modality")]
    labels = ["Valid · local window", "Valid · whole modality",
              "Attachment 4 · local", "Attachment 4 · whole"]
    yy = np.arange(len(conditions))[::-1]
    colors = ["#245D83", "#B57129", "#245D83", "#B57129"]
    for ax, value_key, lo_key, hi_key, title, xlabel in (
            (axes[0], "top1_agreement", "top1_ci95_low", "top1_ci95_high",
             "Top-ranked modality", "Gate / deletion top-1 agreement"),
            (axes[1], "mean_spearman_rho", "rho_ci95_low", "rho_ci95_high",
             "Three-modality ordering", "Mean Spearman ρ")):
        for i, (split, kind) in enumerate(conditions):
            r = df[(df.split == split) & (df.perturbation == kind)].iloc[0]
            value = float(r[value_key])
            observed_y = yy[i] - (.065 if value_key == "top1_agreement" else 0)
            ax.errorbar(value, observed_y,
                        xerr=[[value-float(r[lo_key])], [float(r[hi_key])-value]],
                        fmt="o", color=colors[i], capsize=2, markersize=4, linewidth=1.2,
                        label="Observed" if i == 0 and value_key == "top1_agreement" else None)
            if value_key == "top1_agreement":
                ax.scatter(float(r.independent_marginal_expected_top1), yy[i] + .065,
                           marker="D", s=18, facecolor="white", edgecolor="#424D57",
                           linewidth=.9, zorder=4,
                           label="Marginal baseline" if i == 0 else None)
        ax.set_yticks(yy, labels)
        ax.set_xlabel(xlabel)
        panel(ax, "a" if value_key == "top1_agreement" else "b", title)
    axes[0].set_xlim(0, 1.03)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper center",
                   bbox_to_anchor=(1.05, -.30), ncol=2)
    axes[1].axvline(0, color="#62717D", linestyle="--", linewidth=.8)
    axes[1].set_xlim(-1.05, 1.05)
    fig.suptitle("Gate ranks versus signed deletion | top gate is Text for every sample",
                 x=.03, ha="left", fontsize=10, fontweight="bold")
    save(fig, figs, "fig09_gate_response_rank")


def case_card(out, figs, sid):
    pred = pd.read_csv(out / "tables/attachment4_predictions.csv", dtype={"sample_id": str})
    pred.sample_id = pred.sample_id.str.zfill(2)
    e = pd.read_csv(out / "tables/attachment4_evidence.csv", dtype={"sample_id": str})
    e.sample_id = e.sample_id.str.zfill(2)
    pos = pd.read_csv(out / "tables/attachment4_position_ig.csv", dtype={"sample_id": str})
    pos.sample_id = pos.sample_id.str.zfill(2)
    audit = pd.read_csv(out / "tables/attachment4_alignment_audit.csv", dtype={"sample_id": str})
    audit.sample_id = audit.sample_id.str.zfill(2)
    q = pred[pred.sample_id == sid].iloc[0]
    rows = e[e.sample_id == sid]
    a = audit[audit.sample_id == sid].iloc[0]
    fig = plt.figure(figsize=(7.15, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[.65, 1.6, 1.15], width_ratios=[1.03, 1])
    title = fig.add_subplot(gs[0, :]); title.axis("off")
    strength = (f"{q.sentiment_strength:+.4f}" if 0 < abs(q.sentiment_strength) < .005
                else f"{q.sentiment_strength:+.2f}")
    title.text(0, .95, f"Sample {sid}  |  {q.annotation}  |  strength {strength}",
               fontweight="bold", fontsize=12, va="top")
    title.text(0, .66, f"P(N/U/P) = {q.prob_negative:.2f} / {q.prob_neutral:.2f} / {q.prob_positive:.2f}"
               f"    Predicted-class margin = {q.margin:.2f}", fontsize=8)
    conflict = "  |  polarity / regression conflict" if "classification_regression_sign_conflict" in str(q.explanation_quality_flags) else ""
    title.text(0, .39, f"Raw regression = {q.raw_strength:+.3f}  |  Main response = {q.dominant_modality}{conflict}", fontsize=8)
    title.text(0, .12, f"Alignment: {a.alignment_status}  |  ASR agreement {a.asr_similarity:.2f}"
               f"  |  text truncated: {bool(a.text_truncated)}", fontsize=8)
    ax = fig.add_subplot(gs[1, 0])
    for mod in MODS:
        sub = pos[(pos.sample_id == sid) & (pos.modality == mod) & (pos.native_available == True)]
        if len(sub):
            ax.plot(sub.token_index, sub.ig_signed, color=COLORS[mod], lw=1.15,
                    marker="o", ms=2, label=mod.title())
        er = rows[rows.modality == mod].iloc[0]
        if bool(er.available):
            ax.axvspan(er.token_start - .4, er.token_end + .4,
                       color=COLORS[mod], alpha=.09)
    ax.axhline(0, color="#63727E", linewidth=.7)
    ax.set_xlabel("BERT content position"); ax.set_ylabel("Signed Integrated Gradient")
    panel(ax, "a", "Local feature attribution")
    ax.legend(frameon=False, loc="best")
    right = fig.add_subplot(gs[1, 1]); right.axis("off")
    right.set_title("b  Auditable evidence", loc="left", fontweight="bold", pad=10)
    for i, mod in enumerate(MODS):
        er = rows[rows.modality == mod].iloc[0]
        yy = 1 - i * .31
        right.text(0, yy, mod.title(), color=COLORS[mod], fontweight="bold", va="top")
        if not bool(er.available):
            desc = "Native features absent; no attribution."
        else:
            excerpt = str(er.source_excerpt) if pd.notna(er.source_excerpt) else ""
            clock = (f"{er.start_sec:.2f}–{er.end_sec:.2f} s" if pd.notna(er.start_sec)
                     else f"No trusted clock ({er.localization_status})")
            desc = f"Tokens {int(er.token_start)}–{int(er.token_end)}; {clock}\n“{excerpt}”\nΔmargin {er.top_delta_margin:+.3f}; random {er.random_delta_margin:+.3f}"
        right.text(.25, yy, textwrap.fill(desc, 39, break_long_words=False), va="top", fontsize=7)
    bars = fig.add_subplot(gs[2, 0])
    effects = [q[f"top_delta_margin_{m}"] for m in MODS]
    bars.barh(np.arange(3), effects, color=[COLORS[m] for m in MODS], height=.55)
    bars.axvline(0, color="#63727E", linewidth=.8)
    bars.set_yticks(range(3), [m.title() for m in MODS]); bars.invert_yaxis()
    bars.set_xlabel("Margin drop after selected-span deletion")
    panel(bars, "c", "Observed perturbation response")
    frame_ax = fig.add_subplot(gs[2, 1]); frame_ax.axis("off")
    er = rows[rows.modality == "vision"].iloc[0]
    if isinstance(er.keyframe_path, str) and er.keyframe_path and source_path(er.keyframe_path).exists():
        frame_ax.imshow(plt.imread(source_path(er.keyframe_path)))
        frame_ax.set_title(f"d  Source frame at {er.keyframe_sec:.2f} s", loc="left", fontweight="bold")
    else:
        frame_ax.set_facecolor("#E6E9EC")
        frame_ax.text(.5, .5, "No trustworthy visual-frame localization", ha="center", va="center",
                      transform=frame_ax.transAxes, fontsize=8, wrap=True)
        frame_ax.set_title("d  Visual evidence status", loc="left", fontweight="bold")
    save(fig, figs, f"case_{sid}")


def write_report(out):
    cfg = json.loads((out / "config.json").read_text())
    perf = pd.read_csv(out / "tables/model_performance.csv")
    f = pd.read_csv(out / "tables/validation_explanation_metrics.csv")
    q = pd.read_csv(out / "tables/attachment4_predictions.csv", dtype={"sample_id": str})
    a = pd.read_csv(out / "tables/attachment4_alignment_audit.csv", dtype={"sample_id": str})
    e = pd.read_csv(out / "tables/attachment4_evidence.csv", dtype={"sample_id": str})
    explained_n = len(pd.read_csv(out / "tables/validation_predictions_explained.csv"))
    completeness = pd.read_csv(out / "tables/attachment4_ig_completeness.csv")
    valid_completeness = pd.read_csv(out / "tables/validation_ig_completeness.csv")
    sensitivity = json.loads((out / "model_sensitivity.json").read_text())
    variant = pd.read_csv(out / "tables/validation_fidelity_model_summary.csv")
    full_ig = pd.read_csv(out / "tables/full_validation_ig_completeness.csv")
    quadrature = pd.read_csv(out / "tables/full_validation_ig_quadrature_check.csv")
    rank = pd.read_csv(out / "tables/gate_response_rank_summary.csv")
    all_random = f[(f.stratum == "all") & (f.comparison == "top-minus-random")]
    lines = ["# 问题三：预测解释实验结果", "",
             "冻结问题二的 MRRFNet no_recon 检查点，以三分类 logit margin 为解释目标；附件四的 20 个无标签样本没有参与模型训练、选模或总体忠实度估计，仅用于最终推理及无标签的数值、对齐核验。", "",
             "## 预测性能", "",
             "|划分|样本数|Accuracy|Macro-F1|MAE|Pearson r|", "|---|---:|---:|---:|---:|---:|"]
    for _, x in perf.iterrows():
        lines.append(f"|{x['split']}|{int(x.n)}|{x.accuracy:.3f}|{x.macro_f1:.3f}|{x.mae:.3f}|{x.pearson:.3f}|")
    lines += ["", "情感强度使用问题二既定的极性一致化输出；原始回归值仍在逐样本表中。", "",
              "## 解释忠实度", "",
              f"验证集按真实类别比例、固定种子预先确定 {explained_n} 条（本次覆盖全部验证集）。每模态以 {cfg['ig_steps']} 步 IG 从输入的 10% 衰减参考点出发，提出 20% 内容长度的连续窗口，再独立删除该窗口，并和同长度、相同原生有效位置数的 8 次随机窗口及低 IG 窗口比较。遮挡同步将该模态对应位置的可用掩码置零；没有至少两个可匹配窗口的样本不计入该模态忠实度比较。下表报告关键段相对随机段的配对 logit-margin 下降差异；95% 区间为 2000 次样本 bootstrap。", "",
              "|模态|n|关键−随机|95% CI|关键优于随机比例|关键段自身平均下降|", "|---|---:|---:|---:|---:|---:|"]
    for _, x in all_random.iterrows():
        lines.append(f"|{x.modality}|{int(x.n)}|{x.mean_delta_margin_difference:+.3f}|[{x.ci95_low:+.3f}, {x.ci95_high:+.3f}]|{x.top_better_fraction:.1%}|{x.mean_top_delta_margin:+.3f}|")
    lines += ["", f"数值完备性检查：全验证集 |ΣIG−(margin(x)−margin(0.1x))| 平均 {valid_completeness.absolute_completeness_error.mean():.4f}、最大 {valid_completeness.absolute_completeness_error.max():.4f}；附件四 20 条平均 {completeness.absolute_completeness_error.mean():.4f}、最大 {completeness.absolute_completeness_error.max():.4f}。精确零特征基线附近的归一化路径数值变化很陡，因此固定采用 10% 衰减参考点；该参考点仍含部分原始信息，IG 代表相对衰减参考点的贡献，不代表从绝对无信息状态的因果效果。", "",
              "视觉关键段相对随机段的配对优势 95% 区间跨零，且其自身平均遮挡响应为负：删除该段反而提高原预测类别的 margin。因此本实验没有证明视觉局部解释具有稳定正向忠实度，不能把视觉 IG 或内部门控权重直接表述为可靠的因果贡献。音频有微弱正向优势，但明显弱于文本。", "",
              f"冻结问题二 full 变体在附件四 20 条中极性一致 {sensitivity['class_agreement_count']}/20，强度平均绝对差 {sensitivity['mean_absolute_strength_difference']:.3f}。附件四无标签，这一比较只描述模型选择不确定性，不证明哪一个变体更准确。", "",
              "## 同一验证集的冻结变体解释对照", "",
              "对 no_recon 与 full 的种子 2026 检查点，在相同 728 条验证样本、48 步 IG、20% 连续窗口和随机种子下重跑解释。每个模型解释自己的预测类别；下表差值为 full 减 no_recon 的关键段相对随机段 margin 下降优势，2000 次配对样本 bootstrap 给出 95% 区间。不同模型 logit 尺度可能不同，因此同时保存概率下降差值，并单列两模型预测类别相同的样本。此为既定选模后的诊断，不反向改选问题三冻结检查点。", "",
              "|模态|配对 n|no_recon|full|full−no_recon|95% CI|",
              "|---|---:|---:|---:|---:|---:|"]
    for mod in MODS:
        x = variant[(variant.modality == mod) & (variant.stratum == "all") &
                    (variant.metric == "margin")].iloc[0]
        lines.append(f"|{mod}|{int(x.n)}|{x.no_recon_mean:+.3f}|{x.full_mean:+.3f}|"
                     f"{x.full_minus_no_recon:+.3f}|[{x.ci95_low:+.3f}, {x.ci95_high:+.3f}]|")
    lines += ["", "原始逐样本配对表见 validation_fidelity_model_paired.csv；同类预测子集和概率尺度结果见 validation_fidelity_model_summary.csv，图 8 展示两种口径。不能把一个模态的忠实度优势自动解释为整体预测准确率优势。", "",
              f"full 变体的 728 条 IG 完备性绝对残差平均 {full_ig.absolute_completeness_error.mean():.4f}；48 步下有 {int((full_ig.absolute_completeness_error >= .1).sum())} 条超过预定的 0.1 margin 容差（最大 {full_ig.absolute_completeness_error.max():.4f}）。对残差最大者使用 96/192 步复核，192 步残差降至 {quadrature.absolute_completeness_error.iloc[-1]:.4f}；见 full_validation_ig_quadrature_check.csv。这说明该例主要受积分离散误差影响，但不等于对全部样本做了高步数积分。", "",
              "## 门控权重与删除响应的一致性", "",
              "在原生可用模态中，比较内部门控权重与有符号删除响应的排序。局部实验删除 IG 候选窗口；补充实验清零整个模态及其可用掩码。后者超出问题二主要训练缺失比例，仅作敏感性诊断。下表的 top-1 一致率和平均 Spearman ρ 均给出 2000 次样本 bootstrap 区间；附件四结果无标签。", "",
              "|数据|删除范围|n|top-1 一致率 [95% CI]|边际基线|κ|平均 Spearman ρ [95% CI]|",
              "|---|---|---:|---:|---:|---:|---:|"]
    for _, x in rank.iterrows():
        label = "验证集" if x.split == "validation" else "附件四"
        kind = "局部窗口" if x.perturbation == "top_window" else "整个模态"
        lines.append(f"|{label}|{kind}|{int(x.n)}|{x.top1_agreement:.1%} "
                     f"[{x.top1_ci95_low:.1%}, {x.top1_ci95_high:.1%}]|"
                     f"{x.independent_marginal_expected_top1:.1%}|{x.cohen_kappa:+.3f}|"
                     f"{x.mean_spearman_rho:+.3f} "
                     f"[{x.rho_ci95_low:+.3f}, {x.rho_ci95_high:+.3f}]|")
    lines += ["", "门控最高权重在验证集 728 条及附件四 20 条中均为文本，因此表面的 top-1 一致率恰好等于按两侧边际分布计算的基线，κ=0。门控没有识别样本间主导模态的变化；平均 Spearman ρ 只描述三模态内部排序的部分相关，不改变这一结论。图 9 将观测一致率与边际基线并列展示。门控权重不能当成因果贡献百分比，主要参考模态继续按有符号删除响应和质量门限给出。", "",
              "## 附件四定位与异常", "",
              f"20/20 条完成极性、强度及三模态响应输出；{sum(a.alignment_status == 'trusted')}/20 条通过样本级词对齐与 ASR 一致性筛查。可用模态关键段中，{sum(e.localization_status == 'trusted_word_alignment')}/{int(e.available.sum())} 条获得可信秒数；另有 1 条原生视觉不可用。", "",
              "02 的原始回归值为负而分类为 Positive，按既定极性一致化规则导出强度 +0.0001，并保留分类/回归冲突标志；不能把它解读为稳健的正向强度。07、18 的文本超过 BERT 50 位置上限，解释只涵盖前 48 个内容 token。13 的视觉特征全零，视觉证据留空。15 的语音与官方文本大部分不对应，拒绝所有时钟定位。其余样本若关键段包含低置信词，也只保留 token 和原文，不填写秒数。", "",
              "## 图表与回查", "",
              "- [图1：预测性能](../figures/fig01_performance.pdf)；[图2：解释忠实度](../figures/fig02_fidelity.pdf)。",
              "- [图3：20 条模态作用](../figures/fig03_modality_contributions.pdf)；[图4：局部 IG](../figures/fig04_local_importance.pdf)；[图5：对齐质量](../figures/fig05_alignment_quality.pdf)；[图6：原音轨与视频时间证据](../figures/fig06_timed_evidence.pdf)；[图7：模型敏感性](../figures/fig07_model_sensitivity.pdf)。",
              "- [图8：同一验证集的变体解释对照](../figures/fig08_validation_variant_fidelity.pdf)；[图9：门控与扰动排序核查](../figures/fig09_gate_response_rank.pdf)。",
              "- 典型解释卡：01、07（截断）、13（视觉缺失）、14（无可信局部秒数）、15（对齐拒绝）、19（视觉响应且可定位）；附件四全部 20 条的独立解释卡见下列链接：",
              "- [补充材料：20 页附件四逐样本解释卡合订 PDF](attachment4_explanation_cards.pdf)。",
              "- 逐样本结果在 `tables/attachment4_predictions.csv`、`tables/attachment4_evidence.csv`、`tables/attachment4_word_map.csv` 与 `tables/attachment4_alignment_audit.csv`。", "",
              "图中关键帧是依据同一 token 对应原词的强制对齐时段截取的原视频画面，是时间定位示例，不证明模型关注了画面中的某个物体。音频/视觉原始特征文件没有逐位置秒数，token→词→秒的对应是可审核的候选映射；对齐不可信时不输出时间。", ""]
    card_index = next(i for i, line in enumerate(lines) if line.startswith("- 逐样本结果"))
    lines.insert(card_index, "  " + "、".join(
        f"[{sid}](../figures/case_{sid}.pdf)"
        for sid in sorted(q.sample_id.astype(str).str.zfill(2))))
    lines += ["## 附件四逐样本摘要", "",
              "|ID|极性|强度|主要参考模态|可信关键段数|质量标志|",
              "|---|---|---:|---|---:|---|"]
    for _, r in q.iterrows():
        strength_text = (f"{r.sentiment_strength:+.4f}" if 0 < abs(r.sentiment_strength) < .001
                         else f"{r.sentiment_strength:+.3f}")
        lines.append(f"|{str(r.sample_id).zfill(2)}|{r.annotation}|{strength_text}|{r.dominant_modality}|{int(r.trusted_evidence_count)}/3|{r.explanation_quality_flags}|")
    lines += ["", "## 可直接嵌入论文的图", "",
              "![图1：预测性能](../figures/fig01_performance.png)", "",
              "![图2：解释忠实度](../figures/fig02_fidelity.png)", "",
              "![图3：模态作用](../figures/fig03_modality_contributions.png)", "",
              "![图4：局部重要性](../figures/fig04_local_importance.png)", "",
              "![图5：对齐质量](../figures/fig05_alignment_quality.png)", ""]
    lines += ["![图6：原音轨与视频时间证据](../figures/fig06_timed_evidence.png)", ""]
    lines += ["![图7：冻结模型敏感性](../figures/fig07_model_sensitivity.png)", "",
              "![图8：验证集变体解释对照](../figures/fig08_validation_variant_fidelity.png)", "",
              "![图9：门控与扰动排序核查](../figures/fig09_gate_response_rank.png)", ""]
    paper = out / "paper"; paper.mkdir(parents=True, exist_ok=True)
    (paper / "q3_results.md").write_text("\n".join(lines), encoding="utf-8")
    captions = [
        "# 问题三图注（适用于期刊双栏排版）", "",
        "**图1.** 冻结模型的分类与强度预测性能及解释优势。验证集和留出测试集使用附件二的固定划分；b 面板为验证集关键段相对等长度随机段的平均 margin 下降差异。MAE 越低越好，其余指标越高越好。",
        "", "**图2.** 全验证集独立删除检验。点为关键段与随机段/低 IG 段的配对平均差，误差线为 2000 次样本 bootstrap 的 95% 分位数区间。比较窗口的内容宽度及原生有效位置数相同；零线表示无优势。视觉关键段自身平均下降为负，应结合原始有符号变化解读。",
        "", "**图3.** 附件四 20 条样本的模态作用差异。左侧为每模态关键段删除前后预测类别 logit margin 的有符号下降；负值表示删除该段反而增强原预测，数值标注为未截断真实值。右侧为模型内部门控权重，不作为因果贡献。‘*’表示主导模态不明确。",
        "", "**图4.** 附件四局部归因图。每个样本、每个模态独立以最大绝对 IG 标准化，仅比较该行的位置分布，不跨行比较绝对幅度；黑框为预先固定比例的候选证据窗口。灰色为空白、特殊 token 或原生不可用位置。07、18 仅展示进入模型的截断文本。",
        "", "**图5.** 附件四的原视频定位质量。WhisperX 词覆盖率和独立 ASR 转写相似度用于样本级审核；红色样本 15 的官方文本与音轨大部分不一致，所有关键证据均保留原文/token 而不填秒数。",
        "", "**图6.** 样本 19 的可审核原视频时间证据。a 为原视频音轨每 20 ms 的峰值包络，阴影为模型选出的文本/音频/视觉 token 窗口经词级强制对齐得到的候选秒段；b 为对应原文；c 为视觉窗口中点截取的视频帧。三模态的关键段均通过局部词级时间审核。",
        "", "**图7.** 冻结模型变体的敏感性。左侧比较首选 no_recon 与 full 变体在附件四无标签样本上的情感强度，灰虚线为一致；红点及样本编号表示极性不同。右侧为两变体的极性列联表。该图仅反映模型选择的敏感性，不能以无标签附件四判定哪一个模型更准确。",
        "", "**图8.** 两个冻结检查点在同一 728 条验证集的局部解释忠实度差值，full 减 no_recon；a 为预测类别 margin 尺度，b 为预测类别概率尺度。两色区分全部可比样本与两模型预测类别相同的子集，误差线为 2000 次配对 bootstrap 的 95% 区间。分析发生在原选模之后，不用来回选测试模型。",
        "", "**图9.** 内部门控权重与有符号删除响应的模态排序一致性。a 为最高模态一致率及由两侧边际分布计算的基线，b 为平均 Spearman 排序相关；误差线为样本 bootstrap 95% 区间。所有样本的门控最高模态均为文本，故最高模态一致率不高于边际基线（κ=0）。整模态删除超出主要训练遮挡范围，仅供敏感性诊断；门控权重不等于因果作用比例。",
        "", "**附件四 20 条解释卡.** a 为三模态有符号局部 IG，半透明背景为选中窗口；b 为 token、原文和可信秒段；c 为删除后的预测类别 margin 下降（负值表示删除提高原预测 margin）；d 为在可信词时钟上抽取的原视频画面。01、07、13、14、15、19 可作为正文典型样本，其余卡片置于补充材料。关键帧只表示时间对应关系，不表示帧内物体归因。", ""
    ]
    (paper / "figure_captions.md").write_text("\n".join(captions), encoding="utf-8")


def run(out: Path):
    figs = out / "figures"
    performance(out, figs); fidelity(out, figs); contributions(out, figs)
    position_overview(out, figs); alignment_quality(out, figs)
    timed_evidence(out, figs); model_sensitivity(out, figs)
    validation_variant_fidelity(out, figs); gate_rank_diagnostic(out, figs)
    sample_ids = pd.read_csv(out / "tables/attachment4_predictions.csv",
                             dtype={"sample_id": str}).sample_id.str.zfill(2).tolist()
    for sid in sample_ids:
        case_card(out, figs, sid)
    write_report(out)
    subprocess.run(["pdfunite", *[str(figs / f"case_{sid}.pdf") for sid in sorted(sample_ids)],
                    str(out / "paper/attachment4_explanation_cards.pdf")], check=True)
    print(f"Saved {len(list(figs.glob('*.pdf')))} PDF/PNG figure pairs", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Q3_OUTPUT_DIR)
    run(p.parse_args().output)
