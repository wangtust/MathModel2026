"""Final publication-facing Q3 renderer.

This module keeps the scientific definitions from ``main_text.py`` but tightens
layout for manuscript insertion: no negative-y footers, smaller canvases,
inside-canvas legends, a 2+1 representative-case design, and a compact 2x2
localization audit figure.
"""
from __future__ import annotations

import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from common.paths import Q3_OUTPUT_DIR
from common.plotting import (
    ACCENT, DANGER, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT,
    MODALITY_COLORS, MUTED, OCHRE, apply_paper_style, panel_title,
    save_figure, style_axis,
)
from question3.visualization.main_text import (
    MODS, MOD_LABELS, UNCLEAR, _decode_waveform, _localization_state,
    _normalized_track, _resolve_recorded_path, _table, write_paper_tables,
)


OUT = Q3_OUTPUT_DIR / "paper_figures"


def _box(ax, x, y, w, h, text, color, *, fill="white", dashed=False, fontsize=6.8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fill, edgecolor=color, linewidth=.85,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight="medium")


def _arrow(ax, a, b, *, color=EDGE, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=8, linewidth=.75,
        color=color, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2,
    ))


def framework_figure() -> None:
    """Q3-1: compact four-stage explanation and audit pipeline."""
    apply_paper_style(chinese=True)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 3.82))
    fig.subplots_adjust(left=.025, right=.985, top=.95, bottom=.055)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    # Four visually strict stages: input -> frozen predictor -> explanation -> audit/card.
    _box(ax, .25, 2.25, 1.55, 1.18, "T / A / V\nfeatures", ACCENT,
         fill="#F3F8F7", fontsize=7.2)
    _box(ax, 2.30, 2.10, 1.85, 1.48, "Frozen Q2\nMRRF predictor", ACCENT,
         fill="#F3F8F7", fontsize=7.4)
    _arrow(ax, (1.84, 2.84), (2.25, 2.84), color=ACCENT)

    _box(ax, 4.72, 3.30, 1.65, .92, "Polarity +\nstrength", MODALITY_COLORS["text"])
    _box(ax, 4.72, 1.72, 1.65, .92, "Predicted-class\nlogit margin", OCHRE,
         fill="#FCFAF4")
    _arrow(ax, (4.18, 2.95), (4.67, 3.70), color=MODALITY_COLORS["text"])
    _arrow(ax, (4.18, 2.70), (4.67, 2.18), color=OCHRE)

    _box(ax, 7.00, 3.48, 1.65, .92, "Integrated\nGradients", OCHRE, fill="#FCFAF4")
    _box(ax, 9.00, 3.48, 1.55, .92, "Top 20%\nwindow W*", OCHRE, fill="#FCFAF4")
    _arrow(ax, (6.42, 2.22), (6.96, 3.83), color=OCHRE)
    _arrow(ax, (8.70, 3.94), (8.96, 3.94), color=OCHRE)

    _box(ax, 7.00, 1.64, 1.65, .92, "Independent\ndeletion", DANGER, fill="#FCF6F5")
    _box(ax, 9.00, 1.64, 1.55, .92, "Signed Δ\nmargin", DANGER, fill="#FCF6F5")
    _arrow(ax, (9.76, 3.43), (8.50, 2.60), color=DANGER)
    _arrow(ax, (8.70, 2.10), (8.96, 2.10), color=DANGER)

    _box(ax, 10.95, 3.48, 1.55, .92, "Token → word\nsource clock", MODALITY_COLORS["audio"])
    _arrow(ax, (10.60, 3.94), (10.91, 3.94), color=MODALITY_COLORS["audio"])
    _box(ax, 10.95, 1.64, 1.55, .92, "Audit clock\naccept / reject", DANGER,
         fill="#FCF6F5")
    _arrow(ax, (11.72, 3.43), (11.72, 2.60), color=DANGER)

    _box(ax, 7.00, .22, 5.50, .86,
         "Explanation card: prediction · local evidence · signed deletion response · audited source localization",
         ACCENT, fill="#F3F8F7", fontsize=6.7)
    _arrow(ax, (9.78, 1.58), (9.78, 1.12), color=ACCENT)
    _arrow(ax, (11.72, 1.58), (11.72, 1.12), color=ACCENT)
    _arrow(ax, (5.55, 3.26), (7.40, 1.10), color=ACCENT, dashed=True)

    ax.text(.25, 5.50, "Prediction → Where? → How much? → Where in source media?",
            fontsize=9.0, fontweight="semibold", color=INK)
    ax.text(.25, 5.08,
            "IG locates evidence; deletion measures perturbation response; alignment audit decides whether seconds/frames are reportable.",
            fontsize=6.2, color=MUTED)
    save_figure(fig, OUT, "fig01_q3_explainability_framework")


def fidelity_figure() -> None:
    """Q3-2: overall fidelity and audio/vision zoom without outside-canvas prose."""
    df = _table("validation_explanation_metrics.csv")
    rows = df[(df.stratum == "all") & (df.comparison == "top-minus-random")].copy()
    rows = rows.set_index("modality").loc[list(MODS)].reset_index()

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.72),
                             gridspec_kw={"width_ratios": [1.18, 1]})
    fig.subplots_adjust(left=.10, right=.985, top=.90, bottom=.21, wspace=.34)

    def forest(ax, data: pd.DataFrame, letter: str, title: str, xlim=None):
        yy = np.arange(len(data))[::-1]
        for y, (_, r) in zip(yy, data.iterrows()):
            value = float(r.mean_delta_margin_difference)
            low = float(r.ci95_low)
            high = float(r.ci95_high)
            mod = str(r.modality)
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o",
                        capsize=2.3, markersize=4.6, linewidth=1.15,
                        color=MODALITY_COLORS[mod])
            ax.text(value, y + .18, f"{value:+.3f}", ha="center", va="bottom",
                    fontsize=5.5, color=MUTED)
        ax.axvline(0, color=EDGE, lw=.8, linestyle="--")
        ax.set_yticks(yy, [
            f"{MOD_LABELS[m]}  n={int(data[data.modality == m].n.iloc[0])}"
            for m in data.modality
        ])
        ax.set_xlabel("Top − matched-random margin drop")
        panel_title(ax, letter, title, pad=4)
        style_axis(ax, grid_axis="x")
        if xlim is not None:
            ax.set_xlim(*xlim)

    forest(axes[0], rows, "a", "全验证集解释忠实度")
    av = rows[rows.modality.isin(["audio", "vision"])].copy()
    lo = min(-.008, float(av.ci95_low.min()) - .004)
    hi = max(.025, float(av.ci95_high.max()) + .006)
    forest(axes[1], av, "b", "Audio / Vision 放大", (lo, hi))
    axes[1].axvspan(lo, 0, color=DANGER, alpha=.045, zorder=-1)
    axes[1].axvspan(0, hi, color=ACCENT, alpha=.035, zorder=-1)
    save_figure(fig, OUT, "fig02_q3_fidelity")


def modality_effects_figure() -> None:
    """Q3-3: signed deletion effects + perturbation-derived dominant modality."""
    q = _table("attachment4_predictions.csv", dtype={"sample_id": str})
    q["sample_id"] = q.sample_id.str.zfill(2)
    q = q.sort_values("sample_id").reset_index(drop=True)
    delta = q[[f"top_delta_margin_{m}" for m in MODS]].to_numpy(float)
    bound = max(.10, float(np.nanpercentile(np.abs(delta), 95)))
    dom_codes = {"text": 0, "audio": 1, "vision": 2, "unclear": 3}
    dom_values = [dom_codes.get(v, 3) for v in q.dominant_modality.astype(str).str.lower()]

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 4.72),
                             gridspec_kw={"width_ratios": [3.55, .70]}, sharey=True)
    fig.subplots_adjust(left=.09, right=.96, top=.93, bottom=.10, wspace=.20)

    ax = axes[0]
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                   interpolation="nearest")
    ax.set_xticks(range(3), [MOD_LABELS[m] for m in MODS])
    ax.set_yticks(range(len(q)), q.sample_id)
    ax.set_ylabel("Attachment 4 sample")
    panel_title(ax, "a", "关键窗口有符号删除响应", pad=4)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            value = delta[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if abs(value) > .64*bound else INK)
    fig.colorbar(im, ax=ax, shrink=.84, pad=.018, label="Δ margin")

    ax = axes[1]
    cmap = ListedColormap([
        MODALITY_COLORS["text"], MODALITY_COLORS["audio"],
        MODALITY_COLORS["vision"], UNCLEAR,
    ])
    norm = BoundaryNorm((-.5, .5, 1.5, 2.5, 3.5), cmap.N)
    ax.imshow(np.asarray(dom_values)[:, None], cmap=cmap, norm=norm,
              aspect="auto", interpolation="nearest")
    ax.set_xticks([0], ["Main"])
    ax.tick_params(axis="y", left=False, labelleft=False)
    panel_title(ax, "b", "主要参考模态", pad=4)
    for i, value in enumerate(q.dominant_modality.astype(str).str.lower()):
        code = {"text": "T", "audio": "A", "vision": "V", "unclear": "?"}.get(value, "?")
        ax.text(0, i, code, ha="center", va="center", fontsize=6.0,
                color="white" if code != "?" else INK, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    save_figure(fig, OUT, "fig03_q3_modality_effects")


def _case_context(q: pd.DataFrame, ev: pd.DataFrame, audit: pd.DataFrame, sid: str):
    row = q[q.sample_id == sid].iloc[0]
    erows = ev[ev.sample_id == sid]
    ar = audit[audit.sample_id == sid].iloc[0]
    return row, erows, ar


def _source_summary(row: pd.Series, erows: pd.DataFrame, ar: pd.Series) -> tuple[str, str]:
    dominant = str(row.dominant_modality)
    selected_mod = dominant if dominant in MODS else "text"
    selected = erows[erows.modality == selected_mod].iloc[0]
    excerpt = str(selected.source_excerpt) if pd.notna(selected.source_excerpt) else ""
    if str(ar.alignment_status) == "trusted" and pd.notna(selected.start_sec):
        clock = f"trusted {float(selected.start_sec):.2f}–{float(selected.end_sec):.2f}s"
    elif str(ar.alignment_status) != "trusted":
        clock = "clock rejected; token/text evidence only"
    else:
        clock = f"{selected.localization_status}; no trusted seconds"
    return excerpt, clock


def _draw_tracks(ax, pos: pd.DataFrame, sid: str, erows: pd.DataFrame, *, title=True) -> None:
    offsets = np.array([2., 1., 0.])
    for off, mod in zip(offsets, MODS):
        x, y, _ = _normalized_track(pos, sid, mod)
        if len(x):
            ax.plot(x, y*.30 + off, lw=1.0, color=MODALITY_COLORS[mod])
            er = erows[erows.modality == mod].iloc[0]
            if bool(er.available):
                ax.axvspan(float(er.token_start), float(er.token_end),
                           ymin=max(0, (off-.40)/2.5), ymax=min(1, (off+.40)/2.5),
                           color=MODALITY_COLORS[mod], alpha=.09)
        else:
            ax.text(.5, off, "native unavailable", fontsize=5.2, color=MUTED,
                    transform=ax.get_yaxis_transform())
    ax.set_yticks(offsets, ["Text", "Audio", "Vision"])
    ax.set_ylim(-.50, 2.50)
    ax.set_xlabel("BERT content position", fontsize=5.9)
    if title:
        ax.set_title("Where?  normalized local IG", loc="left", fontsize=6.5)
    ax.grid(axis="x", color=GRID, lw=.42)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=5.7)


def _draw_full_case(fig, spec, q, ev, pos, audit, sid: str, letter: str, title: str) -> None:
    sub = spec.subgridspec(3, 1, height_ratios=[.92, 1.45, .92], hspace=.18)
    row, erows, ar = _case_context(q, ev, audit, sid)
    excerpt, clock = _source_summary(row, erows, ar)

    info = fig.add_subplot(sub[0])
    info.axis("off")
    strength = (f"{row.sentiment_strength:+.4f}"
                if 0 < abs(float(row.sentiment_strength)) < .005
                else f"{row.sentiment_strength:+.2f}")
    info.text(0, .95, f"({letter}) Sample {sid} · {title}", va="top",
              fontsize=7.8, fontweight="semibold")
    info.text(0, .58, f"{row.annotation} · strength {strength} · dominant {row.dominant_modality}",
              va="top", fontsize=6.2)
    evidence = textwrap.shorten(excerpt, width=72, placeholder="…")
    info.text(0, .28, evidence, va="top", fontsize=5.5, color=MUTED)
    info.text(0, .02, clock, va="bottom", fontsize=5.5,
              color=DANGER if "reject" in clock or "no trusted" in clock else ACCENT)

    tracks = fig.add_subplot(sub[1])
    _draw_tracks(tracks, pos, sid, erows)

    bars = fig.add_subplot(sub[2])
    effects = np.array([float(row[f"top_delta_margin_{m}"]) for m in MODS])
    yy = np.arange(3)
    bars.barh(yy, effects, color=[MODALITY_COLORS[m] for m in MODS], height=.56)
    bars.axvline(0, color=EDGE, lw=.7)
    bars.set_yticks(yy, ["T", "A", "V"])
    bars.invert_yaxis()
    bars.set_xlabel("signed deletion Δ margin", fontsize=5.9)
    bars.grid(axis="x", color=GRID, lw=.42)
    bars.set_axisbelow(True)
    for y, value in zip(yy, effects):
        bars.text(value + (.010 if value >= 0 else -.010), y, f"{value:+.2f}",
                  va="center", ha="left" if value >= 0 else "right", fontsize=5.3)


def _draw_rejected_strip(fig, spec, q, ev, pos, audit, sid: str = "15") -> None:
    row, erows, ar = _case_context(q, ev, audit, sid)
    sub = spec.subgridspec(1, 3, width_ratios=[1.05, 2.2, 1.30], wspace=.24)

    info = fig.add_subplot(sub[0])
    info.axis("off")
    info.text(0, .92, f"(c) Sample {sid}", fontsize=7.7, fontweight="semibold", va="top")
    info.text(0, .62, "对齐拒绝案例", fontsize=6.5, color=MUTED, va="top")
    info.text(0, .36, f"prediction: {row.annotation}", fontsize=5.8, va="top")
    info.text(0, .10, "保留 token/IG；不报告秒级定位", fontsize=5.6,
              color=DANGER, va="bottom")

    tracks = fig.add_subplot(sub[1])
    _draw_tracks(tracks, pos, sid, erows, title=False)
    tracks.set_title("模型空间证据仍保留", loc="left", fontsize=6.4)

    audit_ax = fig.add_subplot(sub[2])
    audit_ax.axis("off")
    audit_ax.text(0, .92, "Alignment audit", fontsize=6.5, fontweight="semibold", va="top")
    audit_ax.text(0, .65, f"status: {ar.alignment_status}", fontsize=5.8, color=DANGER, va="top")
    audit_ax.text(0, .45, f"coverage: {float(ar.word_coverage):.2f}", fontsize=5.7, va="top")
    audit_ax.text(0, .28, f"ASR agreement: {float(ar.asr_similarity):.2f}", fontsize=5.7, va="top")
    audit_ax.text(0, .04, "Decision: reject source clock", fontsize=5.8,
                  color=DANGER, fontweight="semibold", va="bottom")


def representative_cases_figure() -> None:
    """Q3-4: 2+1 layout for two full cases plus one rejection strip."""
    q = _table("attachment4_predictions.csv", dtype={"sample_id": str})
    q.sample_id = q.sample_id.str.zfill(2)
    ev = _table("attachment4_evidence.csv", dtype={"sample_id": str})
    ev.sample_id = ev.sample_id.str.zfill(2)
    pos = _table("attachment4_position_ig.csv", dtype={"sample_id": str})
    pos.sample_id = pos.sample_id.str.zfill(2)
    audit = _table("attachment4_alignment_audit.csv", dtype={"sample_id": str})
    audit.sample_id = audit.sample_id.str.zfill(2)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.05))
    outer = fig.add_gridspec(
        2, 2, left=.07, right=.985, top=.96, bottom=.08,
        height_ratios=[2.55, 1.00], hspace=.30, wspace=.18,
    )
    _draw_full_case(fig, outer[0, 0], q, ev, pos, audit, "01", "a", "典型 Text 主导")
    _draw_full_case(fig, outer[0, 1], q, ev, pos, audit, "19", "b", "Vision 主导且可定位")
    _draw_rejected_strip(fig, outer[1, :], q, ev, pos, audit, "15")
    save_figure(fig, OUT, "fig04_q3_representative_cases")


def localization_figure(sid: str = "19") -> None:
    """Q3-5: compact 2x2 alignment audit and source localization."""
    audit = _table("attachment4_alignment_audit.csv", dtype={"sample_id": str})
    audit.sample_id = audit.sample_id.str.zfill(2)
    ev = _table("attachment4_evidence.csv", dtype={"sample_id": str})
    ev.sample_id = ev.sample_id.str.zfill(2)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.00))
    gs = fig.add_gridspec(
        2, 2, left=.08, right=.985, top=.95, bottom=.10,
        height_ratios=[1.0, 1.03], width_ratios=[1.08, .92],
        hspace=.34, wspace=.27,
    )

    ax = fig.add_subplot(gs[0, 0])
    trusted = audit.alignment_status.astype(str) == "trusted"
    ax.scatter(audit.loc[trusted, "word_coverage"], audit.loc[trusted, "asr_similarity"],
               s=25, facecolor=ACCENT, edgecolor="white", linewidth=.4, alpha=.85,
               label="trusted")
    ax.scatter(audit.loc[~trusted, "word_coverage"], audit.loc[~trusted, "asr_similarity"],
               s=39, marker="X", color=DANGER, label="rejected")
    ax.axvline(.85, color=EDGE, lw=.8, linestyle="--")
    ax.axhline(.35, color=EDGE, lw=.8, linestyle="--")
    for _, r in audit.iterrows():
        if str(r.sample_id) in {"15", "07", "18"} or str(r.alignment_status) != "trusted":
            ax.text(float(r.word_coverage)+.010, float(r.asr_similarity)+.014,
                    str(r.sample_id), fontsize=5.8)
    ax.set_xlim(.78, 1.02)
    ax.set_ylim(.12, 1.04)
    ax.set_xlabel("WhisperX word coverage")
    ax.set_ylabel("Independent ASR agreement")
    panel_title(ax, "a", "样本级时间定位审核", pad=4)
    style_axis(ax, grid_axis="both")
    ax.legend(loc="lower right", frameon=False, fontsize=5.8)

    ax = fig.add_subplot(gs[0, 1])
    ids = sorted(ev.sample_id.unique())
    matrix = np.full((len(ids), 3), 1, dtype=int)
    for i, sample_id in enumerate(ids):
        rows = ev[ev.sample_id == sample_id]
        for j, mod in enumerate(MODS):
            matrix[i, j] = _localization_state(rows[rows.modality == mod].iloc[0])
    cmap = ListedColormap((ACCENT, OCHRE, DANGER, "#B8BEC3"))
    norm = BoundaryNorm((-.5, .5, 1.5, 2.5, 3.5), cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(3), [MOD_LABELS[m] for m in MODS])
    ax.set_yticks(range(len(ids)), ids)
    panel_title(ax, "b", "关键窗口原素材定位状态", pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [
        Patch(facecolor=ACCENT, label="Trusted time"),
        Patch(facecolor=OCHRE, label="Token/text only"),
        Patch(facecolor=DANGER, label="Rejected"),
        Patch(facecolor="#B8BEC3", label="Unavailable"),
    ]
    ax.legend(handles=legend, loc="upper right", ncol=2, frameon=False, fontsize=5.2)

    rows = ev[ev.sample_id == sid]
    ar = audit[audit.sample_id == sid].iloc[0]
    ax = fig.add_subplot(gs[1, 0])
    video = _resolve_recorded_path(rows.video_path.iloc[0])
    ts, levels = _decode_waveform(video)
    duration = float(ar.duration_sec)
    if ts is not None:
        peak = max(float(np.nanmax(levels)), 1e-6)
        ax.plot(ts, levels/peak*.34 + 3.0, color="#75838D", lw=.55)
    else:
        ax.hlines(3.0, 0, duration, color="#75838D", lw=.7)
    ax.text(-duration*.02, 3.0, "Wave", ha="right", va="center", fontsize=5.8)
    for y, mod in zip((2.0, 1.0, 0.0), MODS):
        r = rows[rows.modality == mod].iloc[0]
        ax.hlines(y, 0, duration, color=GRID, lw=1.1)
        if pd.notna(r.start_sec):
            ax.plot([float(r.start_sec), float(r.end_sec)], [y, y],
                    color=MODALITY_COLORS[mod], lw=6.5, solid_capstyle="butt")
            ax.text((float(r.start_sec)+float(r.end_sec))/2, y+.17,
                    f"{float(r.start_sec):.1f}–{float(r.end_sec):.1f}s",
                    ha="center", fontsize=5.2, color=MODALITY_COLORS[mod])
        ax.text(-duration*.02, y, MOD_LABELS[mod], ha="right", va="center",
                fontsize=5.8, color=MODALITY_COLORS[mod])
    ax.set_xlim(0, duration)
    ax.set_ylim(-.50, 3.50)
    ax.set_yticks([])
    ax.set_xlabel("Clip-relative time (s)")
    panel_title(ax, "c", f"Sample {sid}：审核后的 source-time localization", pad=4)
    style_axis(ax, grid_axis="x")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    vision = rows[rows.modality == "vision"].iloc[0]
    frame = _resolve_recorded_path(vision.keyframe_path)
    if frame is not None and frame.exists():
        image = plt.imread(frame)
        ax.imshow(image, aspect="auto")
        ax.set_box_aspect(.62)
        ax.set_title(f"(d)  Sample {sid} 原视频帧 · {float(vision.keyframe_sec):.2f}s",
                     loc="left", fontsize=7.0, fontweight="semibold")
    else:
        ax.add_patch(Rectangle((0, .18), 1, .72, transform=ax.transAxes,
                               facecolor=LIGHT, edgecolor=EDGE, linewidth=.7))
        ax.text(.5, .54, "Key frame unavailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=6.8, color=MUTED)
        ax.set_title(f"(d)  Sample {sid} 原素材核验", loc="left",
                     fontsize=7.0, fontweight="semibold")
    save_figure(fig, OUT, "fig05_q3_localization_and_audit")


def write_manifest() -> None:
    manifest = {
        "version": "v2_final_publication_layout",
        "main_text_figures": {
            "fig01_q3_explainability_framework": "compact prediction -> IG -> deletion/source-audit framework",
            "fig02_q3_fidelity": "overall fidelity plus audio/vision zoom without outside-canvas footer",
            "fig03_q3_modality_effects": "signed deletion response and perturbation-derived dominant modality",
            "fig04_q3_representative_cases": "2+1 layout: Samples 01 and 19 full cases plus Sample 15 rejection strip",
            "fig05_q3_localization_and_audit": "compact 2x2 alignment audit, localization states, source-time tracks and key frame",
        },
        "main_text_table": "table01_q3_frozen_model_performance",
        "interpretation_rules": [
            "Integrated Gradients answers where local evidence lies within each modality.",
            "Signed deletion response is used for cross-modality perturbation effect size.",
            "Source seconds/frames are emitted only after alignment audit; failed audits keep token/text evidence only.",
            "Long methodological explanations belong in the manuscript caption rather than below the exported canvas.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    framework_figure()
    fidelity_figure()
    modality_effects_figure()
    representative_cases_figure()
    localization_figure("19")
    write_paper_tables()
    write_manifest()
    print(f"Wrote final Q3 publication figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
