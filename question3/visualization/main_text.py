"""Build the compact journal-style Q3 main-text figure set from saved outputs.

This module deliberately separates four questions that are easy to conflate:

- IG: where is local evidence located?
- signed deletion response: how much does a selected modality window affect the prediction?
- audited alignment: where can that model-space window be mapped in the source media?
- prediction: what did the frozen Q2 model output?

No training, model selection or attribution recomputation is performed here.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from common.paths import Q3_OUTPUT_DIR
from common.plotting import (
    ACCENT, BG, DANGER, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT,
    MODALITY_COLORS, MUTED, OCHRE, apply_paper_style, panel_title,
    save_figure, style_axis,
)


OUT = Q3_OUTPUT_DIR / "paper_figures"
MODS = ("text", "audio", "vision")
MOD_LABELS = {"text": "Text", "audio": "Audio", "vision": "Vision"}
UNCLEAR = "#9AA3AA"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Q3 result is missing: {path}\n"
            "Run the Q3 pipeline/legacy figures first; paper-figures only reorganizes saved results."
        )
    return path


def _table(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(_require(Q3_OUTPUT_DIR / "tables" / name), **kwargs)


def _resolve_recorded_path(value: object) -> Path | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    # Old runs can record an absolute workspace path. Rebase generated Q3 assets
    # by basename so figures remain portable after the repository is moved.
    for candidate in (
        Q3_OUTPUT_DIR / "frames" / path.name,
        Q3_OUTPUT_DIR / path.name,
    ):
        if candidate.exists():
            return candidate
    return path


def _box(ax, x, y, w, h, text, color, *, dashed=False, fill="white", fontsize=7.1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=.9, edgecolor=color, facecolor=fill,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight="medium")
    return patch


def _arrow(ax, a, b, *, color=EDGE, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=8.5, linewidth=.8,
        color=color, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2,
    ))


def framework_figure() -> None:
    """Q3-1: explainability generation, validation and source-audit pipeline."""
    apply_paper_style(chinese=True)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 4.55))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 7.6); ax.axis("off")

    # Frozen prediction path.
    _box(ax, .25, 4.95, 1.45, 1.15, "T / A / V\nfeatures", ACCENT,
         fill="#F3F8F7")
    _box(ax, 2.20, 4.78, 1.80, 1.48, "Frozen Q2\nMRRF predictor", ACCENT,
         fill="#F3F8F7", fontsize=7.5)
    _arrow(ax, (1.72, 5.52), (2.16, 5.52), color=ACCENT)

    _box(ax, 4.63, 5.48, 1.55, .82, "Polarity +\nstrength", MODALITY_COLORS["text"])
    _box(ax, 4.63, 4.35, 1.55, .82, "Predicted-class\nlogit margin", OCHRE,
         fill="#FCFAF4")
    _arrow(ax, (4.04, 5.55), (4.58, 5.88), color=MODALITY_COLORS["text"])
    _arrow(ax, (4.04, 5.40), (4.58, 4.76), color=OCHRE)

    # Attribution path.
    _box(ax, 6.85, 4.23, 1.70, 1.05, "Integrated\nGradients", OCHRE,
         fill="#FCFAF4")
    _arrow(ax, (6.20, 4.76), (6.80, 4.76), color=OCHRE)
    _box(ax, 9.12, 4.23, 1.68, 1.05, "Position scores\nper modality", OCHRE,
         fill="#FCFAF4")
    _arrow(ax, (8.58, 4.76), (9.07, 4.76), color=OCHRE)
    _box(ax, 11.35, 4.23, 1.36, 1.05, "Top 20%\nwindow W*", OCHRE,
         fill="#FCFAF4")
    _arrow(ax, (10.84, 4.76), (11.30, 4.76), color=OCHRE)

    # Independent perturbation branch.
    _box(ax, 7.05, 2.30, 1.66, 1.05, "Independent\ndeletion", DANGER,
         fill="#FCF6F5")
    _arrow(ax, (12.02, 4.18), (8.67, 3.38), color=DANGER)
    _box(ax, 9.20, 2.30, 1.73, 1.05, "Δ logit margin\nvs random / low", DANGER,
         fill="#FCF6F5")
    _arrow(ax, (8.75, 2.82), (9.15, 2.82), color=DANGER)
    _box(ax, 11.45, 2.30, 1.22, 1.05, "Fidelity\ncheck", DANGER,
         fill="#FCF6F5")
    _arrow(ax, (10.97, 2.82), (11.40, 2.82), color=DANGER)

    # Source mapping branch.
    _box(ax, 2.55, 2.28, 1.55, 1.08, "Token → word\nmapping", MODALITY_COLORS["audio"])
    _arrow(ax, (11.72, 4.18), (4.05, 3.40), color=MODALITY_COLORS["audio"])
    _box(ax, 4.62, 2.28, 1.66, 1.08, "WhisperX clock\n+ ASR audit", MODALITY_COLORS["audio"])
    _arrow(ax, (4.14, 2.82), (4.57, 2.82), color=MODALITY_COLORS["audio"])
    _box(ax, 2.58, .62, 1.55, 1.02, "Trusted source\ntime / frame", MODALITY_COLORS["vision"])
    _box(ax, 4.66, .62, 1.58, 1.02, "Reject clock if\naudit fails", DANGER,
         fill="#FCF6F5")
    _arrow(ax, (5.10, 2.24), (3.78, 1.69), color=MODALITY_COLORS["vision"])
    _arrow(ax, (5.74, 2.24), (5.48, 1.69), color=DANGER)

    # Final explanation card.
    _box(ax, 7.38, .48, 5.05, 1.18,
         "Explanation card\nPrediction  ·  dominant reference modality  ·  deletion response\nlocal evidence  ·  audited source localization  ·  quality flags",
         ACCENT, fill="#F3F8F7", fontsize=7.2)
    _arrow(ax, (4.17, 1.13), (7.32, 1.13), color=ACCENT)
    _arrow(ax, (6.28, 1.13), (7.32, 1.13), color=ACCENT)
    _arrow(ax, (12.06, 2.25), (11.35, 1.70), color=ACCENT)
    _arrow(ax, (5.42, 5.43), (8.50, 1.72), color=ACCENT, dashed=True)

    ax.text(.25, 7.20,
            "预测是什么 → 证据在哪里 → 删除后影响多大 → 能否回查原始素材",
            fontsize=9.1, fontweight="semibold", color=INK)
    ax.text(.25, 6.82,
            "IG 负责 Where；有符号删除响应负责 How much；对齐审核负责 Where in source media。",
            fontsize=6.6, color=MUTED)
    ax.text(.25, .10,
            "当词时钟审核不通过时保留 token/文本证据与模型响应，但不输出伪精确秒数。",
            fontsize=6.4, color=MUTED)
    save_figure(fig, OUT, "fig01_q3_explainability_framework")


def fidelity_figure() -> None:
    """Q3-2: main fidelity estimate plus a non-text zoom."""
    df = _table("validation_explanation_metrics.csv")
    rows = df[(df.stratum == "all") & (df.comparison == "top-minus-random")].copy()
    rows = rows.set_index("modality").loc[list(MODS)].reset_index()

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.10),
                             gridspec_kw={"width_ratios": [1.15, 1]},
                             constrained_layout=True)

    def forest(ax, data: pd.DataFrame, letter: str, title: str, xlim=None):
        yy = np.arange(len(data))[::-1]
        for y, (_, r) in zip(yy, data.iterrows()):
            value = float(r.mean_delta_margin_difference)
            low = float(r.ci95_low); high = float(r.ci95_high)
            mod = str(r.modality)
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o",
                        capsize=2.5, markersize=5.0, linewidth=1.25,
                        color=MODALITY_COLORS[mod])
            ax.text(value, y + .20,
                    f"{value:+.3f}  [{low:+.3f}, {high:+.3f}]",
                    ha="center", va="bottom", fontsize=5.8, color=MUTED)
        ax.axvline(0, color=EDGE, lw=.8, linestyle="--")
        ax.set_yticks(yy, [f"{MOD_LABELS[m]}  n={int(data[data.modality==m].n.iloc[0])}"
                           for m in data.modality])
        ax.set_xlabel("Top-window − matched-random margin drop")
        panel_title(ax, letter, title)
        style_axis(ax, grid_axis="x")
        if xlim is not None:
            ax.set_xlim(*xlim)

    forest(axes[0], rows, "a", "全验证集解释忠实度")
    av = rows[rows.modality.isin(["audio", "vision"])].copy()
    lo = min(-.008, float(av.ci95_low.min()) - .004)
    hi = max(.025, float(av.ci95_high.max()) + .006)
    forest(axes[1], av, "b", "非文本模态放大视图", (lo, hi))
    axes[1].axvspan(lo, 0, color=DANGER, alpha=.045, zorder=-1)
    axes[1].axvspan(0, hi, color=ACCENT, alpha=.035, zorder=-1)

    fig.text(.5, -.035,
             "窗口长度及原生有效位置数匹配；误差线为 2000 次样本 bootstrap 95% CI。正值表示 IG 选出的窗口比随机窗口造成更大的预测 margin 下降。",
             ha="center", fontsize=6.0, color=MUTED)
    save_figure(fig, OUT, "fig02_q3_fidelity")


def modality_effects_figure() -> None:
    """Q3-3: signed deletion response + perturbation-derived dominant modality."""
    q = _table("attachment4_predictions.csv", dtype={"sample_id": str})
    q["sample_id"] = q.sample_id.str.zfill(2)
    q = q.sort_values("sample_id").reset_index(drop=True)
    delta = q[[f"top_delta_margin_{m}" for m in MODS]].to_numpy(float)
    bound = max(.10, float(np.nanpercentile(np.abs(delta), 95)))

    dom_values = []
    dom_codes = {"text": 0, "audio": 1, "vision": 2, "unclear": 3}
    for value in q.dominant_modality.astype(str).str.lower():
        dom_values.append(dom_codes.get(value, 3))

    apply_paper_style(chinese=True)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 5.25),
                             gridspec_kw={"width_ratios": [3.4, .85]},
                             sharey=True, constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                   interpolation="nearest")
    ax.set_xticks(range(3), [MOD_LABELS[m] for m in MODS])
    ax.set_yticks(range(len(q)), q.sample_id)
    ax.set_ylabel("Attachment 4 sample")
    panel_title(ax, "a", "有符号关键窗口删除响应")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            value = delta[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(value) > .64*bound else INK)
    fig.colorbar(im, ax=ax, shrink=.78, pad=.025,
                 label="Δ margin after selected-window deletion")

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
    panel_title(ax, "b", "主要参考模态")
    for i, value in enumerate(q.dominant_modality.astype(str)):
        ax.text(0, i, value.replace("vision", "V").replace("audio", "A")
                .replace("text", "T").replace("unclear", "?"),
                ha="center", va="center", fontsize=6.1,
                color="white" if value.lower() != "unclear" else INK,
                fontweight="semibold")
    legend = [
        Patch(facecolor=MODALITY_COLORS["text"], label="Text"),
        Patch(facecolor=MODALITY_COLORS["audio"], label="Audio"),
        Patch(facecolor=MODALITY_COLORS["vision"], label="Vision"),
        Patch(facecolor=UNCLEAR, label="Unclear"),
    ]
    fig.legend(handles=legend, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, -.03), frameon=False)
    fig.text(.5, -.068,
             "主要参考模态由独立有符号删除响应确定，不由 internal gate 最大值替代；负响应表示删除该窗口反而增强原预测。",
             ha="center", fontsize=6.0, color=MUTED)
    save_figure(fig, OUT, "fig03_q3_modality_effects")


def _normalized_track(pos: pd.DataFrame, sid: str, modality: str):
    sub = pos[(pos.sample_id == sid) & (pos.modality == modality) &
              (pos.native_available == True)].sort_values("token_index")
    if sub.empty:
        return np.array([]), np.array([]), 0.0
    x = sub.token_index.to_numpy(float)
    y = sub.ig_signed.to_numpy(float)
    scale = float(np.max(np.abs(y))) if len(y) else 0.0
    return x, (y / scale if scale > 0 else y), scale


def representative_cases_figure() -> None:
    """Q3-4: typical text, vision-dominant and alignment-rejection cases."""
    q = _table("attachment4_predictions.csv", dtype={"sample_id": str})
    q.sample_id = q.sample_id.str.zfill(2)
    ev = _table("attachment4_evidence.csv", dtype={"sample_id": str})
    ev.sample_id = ev.sample_id.str.zfill(2)
    pos = _table("attachment4_position_ig.csv", dtype={"sample_id": str})
    pos.sample_id = pos.sample_id.str.zfill(2)
    audit = _table("attachment4_alignment_audit.csv", dtype={"sample_id": str})
    audit.sample_id = audit.sample_id.str.zfill(2)

    cases = (
        ("01", "典型 Text 主导"),
        ("19", "Vision 主导且可定位"),
        ("15", "对齐拒绝：不输出秒数"),
    )
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 6.0), constrained_layout=True)
    outer = fig.add_gridspec(1, 3, wspace=.08)

    for c, (sid, case_title) in enumerate(cases):
        sub = outer[c].subgridspec(4, 1, height_ratios=[.82, 1.68, 1.00, 1.15], hspace=.12)
        row = q[q.sample_id == sid].iloc[0]
        erows = ev[ev.sample_id == sid]
        ar = audit[audit.sample_id == sid].iloc[0]

        info = fig.add_subplot(sub[0]); info.axis("off")
        strength = (f"{row.sentiment_strength:+.4f}" if 0 < abs(float(row.sentiment_strength)) < .005
                    else f"{row.sentiment_strength:+.2f}")
        info.text(0, .95, f"({chr(ord('a')+c)})  Sample {sid}", va="top",
                  fontsize=8.5, fontweight="semibold")
        info.text(0, .64, case_title, va="top", fontsize=6.7, color=MUTED)
        info.text(0, .37, f"{row.annotation} · strength {strength}", fontsize=6.8)
        info.text(0, .10, f"dominant: {row.dominant_modality}", fontsize=6.8,
                  color=(MODALITY_COLORS.get(str(row.dominant_modality), DANGER)
                         if str(row.dominant_modality) != "unclear" else UNCLEAR),
                  fontweight="semibold")

        tracks = fig.add_subplot(sub[1])
        offsets = np.array([2., 1., 0.])
        scales = []
        for off, mod in zip(offsets, MODS):
            x, y, scale = _normalized_track(pos, sid, mod)
            scales.append(scale)
            if len(x):
                tracks.plot(x, y*.32 + off, lw=1.0, color=MODALITY_COLORS[mod])
                er = erows[erows.modality == mod].iloc[0]
                if bool(er.available):
                    tracks.axvspan(float(er.token_start), float(er.token_end),
                                   ymin=max(0, (off-.42)/2.5),
                                   ymax=min(1, (off+.42)/2.5),
                                   color=MODALITY_COLORS[mod], alpha=.09)
            else:
                tracks.text(.5, off, "native unavailable", fontsize=5.5,
                            color=MUTED, transform=tracks.get_yaxis_transform())
        tracks.set_yticks(offsets, ["Text", "Audio", "Vision"])
        tracks.set_ylim(-.55, 2.55)
        tracks.set_xlabel("BERT content position", fontsize=6.2)
        tracks.set_title("Where?  normalized local IG tracks", loc="left", fontsize=6.8)
        tracks.grid(axis="x", color=GRID, lw=.45)
        tracks.spines[["top", "right", "left"]].set_visible(False)
        tracks.tick_params(axis="y", length=0)
        tracks.text(.99, .02,
                    "max|IG| " + " / ".join(f"{s:.3g}" for s in scales),
                    ha="right", va="bottom", transform=tracks.transAxes,
                    fontsize=5.2, color=MUTED)

        bars = fig.add_subplot(sub[2])
        effects = np.array([float(row[f"top_delta_margin_{m}"]) for m in MODS])
        yy = np.arange(3)
        bars.barh(yy, effects, color=[MODALITY_COLORS[m] for m in MODS], height=.56)
        bars.axvline(0, color=EDGE, lw=.7)
        bars.set_yticks(yy, ["T", "A", "V"]); bars.invert_yaxis()
        bars.set_title("How much?  signed deletion response", loc="left", fontsize=6.8)
        bars.set_xlabel("Δ margin", fontsize=6.2)
        bars.grid(axis="x", color=GRID, lw=.45); bars.set_axisbelow(True)
        for y, v in zip(yy, effects):
            bars.text(v + (.012 if v >= 0 else -.012), y, f"{v:+.2f}",
                      va="center", ha="left" if v >= 0 else "right", fontsize=5.5)

        note = fig.add_subplot(sub[3]); note.axis("off")
        dominant = str(row.dominant_modality)
        selected = erows[erows.modality == (dominant if dominant in MODS else "text")].iloc[0]
        excerpt = str(selected.source_excerpt) if pd.notna(selected.source_excerpt) else ""
        if str(ar.alignment_status) == "trusted" and pd.notna(selected.start_sec):
            clock = f"trusted {float(selected.start_sec):.2f}–{float(selected.end_sec):.2f}s"
        elif str(ar.alignment_status) != "trusted":
            clock = "clock rejected; token/text evidence only"
        else:
            clock = f"{selected.localization_status}; no trusted seconds"
        note.text(0, .92, "Source evidence / audit", fontweight="semibold", fontsize=6.8, va="top")
        note.text(0, .67, textwrap.fill(excerpt, width=31, placeholder="…"),
                  fontsize=5.9, va="top")
        note.text(0, .17, textwrap.fill(clock, width=34), fontsize=5.8,
                  color=DANGER if "reject" in clock or "no trusted" in clock else ACCENT,
                  va="top")

    fig.text(.5, -.018,
             "每条 IG 轨道仅在该模态内部按 max|IG| 归一化，用来回答“哪里重要”；模态作用大小只由下方有符号删除响应比较。",
             ha="center", fontsize=6.0, color=MUTED)
    save_figure(fig, OUT, "fig04_q3_representative_cases")


def _localization_state(row: pd.Series) -> int:
    if not bool(row.available):
        return 3  # native unavailable
    status = str(row.localization_status)
    if pd.notna(row.start_sec) and status.startswith("trusted"):
        return 0  # trusted time
    if str(row.alignment_status) != "trusted":
        return 2  # rejected sample-level clock
    return 1      # token/text only or local timing unavailable


def _decode_waveform(video: Path | None):
    if video is None or not video.exists() or shutil.which("ffmpeg") is None:
        return None, None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-f", "f32le",
             "-ac", "1", "-ar", "16000", "pipe:1"],
            capture_output=True, check=True,
        )
    except Exception:
        return None, None
    wav = np.frombuffer(proc.stdout, dtype="<f4")
    if len(wav) < 320:
        return None, None
    block = 320
    usable = wav[:len(wav)//block*block].reshape(-1, block)
    levels = np.max(np.abs(usable), axis=1)
    ts = (np.arange(len(levels)) + .5) * block / 16000
    return ts, levels


def localization_figure(sid: str = "19") -> None:
    """Q3-5: alignment audit, localization states and a source-time example."""
    audit = _table("attachment4_alignment_audit.csv", dtype={"sample_id": str})
    audit.sample_id = audit.sample_id.str.zfill(2)
    ev = _table("attachment4_evidence.csv", dtype={"sample_id": str})
    ev.sample_id = ev.sample_id.str.zfill(2)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.05, .95])

    # (a) Two-dimensional sample-level audit.
    ax = fig.add_subplot(gs[0, 0])
    trusted = audit.alignment_status.astype(str) == "trusted"
    ax.scatter(audit.loc[trusted, "word_coverage"], audit.loc[trusted, "asr_similarity"],
               s=27, facecolor=ACCENT, edgecolor="white", linewidth=.4, alpha=.85,
               label="trusted")
    ax.scatter(audit.loc[~trusted, "word_coverage"], audit.loc[~trusted, "asr_similarity"],
               s=42, marker="X", color=DANGER, label="rejected")
    ax.axvline(.85, color=EDGE, lw=.8, linestyle="--")
    ax.axhline(.35, color=EDGE, lw=.8, linestyle="--")
    for _, r in audit.iterrows():
        if str(r.sample_id) in {"15", "07", "18"} or str(r.alignment_status) != "trusted":
            ax.text(float(r.word_coverage)+.012, float(r.asr_similarity)+.015,
                    str(r.sample_id), fontsize=6.1)
    ax.set_xlim(.78, 1.02); ax.set_ylim(.12, 1.04)
    ax.set_xlabel("WhisperX word coverage")
    ax.set_ylabel("Independent ASR agreement")
    panel_title(ax, "a", "样本级时间定位审核")
    style_axis(ax, grid_axis="both")
    ax.legend(loc="lower right", frameon=False)

    # (b) Per-modality localization status matrix.
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
    panel_title(ax, "b", "关键窗口的原素材定位状态")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [
        Patch(facecolor=ACCENT, label="Trusted time"),
        Patch(facecolor=OCHRE, label="Token/text only"),
        Patch(facecolor=DANGER, label="Alignment rejected"),
        Patch(facecolor="#B8BEC3", label="Native unavailable"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.11),
              ncol=2, frameon=False, fontsize=5.8)

    # (c) Audited source-time tracks for a non-text-dominant example.
    rows = ev[ev.sample_id == sid]
    ar = audit[audit.sample_id == sid].iloc[0]
    ax = fig.add_subplot(gs[1, 0])
    video = _resolve_recorded_path(rows.video_path.iloc[0])
    ts, levels = _decode_waveform(video)
    duration = float(ar.duration_sec)
    if ts is not None:
        peak = max(float(np.nanmax(levels)), 1e-6)
        ax.plot(ts, levels/peak*.36 + 3.0, color="#75838D", lw=.55)
    else:
        ax.hlines(3.0, 0, duration, color="#75838D", lw=.7)
        ax.text(duration*.02, 3.24, "waveform unavailable in this runtime",
                fontsize=5.4, color=MUTED)
    ax.text(-duration*.02, 3.0, "Wave", ha="right", va="center", fontsize=6)
    for y, mod in zip((2.0, 1.0, 0.0), MODS):
        r = rows[rows.modality == mod].iloc[0]
        ax.hlines(y, 0, duration, color=GRID, lw=1.1)
        if pd.notna(r.start_sec):
            ax.plot([float(r.start_sec), float(r.end_sec)], [y, y],
                    color=MODALITY_COLORS[mod], lw=7, solid_capstyle="butt")
            ax.text((float(r.start_sec)+float(r.end_sec))/2, y+.18,
                    f"{float(r.start_sec):.1f}–{float(r.end_sec):.1f}s",
                    ha="center", fontsize=5.5, color=MODALITY_COLORS[mod])
        ax.text(-duration*.02, y, MOD_LABELS[mod], ha="right", va="center",
                fontsize=6, color=MODALITY_COLORS[mod])
    ax.set_xlim(0, duration); ax.set_ylim(-.55, 3.55)
    ax.set_yticks([]); ax.set_xlabel("Clip-relative time (s)")
    panel_title(ax, "c", f"Sample {sid}：审核后的 source-time localization")
    style_axis(ax, grid_axis="x")

    # (d) Key frame + audit wording.
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    vision = rows[rows.modality == "vision"].iloc[0]
    frame = _resolve_recorded_path(vision.keyframe_path)
    if frame is not None and frame.exists():
        ax.imshow(plt.imread(frame))
        ax.set_title(f"(d)  Sample {sid} 对应原视频帧 · {float(vision.keyframe_sec):.2f}s",
                     loc="left", fontsize=7.2, fontweight="semibold")
    else:
        ax.add_patch(Rectangle((0, .28), 1, .62, transform=ax.transAxes,
                               facecolor=LIGHT, edgecolor=EDGE, linewidth=.7))
        ax.text(.5, .59, "Key frame not available\nin this runtime", ha="center",
                va="center", transform=ax.transAxes, fontsize=7, color=MUTED)
        ax.set_title(f"(d)  Sample {sid} 原素材核验", loc="left",
                     fontsize=7.2, fontweight="semibold")
    ax.text(0, .16,
            "时间区间来自 token → word → forced-alignment 映射；\n视频帧仅用于核验时间对应关系，不表示帧内物体归因。",
            transform=ax.transAxes, fontsize=5.9, color=MUTED, va="top")

    fig.text(.5, -.025,
             "仅当样本级和局部词时钟审核通过时输出秒级定位；例如 Sample 15 的 ASR 不一致会触发拒绝定位，而不是填入伪精确时间。",
             ha="center", fontsize=6.0, color=MUTED)
    save_figure(fig, OUT, "fig05_q3_localization_and_audit")


def write_paper_tables() -> None:
    """Move frozen-model performance out of the main figure set into a compact table."""
    perf = _table("model_performance.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("split", "n", "accuracy", "macro_f1", "mae", "pearson") if c in perf]
    perf[keep].to_csv(OUT / "table01_q3_frozen_model_performance.csv",
                      index=False, encoding="utf-8-sig")
    lines = [
        "| Split | n | Accuracy ↑ | Macro-F1 ↑ | MAE ↓ | Pearson ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in perf.iterrows():
        lines.append(
            f"| {r['split']} | {int(r['n'])} | {r['accuracy']:.3f} | "
            f"{r['macro_f1']:.3f} | {r['mae']:.3f} | {r['pearson']:.3f} |"
        )
    (OUT / "table01_q3_frozen_model_performance.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "main_text_figures": {
            "fig01_q3_explainability_framework": "explanation generation, deletion validation, and source-audit framework",
            "fig02_q3_fidelity": "validation top-window versus matched-random fidelity with non-text zoom",
            "fig03_q3_modality_effects": "Attachment-4 signed deletion effects and perturbation-derived dominant modality",
            "fig04_q3_representative_cases": "Samples 01, 19, 15: typical, vision-dominant, and clock-rejection cases",
            "fig05_q3_localization_and_audit": "alignment audit, localization-state matrix, and Sample-19 source-time evidence",
        },
        "main_text_table": "table01_q3_frozen_model_performance",
        "supplementary_legacy_figures": [
            "fig01_performance",
            "fig02_fidelity",
            "fig03_modality_contributions",
            "fig04_local_importance",
            "fig05_alignment_quality",
            "fig06_timed_evidence",
            "fig07_model_sensitivity",
            "fig08_validation_variant_fidelity",
            "fig09_gate_response_rank",
            "case_01 ... case_20",
            "attachment4_explanation_cards.pdf",
        ],
        "interpretation_rules": [
            "Integrated Gradients identifies local positions (Where), not cross-modality effect size.",
            "Signed deletion response estimates perturbation-derived modality effect (How much).",
            "Internal gate weights are retained only as a model diagnostic and are not treated as causal contribution percentages.",
            "Source-time intervals are emitted only after alignment audit; failed audits retain token/text evidence without fabricated seconds.",
            "Video frames verify temporal correspondence and are not object-level visual attribution.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    framework_figure()
    fidelity_figure()
    modality_effects_figure()
    representative_cases_figure()
    localization_figure("19")
    write_paper_tables()
    write_manifest()
    print(f"Wrote compact Q3 main-text figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
