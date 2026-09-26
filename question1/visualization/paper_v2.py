"""Publication-facing Q1 figures in a compact evidence-card style.

The figures deliberately use the visual grammar of the supplied reference paper:
small statistical panels, long modality/status strips, source-frame evidence and
worked sample cards.  Experiment definitions and saved features are unchanged.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Polygon, Rectangle
import numpy as np
import pandas as pd

from common.plotting import (
    ACCENT, DANGER, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT, MODALITY_COLORS,
    MUTED, OCHRE, apply_paper_style, panel_title, save_figure,
)
from .. import config
from .journal_cases import (
    AUDIO, TEXT, VISION, _feature_activity, _open_case, _source_frame, _timeline,
)
from .cases import _decode_word_audio, _letterbox_frame, _read_frame_pts, _representative_frames


OUT = config.OUT / "paper_figures"


def _case_map() -> dict[str, str]:
    path = config.OUT / "tables/10_journal_case_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Q1 journal case manifest: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    result: dict[str, str] = {}
    for _, row in df.iterrows():
        finding = str(row.get("finding", ""))
        sid = str(row.get("sample_id", ""))
        if finding and sid:
            result[finding] = sid
    return result


def _crop_16_9(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    target = 16 / 9
    aspect = w / max(h, 1)
    if aspect > target:
        nw = int(round(h * target))
        x0 = max(0, (w - nw) // 2)
        return frame[:, x0:x0 + nw]
    nh = int(round(w / target))
    y0 = max(0, (h - nh) // 2)
    return frame[y0:y0 + nh, :]


def _mix(color: str, strength: float) -> tuple[float, float, float]:
    """Mix a semantic color with white; strength=0 gives almost white."""
    rgb = np.asarray(to_rgb(color), float)
    s = float(np.clip(strength, 0.0, 1.0))
    return tuple(1.0 - (1.0 - rgb) * (0.10 + 0.90 * s))


def _feature_rgb(values: np.ndarray) -> np.ndarray:
    colors = (TEXT, AUDIO, VISION)
    image = np.ones((3, values.shape[1], 3), float)
    for i, color in enumerate(colors):
        for j, value in enumerate(values[i]):
            image[i, j] = _mix(color, float(value))
    return image


def _sample_excerpt(words: list[str], limit: int = 13) -> str:
    shown = words[:limit]
    suffix = " …" if len(words) > limit else ""
    return " ".join(shown) + suffix


def framework_figure(case_id: str) -> None:
    """Q1-1: worked processing example instead of a generic box-only flowchart."""
    meta, z = _open_case(case_id)
    count = int(meta["word_count"])
    if count <= 0:
        raise RuntimeError(f"Q1 framework sample {case_id} has no official words")
    n = min(count, 12)
    words = list(meta["official_words"][:n])

    ws = z["word_start"][:n].astype(float)
    we = z["word_end"][:n].astype(float)
    text_clock = np.isfinite(ws) & np.isfinite(we) & (we > ws)
    audio_mask = z["word_audio_mask"][:n].astype(bool)
    vision_mask = z["word_vision_mask"][:n].astype(bool)
    activities = np.vstack([
        _feature_activity(z["text_word_native"][:n]),
        _feature_activity(z["word_audio"][:n]),
        _feature_activity(z["word_vision"][:n]),
    ])

    pts = _read_frame_pts(meta["video_path"])
    frames = _representative_frames(meta["video_path"], pts, (0.0, float(meta["duration"])), number=3)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.10), facecolor="white")
    gs = fig.add_gridspec(
        2, 3, left=.065, right=.985, top=.95, bottom=.10,
        height_ratios=[1.05, .95], width_ratios=[1.25, 1.05, .88],
        hspace=.34, wspace=.30,
    )

    # (a) Real source material.
    ax = fig.add_subplot(gs[0, 0])
    panel_title(ax, "a", "原始样本：视频 + 官方文本", pad=4)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.18)
    for i, (_, t, rgb) in enumerate(frames):
        ax.imshow(_crop_16_9(rgb), extent=(i+.04, i+.96, .40, 1.10),
                  aspect="auto", interpolation="bilinear")
        ax.text(i+.50, .34, f"{t:.2f}s", ha="center", va="top", fontsize=5.3, color=MUTED)
    ax.text(.02, .08, textwrap.fill(_sample_excerpt(words), width=46),
            transform=ax.transAxes, fontsize=5.7, color=INK, va="bottom")
    ax.axis("off")

    # (b) Three feature streams use the same word axis.
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_feature_rgb(activities), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(3), ["Text 768-D", "Audio 25-D", "Vision 52-D"])
    ax.set_xticks(range(n), [str(i+1) for i in range(n)])
    ax.set_xlabel("official word index")
    panel_title(ax, "b", "三模态特征提取摘要", pad=4)
    ax.tick_params(length=0, labelsize=5.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.99, .03, "BERT / eGeMAPS / blendshape",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=MUTED)

    # (c) Tensor + audit outputs, compact rather than another process diagram.
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    panel_title(ax, "c", "标准化输出", pad=4)
    rows = [
        (TEXT, f"Text   {config.MAX_WORDS} × 768"),
        (AUDIO, f"Audio  {config.MAX_WORDS} × 25"),
        (VISION, f"Vision {config.MAX_WORDS} × 52"),
        (OCHRE, "word clock + validity masks"),
    ]
    for i, (color, label) in enumerate(rows):
        y = .83 - i*.205
        ax.add_patch(Rectangle((.05, y-.08), .90, .13, transform=ax.transAxes,
                               facecolor=_mix(color, .26), edgecolor=color, linewidth=.75))
        ax.text(.09, y-.015, label, transform=ax.transAxes, fontsize=6.0,
                va="center", color=INK)
    ax.text(.05, .03, "NPZ feature · sample log · word mapping · quality flags",
            transform=ax.transAxes, fontsize=5.2, color=MUTED)

    # (d) The core mechanism: official words aligned to Audio/Vision validity.
    ax = fig.add_subplot(gs[1, :])
    panel_title(ax, "d", "共享词级时间轴：词边界、语音聚合与视觉聚合一一对应", pad=4)
    ax.set_xlim(-.55, n-.45)
    ax.set_ylim(-.55, 3.10)
    rows_y = (2.15, 1.15, .15)
    labels = ("Text clock", "Audio", "Vision")
    colors = (OCHRE, AUDIO, VISION)
    masks = (text_clock, audio_mask, vision_mask)
    for y, label, color, mask in zip(rows_y, labels, colors, masks):
        ax.text(-.62, y, label, ha="right", va="center", fontsize=6.1,
                color=color, fontweight="semibold")
        for j, valid in enumerate(mask):
            ax.add_patch(Rectangle((j-.45, y-.30), .90, .60,
                                   facecolor=_mix(color, .76 if valid else .03),
                                   edgecolor="white", linewidth=.5))
            if not valid:
                ax.add_patch(Rectangle((j-.45, y-.30), .90, .60, fill=False,
                                       edgecolor="#A85E55", hatch="////", linewidth=.45))
    for j, word in enumerate(words):
        ax.text(j, 2.75, word, rotation=55, ha="left", va="bottom", fontsize=5.3, color=INK)
    ax.set_xticks(range(n), [str(i+1) for i in range(n)])
    ax.set_xlabel("official word index")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelsize=5.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [
        Patch(facecolor=_mix(ACCENT, .60), edgecolor="none", label="valid / aggregated"),
        Patch(facecolor="white", edgecolor="#A85E55", hatch="////", label="untrusted → zero + mask"),
    ]
    ax.legend(handles=legend, loc="lower right", ncol=2, frameon=False, fontsize=5.5)

    save_figure(fig, OUT, "fig01_q1_framework")


def _alignment_status_panel(ax, table: pd.DataFrame) -> None:
    status = table["text_audio_status"].fillna("unknown").astype(str)
    counts = status.value_counts()
    preferred = [
        "matched", "crosslingual_translation_candidate", "mismatch_candidate",
        "no_speech", "asr_unavailable", "asr_unreliable",
    ]
    keys = [k for k in preferred if k in counts.index]
    keys += [k for k in counts.index if k not in keys]
    names = {
        "matched": "matched", "crosslingual_translation_candidate": "cross-lingual",
        "mismatch_candidate": "mismatch", "no_speech": "no speech",
        "asr_unavailable": "ASR unavailable", "asr_unreliable": "ASR unreliable",
    }
    vals = np.array([int(counts[k]) for k in keys], int)
    x = np.arange(len(keys))
    ax.bar(x, vals, color=[ACCENT if k == "matched" else "#A9B1B7" for k in keys], width=.68)
    ax.set_xticks(x, [names.get(k, k) for k in keys], rotation=34, ha="right")
    ax.set_ylabel("样本数")
    panel_title(ax, "b", "词时钟审核状态", pad=4)
    for xx, value in zip(x, vals):
        ax.text(xx, value+.7, str(value), ha="center", fontsize=5.6, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=.45)
    ax.set_axisbelow(True)


def _coverage_hist_panel(ax, table: pd.DataFrame) -> None:
    paired = np.minimum(
        table.word_audio_valid_fraction.fillna(0).to_numpy(float),
        table.word_vision_valid_fraction.fillna(0).to_numpy(float),
    )
    bins = np.linspace(0, 1, 6)
    hist, _ = np.histogram(paired, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, hist, width=.17, color="#5F8CA8", edgecolor="white", linewidth=.35)
    ax.set_xticks(centers, ["0–20", "20–40", "40–60", "60–80", "80–100"])
    ax.set_xlabel("Audio–Vision 同时有效词比例 / %")
    ax.set_ylabel("样本数")
    panel_title(ax, "c", "联合逐词覆盖分布", pad=4)
    ax.grid(axis="y", color=GRID, lw=.45)
    ax.set_axisbelow(True)


def _quality_strip(ax, table: pd.DataFrame) -> None:
    text_trusted = table.text_audio_match.fillna(False).astype(bool).to_numpy()
    audio = table.word_audio_valid_fraction.fillna(0).to_numpy(float)
    vision = table.word_vision_valid_fraction.fillna(0).to_numpy(float)
    uncertain = table.target_face_uncertain.fillna(False).astype(bool).to_numpy()

    severity = (~text_trusted).astype(int) * 2 + (audio < .999).astype(int) + (vision < .999).astype(int) + uncertain.astype(int)
    order = np.lexsort((np.arange(len(table)), -vision, -audio, -severity))
    pd.DataFrame({
        "figure_column_1based": np.arange(1, len(order)+1),
        "sample_id": table.iloc[order].sample_id.to_numpy(),
        "text_clock_trusted": text_trusted[order],
        "audio_word_valid_fraction": audio[order],
        "vision_word_valid_fraction": vision[order],
    }).to_csv(OUT / "fig02_quality_strip_order.csv", index=False, encoding="utf-8-sig")

    values = np.vstack([text_trusted.astype(float), audio, vision])[:, order]
    colors = (TEXT, AUDIO, VISION)
    rgb = np.ones((3, len(order), 3), float)
    for i, color in enumerate(colors):
        for j, value in enumerate(values[i]):
            rgb[i, j] = _mix(color, float(value))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(3), ["Text clock", "Audio", "Vision"])
    ticks = [0, 19, 39, 59, 79, min(99, len(order)-1)]
    ticks = sorted(set(t for t in ticks if 0 <= t < len(order)))
    ax.set_xticks(ticks, [str(t+1) for t in ticks])
    ax.set_xlabel("100 条样本（按审核异常与覆盖不足排序）")
    panel_title(ax, "d", "全量样本的词级可用性 / 审核状态条带", pad=4)
    ax.tick_params(length=0, labelsize=5.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for j, original in enumerate(order):
        if not text_trusted[original]:
            ax.add_patch(Rectangle((j-.5, -.5), 1, 1, fill=False,
                                   edgecolor="#A85E55", hatch="////", linewidth=.25))
        if uncertain[original]:
            ax.add_patch(Rectangle((j-.5, 1.5), 1, 1, fill=False,
                                   edgecolor="#A85E55", hatch="\\\\", linewidth=.25))
    legend = [
        Patch(facecolor=_mix(TEXT, .90), label="高覆盖 / 可信"),
        Patch(facecolor=_mix(TEXT, .28), label="局部有效"),
        Patch(facecolor="white", edgecolor="#A85E55", hatch="////", label="审核拒绝 / 歧义"),
    ]
    ax.legend(handles=legend, ncol=3, loc="upper right", frameon=False, fontsize=5.5)


def quality_overview_figure(table: pd.DataFrame) -> None:
    """Q1-2: reference-style three statistics + one long dataset status strip."""
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.65), facecolor="white")
    gs = fig.add_gridspec(
        2, 3, left=.075, right=.985, top=.95, bottom=.10,
        height_ratios=[1.05, .78], hspace=.40, wspace=.35,
    )

    ax = fig.add_subplot(gs[0, 0])
    labels = ["真人语音", "可信词时钟", "有效人脸"]
    vals = np.array([
        table.speech_present.fillna(False).astype(bool).mean(),
        table.text_audio_match.fillna(False).astype(bool).mean(),
        table.face_feature_valid.fillna(False).astype(bool).mean(),
    ]) * 100
    ax.bar(np.arange(3), vals, color=[AUDIO, OCHRE, VISION], width=.64)
    ax.set_xticks(range(3), labels, rotation=18)
    ax.set_ylim(0, 105)
    ax.set_ylabel("样本占比 / %")
    panel_title(ax, "a", "三类关键可用率", pad=4)
    for i, value in enumerate(vals):
        ax.text(i, value+2.0, f"{value:.1f}%", ha="center", fontsize=5.8, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=.45)
    ax.set_axisbelow(True)

    _alignment_status_panel(fig.add_subplot(gs[0, 1]), table)
    _coverage_hist_panel(fig.add_subplot(gs[0, 2]), table)
    _quality_strip(fig.add_subplot(gs[1, :]), table)
    save_figure(fig, OUT, "fig02_q1_quality_overview")


def _quality_bars(ax, meta: dict, z: dict, count: int) -> None:
    audio_mask = z["word_audio_mask"][:count].astype(bool)
    vision_mask = z["word_vision_mask"][:count].astype(bool)
    timed = np.isfinite(z["word_start"][:count]) & np.isfinite(z["word_end"][:count])
    values = np.array([timed.mean(), audio_mask.mean(), vision_mask.mean()]) if count else np.zeros(3)
    labels = ["word clock", "Audio valid", "Vision valid"]
    colors = [OCHRE, AUDIO, VISION]
    y = np.arange(3)
    ax.barh(y, values, color=colors, height=.52)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, .5, 1], ["0", "50%", "100%"])
    panel_title(ax, "a", "典型样本对齐质量", pad=4)
    for yy, value in zip(y, values):
        ax.text(min(1.0, value+.025), yy, f"{value:.0%}", va="center", fontsize=5.7)
    ax.grid(axis="x", color=GRID, lw=.45)
    ax.set_axisbelow(True)
    ax.text(.02, -.25, f"sample {meta['sample_id']} · {float(meta['duration']):.2f}s · {count} words",
            transform=ax.transAxes, fontsize=5.2, color=MUTED)


def representative_alignment_figure(case_id: str) -> None:
    """Q1-3: representative alignment card inspired by the supplied explanation-card style."""
    meta, z = _open_case(case_id)
    count = int(meta["word_count"])
    ws = z["word_start"][:count].astype(float)
    we = z["word_end"][:count].astype(float)
    valid = np.isfinite(ws) & np.isfinite(we) & (we > ws)
    if count == 0 or not valid.all():
        raise RuntimeError(f"Q1 representative sample {case_id} lacks complete word timing")

    display_count = min(count, 14)
    words = list(meta["official_words"][:display_count])
    ws_d, we_d = ws[:display_count], we[:display_count]
    activities = np.vstack([
        _feature_activity(z["text_word_native"][:display_count]),
        _feature_activity(z["word_audio"][:display_count]),
        _feature_activity(z["word_vision"][:display_count]),
    ])
    audio_mask = z["word_audio_mask"][:display_count].astype(bool)
    vision_mask = z["word_vision_mask"][:display_count].astype(bool)
    duration = float(meta["duration"])

    pts = _read_frame_pts(meta["video_path"])
    frames = _representative_frames(meta["video_path"], pts,
                                    (float(ws_d.min()), float(we_d.max())), number=3)
    audio_path = config.OUT / "case_assets" / case_id / "audio_16k.wav"
    _, waveform = _decode_word_audio(meta["video_path"], audio_path)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.75), facecolor="white")
    gs = fig.add_gridspec(
        3, 2, left=.075, right=.985, top=.95, bottom=.09,
        height_ratios=[.86, 1.02, 1.12], width_ratios=[.85, 1.65],
        hspace=.38, wspace=.28,
    )

    _quality_bars(fig.add_subplot(gs[0, 0]), meta, z, count)

    ax = fig.add_subplot(gs[0, 1])
    panel_title(ax, "b", "与共享时间轴对应的原视频帧", pad=4)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    for i, (_, t, rgb) in enumerate(frames):
        ax.imshow(_crop_16_9(rgb), extent=(i+.03, i+.97, .08, .92),
                  aspect="auto", interpolation="bilinear")
        ax.text(i+.50, .02, f"{t:.2f}s", ha="center", va="bottom", fontsize=5.3, color=MUTED)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(_feature_rgb(activities), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(3), ["Text", "Audio", "Vision"])
    ax.set_xticks(range(display_count), words, rotation=58, ha="right")
    panel_title(ax, "c", "逐词三模态特征活动；黑框表示 Audio 与 Vision 同时有效", pad=4)
    both = audio_mask & vision_mask
    for j, ok in enumerate(both):
        if ok:
            ax.add_patch(Rectangle((j-.5, -.5), 1, 3, fill=False, edgecolor="#34434D", linewidth=.55))
        else:
            if not audio_mask[j]:
                ax.add_patch(Rectangle((j-.5, .5), 1, 1, fill=False,
                                       edgecolor="#A85E55", hatch="////", linewidth=.35))
            if not vision_mask[j]:
                ax.add_patch(Rectangle((j-.5, 1.5), 1, 1, fill=False,
                                       edgecolor="#A85E55", hatch="\\\\", linewidth=.35))
    ax.tick_params(length=0, labelsize=5.4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Word-index -> source-time ribbon.  The top is equal-width linguistic order;
    # the bottom uses the actual [start,end) positions on the clip timeline.
    ax = fig.add_subplot(gs[2, :])
    panel_title(ax, "d", "official word → 原视频真实时间的可核验映射", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (start, end, word) in enumerate(zip(ws_d, we_d, words)):
        tx0, tx1 = i/display_count, (i+1)/display_count
        sx0, sx1 = float(start)/duration, float(end)/duration
        ax.add_patch(Polygon([(tx0, .78), (tx1, .78), (sx1, .30), (sx0, .30)],
                             closed=True, facecolor=_mix(OCHRE, .42), edgecolor="none", alpha=.58))
        ax.add_patch(Rectangle((tx0, .78), tx1-tx0, .12,
                               facecolor=_mix(TEXT, .26), edgecolor="white", linewidth=.3))
        ax.text((tx0+tx1)/2, .92, word, rotation=60, ha="left", va="bottom",
                fontsize=4.9, color=INK)
        ax.add_patch(Rectangle((sx0, .24), max(sx1-sx0, .002), .09,
                               facecolor=OCHRE, edgecolor="none", alpha=.88))

    step = max(1, int(np.ceil(len(waveform)/900)))
    usable = len(waveform)//step*step
    if usable:
        binned = waveform[:usable].reshape(-1, step)
        env = np.maximum(np.abs(binned.min(axis=1)), np.abs(binned.max(axis=1)))
        scale = max(float(env.max()), 1e-6)
        xx = (np.arange(len(env))+.5)*step/16000/max(duration, 1e-6)
        ax.plot(xx, .12 + .07*env/scale, color="#7F8B93", lw=.55)
    ax.text(-.01, .84, "word", ha="right", va="center", fontsize=5.6, color=TEXT)
    ax.text(-.01, .28, "time", ha="right", va="center", fontsize=5.6, color=OCHRE)
    for t in np.linspace(0, duration, 5):
        x = t/duration
        ax.plot([x, x], [.20, .22], color=EDGE, lw=.5)
        ax.text(x, .02, f"{t:.1f}s", ha="center", va="bottom", fontsize=5.2, color=MUTED)
    ax.axis("off")

    save_figure(fig, OUT, "fig03_q1_representative_alignment")


def quality_control_figure(cases: dict[str, str]) -> None:
    """Q1-4: four source-evidence cards with the same compact visual grammar."""
    kinds = [
        ("no_face", "无可用目标人脸", "Vision mask=0"),
        ("no_speech", "无真人语音", "Audio mask=0"),
        ("multiple_faces", "多人脸目标歧义", "歧义区间剔除"),
        ("mismatch", "官方文本与原声不符", "拒绝词时钟"),
    ]
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.55), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=.065, right=.985, top=.95, bottom=.08,
                             hspace=.30, wspace=.18)
    for index, (kind, title, action) in enumerate(kinds):
        row, col = divmod(index, 2)
        sub = outer[row, col].subgridspec(2, 1, height_ratios=[1.05, .72], hspace=.05)
        image_ax = fig.add_subplot(sub[0])
        timeline_ax = fig.add_subplot(sub[1])
        sid = cases[kind]
        meta, z = _open_case(sid)
        _, t, frame = _source_frame(meta, z, kind)
        image_ax.imshow(_crop_16_9(frame), aspect="auto")
        image_ax.axis("off")
        panel_title(image_ax, chr(ord("a")+index), title, pad=2)
        image_ax.text(.015, .035, f"{sid} · {t:.2f}s", transform=image_ax.transAxes,
                      fontsize=5.1, color="white",
                      bbox=dict(facecolor="#26343B", alpha=.66, edgecolor="none", pad=1.4))
        image_ax.text(.985, .965, action, transform=image_ax.transAxes,
                      ha="right", va="top", fontsize=5.4,
                      color="white", fontweight="semibold",
                      bbox=dict(facecolor="#A85E55" if kind in {"mismatch", "multiple_faces"} else ACCENT,
                                alpha=.88, edgecolor="none", pad=1.5))
        _timeline(timeline_ax, meta, z, compact=True)
        if row == 0:
            timeline_ax.set_xlabel("")
        timeline_ax.tick_params(axis="y", labelsize=5.6)
    save_figure(fig, OUT, "fig04_q1_quality_control")


def write_manifest() -> None:
    payload = {
        "version": "v3_reference_style_evidence_cards",
        "main_text_figures": {
            "fig01_q1_framework": "worked sample pipeline: source media + feature streams + tensors + word-level validity strip",
            "fig02_q1_quality_overview": "three compact dataset statistics + 100-sample modality/audit strip",
            "fig03_q1_representative_alignment": "quality bars + source frames + word-level feature map + word-to-time ribbon",
            "fig04_q1_quality_control": "four compact complex-scene source-evidence cards",
        },
        "layout_notes": [
            "The Q1 main figures use compact multi-panel evidence cards rather than large generic flowcharts.",
            "Semantic colors remain fixed across Text, Audio and Vision.",
            "Hatching denotes rejected or ambiguous evidence; complete numeric values remain in saved tables.",
            "No experiment value, feature definition or alignment rule is changed by this visualization redesign.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    summary = config.OUT / "tables/02_sample_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing Q1 summary: {summary}")
    table = pd.read_csv(summary)
    cases = _case_map()
    required = {"direct_word_alignment", "no_face", "no_speech", "multiple_faces", "mismatch"}
    if not required.issubset(cases):
        raise RuntimeError(f"Q1 case manifest missing: {sorted(required-set(cases))}")

    OUT.mkdir(parents=True, exist_ok=True)
    direct = cases["direct_word_alignment"]
    framework_figure(direct)
    quality_overview_figure(table)
    representative_alignment_figure(direct)
    quality_control_figure(cases)
    write_manifest()
    print(f"Wrote redesigned Q1 publication figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
