"""V2 publication-facing Q1 figures with tighter journal layouts.

Legacy figures remain untouched for auditability.  This module only repackages
already validated Q1 outputs into a compact main-text layer under
``outputs/q1/paper_figures``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import (
    ACCENT, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT, MODALITY_COLORS, MUTED,
    OCHRE, apply_paper_style, panel_title, save_figure,
)
from .. import config
from .overview import _availability_panel, _coverage_panel, _matrix_panel, _box, _arrow
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


def framework_figure() -> None:
    apply_paper_style(chinese=True)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 3.72))
    fig.subplots_adjust(left=.025, right=.985, top=.97, bottom=.045)
    ax.set_axis_off()

    tc = MODALITY_COLORS["text"]
    ac = MODALITY_COLORS["audio"]
    vc = MODALITY_COLORS["vision"]

    _box(ax, (.025, .38), .15, .34, "Input",
         ["MP4 video", "official transcript", "sentiment label", "sample ID"],
         color=ACCENT, fill_alpha=.07)
    _box(ax, (.215, .69), .25, .19, "Text",
         ["words -> WordPiece", "BERT pooling -> 768-D"], color=tc)
    _box(ax, (.215, .405), .25, .19, "Audio",
         ["VAD + WhisperX clock", "eGeMAPS pooling -> 25-D"], color=ac)
    _box(ax, (.215, .12), .25, .19, "Vision",
         [f"frames @ {config.FPS:g} fps + face filter", "blendshape pooling -> 52-D"], color=vc)

    _box(ax, (.515, .32), .245, .46, "Word-level alignment",
         ["official word sequence", "accepted interval [s_i, e_i)",
          "audio: interval mean", "vision: overlap-weighted mean",
          "untrusted evidence -> 0 + mask"], color=OCHRE, fill_alpha=.08)
    _box(ax, (.81, .50), .165, .25, "Aligned features",
         ["T_i: R^768", "A_i: R^25", "V_i: R^52", f"length -> {config.MAX_WORDS}"],
         color=ACCENT, fill_alpha=.07)
    _box(ax, (.81, .17), .165, .22, "Audit outputs",
         ["NPZ feature", "sample log", "word mapping", "quality flags"],
         color=ACCENT, fill_alpha=.07)

    _arrow(ax, (.175, .55), (.215, .785))
    _arrow(ax, (.175, .55), (.215, .50))
    _arrow(ax, (.175, .55), (.215, .215))
    _arrow(ax, (.465, .785), (.515, .66), color=tc)
    _arrow(ax, (.465, .50), (.515, .55), color=ac)
    _arrow(ax, (.465, .215), (.515, .43), color=vc)
    _arrow(ax, (.76, .58), (.81, .625))
    _arrow(ax, (.76, .42), (.81, .28))

    # Small shared-time sketch: enough to communicate the alignment mechanism
    # without creating a second explanatory panel.
    ax.text(.515, .245, "word", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    for word, a, b in (("I", .575, .61), ("think", .615, .66),
                       ("social", .665, .71), ("justice", .715, .76)):
        ax.plot([a, b], [.255, .255], transform=ax.transAxes, color=OCHRE,
                lw=3.5, solid_capstyle="butt")
        ax.text((a+b)/2, .222, word, transform=ax.transAxes, fontsize=5.0,
                ha="center", color=INK)
    ax.text(.515, .185, "audio", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    ax.plot([.575, .76], [.195, .195], transform=ax.transAxes, color=ac, lw=1.8)
    ax.text(.515, .135, "video", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    for x in np.linspace(.575, .76, 9):
        ax.plot(x, .145, marker="o", ms=2.1, color=vc, transform=ax.transAxes)

    save_figure(fig, OUT, "fig01_q1_framework")


def quality_overview_figure(table: pd.DataFrame) -> None:
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.72))
    gs = fig.add_gridspec(
        2, 2, left=.10, right=.975, top=.965, bottom=.085,
        height_ratios=[1.0, .70], width_ratios=[1, 1.05],
        hspace=.34, wspace=.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    _availability_panel(ax_a, table)
    _coverage_panel(ax_c, table)
    _matrix_panel(ax_b, table)
    save_figure(fig, OUT, "fig02_q1_quality_overview")


def _feature_tracks(ax, intervals, text_values, audio_values, vision_values) -> None:
    centers = np.array([(a+b)/2 for a, b in intervals], float)
    offsets = (2.0, 1.0, 0.0)
    series = ((text_values, TEXT, "Text"), (audio_values, AUDIO, "Audio"),
              (vision_values, VISION, "Vision"))
    for off, (values, color, _) in zip(offsets, series):
        ax.axhline(off, color=GRID, lw=.65, zorder=0)
        ax.plot(centers, off + .55*values, color=color, lw=1.15, marker="o",
                ms=2.5, zorder=2)
        for (a, b), value in zip(intervals, values):
            ax.hlines(off + .55*value, a, b, color=color, lw=1.7, alpha=.70)
    ax.set_yticks(offsets, [x[2] for x in series])
    ax.set_ylim(-.15, 2.72)
    ax.set_ylabel("within-modality\nrelative activity", fontsize=5.8)
    panel_title(ax, "d", "三模态词级特征活动摘要", pad=3)
    ax.grid(axis="x", color=GRID, linewidth=.4)
    ax.tick_params(axis="both", labelsize=5.8, length=2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def representative_alignment_figure(case_id: str) -> None:
    meta, z = _open_case(case_id)
    count = int(meta["word_count"])
    ws = z["word_start"][:count].astype(float)
    we = z["word_end"][:count].astype(float)
    valid = np.isfinite(ws) & np.isfinite(we) & (we > ws)
    if count == 0 or not valid.all():
        raise RuntimeError(f"Q1 representative sample {case_id} lacks complete word timing")
    intervals = list(zip(ws, we))
    duration = float(meta["duration"])

    pts = _read_frame_pts(meta["video_path"])
    frames = _representative_frames(meta["video_path"], pts,
                                    (float(ws.min()), float(we.max())), number=5)
    audio_path = config.OUT / "case_assets" / case_id / "audio_16k.wav"
    _, waveform = _decode_word_audio(meta["video_path"], audio_path)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.92), facecolor="white")
    gs = fig.add_gridspec(4, 1, left=.105, right=.975, top=.965, bottom=.085,
                          height_ratios=[1.12, .78, .95, 1.05], hspace=.31)
    video_ax, wave_ax, word_ax, feature_ax = [fig.add_subplot(gs[i]) for i in range(4)]

    panel_title(video_ax, "a", "原视频帧", pad=3)
    video_ax.set_ylim(0, 1)
    card_width = duration / max(5, len(frames)) * .88
    for _, t, rgb in frames:
        left, right = max(0, t-card_width/2), min(duration, t+card_width/2)
        video_ax.imshow(_letterbox_frame(rgb), extent=(left, right, .14, .92),
                        aspect="auto", interpolation="bilinear")
        video_ax.text(t, .035, f"{t:.2f}s", ha="center", fontsize=5.3, color=MUTED)
    video_ax.set_yticks([])

    panel_title(wave_ax, "b", "原声音频与真人语音区间", pad=3)
    step = max(1, int(np.ceil(len(waveform)/1600)))
    usable = len(waveform)//step*step
    binned = waveform[:usable].reshape(-1, step)
    times = (np.arange(len(binned))+.5)*step/16000
    for start, end in meta["speech_segments"]:
        wave_ax.axvspan(float(start), float(end), color=AUDIO, alpha=.11, linewidth=0)
    wave_ax.fill_between(times, binned.min(axis=1), binned.max(axis=1),
                         color="#70808D", linewidth=0, alpha=.80)
    peak = max(.05, float(np.max(np.abs(waveform))))
    wave_ax.set_ylim(-peak*1.06, peak*1.06)
    wave_ax.set_ylabel("幅值", fontsize=5.8)

    panel_title(word_ax, "c", "官方词与真实发音区间 [s_i, e_i)", pad=3)
    word_ax.set_ylim(0, 1)
    lanes = (.34, .61, .84)
    for i, ((start, end), word) in enumerate(zip(intervals, meta["official_words"][:count])):
        center = (start+end)/2
        word_ax.barh(.11, end-start, left=start, height=.15, color=OCHRE, alpha=.82,
                     edgecolor="white", linewidth=.2)
        lane = lanes[i % 3]
        word_ax.plot([center, center], [.21, lane-.055], color="#A4AEB2", lw=.42)
        word_ax.text(center, lane, f"{i+1:02d} {word}", ha="center", va="center",
                     fontsize=5.4, color=INK)
    word_ax.set_yticks([])

    _feature_tracks(
        feature_ax, intervals,
        _feature_activity(z["text_word_native"][:count]),
        _feature_activity(z["word_audio"][:count]),
        _feature_activity(z["word_vision"][:count]),
    )

    for ax in (video_ax, wave_ax, word_ax, feature_ax):
        ax.set_xlim(0, duration)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=5.6, length=2)
    for ax in (video_ax, wave_ax, word_ax):
        ax.tick_params(axis="x", labelbottom=False)
    feature_ax.set_xlabel("共享原视频时间轴 / s")
    save_figure(fig, OUT, "fig03_q1_representative_alignment")


def _crop_16_9(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    target = 16/9
    aspect = w / max(h, 1)
    if aspect > target:
        nw = int(round(h * target)); x0 = max(0, (w-nw)//2)
        return frame[:, x0:x0+nw]
    nh = int(round(w / target)); y0 = max(0, (h-nh)//2)
    return frame[y0:y0+nh, :]


def quality_control_figure(cases: dict[str, str]) -> None:
    kinds = [
        ("no_face", "无可用目标人脸", "语音保留；Vision mask=0"),
        ("no_speech", "无真人语音", "视觉保留；Audio mask=0"),
        ("multiple_faces", "多人脸目标歧义", "歧义区间不参与视觉聚合"),
        ("mismatch", "官方文本与原声不符", "停止英语逐词强制配对"),
    ]
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.82), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=.07, right=.985, top=.975, bottom=.07,
                             hspace=.26, wspace=.20)
    for index, (kind, title, action) in enumerate(kinds):
        row, col = divmod(index, 2)
        sub = outer[row, col].subgridspec(2, 1, height_ratios=[1.0, .88], hspace=.04)
        image_ax = fig.add_subplot(sub[0])
        timeline_ax = fig.add_subplot(sub[1])
        sid = cases[kind]
        meta, z = _open_case(sid)
        _, t, frame = _source_frame(meta, z, kind)
        image_ax.imshow(_crop_16_9(frame), aspect="auto")
        image_ax.axis("off")
        panel_title(image_ax, chr(ord("a")+index), title, pad=2)
        image_ax.text(.01, .03, f"{sid} · {t:.2f}s", transform=image_ax.transAxes,
                      fontsize=5.4, color="white",
                      bbox=dict(facecolor="#26343B", alpha=.65, edgecolor="none", pad=1.5))
        _timeline(timeline_ax, meta, z, compact=True)
        timeline_ax.text(.99, .94, action, transform=timeline_ax.transAxes,
                         ha="right", va="top", fontsize=5.5,
                         color=ACCENT if kind != "mismatch" else "#A85E55")
    save_figure(fig, OUT, "fig04_q1_quality_control")


def write_manifest() -> None:
    payload = {
        "version": "v2_compact_journal_layout",
        "main_text_figures": {
            "fig01_q1_framework": "feature extraction + word-level alignment + auditable output",
            "fig02_q1_quality_overview": "availability + word coverage + full-dataset quality-event matrix",
            "fig03_q1_representative_alignment": "source frames + waveform + word timing + compact three-modality feature activity",
            "fig04_q1_quality_control": "four complex-scene quality-control cases with source frames and timelines",
        },
        "layout_notes": [
            "Long explanatory prose is intentionally kept out of figure canvases and belongs in the manuscript caption.",
            "Full-dimensional feature heatmaps remain legacy/supplementary audit material.",
            "All main-text figures use the shared 7.15-inch double-column width.",
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
    framework_figure()
    quality_overview_figure(table)
    representative_alignment_figure(cases["direct_word_alignment"])
    quality_control_figure(cases)
    write_manifest()
    print(f"Wrote compact Q1 V2 paper figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
