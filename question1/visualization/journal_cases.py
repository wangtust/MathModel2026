"""Compact main-text Q1 case figures built from auditable source-media mappings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import (
    DANGER, DOUBLE_COLUMN, GRID, INK, LIGHT, MODALITY_COLORS, MUTED, OCHRE,
    apply_paper_style, panel_title, save_figure,
)
from .. import config
from .cases import (
    _decode_word_audio, _letterbox_frame, _read_frame_pts, _representative_frames,
)


TEXT = MODALITY_COLORS["text"]
AUDIO = MODALITY_COLORS["audio"]
VISION = MODALITY_COLORS["vision"]


def _open_case(sid):
    meta = json.loads((config.OUT / "logs/samples" / f"{sid}.json").read_text())
    with np.load(config.OUT / "features" / f"{sid}.npz") as archive:
        z = {key: archive[key] for key in archive.files}
    return meta, z


def _intervals(times, mask, half, duration):
    selected = []
    for t, valid in zip(times, mask):
        if valid:
            start, end = max(0., float(t)-half), min(duration, float(t)+half)
            if selected and start <= selected[-1][1] + .025:
                selected[-1] = (selected[-1][0], max(end, selected[-1][1]))
            else:
                selected.append((start, end))
    return selected


def _bands(ax, intervals, y, color, height=.38):
    if intervals:
        ax.broken_barh([(float(a), float(b-a)) for a, b in intervals if b > a],
                       (y-height/2, height), facecolors=color, edgecolors="none")


def _timeline(ax, meta, z, *, compact=True, number_words=False):
    duration = float(meta["duration"])
    lanes = [2.45, 1.45, .45]
    for y in lanes:
        ax.axhspan(y-.23, y+.23, color=LIGHT, zorder=-2)
    speech = [(float(a), float(b)) for a, b in meta["speech_segments"]]
    face = _intervals(z["vision_time"], z["vision_valid_mask"], .5/config.FPS, duration)
    multiple = _intervals(z["vision_time"], z["vision_multiface_mask"], .5/config.FPS, duration)
    ws, we = z["word_start"], z["word_end"]
    words = [(float(a), float(b)) for a, b in zip(ws, we)
             if np.isfinite(a) and np.isfinite(b) and b > a]
    _bands(ax, speech, lanes[0], AUDIO)
    _bands(ax, face, lanes[1], VISION)
    _bands(ax, multiple, lanes[1], DANGER, height=.47)
    _bands(ax, words, lanes[2], OCHRE)
    if number_words:
        for i, (a, b) in enumerate(zip(ws, we)):
            if np.isfinite(a) and np.isfinite(b) and b > a and b-a > .095:
                ax.text((a+b)/2, lanes[2], str(i+1), ha="center", va="center",
                        fontsize=5.6, color="white", fontweight="semibold")
    if not speech:
        ax.text(duration/2, lanes[0], "无真人语音", ha="center", va="center",
                fontsize=6.2, color=MUTED)
    if not face and not multiple:
        message = "检测到人脸但表情无效" if meta["face_present"] else "无可用目标人脸"
        ax.text(duration/2, lanes[1], message, ha="center", va="center",
                fontsize=6.2, color=MUTED)
    if not words:
        ax.text(duration/2, lanes[2], "无可靠官方词时间", ha="center", va="center",
                fontsize=6.2, color=MUTED)
    ax.set_xlim(0, max(duration, .5)); ax.set_ylim(0, 3.0)
    ax.set_yticks(lanes, ["语音", "人脸", "词时间"] if compact else
                  ["真人语音", "可用人脸", "官方词时间"])
    ax.set_xlabel("时间 / s" if compact else "原视频时间 / s")
    ax.tick_params(axis="both", labelsize=6, length=2)
    ax.grid(axis="x", color=GRID, lw=.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _source_frame(meta, z, kind):
    times = z["vision_time"]
    if kind == "multiple_faces":
        mask = z["vision_multiface_mask"].astype(bool)
    elif kind in ("typical", "no_speech"):
        mask = z["vision_valid_mask"].astype(bool)
    else:
        mask = np.ones(len(times), bool)
    idx = np.flatnonzero(mask)
    target = float(times[idx[len(idx)//2]]) if len(idx) else float(meta["duration"])/2
    pts = _read_frame_pts(meta["video_path"])
    frame_index, t, frame = _representative_frames(
        meta["video_path"], pts, (target, target+.0001), number=1)[0]
    return frame_index, t, frame


def _feature_activity(matrix: np.ndarray) -> np.ndarray:
    """L2 magnitude normalized within one modality; visualization only."""
    values = np.linalg.norm(np.asarray(matrix, np.float32), axis=1)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    scale = float(np.nanmax(values[finite]))
    return values / scale if scale > 1e-12 else np.zeros_like(values)


def _plot_activity(ax, intervals, values, *, color, title, letter):
    centers = np.array([(a+b)/2 for a, b in intervals], float)
    ax.plot(centers, values, marker="o", markersize=2.7, linewidth=1.15, color=color)
    for (a, b), value in zip(intervals, values):
        ax.hlines(value, a, b, color=color, linewidth=2.0, alpha=.72)
    ax.set_ylim(-.04, 1.06)
    ax.set_ylabel("相对幅值", fontsize=6.2)
    panel_title(ax, letter, title, pad=3)
    ax.grid(axis="y", color=GRID, linewidth=.45)
    ax.tick_params(axis="both", labelsize=5.8, length=2)
    ax.spines[["top", "right"]].set_visible(False)


def _representative_alignment(meta, z, out):
    """Main-text typical sample: source frames, waveform, words and compact feature summaries."""
    sid = meta["sample_id"]
    count = int(meta["word_count"])
    ws = z["word_start"][:count].astype(float)
    we = z["word_end"][:count].astype(float)
    valid = np.isfinite(ws) & np.isfinite(we) & (we > ws)
    if not valid.all() or count == 0:
        return None
    intervals = list(zip(ws, we))
    duration = float(meta["duration"])

    pts = _read_frame_pts(meta["video_path"])
    frames = _representative_frames(meta["video_path"], pts,
                                    (float(ws.min()), float(we.max())), number=5)
    audio_path = config.OUT / "case_assets" / sid / "audio_16k.wav"
    _, waveform = _decode_word_audio(meta["video_path"], audio_path)

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 6.15), facecolor="white")
    gs = fig.add_gridspec(6, 1, left=.105, right=.975, top=.965, bottom=.085,
                          height_ratios=[1.28, .94, 1.04, .60, .60, .60], hspace=.42)
    video_ax, wave_ax, word_ax, text_ax, audio_ax, vision_ax = [fig.add_subplot(gs[i]) for i in range(6)]

    panel_title(video_ax, "a", "原视频帧", pad=3)
    video_ax.set_ylim(0, 1)
    card_width = duration / max(5, len(frames)) * .88
    for _, t, rgb in frames:
        left, right = max(0, t-card_width/2), min(duration, t+card_width/2)
        video_ax.imshow(_letterbox_frame(rgb), extent=(left, right, .16, .91),
                        aspect="auto", interpolation="bilinear")
        video_ax.text(t, .05, f"{t:.2f}s", ha="center", fontsize=5.6, color=MUTED)
    video_ax.set_yticks([])

    panel_title(wave_ax, "b", "原声音频与真人语音区间", pad=3)
    step = max(1, int(np.ceil(len(waveform)/1800)))
    usable = len(waveform)//step*step
    binned = waveform[:usable].reshape(-1, step)
    times = (np.arange(len(binned))+.5)*step/16000
    for start, end in meta["speech_segments"]:
        wave_ax.axvspan(float(start), float(end), color=AUDIO, alpha=.12, linewidth=0)
    wave_ax.fill_between(times, binned.min(axis=1), binned.max(axis=1),
                         color="#70808D", linewidth=0, alpha=.82)
    peak = max(.05, float(np.max(np.abs(waveform))))
    wave_ax.set_ylim(-peak*1.08, peak*1.08); wave_ax.set_ylabel("幅值", fontsize=6.2)

    panel_title(word_ax, "c", "官方词与真实发音区间 [sᵢ, eᵢ)", pad=3)
    word_ax.set_ylim(0, 1)
    lanes = (.36, .62, .84)
    for i, ((start, end), word) in enumerate(zip(intervals, meta["official_words"][:count])):
        center = (start+end)/2
        word_ax.barh(.12, end-start, left=start, height=.16, color=OCHRE, alpha=.82,
                     edgecolor="white", linewidth=.2)
        lane = lanes[i % len(lanes)]
        word_ax.plot([center, center], [.22, lane-.06], color="#A4AEB2", lw=.45)
        word_ax.text(center, lane, f"{i+1:02d} {word}", ha="center", va="center",
                     fontsize=5.7, color=INK)
    word_ax.set_yticks([])

    text_values = _feature_activity(z["text_word_native"][:count])
    audio_values = _feature_activity(z["word_audio"][:count])
    vision_values = _feature_activity(z["word_vision"][:count])
    _plot_activity(text_ax, intervals, text_values, color=TEXT,
                   title="Text：768-D BERT 词特征 L2 摘要", letter="d")
    _plot_activity(audio_ax, intervals, audio_values, color=AUDIO,
                   title="Audio：25-D eGeMAPS 词特征 L2 摘要", letter="e")
    _plot_activity(vision_ax, intervals, vision_values, color=VISION,
                   title="Vision：52-D 表情词特征 L2 摘要", letter="f")

    for ax in (video_ax, wave_ax, word_ax, text_ax, audio_ax, vision_ax):
        ax.set_xlim(0, duration)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=5.8, length=2)
    for ax in (video_ax, wave_ax, word_ax, text_ax, audio_ax):
        ax.tick_params(axis="x", labelbottom=False)
    vision_ax.set_xlabel("共享原视频时间轴 / s")
    fig.text(.105, .025,
             f"sample {sid}；{count}/{count} 个官方词获得可靠真实发音时间。三类 L2 摘要均按本模态片段最大值归一化，仅用于可视化；模型与附件保留原始 768/25/52 维特征。",
             fontsize=6.15, color=MUTED)

    save_figure(fig, out, "fig03_q1_representative_alignment")
    return {"figure": "fig03_q1_representative_alignment", "panel": "(a)-(f)",
            "sample_id": sid, "finding": "direct_word_alignment"}


def _quality_control(cases, out):
    """Merge source evidence and validity timelines for four complex-scene failure modes."""
    kinds = [
        ("no_face", "无可用目标人脸", "保留语音；视觉 mask=0"),
        ("no_speech", "无真人语音", "保留视觉；语音 mask=0"),
        ("multiple_faces", "多人脸目标歧义", "不确定区间不用于视觉聚合"),
        ("mismatch", "官方文本与原声不符", "停止英语逐词强制配对"),
    ]
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.75), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=.075, right=.985, top=.97, bottom=.08,
                             hspace=.32, wspace=.22)
    records = []
    for index, (kind, title, action) in enumerate(kinds):
        row, col = divmod(index, 2)
        sub = outer[row, col].subgridspec(2, 1, height_ratios=[1.12, .88], hspace=.05)
        image_ax = fig.add_subplot(sub[0])
        timeline_ax = fig.add_subplot(sub[1])
        meta, z = _open_case(cases[kind])
        frame_index, t, frame = _source_frame(meta, z, kind)
        image_ax.imshow(frame); image_ax.axis("off")
        panel_title(image_ax, chr(ord("a")+index), title, pad=3)
        image_ax.text(0, -.045, f"{meta['sample_id']} · PTS {t:.2f}s · {action}",
                      transform=image_ax.transAxes, fontsize=5.9, color=MUTED, va="top")
        _timeline(timeline_ax, meta, z, compact=True)
        if kind == "mismatch":
            timeline_ax.text(.99, .94, "Official ≠ ASR → no accepted word alignment",
                             transform=timeline_ax.transAxes, ha="right", va="top",
                             fontsize=5.6, color=DANGER)
        records.append({"figure": "fig04_q1_quality_control",
                        "panel": f"({chr(ord('a')+index)})",
                        "sample_id": meta["sample_id"],
                        "original_frame_pts_s": t,
                        "original_frame_index_0based": frame_index,
                        "finding": kind})
    fig.text(.075, .018,
             "蓝=真人语音，绿=可用人脸，黄=已接受官方词时间，红=多人脸目标不确定。异常只屏蔽不可信局部证据，不删除原始样本。",
             fontsize=6.2, color=MUTED)
    save_figure(fig, out, "fig04_q1_quality_control")
    return records


def create_journal_case_figures(cases: dict, out: Path) -> list[dict]:
    """Create the two main-text case figures; detailed three-case heatmaps remain supplementary."""
    apply_paper_style(chinese=True)
    out.mkdir(parents=True, exist_ok=True)
    needed = {"typical", "no_face", "multiple_faces", "no_speech", "mismatch"}
    if not needed.issubset(cases):
        return []
    records = []
    meta, z = _open_case(cases["typical"])
    typical = _representative_alignment(meta, z, out)
    if typical:
        records.append(typical)
    records.extend(_quality_control(cases, out))
    pd.DataFrame(records).to_csv(config.OUT / "tables/10_journal_case_manifest.csv", index=False)
    return records
