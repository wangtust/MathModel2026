"""Publication-facing Q1 figures: compact, chart-first manuscript layout.

Design goals:
- Prefer standard statistical charts, heatmaps and time strips to prose-heavy panels.
- Keep Text/Audio/Vision colors stable across all figures.
- Use real saved experiment outputs only; no experiment definition is changed.
- Fill the canvas efficiently and move methodological explanation to the paper caption.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from common.plotting import (
    ACCENT, DANGER, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT, MODALITY_COLORS,
    MUTED, OCHRE, apply_paper_style, panel_title, save_figure,
)
from .. import config
from .journal_cases import (
    AUDIO, TEXT, VISION, _feature_activity, _open_case, _source_frame,
)
from .cases import (
    _decode_word_audio, _letterbox_frame, _read_frame_pts, _representative_frames,
)


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
    rgb = np.asarray(to_rgb(color), float)
    s = float(np.clip(strength, 0.0, 1.0))
    return tuple(1.0 - (1.0 - rgb) * (0.08 + 0.92 * s))


def _feature_rgb(values: np.ndarray) -> np.ndarray:
    colors = (TEXT, AUDIO, VISION)
    image = np.ones((3, values.shape[1], 3), float)
    for i, color in enumerate(colors):
        for j, value in enumerate(values[i]):
            image[i, j] = _mix(color, float(value))
    return image


def _wave_envelope(waveform: np.ndarray, duration: float, bins: int = 900):
    if waveform is None or len(waveform) == 0 or duration <= 0:
        return np.array([]), np.array([])
    step = max(1, int(np.ceil(len(waveform) / bins)))
    usable = len(waveform) // step * step
    if usable <= 0:
        return np.array([]), np.array([])
    block = waveform[:usable].reshape(-1, step)
    env = np.maximum(np.abs(block.min(axis=1)), np.abs(block.max(axis=1)))
    scale = max(float(np.nanmax(env)), 1e-8)
    t = (np.arange(len(env)) + .5) * duration / len(env)
    return t, env / scale


def _validity_rgb(text_mask: np.ndarray, audio_mask: np.ndarray, vision_mask: np.ndarray) -> np.ndarray:
    masks = np.vstack([text_mask, audio_mask, vision_mask]).astype(float)
    colors = (TEXT, AUDIO, VISION)
    rgb = np.ones((3, masks.shape[1], 3), float)
    for i, color in enumerate(colors):
        for j, value in enumerate(masks[i]):
            rgb[i, j] = _mix(color, .85 if value else .02)
    return rgb


def framework_figure(case_id: str) -> None:
    """Q1-1: source -> features -> aligned output, using one real sample."""
    meta, z = _open_case(case_id)
    count = int(meta["word_count"])
    if count <= 0:
        raise RuntimeError(f"Q1 framework sample {case_id} has no official words")

    n = min(count, 12)
    words = list(meta["official_words"][:n])
    ws = z["word_start"][:n].astype(float)
    we = z["word_end"][:n].astype(float)
    text_mask = np.isfinite(ws) & np.isfinite(we) & (we > ws)
    audio_mask = z["word_audio_mask"][:n].astype(bool)
    vision_mask = z["word_vision_mask"][:n].astype(bool)
    activities = np.vstack([
        _feature_activity(z["text_word_native"][:n]),
        _feature_activity(z["word_audio"][:n]),
        _feature_activity(z["word_vision"][:n]),
    ])

    pts = _read_frame_pts(meta["video_path"])
    frames = _representative_frames(
        meta["video_path"], pts, (0.0, float(meta["duration"])), number=3
    )
    audio_path = config.OUT / "case_assets" / case_id / "audio_16k.wav"
    _, waveform = _decode_word_audio(meta["video_path"], audio_path)
    t_env, env = _wave_envelope(waveform, float(meta["duration"]))

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 3.58), facecolor="white")
    gs = fig.add_gridspec(
        2, 3, left=.065, right=.985, top=.95, bottom=.10,
        height_ratios=[1.05, .72], width_ratios=[1.25, 1.08, .76],
        hspace=.34, wspace=.30,
    )

    ax = fig.add_subplot(gs[0, 0])
    panel_title(ax, "a", "原始样本", pad=3)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.10)
    for i, (_, t, rgb) in enumerate(frames):
        ax.imshow(
            _crop_16_9(rgb), extent=(i+.03, i+.97, .38, 1.05),
            aspect="auto", interpolation="bilinear"
        )
        ax.text(i+.5, .34, f"{t:.1f}s", ha="center", va="top", fontsize=5.0, color=MUTED)
    if len(t_env):
        xx = t_env / max(float(meta["duration"]), 1e-8) * 3.0
        ax.fill_between(xx, .05, .05 + .18*env, color="#7E8A92", linewidth=0, alpha=.80)
    ax.text(.01, .01, f"{count} words", transform=ax.transAxes, fontsize=5.2, color=MUTED)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_feature_rgb(activities), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(3), ["Text", "Audio", "Vision"])
    ax.set_xticks(range(n), [str(i+1) for i in range(n)])
    panel_title(ax, "b", "逐词特征活动", pad=3)
    ax.set_xlabel("word index")
    ax.tick_params(length=0, labelsize=5.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    dims = np.array([768, 25, 52])
    yy = np.arange(3)
    ax.barh(yy, dims, color=[TEXT, AUDIO, VISION], height=.52)
    ax.set_yticks(yy, ["T", "A", "V"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(15, 1200)
    ax.set_xticks([25, 50, 100, 500], ["25", "50", "100", "500"])
    panel_title(ax, "c", "特征维数", pad=3)
    for y, d in zip(yy, dims):
        ax.text(d*1.05, y, str(d), va="center", fontsize=5.4, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=.45)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(
        _validity_rgb(text_mask, audio_mask, vision_mask),
        aspect="auto", interpolation="nearest"
    )
    ax.set_yticks(range(3), ["Text", "Audio", "Vision"])
    ax.set_xticks(range(n), words, rotation=45, ha="right")
    panel_title(ax, "d", "词级对齐与有效掩码", pad=3)
    ax.tick_params(length=0, labelsize=5.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for j in range(n):
        if not text_mask[j]:
            ax.add_patch(Rectangle((j-.5, -.5), 1, 1, fill=False,
                                   edgecolor=DANGER, hatch="////", linewidth=.35))
        if not audio_mask[j]:
            ax.add_patch(Rectangle((j-.5, .5), 1, 1, fill=False,
                                   edgecolor=DANGER, hatch="////", linewidth=.35))
        if not vision_mask[j]:
            ax.add_patch(Rectangle((j-.5, 1.5), 1, 1, fill=False,
                                   edgecolor=DANGER, hatch="////", linewidth=.35))

    save_figure(fig, OUT, "fig01_q1_framework")


def _short_status_name(name: str) -> str:
    mapping = {
        "matched": "matched",
        "crosslingual_translation_candidate": "cross-lang",
        "mismatch_candidate": "mismatch",
        "no_speech": "no speech",
        "asr_unavailable": "ASR N/A",
        "asr_unreliable": "ASR low",
    }
    return mapping.get(name, name.replace("_", " ")[:12])


def _quality_strip(ax, table: pd.DataFrame) -> None:
    text = table.text_audio_match.fillna(False).astype(bool).to_numpy()
    audio = table.word_audio_valid_fraction.fillna(0).to_numpy(float)
    vision = table.word_vision_valid_fraction.fillna(0).to_numpy(float)
    uncertain = table.target_face_uncertain.fillna(False).astype(bool).to_numpy()

    severity = (
        (~text).astype(int)*3
        + (audio < .999).astype(int)
        + (vision < .999).astype(int)
        + uncertain.astype(int)*2
    )
    order = np.lexsort((np.arange(len(table)), -vision, -audio, -severity))
    pd.DataFrame({
        "figure_column_1based": np.arange(1, len(order)+1),
        "sample_id": table.iloc[order].sample_id.to_numpy(),
        "text_clock_trusted": text[order],
        "audio_word_valid_fraction": audio[order],
        "vision_word_valid_fraction": vision[order],
    }).to_csv(OUT / "fig02_quality_strip_order.csv", index=False, encoding="utf-8-sig")

    values = np.vstack([text.astype(float), audio, vision])[:, order]
    colors = (TEXT, AUDIO, VISION)
    rgb = np.ones((3, len(order), 3), float)
    for i, color in enumerate(colors):
        for j, value in enumerate(values[i]):
            rgb[i, j] = _mix(color, float(value))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(3), ["Text", "Audio", "Vision"])
    ticks = np.linspace(0, len(order)-1, 6).round().astype(int)
    ax.set_xticks(ticks, [str(t+1) for t in ticks])
    ax.set_xlabel("sample rank")
    panel_title(ax, "d", "100 条样本审核状态", pad=3)
    ax.tick_params(length=0, labelsize=5.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for j, original in enumerate(order):
        if not text[original]:
            ax.add_patch(Rectangle((j-.5, -.5), 1, 1, fill=False,
                                   edgecolor=DANGER, hatch="////", linewidth=.20))
        if uncertain[original]:
            ax.add_patch(Rectangle((j-.5, 1.5), 1, 1, fill=False,
                                   edgecolor=DANGER, hatch="\\\\", linewidth=.20))


def quality_overview_figure(table: pd.DataFrame) -> None:
    """Q1-2: three compact statistics above one full-dataset status strip."""
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 3.70), facecolor="white")
    gs = fig.add_gridspec(
        2, 3, left=.075, right=.985, top=.95, bottom=.105,
        height_ratios=[1.0, .66], hspace=.34, wspace=.34,
    )

    ax = fig.add_subplot(gs[0, 0])
    labels = ["Speech", "Clock", "Face"]
    vals = np.array([
        table.speech_present.fillna(False).astype(bool).mean(),
        table.text_audio_match.fillna(False).astype(bool).mean(),
        table.face_feature_valid.fillna(False).astype(bool).mean(),
    ]) * 100
    xx = np.arange(3)
    ax.bar(xx, vals, color=[AUDIO, TEXT, VISION], width=.62)
    ax.set_xticks(xx, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    panel_title(ax, "a", "关键可用率", pad=3)
    for x, v in zip(xx, vals):
        ax.text(x, min(v+2.0, 102), f"{v:.0f}", ha="center", fontsize=5.6)
    ax.grid(axis="y", color=GRID, lw=.45)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 1])
    counts = table["text_audio_status"].fillna("unknown").astype(str).value_counts()
    keys = list(counts.index)
    keys.sort(key=lambda k: (-int(counts[k]), k))
    vals_b = np.array([int(counts[k]) for k in keys])
    x = np.arange(len(keys))
    ax.bar(x, vals_b, color=[ACCENT if k == "matched" else "#A9B1B7" for k in keys], width=.68)
    ax.set_xticks(x, [_short_status_name(k) for k in keys], rotation=36, ha="right")
    ax.set_ylabel("n")
    panel_title(ax, "b", "词时钟状态", pad=3)
    for xx_, value in zip(x, vals_b):
        ax.text(xx_, value+.6, str(value), ha="center", fontsize=5.2)
    ax.grid(axis="y", color=GRID, lw=.45)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 2])
    paired = np.minimum(
        table.word_audio_valid_fraction.fillna(0).to_numpy(float),
        table.word_vision_valid_fraction.fillna(0).to_numpy(float),
    )
    bins = np.linspace(0, 1, 6)
    hist, _ = np.histogram(paired, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, hist, width=.18, color="#5F8CA8", edgecolor="white", linewidth=.35)
    ax.set_xticks(centers, ["0–20", "20–40", "40–60", "60–80", "80–100"], rotation=24)
    ax.set_xlabel("%")
    ax.set_ylabel("n")
    panel_title(ax, "c", "联合逐词覆盖", pad=3)
    ax.grid(axis="y", color=GRID, lw=.45)
    ax.set_axisbelow(True)

    _quality_strip(fig.add_subplot(gs[1, :]), table)
    save_figure(fig, OUT, "fig02_q1_quality_overview")


def representative_alignment_figure(case_id: str) -> None:
    """Q1-3: one dense alignment card with frames, waveform, words and feature heatmap."""
    meta, z = _open_case(case_id)
    count = int(meta["word_count"])
    ws = z["word_start"][:count].astype(float)
    we = z["word_end"][:count].astype(float)
    valid = np.isfinite(ws) & np.isfinite(we) & (we > ws)
    if count == 0 or not valid.all():
        raise RuntimeError(f"Q1 representative sample {case_id} lacks complete word timing")

    n = min(count, 14)
    words = list(meta["official_words"][:n])
    ws = ws[:n]
    we = we[:n]
    audio_mask = z["word_audio_mask"][:n].astype(bool)
    vision_mask = z["word_vision_mask"][:n].astype(bool)
    activities = np.vstack([
        _feature_activity(z["text_word_native"][:n]),
        _feature_activity(z["word_audio"][:n]),
        _feature_activity(z["word_vision"][:n]),
    ])
    duration = float(meta["duration"])

    pts = _read_frame_pts(meta["video_path"])
    frames = _representative_frames(
        meta["video_path"], pts, (float(ws.min()), float(we.max())), number=4
    )
    audio_path = config.OUT / "case_assets" / case_id / "audio_16k.wav"
    _, waveform = _decode_word_audio(meta["video_path"], audio_path)
    t_env, env = _wave_envelope(waveform, duration, bins=1200)

    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.08), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, left=.085, right=.985, top=.95, bottom=.105,
        height_ratios=[.92, .82, 1.05], hspace=.30,
    )

    ax = fig.add_subplot(gs[0])
    panel_title(ax, "a", "原视频帧", pad=3)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    for i, (_, t, rgb) in enumerate(frames):
        ax.imshow(_crop_16_9(rgb), extent=(i+.02, i+.98, .08, .93),
                  aspect="auto", interpolation="bilinear")
        ax.text(i+.5, .01, f"{t:.2f}s", ha="center", va="bottom", fontsize=5.2, color=MUTED)
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    if len(t_env):
        ax.fill_between(t_env, 0, env, color="#7F8B93", alpha=.78, linewidth=0)
    for i, (start, end, word) in enumerate(zip(ws, we, words)):
        y = 1.03 + .14*(i % 2)
        ax.plot([start, end], [y, y], color=OCHRE, lw=3.0, solid_capstyle="butt")
        ax.text((start+end)/2, y+.035, word, ha="center", va="bottom",
                fontsize=4.9, rotation=34)
    for start, end in meta.get("speech_segments", []):
        ax.axvspan(float(start), float(end), color=AUDIO, alpha=.07, linewidth=0)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.40)
    ax.set_yticks([])
    ax.set_xlabel("time / s")
    panel_title(ax, "b", "语音与词边界", pad=3)
    ax.grid(axis="x", color=GRID, lw=.42)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = fig.add_subplot(gs[2])
    rows = [
        (2.0, TEXT, activities[0], np.ones(n, dtype=bool)),
        (1.0, AUDIO, activities[1], audio_mask),
        (0.0, VISION, activities[2], vision_mask),
    ]
    for y, color, values, mask in rows:
        for start, end, value, ok in zip(ws, we, values, mask):
            ax.add_patch(Rectangle(
                (float(start), y-.38), float(end-start), .76,
                facecolor=_mix(color, float(value)), edgecolor="white", linewidth=.35
            ))
            if not ok:
                ax.add_patch(Rectangle(
                    (float(start), y-.38), float(end-start), .76, fill=False,
                    edgecolor=DANGER, hatch="////", linewidth=.30
                ))
    ax.set_xlim(0, duration)
    ax.set_ylim(-.55, 2.55)
    ax.set_yticks([2, 1, 0], ["Text", "Audio", "Vision"])
    ax.set_xlabel("time / s")
    panel_title(ax, "c", "三模态逐词特征活动", pad=3)
    ax.grid(axis="x", color=GRID, lw=.42)
    ax.tick_params(length=0, labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_figure(fig, OUT, "fig03_q1_representative_alignment")


def _interval_mask_on_bins(intervals, duration: float, bins: int) -> np.ndarray:
    centers = (np.arange(bins) + .5) * duration / bins
    mask = np.zeros(bins, dtype=bool)
    for a, b in intervals:
        mask |= (centers >= float(a)) & (centers < float(b))
    return mask


def _sample_case_states(meta: dict, z: dict, bins: int = 44) -> tuple[np.ndarray, np.ndarray]:
    duration = max(float(meta["duration"]), .5)
    speech = _interval_mask_on_bins(meta.get("speech_segments", []), duration, bins)

    vt = np.asarray(z["vision_time"], float)
    face_valid = np.asarray(z["vision_valid_mask"], bool)
    multiface = np.asarray(z["vision_multiface_mask"], bool)
    centers = (np.arange(bins) + .5) * duration / bins
    face = np.zeros(bins, dtype=bool)
    multi = np.zeros(bins, dtype=bool)
    if len(vt):
        nearest = np.abs(vt[:, None] - centers[None, :]).argmin(axis=0)
        face = face_valid[nearest]
        multi = multiface[nearest]

    ws = np.asarray(z["word_start"], float)
    we = np.asarray(z["word_end"], float)
    word_intervals = [(a, b) for a, b in zip(ws, we)
                      if np.isfinite(a) and np.isfinite(b) and b > a]
    clock = _interval_mask_on_bins(word_intervals, duration, bins)

    states = np.vstack([speech, face, clock]).astype(int)
    flags = np.zeros_like(states, dtype=int)
    flags[1, multi] = 1
    return states, flags


def quality_control_figure(cases: dict[str, str]) -> None:
    """Q1-4: four compact source-evidence cards with status strips, not prose."""
    kinds = [
        ("no_face", "无目标脸"),
        ("no_speech", "无真人语音"),
        ("multiple_faces", "多人脸"),
        ("mismatch", "文本不符"),
    ]
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.18), facecolor="white")
    outer = fig.add_gridspec(
        2, 2, left=.065, right=.985, top=.94, bottom=.10,
        hspace=.30, wspace=.18
    )

    for index, (kind, title) in enumerate(kinds):
        row, col = divmod(index, 2)
        sub = outer[row, col].subgridspec(2, 1, height_ratios=[1.15, .42], hspace=.05)
        img_ax = fig.add_subplot(sub[0])
        strip_ax = fig.add_subplot(sub[1])

        sid = cases[kind]
        meta, z = _open_case(sid)
        _, t, frame = _source_frame(meta, z, kind)

        img_ax.imshow(_crop_16_9(frame), aspect="auto")
        img_ax.axis("off")
        panel_title(img_ax, chr(ord("a")+index), title, pad=2)
        img_ax.text(.015, .035, f"{t:.1f}s", transform=img_ax.transAxes,
                    fontsize=5.0, color="white",
                    bbox=dict(facecolor="#25343C", alpha=.65, edgecolor="none", pad=1.2))

        states, flags = _sample_case_states(meta, z)
        colors = (AUDIO, VISION, TEXT)
        rgb = np.ones((3, states.shape[1], 3), float)
        for r, color in enumerate(colors):
            for j, value in enumerate(states[r]):
                rgb[r, j] = _mix(color, .82 if value else .01)

        strip_ax.imshow(rgb, aspect="auto", interpolation="nearest")
        strip_ax.set_yticks(range(3), ["Speech", "Face", "Clock"])
        strip_ax.set_xticks([])
        strip_ax.tick_params(length=0, labelsize=5.3)
        for spine in strip_ax.spines.values():
            spine.set_visible(False)

        for j in np.flatnonzero(flags[1]):
            strip_ax.add_patch(Rectangle((j-.5, .5), 1, 1, fill=False,
                                         edgecolor=DANGER, hatch="////", linewidth=.22))

    legend = [
        Patch(facecolor="#697780", edgecolor="none", label="深色=有效"),
        Patch(facecolor="white", edgecolor=DANGER, hatch="////", label="斜线=歧义"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.76, .995), fontsize=5.5)
    save_figure(fig, OUT, "fig04_q1_quality_control")


def write_manifest() -> None:
    payload = {
        "version": "v4_compact_chart_first",
        "main_text_figures": {
            "fig01_q1_framework": "real sample + feature heat strip + feature dimensions + word-level masks",
            "fig02_q1_quality_overview": "three compact dataset statistics + full 100-sample audit strip",
            "fig03_q1_representative_alignment": "frames + waveform/word clock + source-time feature heatmap",
            "fig04_q1_quality_control": "four compact complex-scene cards with three-row status strips",
        },
        "layout_notes": [
            "Main-text figures prioritize charts, heatmaps, time strips and source frames over explanatory prose.",
            "Large empty side annotations are removed; explanations belong in the manuscript caption.",
            "Semantic colors remain fixed across Text, Audio and Vision.",
            "Hatching marks rejected or ambiguous evidence; saved numeric tables remain the audit source.",
            "Experiment values, features and alignment rules are unchanged.",
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
    print(f"Wrote compact chart-first Q1 figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
