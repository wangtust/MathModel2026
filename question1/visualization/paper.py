"""Main-text Q1 figures for the paper.

The module reads already-saved Q1 audit outputs and feature files. It does not
rerun extraction. Detailed diagnostic figures remain available as supplementary
material, while this module produces the compact main-text figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from common.paths import Q1_OUTPUT_DIR
from common.publication import (
    AUDIO, GRID, INK, MUTED, TEXT, VISION,
    apply_publication_style, light_grid, panel_label, save_figure,
)

OUT = Q1_OUTPUT_DIR / "paper_figures"
TABLE = Q1_OUTPUT_DIR / "tables" / "02_sample_summary.csv"
RUN_MANIFEST = Q1_OUTPUT_DIR / "run_manifest.json"


def _bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _load_case(sid: str):
    meta_path = Q1_OUTPUT_DIR / "logs" / "samples" / f"{sid}.json"
    feature_path = Q1_OUTPUT_DIR / "features" / f"{sid}.npz"
    if not meta_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"Missing Q1 case artifacts for {sid}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(feature_path) as archive:
        feat = {k: archive[k] for k in archive.files}
    return meta, feat


def _case_ids(df: pd.DataFrame) -> dict[str, str]:
    if RUN_MANIFEST.exists():
        manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
        found = dict(manifest.get("case_ids", {}))
    else:
        found = {}

    def first(mask, fallback=None):
        rows = df[mask]
        return str(rows.iloc[0].sample_id) if len(rows) else fallback

    found.setdefault("typical", first(_bool(df.speech_present) & _bool(df.face_feature_valid) & _bool(df.text_audio_match)))
    found.setdefault("no_face", first(~_bool(df.face_present)))
    found.setdefault("no_speech", first(~_bool(df.speech_present)))
    found.setdefault("multiple_faces", first(_bool(df.target_face_uncertain)))
    found.setdefault("mismatch", first(df.text_audio_status.eq("mismatch_candidate")))
    return {k: v for k, v in found.items() if v}


def plot_pipeline() -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.15, 3.45))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, edge, face="#F8FAFB", lw=1.0, fs=7.2):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.06",
            linewidth=lw, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#74818A"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, linewidth=.8, color=color))

    box(.25, 2.65, 1.55, 1.0, "Raw video\n+ official text", "#5D6B75", face="#F2F5F7", fs=8)
    lanes = [
        ("Text", "Official words\n→ BERT", "768-D", TEXT, 5.45),
        ("Audio", "Audio track\n→ VAD + ASR\n→ eGeMAPS", "25-D", AUDIO, 3.55),
        ("Vision", "Frames\n→ face selection\n→ MediaPipe", "52-D", VISION, 1.55),
    ]
    for name, process, dim, color, y in lanes:
        arrow(1.8, 3.15, 2.35, y + .45, color)
        box(2.35, y, 1.45, .9, name, color, face="white", fs=8)
        arrow(3.8, y + .45, 4.25, y + .45, color)
        box(4.25, y - .15, 2.0, 1.2, process, color, face="#FBFCFD")
        arrow(6.25, y + .45, 6.7, y + .45, color)
        box(6.7, y, 1.0, .9, dim, color, face="white", fs=8)

    box(8.3, 2.65, 1.8, 1.0, "Shared word timeline\n$[s_i,e_i)$", "#566570", face="#F2F5F7", fs=8)
    for _, _, _, color, y in lanes:
        arrow(7.7, y + .45, 8.3, 3.15, color)
    box(10.45, 4.55, 1.25, .9, "$T_i$", TEXT, face="white", fs=9)
    box(10.45, 3.00, 1.25, .9, "$A_i$", AUDIO, face="white", fs=9)
    box(10.45, 1.45, 1.25, .9, "$V_i$", VISION, face="white", fs=9)
    for y in (5.0, 3.45, 1.9):
        arrow(10.1, 3.15, 10.45, y)

    ax.text(8.35, 1.0, "overlap pooling  →  validity masks  →  fixed-length feature file", fontsize=7.2, color="#51606A")
    ax.text(.25, 6.55, "Word-level multimodal feature extraction and temporal alignment", fontsize=10, fontweight="bold", color=INK)
    ax.text(.25, 6.15, "Model inputs retain original feature dimensions; masks record unavailable local evidence.", fontsize=7.2, color=MUTED)
    save_figure(fig, OUT, "q1_fig01_pipeline")


def plot_quality_overview() -> None:
    apply_publication_style()
    df = pd.read_csv(TABLE)
    n = len(df)
    fig = plt.figure(figsize=(7.15, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[.85, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    fields = [
        ("Audio stream", "audio_stream_present", "#71808A"),
        ("Speech", "speech_present", AUDIO),
        ("Face detected", "face_present", VISION),
        ("Valid face feat.", "face_feature_valid", VISION),
        ("Trusted word align.", "text_audio_match", TEXT),
    ]
    vals = [int(_bool(df[col]).sum()) for _, col, _ in fields]
    y = np.arange(len(fields))
    ax.barh(y, vals, color=[c for _, _, c in fields], height=.55)
    ax.set_yticks(y, [name for name, _, _ in fields]); ax.invert_yaxis(); ax.set_xlim(0, n + max(4, int(n*.08)))
    ax.set_xlabel(f"Samples (n={n})")
    for yi, value in zip(y, vals):
        ax.text(value + .8, yi, str(value), va="center", fontsize=7)
    light_grid(ax, "x"); panel_label(ax, "a", "Signal and evidence availability")

    ax = fig.add_subplot(gs[0, 1])
    events = np.column_stack([
        ~_bool(df.speech_present), ~_bool(df.face_present), ~_bool(df.face_feature_valid),
        ~_bool(df.text_audio_match), df.text_audio_status.eq("crosslingual_translation_candidate"),
        _bool(df.target_face_uncertain),
    ]).astype(int)
    labels = ["No speech", "No face", "Invalid face", "No trusted align.", "Cross-lingual", "Face uncertain"]
    severity = events.sum(axis=1)
    order = np.lexsort((np.arange(n), -severity))
    matrix = events[order].T
    cmap = ListedColormap(["#F5F7F8", "#426E68"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)), labels)
    ticks = np.linspace(0, n - 1, min(6, n), dtype=int)
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.set_xlabel("Samples ordered by number of quality events")
    for i, count in enumerate(matrix.sum(axis=1)):
        ax.text(n + 1, i, f"n={count}", va="center", fontsize=6.7, color="#4F5E67")
    ax.set_xlim(-.5, n + 9); panel_label(ax, "b", "Quality-control events")

    ax = fig.add_subplot(gs[1, :])
    x = df.word_audio_valid_fraction.fillna(0).to_numpy(float)
    yv = df.word_vision_valid_fraction.fillna(0).to_numpy(float)
    trusted = _bool(df.text_audio_match).to_numpy() & ~_bool(df.target_face_uncertain).to_numpy()

    def bubble(mask, edge, face, label):
        xy = np.column_stack([x[mask], yv[mask]])
        if not len(xy):
            return
        points, counts = np.unique(np.round(xy, 5), axis=0, return_counts=True)
        sizes = 22 + 10 * counts
        ax.scatter(points[:, 0], points[:, 1], s=sizes, edgecolor=edge, facecolor=face, linewidth=.75, alpha=.85, label=label)
        for (px, py), count in zip(points, counts):
            if count >= 4:
                ax.text(px, py, str(count), ha="center", va="center", fontsize=6.2, color="white" if face != "none" else edge)

    bubble(trusted & ((x > 0) | (yv > 0)), "#2E6D67", "#4D8A82", "Trusted word timing")
    bubble((~trusted) & ((x > 0) | (yv > 0)), "#A16055", "none", "Needs review")
    zero = (x == 0) & (yv == 0)
    if zero.any():
        ax.scatter([0], [0], s=22 + 10 * int(zero.sum()), color="#89949C", alpha=.8, edgecolor="white", linewidth=.6, label="Zero A/V coverage")
        ax.text(0, 0, str(int(zero.sum())), ha="center", va="center", fontsize=6.2, color="white")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8F9AA2", linewidth=.8, label="$y=x$")
    ax.set_xlim(-.04, 1.04); ax.set_ylim(-.04, 1.04)
    ax.set_xlabel("Valid audio words / official words"); ax.set_ylabel("Valid face words / official words")
    ax.legend(frameon=False, ncol=2, loc="lower right"); ax.grid(color=GRID, linewidth=.5)
    panel_label(ax, "c", "Word-level multimodal coverage")
    save_figure(fig, OUT, "q1_fig02_quality_overview")


def plot_typical_alignment() -> None:
    apply_publication_style()
    df = pd.read_csv(TABLE)
    ids = _case_ids(df)
    sid = ids.get("typical")
    if not sid:
        return
    meta, z = _load_case(sid)
    count = int(meta.get("word_count", np.sum(z["word_valid"])))
    starts = np.asarray(z["word_start"][:count], float)
    ends = np.asarray(z["word_end"][:count], float)
    words = list(meta.get("official_words", []))[:count]
    valid_time = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    audio_mask = np.asarray(z["word_audio_mask"][:count]).astype(bool)
    vision_mask = np.asarray(z["word_vision_mask"][:count]).astype(bool)

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 4.6), gridspec_kw={"height_ratios": [1.45, 1]}, constrained_layout=True)
    ax = axes[0]
    lanes = {"Text": 2, "Audio": 1, "Vision": 0}
    for name, lane, color, mask in (
        ("Text", lanes["Text"], TEXT, valid_time),
        ("Audio", lanes["Audio"], AUDIO, valid_time & audio_mask),
        ("Vision", lanes["Vision"], VISION, valid_time & vision_mask),
    ):
        for i in np.flatnonzero(mask):
            ax.broken_barh([(starts[i], ends[i] - starts[i])], (lane - .28, .56), facecolors=color, edgecolors="white", linewidth=.3)
        ax.text(-.01, lane, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color=color, fontweight="bold")
    for i in np.flatnonzero(valid_time):
        if i < len(words) and (count <= 20 or i % max(1, count // 14) == 0):
            label = str(words[i])
            if len(label) > 10:
                label = label[:9] + "…"
            ax.text((starts[i] + ends[i]) / 2, 2.37, label, rotation=45, ha="left", va="bottom", fontsize=5.7, color=INK)
    ax.set_ylim(-.55, 2.9); ax.set_yticks([]); ax.set_xlabel("Original clip time (s)")
    light_grid(ax, "x"); panel_label(ax, "a", f"Representative word-level alignment | {sid}")

    ax = axes[1]
    idx = np.arange(1, count + 1)
    for key, mask, color, label in (
        ("word_text", np.ones(count, bool), TEXT, "Text"),
        ("word_audio", audio_mask, AUDIO, "Audio"),
        ("word_vision", vision_mask, VISION, "Vision"),
    ):
        values = np.linalg.norm(np.asarray(z[key][:count], float), axis=1)
        finite = np.isfinite(values) & mask
        scaled = np.full(count, np.nan)
        if finite.any():
            lo, hi = np.nanpercentile(values[finite], [5, 95])
            denom = hi - lo if hi > lo else max(abs(hi), 1.0)
            scaled[finite] = np.clip((values[finite] - lo) / denom, 0, 1)
        ax.plot(idx, scaled, marker="o", ms=2.5, color=color, label=label)
    ax.set_xlim(.5, count + .5); ax.set_ylim(-.05, 1.05)
    ax.set_xlabel("Official word index"); ax.set_ylabel("Normalized feature magnitude")
    ax.legend(frameon=False, ncol=3); light_grid(ax, "y"); panel_label(ax, "b", "Low-dimensional visual summary of aligned features")
    fig.text(.01, .005, "Feature magnitudes are normalized only for visualization; the model retains the original 768/25/52-D vectors.", fontsize=6.3, color=MUTED)
    save_figure(fig, OUT, "q1_fig03_typical_alignment")


def plot_complex_scene_audit() -> None:
    apply_publication_style()
    df = pd.read_csv(TABLE)
    ids = _case_ids(df)
    wanted = [
        ("no_face", "No target face"),
        ("no_speech", "No human speech"),
        ("multiple_faces", "Multiple / uncertain faces"),
        ("mismatch", "Text–audio mismatch"),
    ]
    available = [(kind, title, ids.get(kind)) for kind, title in wanted if ids.get(kind)]
    if not available:
        return

    from question1.visualization.journal_cases import _open_case, _source_frame, _timeline

    fig = plt.figure(figsize=(7.15, 5.45), constrained_layout=True)
    outer = fig.add_gridspec(2, 2)
    for k, (kind, title, sid) in enumerate(available[:4]):
        sub = outer[k // 2, k % 2].subgridspec(1, 2, width_ratios=[.8, 1.25], wspace=.08)
        image_ax = fig.add_subplot(sub[0, 0]); time_ax = fig.add_subplot(sub[0, 1])
        meta, z = _open_case(sid)
        source_kind = "multiple_faces" if kind == "multiple_faces" else ("no_speech" if kind == "no_speech" else ("mismatch" if kind == "mismatch" else "typical"))
        try:
            _, t, frame = _source_frame(meta, z, source_kind)
            image_ax.imshow(frame); image_ax.set_title(f"({chr(97+k)}) {title}", loc="left", fontsize=8, fontweight="bold")
            image_ax.text(.02, -.05, f"{sid}\nframe @ {t:.2f} s", transform=image_ax.transAxes, va="top", fontsize=5.9, color=MUTED)
        except Exception:
            image_ax.text(.5, .5, "Frame unavailable", ha="center", va="center", fontsize=7, color=MUTED)
            image_ax.set_title(f"({chr(97+k)}) {title}", loc="left", fontsize=8, fontweight="bold")
        image_ax.axis("off")
        _timeline(time_ax, meta, z, compact=True)
        time_ax.set_title("Audited modality timeline", loc="left", fontsize=6.7)
    save_figure(fig, OUT, "q1_fig04_complex_scene_audit")


def generate_all() -> None:
    if not TABLE.exists():
        raise FileNotFoundError(f"Missing Q1 summary table: {TABLE}")
    plot_pipeline()
    plot_quality_overview()
    plot_typical_alignment()
    plot_complex_scene_audit()
    print(f"Q1 main-text figures written to {OUT}")


if __name__ == "__main__":
    generate_all()
