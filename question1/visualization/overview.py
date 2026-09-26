"""Main-text Q1 overview figures: method framework and full-dataset quality audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from common.plotting import (
    ACCENT, BG, DOUBLE_COLUMN, EDGE, GRID, INK, LIGHT, MODALITY_COLORS, MUTED,
    OCHRE, apply_paper_style, panel_title, save_figure,
)
from .. import config


def _box(ax, xy, width, height, title, lines, *, color, fill_alpha=.10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=.9, edgecolor=color, facecolor=color, alpha=fill_alpha,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + .018, y + height - .045, title, transform=ax.transAxes,
            fontsize=8.2, fontweight="semibold", color=color, va="top")
    ax.text(x + .018, y + height - .092, "\n".join(lines), transform=ax.transAxes,
            fontsize=6.8, color=INK, va="top", linespacing=1.38)
    return patch


def _arrow(ax, start, end, *, color=EDGE, lw=.9):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=1, shrinkB=1,
    ))


def create_framework_figure(out: Path) -> Path:
    """Create the compact Q1 processing/alignment framework used in the main text."""
    apply_paper_style(chinese=True)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 4.25))
    fig.subplots_adjust(left=.03, right=.985, top=.94, bottom=.08)
    ax.set_axis_off()

    # Stage labels are intentionally short; the manuscript caption explains details.
    stage_y = .94
    for x, text in [(0.035, "输入"), (0.245, "三模态特征提取"),
                    (0.585, "词级时间锚定与聚合"), (0.835, "可核验输出")]:
        ax.text(x, stage_y, text, transform=ax.transAxes, fontsize=7.2,
                fontweight="semibold", color=MUTED, va="center")

    _box(ax, (.03, .40), .16, .34, "原始素材",
         ["MP4 视频", "官方英文转写", "连续情感标签", "video_id + clip_id"],
         color=ACCENT, fill_alpha=.08)

    text_c = MODALITY_COLORS["text"]
    audio_c = MODALITY_COLORS["audio"]
    vision_c = MODALITY_COLORS["vision"]
    _box(ax, (.235, .68), .265, .20, "Text",
         ["官方词序列 → WordPiece", "BERT 词向量均值 → 768-D"],
         color=text_c)
    _box(ax, (.235, .415), .265, .20, "Audio",
         ["音轨 → Silero VAD + WhisperX", "eGeMAPS LLD 聚合 → 25-D"],
         color=audio_c)
    _box(ax, (.235, .15), .265, .20, "Vision",
         [f"视频帧 @ {config.FPS:g} fps → 目标脸筛选", "MediaPipe blendshape → 52-D"],
         color=vision_c)

    _box(ax, (.555, .30), .235, .43, "共享词级时间轴",
         ["以官方词序列作为主轴", "接受真实发音区间  [sᵢ, eᵢ)",
          "Audio: 区间内帧均值", "Vision: 时间交叠加权均值",
          "无可靠证据 → 0 + validity mask"],
         color=OCHRE, fill_alpha=.09)

    _box(ax, (.825, .48), .15, .27, "标准化序列",
         ["Tᵢ ∈ R⁷⁶⁸", "Aᵢ ∈ R²⁵", "Vᵢ ∈ R⁵²", f"pad/truncate → {config.MAX_WORDS}"],
         color=ACCENT, fill_alpha=.08)
    _box(ax, (.825, .16), .15, .23, "审计材料",
         ["NPZ 特征", "样本日志", "逐词映射表", "质量标记 / mask"],
         color=ACCENT, fill_alpha=.08)

    _arrow(ax, (.19, .57), (.235, .78))
    _arrow(ax, (.19, .57), (.235, .515))
    _arrow(ax, (.19, .57), (.235, .25))
    _arrow(ax, (.50, .78), (.555, .61), color=text_c)
    _arrow(ax, (.50, .515), (.555, .515), color=audio_c)
    _arrow(ax, (.50, .25), (.555, .40), color=vision_c)
    _arrow(ax, (.79, .57), (.825, .615))
    _arrow(ax, (.79, .44), (.825, .275))

    # Tiny shared-time-axis sketch makes the cross-rate alignment visually explicit.
    ax.text(.555, .245, "word", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    words = [("I", .61, .645), ("think", .65, .695), ("social", .70, .745),
             ("justice", .75, .79)]
    for word, a, b in words:
        ax.plot([a, b], [.255, .255], transform=ax.transAxes, color=OCHRE,
                lw=4.1, solid_capstyle="butt")
        ax.text((a+b)/2, .225, word, transform=ax.transAxes, fontsize=5.4,
                ha="center", color=INK)
    ax.text(.555, .195, "audio", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    ax.plot([.61, .79], [.205, .205], transform=ax.transAxes, color=audio_c, lw=2.0)
    ax.text(.555, .155, "video", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    for x in np.linspace(.61, .79, 9):
        ax.plot(x, .165, marker="o", ms=2.4, color=vision_c, transform=ax.transAxes)

    path = save_figure(fig, out, "fig01_q1_framework")[0]
    return path


def _quality_events(table: pd.DataFrame):
    events = np.column_stack([
        ~table.speech_present.astype(bool),
        ~table.face_present.astype(bool),
        ~table.face_feature_valid.astype(bool),
        ~table.text_audio_match.astype(bool),
        table.text_audio_status.eq("crosslingual_translation_candidate"),
        table.target_face_uncertain.astype(bool),
    ]).astype(int)
    names = ["无人声", "无人脸", "无有效表情", "无可信英语词对齐",
             "跨语译文候选", "目标人脸不确定"]
    severity = events.sum(axis=1)
    order = np.lexsort((np.arange(len(table)),
                        *tuple(-events[:, j] for j in reversed(range(events.shape[1]))),
                        -severity))
    return events, names, severity, order


def _availability_panel(ax, table: pd.DataFrame):
    rows = [
        ("可解码音轨", "audio_stream_present", MUTED),
        ("真人语音", "speech_present", MODALITY_COLORS["audio"]),
        ("可靠英语词时间", "text_audio_match", MODALITY_COLORS["audio"]),
        ("检测到人脸", "face_present", MODALITY_COLORS["vision"]),
        ("有效表情特征", "face_feature_valid", MODALITY_COLORS["vision"]),
    ]
    y = np.arange(len(rows))
    values = [int(table[col].fillna(False).astype(bool).sum()) for _, col, _ in rows]
    ax.barh(y, values, color=[color for _, _, color in rows], height=.54, edgecolor="none")
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.invert_yaxis(); ax.set_xlim(0, max(104, len(table)+4))
    ax.set_xlabel("样本数")
    panel_title(ax, "a", f"模态与对齐可用性（n={len(table)}）")
    for yi, value in zip(y, values):
        ax.text(value + 1.0, yi, str(value), va="center", fontsize=7.2, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.set_axisbelow(True)


def _matrix_panel(ax, table: pd.DataFrame):
    events, names, severity, order = _quality_events(table)
    shown = events[order].T
    pd.DataFrame({
        "figure_column_1based": np.arange(1, len(order)+1),
        "sample_id": table.iloc[order].sample_id.to_numpy(),
        "quality_event_count": severity[order],
    }).to_csv(config.OUT / "tables/06_status_matrix_order.csv", index=False)

    ax.pcolormesh(np.arange(len(order)+1), np.arange(len(names)+1), shown,
                  cmap=ListedColormap([BG, ACCENT]), vmin=0, vmax=1,
                  edgecolors="none", rasterized=True)
    ax.set_yticks(np.arange(len(names)) + .5, names)
    ax.invert_yaxis()
    ax.set_xticks([.5, 19.5, 39.5, 59.5, 79.5, 99.5], ["1", "20", "40", "60", "80", "100"])
    ax.set_xlim(0, len(order)+8)
    ax.set_xlabel("样本序号")
    panel_title(ax, "b", "质量事件矩阵")
    for i, n in enumerate(shown.sum(axis=1)):
        ax.text(len(order)+1.1, i+.5, f"n={int(n)}", va="center", fontsize=6.6, color=MUTED)
    ax.tick_params(length=0)


def _coverage_panel(ax, table: pd.DataFrame):
    x = table.word_audio_valid_fraction.fillna(0).to_numpy(float)
    y = table.word_vision_valid_fraction.fillna(0).to_numpy(float)
    paired = table.text_audio_match.astype(bool).to_numpy()
    uncertain = table.target_face_uncertain.astype(bool).to_numpy()

    def bubbles(mask, *, edge, face, label):
        xy = np.column_stack((x[mask], y[mask]))
        if not len(xy):
            return
        points, counts = np.unique(np.round(xy, 6), axis=0, return_counts=True)
        sizes = 14 + 10 * counts
        ax.scatter(points[:, 0], points[:, 1], s=sizes, edgecolor=edge, facecolor=face,
                   linewidth=.75, alpha=.86, label=label, zorder=3)
        for (u, v), count in zip(points, counts):
            if count >= 8:
                ax.text(u, v, str(int(count)), ha="center", va="center", fontsize=5.8,
                        color="white" if face != "none" else edge, zorder=4)

    nonzero = (x > 0) | (y > 0)
    bubbles(nonzero & paired & ~uncertain, edge=ACCENT, face=ACCENT,
            label="已接受英语词时间")
    bubbles(nonzero & (~paired | uncertain), edge="#9A665A", face="none",
            label="待核 / 目标脸不确定")
    zero = ~nonzero
    if zero.any():
        ax.scatter([0], [0], s=14 + 10*int(zero.sum()), color="#8B949C", alpha=.75, zorder=3)
        ax.text(.025, .03, f"n={int(zero.sum())}", fontsize=6.2, color=MUTED)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=.8, color="#9AA6AD", zorder=1)
    ax.set(xlim=(-.035, 1.06), ylim=(-.035, 1.06),
           xlabel="有效语音词 / 官方词", ylabel="有效视觉词 / 官方词")
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    panel_title(ax, "c", "逐词 Audio–Vision 覆盖")
    ax.grid(color=GRID, linewidth=.5); ax.set_axisbelow(True)
    status_legend = ax.legend(loc="lower right", fontsize=6.0, frameon=False)
    ax.add_artist(status_legend)

    # Bubble-size key: enough to decode area without taking over the panel.
    handles = [ax.scatter([], [], s=14+10*n, facecolor="none", edgecolor="#68747C",
                          linewidth=.7, label=f"n={n}") for n in (1, 5, 15, 30)]
    ax.legend(handles=handles, title="重合样本", loc="upper left",
              fontsize=5.8, title_fontsize=6.1, frameon=False, borderaxespad=.2)


def create_quality_overview(table: pd.DataFrame, out: Path) -> Path:
    """Create the three-panel full-dataset quality figure for Q1 main text."""
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 6.15))
    gs = fig.add_gridspec(2, 2, left=.10, right=.975, top=.965, bottom=.09,
                          height_ratios=[.83, 1.17], width_ratios=[1, 1.06],
                          hspace=.42, wspace=.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    _availability_panel(ax_a, table)
    _coverage_panel(ax_c, table)
    _matrix_panel(ax_b, table)
    fig.text(.10, .025,
             "样本按质量事件数量与类型排序；原始 sample_id 映射保存于 tables/06_status_matrix_order.csv。",
             fontsize=6.3, color=MUTED)
    path = save_figure(fig, out, "fig02_q1_quality_overview")[0]
    return path


def create_overview_figures(table: pd.DataFrame, out: Path) -> dict[str, str]:
    return {
        "framework": str(create_framework_figure(out)),
        "quality_overview": str(create_quality_overview(table, out)),
    }
