"""Shared publication-quality plotting helpers for Q1/Q2/Q3.

The original audit figures remain untouched.  The helpers in this module are
used by the dedicated ``visualization.paper`` modules to generate concise main-
text figures from already-saved experiment outputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TEXT = "#2F6B8A"
AUDIO = "#C9822B"
VISION = "#4F8F73"
MODALITY_COLORS = {"text": TEXT, "audio": AUDIO, "vision": VISION}
INK = "#24323D"
MUTED = "#6E7B85"
GRID = "#DDE3E8"
NEUTRAL = "#8B959D"
NEGATIVE = "#B75D57"
POSITIVE = "#397D9F"
BG = "#FFFFFF"


def apply_publication_style() -> None:
    """Apply a compact, journal-friendly Matplotlib style."""
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8C98A1",
        "axes.linewidth": 0.65,
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
    })


def save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax, label: str, title: str | None = None) -> None:
    text = f"({label})"
    if title:
        text += f"  {title}"
    ax.set_title(text, loc="left", fontweight="bold", pad=6)


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.55, alpha=0.9)


def symmetric_limit(values, minimum: float = 0.01, quantile: float | None = None) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return minimum
    bound = np.quantile(np.abs(arr), quantile) if quantile is not None else np.max(np.abs(arr))
    return max(minimum, float(bound))


def add_heatmap_values(ax, matrix, *, fmt="{:+.3f}", threshold=0.55, limit=None, fontsize=6.2):
    matrix = np.asarray(matrix, dtype=float)
    if limit is None:
        limit = symmetric_limit(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if abs(value) >= threshold * limit else INK
            ax.text(j, i, fmt.format(value), ha="center", va="center",
                    fontsize=fontsize, color=color)
