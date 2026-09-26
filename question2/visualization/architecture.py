"""Draw Q2 model/missingness diagrams for main text and supplementary material."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import torch

from common.paths import Q2_OUTPUT_DIR
from common.plotting import (
    ACCENT, DANGER, DOUBLE_COLUMN, EDGE, INK, LIGHT, MODALITY_COLORS, MUTED,
    OCHRE, apply_paper_style, panel_title, save_figure,
)
from question2.config import DATA
from question2.data.dataset import FeatureStore
from question2.data.missingness import hide_batch


OUT = Q2_OUTPUT_DIR / "figures"
PAPER_OUT = Q2_OUTPUT_DIR / "paper_figures"
TEXT = MODALITY_COLORS["text"]
AUDIO = MODALITY_COLORS["audio"]
VISION = MODALITY_COLORS["vision"]
PURPLE = "#75638A"


def box(ax, x, y, w, h, text, color, fontsize=8, dashed=False, fill="white"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.045",
        facecolor=fill, edgecolor=color, linewidth=.9,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, fontweight="medium")
    return rect


def arrow(ax, start, end, color=EDGE, style="-|>", width=.9):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=9,
        linewidth=width, color=color, shrinkA=2, shrinkB=2,
    ))


def save(fig, name):
    apply_paper_style(chinese=False)
    save_figure(fig, OUT, name)


def _selected_variant_note() -> str:
    selected = Q2_OUTPUT_DIR / "analysis" / "selected_model.json"
    if not selected.exists():
        return "最终候选由预设验证规则选择"
    data = json.loads(selected.read_text(encoding="utf-8"))
    value = data.get("selected_variant", "")
    if value == "mrrf_no_recon":
        return "验证选择：no-reconstruction variant"
    if value == "mrrf_full":
        return "验证选择：full MRRF variant"
    return f"验证选择：{value or 'candidate model'}"


def _draw_architecture(ax, *, main_text=False):
    """Draw architecture with reconstruction explicitly optional."""
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.0); ax.axis("off")
    colors = (TEXT, AUDIO, VISION)
    streams = ((4.55, "Text", 768), (3.10, "Audio", 74), (1.65, "Vision", 35))

    box(ax, .20, 2.72, 1.25, 1.15, "Availability\n+ masks", PURPLE, fontsize=7.5,
        fill="#F8F7FA")
    for (y, name, dim), color in zip(streams, colors):
        box(ax, 1.95, y, 1.40, .70, f"{name}\n50 × {dim}", color, fontsize=7.2)
        arrow(ax, (1.48, 3.30), (1.90, y+.35), color=color)
        box(ax, 3.90, y, 1.65, .70, f"MulT\n{name} stream", color, fontsize=7.1)
        arrow(ax, (3.40, y+.35), (3.85, y+.35), color=color)
        arrow(ax, (5.60, y+.35), (7.35, 3.30), color=color)

    # Optional training-only auxiliary branch. It is intentionally not drawn
    # in the main inference path so the framework remains compatible with the
    # validation-selected no-reconstruction candidate.
    box(ax, 5.88, .42, 2.05, .82,
        "Optional auxiliary\nreconstruction heads", OCHRE, fontsize=7.0,
        dashed=True, fill="#FCFAF5")
    for y, _, _ in streams:
        arrow(ax, (4.75, y), (6.55, 1.28), color=OCHRE, style="->", width=.65)
    ax.text(6.90, .18, "training auxiliary; removable at candidate level",
            ha="center", fontsize=5.9, color=MUTED)

    box(ax, 7.42, 2.62, 1.72, 1.35,
        "Reliability gate\n+ adaptive fusion", PURPLE, fontsize=7.5,
        fill="#F8F7FA")
    ax.text(8.28, 4.22, "features + observed fraction", ha="center",
            fontsize=5.9, color=MUTED)
    box(ax, 9.80, 3.78, 1.55, .70, "3-class polarity", TEXT, fontsize=7.1)
    box(ax, 9.80, 2.10, 1.55, .70, "Strength [-3,3]", VISION, fontsize=7.1)
    arrow(ax, (9.18, 3.30), (9.75, 4.13), color=PURPLE)
    arrow(ax, (9.18, 3.30), (9.75, 2.45), color=PURPLE)

    ax.text(11.55, 3.30, "joint\noutput", ha="center", va="center",
            fontsize=6.2, color=MUTED)
    ax.text(.20, 5.60,
            "局部连续缺失 → 可用性显式建模 → 可靠性自适应融合 → 分类 + 回归",
            fontsize=8.2 if main_text else 8.7, fontweight="semibold", color=INK)
    ax.text(.20, 5.25,
            "虚线重建支路是可选训练辅助模块；因此 full 与 no-reconstruction 共用同一主干框架。",
            fontsize=6.3, color=MUTED)
    ax.text(9.45, .62, _selected_variant_note(), fontsize=6.2, color=ACCENT,
            fontweight="semibold")


def _mask_arrays():
    store = FeatureStore(DATA)
    valid = store.splits["valid"]
    choice = next(
        i for i in range(len(valid["id"]))
        if 18 <= valid["content"][i].sum() <= 40
        and 0 < (valid["content"][i] & ~valid["observed_vision"][i]).sum() < 5
    )
    item = store.dataset("valid")[choice]
    original = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in item.items()
    }
    masked = hide_batch(
        original, np.random.default_rng(20260923),
        scenario="audio+vision", rate=.3, position="middle",
    )
    content = original["content"][0].numpy()
    before, after = [], []
    for name in ("text", "audio", "vision"):
        observed = content if name == "text" else original[f"observed_{name}"][0].numpy()
        injected = masked[f"injected_{name}"][0].numpy()
        raw = np.zeros(50, dtype=np.int8)
        raw[content] = 2                    # content but original all-zero
        raw[observed] = 1                  # truly observed
        changed = raw.copy()
        changed[injected] = 3              # synthetic contiguous missing
        before.append(raw); after.append(changed)
    return valid["id"][choice], int(content.sum()), np.stack(before), np.stack(after)


def _draw_mask(ax, matrix, title, *, show_xlabel=False):
    cmap = ListedColormap((LIGHT, "#DCEAE7", OCHRE, DANGER))
    norm = BoundaryNorm((-.5, .5, 1.5, 2.5, 3.5), cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_yticks((0, 1, 2), ("Text", "Audio", "Vision"))
    ax.set_xticks((0, 9, 19, 29, 39, 49), ("1", "10", "20", "30", "40", "50"))
    ax.set_title(title, loc="left", fontsize=7.2, pad=3)
    ax.tick_params(length=0, labelsize=6.2)
    if show_xlabel:
        ax.set_xlabel("Aligned sequence position", fontsize=6.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def architecture():
    """Legacy standalone architecture, kept as supplementary output."""
    apply_paper_style(chinese=False)
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _draw_architecture(ax)
    save_figure(fig, OUT, "fig13_mrrf_architecture")


def mask_example():
    """Legacy standalone traceable contiguous-mask example."""
    sid, length, before, after = _mask_arrays()
    apply_paper_style(chinese=False)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 3.25), sharex=True)
    _draw_mask(axes[0], before, "(a) Before synthetic masking")
    _draw_mask(axes[1], after, "(b) A+V 30% middle-interval masking", show_xlabel=True)
    fig.suptitle(f"Traceable validation example: {sid} | content length {length}",
                 x=.12, ha="left", fontsize=9.5, fontweight="bold")
    legend = [Patch(facecolor=color, label=label) for color, label in (
        (LIGHT, "Special token / padding"), ("#DCEAE7", "Observed"),
        (OCHRE, "Original all-zero"), (DANGER, "Injected missing"),
    )]
    fig.legend(handles=legend, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, -.02), fontsize=7)
    fig.tight_layout(rect=(0, .11, 1, .91), h_pad=.65)
    save_figure(fig, OUT, "fig14_contiguous_mask_example")


def paper_framework():
    """Q2-1 main-text figure: missingness construction + compatible MRRF framework."""
    sid, length, before, after = _mask_arrays()
    apply_paper_style(chinese=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.58, 1.0])

    ax = fig.add_subplot(gs[0])
    _draw_architecture(ax, main_text=True)
    panel_title(ax, "a", "MRRF 总体框架与可选重建支路", pad=2)

    lower = gs[1].subgridspec(1, 2, wspace=.08)
    ax1 = fig.add_subplot(lower[0]); ax2 = fig.add_subplot(lower[1])
    _draw_mask(ax1, before, "(b) 注入缺失前")
    _draw_mask(ax2, after, "(c) A+V 30% 中段连续缺失", show_xlabel=True)
    ax1.set_xlabel("Aligned sequence position", fontsize=6.5)

    legend = [Patch(facecolor=color, label=label) for color, label in (
        (LIGHT, "Padding/special"), ("#DCEAE7", "Observed"),
        (OCHRE, "Original all-zero"), (DANGER, "Injected missing"),
    )]
    fig.legend(handles=legend, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, -.025), fontsize=6.4)
    fig.text(.02, .01,
             f"示例 {sid}，有效正文长度 {length}；原生全零与人工连续遮挡分别记录，避免把数据固有缺失误当作实验注入缺失。",
             fontsize=5.9, color=MUTED)
    save_figure(fig, PAPER_OUT, "fig01_q2_framework")


def main():
    architecture()
    mask_example()
    paper_framework()
    print(f"Wrote Q2 architecture/mask figures to {OUT} and {PAPER_OUT}", flush=True)


if __name__ == "__main__":
    main()
