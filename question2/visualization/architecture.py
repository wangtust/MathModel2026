"""Draw the implemented MRRF structure and a traceable contiguous-mask example."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import torch

from question2.config import DATA
from question2.data.dataset import FeatureStore
from question2.data.missingness import hide_batch


from common.paths import Q2_OUTPUT_DIR
OUT = Q2_OUTPUT_DIR / "figures"
BLUE, ORANGE, TEAL, PURPLE = "#356779", "#a77a47", "#4f827e", "#726c82"


def box(ax, x, y, w, h, text, color, fontsize=9, dashed=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045,rounding_size=0.055",
                          facecolor="#eef3f2" if dashed else "#fafbfa",
                          edgecolor=color, linewidth=.9,
                          linestyle="--" if dashed else "-")
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize,
            color="#253841", fontweight="medium")


def arrow(ax, start, end, color="#595959", style="-|>", width=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10,
                                 linewidth=width, color=color, shrinkA=2, shrinkB=2))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.text(.15, 5.76, "MRRF-Net: masked reconstruction and reliability fusion",
            fontsize=10.5, fontweight="semibold", color="#26363d")
    positions = ((4.57, BLUE, "Text", 768), (3.10, ORANGE, "Audio", 74),
                 (1.63, TEAL, "Vision", 35))
    for y, color, name, dim in positions:
        box(ax, .20, y, 1.22, .75, f"{name}\n50 × {dim}", color)
        arrow(ax, (1.47, y+.375), (2.05, 3.36), color=color)
        box(ax, 3.55, y, 1.75, .75, f"MMSA MulT\n{name} stream", color, fontsize=8.4)
        arrow(ax, (3.13, 3.36), (3.49, y+.375), color=color)
        box(ax, 5.88, y, 1.60, .75, "Reconstruct\n& repair", color, fontsize=8.3,
            dashed=True)
        arrow(ax, (5.35, y+.375), (5.82, y+.375), color=color)
        arrow(ax, (7.54, y+.375), (8.18, 3.36), color=color)
    box(ax, 2.10, 2.75, 1.00, 1.22, "Mask +\navailability", PURPLE, fontsize=8.1)
    box(ax, 8.23, 2.66, 1.55, 1.36, "Reliability\ngate +\nweighted sum", PURPLE, fontsize=9)
    ax.text(9.02, 4.32, "Observed fraction\n+ features", ha="center",
            va="center", fontsize=7.0, color="#555555", linespacing=1.1)
    box(ax, 10.49, 3.86, 1.45, .76, "3-class\nsoftmax", BLUE, fontsize=8.7)
    box(ax, 10.49, 2.20, 1.45, .76, "Strength\n[-3, 3]", TEAL, fontsize=8.7)
    arrow(ax, (9.84, 3.35), (10.43, 4.24), color=PURPLE)
    arrow(ax, (9.84, 3.35), (10.43, 2.58), color=PURPLE)
    ax.text(6.68, .63, "Auxiliary loss on injected, originally observed positions only",
            ha="center", fontsize=8, color="#555555")
    ax.text(6.68, .30, "Training: class + strength + reconstruction + masked/clean consistency",
            ha="center", fontsize=8, color="#555555")
    ax.text(11.22, 1.21, "Negative / Neutral / Positive\nand coherent intensity",
            ha="center", fontsize=7.5, color="#555555")
    save(fig, "fig13_mrrf_architecture")


def mask_example():
    store = FeatureStore(DATA)
    valid = store.splits["valid"]
    choice = next(i for i in range(len(valid["id"]))
                  if 18 <= valid["content"][i].sum() <= 40
                  and 0 < (valid["content"][i] & ~valid["observed_vision"][i]).sum() < 5)
    item = store.dataset("valid")[choice]
    original = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                for key, value in item.items()}
    masked = hide_batch(original, np.random.default_rng(20260923),
                        scenario="audio+vision", rate=.3, position="middle")
    content = original["content"][0].numpy()
    before, after = [], []
    for name in ("text", "audio", "vision"):
        observed = content if name == "text" else original[f"observed_{name}"][0].numpy()
        injected = masked[f"injected_{name}"][0].numpy()
        raw = np.zeros(50, dtype=np.int8)
        raw[content] = 2
        raw[observed] = 1
        changed = raw.copy()
        changed[injected] = 3
        before.append(raw)
        after.append(changed)
    cmap = ListedColormap(("#e9edf0", BLUE, "#d9a24c", "#b85c56"))
    norm = BoundaryNorm((-.5, .5, 1.5, 2.5, 3.5), cmap.N)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 3.25), sharex=True)
    for ax, matrix, title in zip(axes, (before, after),
                                 ("(a) Before synthetic masking", "(b) A+V 30% middle-interval masking")):
        ax.imshow(np.stack(matrix), cmap=cmap, norm=norm,
                  interpolation="nearest", aspect="auto")
        ax.set_yticks((0, 1, 2), ("Text", "Audio", "Vision"))
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xticks((0, 9, 19, 29, 39, 49), (1, 10, 20, 30, 40, 50))
    axes[1].set_xlabel("Aligned sequence position")
    fig.suptitle(f"Traceable validation example: {valid['id'][choice]} | content length {int(content.sum())}",
                 x=.12, ha="left", fontsize=10, fontweight="bold", parse_math=False)
    legend = [Patch(facecolor=color, label=label) for color, label in
              (("#e9edf0", "Special token / padding"), (BLUE, "Observed"),
               ("#d9a24c", "Original all-zero"), ("#b85c56", "Injected missing"))]
    fig.legend(handles=legend, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, -.02), fontsize=7.5)
    fig.tight_layout(rect=(0, .11, 1, .91), h_pad=.65)
    save(fig, "fig14_contiguous_mask_example")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42})
    architecture()
    mask_example()
    print(f"Wrote model and mask diagrams to {OUT}")
