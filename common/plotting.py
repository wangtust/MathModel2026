from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


# Shared visual language for Q1/Q2/Q3. Keep modality colors stable throughout
# the paper so that the same color always carries the same semantic meaning.
MODALITY_COLORS = {
    "text": "#245D83",
    "audio": "#D18532",
    "vision": "#5A9674",
}

INK = "#23313E"
MUTED = "#5E6B75"
GRID = "#E5EAEE"
EDGE = "#9EAAB4"
LIGHT = "#F2F5F6"
BG = "#FFFFFF"
DANGER = "#BC6B5E"
ACCENT = "#39786F"
OCHRE = "#B3834A"

# Approximate journal widths in inches. Most main-text figures use the double
# column width so their typography remains readable after LaTeX insertion.
SINGLE_COLUMN = 3.35
DOUBLE_COLUMN = 7.15


def _pick_font(*, chinese: bool) -> str:
    """Pick one actually installed font instead of asking Matplotlib to try missing families."""
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    requested = os.environ.get("MM_PLOT_FONT", "").strip()
    if requested and requested in available:
        return requested

    candidates = (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ) if chinese else ("DejaVu Sans", "Arial", "Liberation Sans")

    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def apply_paper_style(*, chinese: bool = True, base_font_size: float = 8.0) -> None:
    """Apply one publication-oriented Matplotlib style across all questions."""
    import matplotlib.pyplot as plt

    font = _pick_font(chinese=chinese)
    params = {
        "font.family": font,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .65,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": INK,
        "axes.unicode_minus": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 300,
        "figure.dpi": 130,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
    }
    # MOSEI sample IDs use a literal `$_$` separator.  Matplotlib otherwise
    # interprets the text between dollar signs as math and crashes on `_`.
    # The paper figures use Unicode symbols (Δ, ±, etc.) rather than $...$ math,
    # so disabling automatic math parsing is both safe and more robust.
    if "text.parse_math" in plt.rcParams:
        params["text.parse_math"] = False
    plt.rcParams.update(params)


def save_figure(fig, folder: Path | str, name: str, *,
                formats: Iterable[str] = ("pdf", "png"), dpi: int = 300,
                close: bool = True, pad_inches: float = .04) -> list[Path]:
    """Save a paper figure consistently as vector PDF plus high-DPI PNG."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        path = folder / f"{name}.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": pad_inches,
                  "facecolor": BG}
        if suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(path)
    if close:
        plt.close(fig)
    return written


def panel_title(ax, letter: str, title: str = "", *, pad: float = 7,
                weight: str = "semibold") -> None:
    """Use compact panel labels; the LaTeX caption carries the long title."""
    prefix = f"({letter})" if letter else ""
    text = f"{prefix}  {title}".strip()
    ax.set_title(text, loc="left", fontweight=weight, pad=pad)


def style_axis(ax, *, grid_axis: str | None = None, grid_alpha: float = 1.0) -> None:
    """Apply the shared light-weight axis treatment to an existing axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=.55, alpha=grid_alpha)
        ax.set_axisbelow(True)


def modality_color(name: str) -> str:
    """Return the stable paper color for text/audio/vision aliases."""
    key = name.strip().lower()
    aliases = {
        "t": "text", "textual": "text",
        "a": "audio", "speech": "audio", "acoustic": "audio",
        "v": "vision", "visual": "vision", "face": "vision",
    }
    return MODALITY_COLORS[aliases.get(key, key)]
