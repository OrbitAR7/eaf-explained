"""Figure style: white background, boxed axes with inward ticks, MATLAB colour order."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BLUE, ORANGE, YELLOW, PURPLE, GREEN, CYAN, RED = (
    "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F")
GRAY = "#808080"


def style(font_size: float = 9.0) -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": font_size,
        "mathtext.fontset": "stixsans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.6,
        "axes.titlesize": font_size,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#262626",
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "legend.fontsize": font_size - 1,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save(fig, stem) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_name(f"{stem.name}.{ext}"))
    plt.close(fig)


def panel(ax, tag: str, text: str = "") -> None:
    ax.set_title(f"({tag}) {text}".rstrip(), loc="left")
