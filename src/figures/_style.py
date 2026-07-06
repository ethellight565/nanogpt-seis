"""Shared style for all figures — one coherent visual system.

Palette is the dataviz reference categorical set (CVD-safe, validated):
blue / aqua / yellow / green / violet / red / magenta / orange, on a warm
off-white surface with ink-toned text and a recessive grid.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical hues (fixed order)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

# ink / surface tokens
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a897f"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"
PANEL = "#f2f1ec"        # soft fill for boxes


def apply_base():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.edgecolor": INK2, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
