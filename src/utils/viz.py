"""Shared palette + matplotlib style for all project figures."""

import matplotlib as mpl

# categorical slots (fixed order, light mode)
BLUE = "#2a78d6"    # slot 1: primary series (calibrated model)
AQUA = "#1baf7a"    # slot 2
YELLOW = "#eda100"  # slot 3
VIOLET = "#4a3aa7"  # slot 5
RED = "#e34948"     # slot 6: contrast series (raw model)

# chrome & ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# sequential blue ramp (score-matrix heatmap)
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    """Recessive grid/axes, thin marks, muted chrome."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
        "legend.frameon": False,
    })
