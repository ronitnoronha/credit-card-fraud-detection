"""Shared matplotlib styling so every saved figure looks consistent."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": config.COLOR_SURFACE,
        "axes.facecolor": config.COLOR_SURFACE,
        "savefig.facecolor": config.COLOR_SURFACE,
        "axes.edgecolor": config.COLOR_GRID,
        "axes.labelcolor": config.COLOR_TEXT_MUTED,
        "axes.titlecolor": config.COLOR_TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": config.COLOR_GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": config.COLOR_TEXT_MUTED,
        "ytick.color": config.COLOR_TEXT_MUTED,
        "legend.frameon": False,
        "font.size": 10,
        "lines.linewidth": 2,
    })


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
