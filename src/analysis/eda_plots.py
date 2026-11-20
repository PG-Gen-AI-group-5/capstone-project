"""Plot helpers for the EDA notebooks (notebooks/eda/). The model notebooks use src/analysis/plots.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

FIG_DIR = ROOT / "report" / "figures" / "eda"


def heatmap(ax, matrix: pd.DataFrame, title: str, cmap: str = "viridis", fmt: str | None = None,
            vmin: float | None = None, vmax: float | None = None, cbar_label: str = ""):
    """Rows x columns heat map; NaN cells are left blank. `fmt` (e.g. "{:.0%}") writes values in cells."""
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index], fontsize=7)
    ax.set_title(title, fontsize=10)
    if fmt:
        for (i, j), v in np.ndenumerate(data):
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5, color="white")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return im


def save(fig, name: str, fig_dir: Path | None = None) -> Path:
    """Save a figure as PNG under report/figures/eda/ (created if missing)."""
    d = fig_dir or FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
