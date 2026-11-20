"""One chart style for every notebook figure (light mode, for the report).

Colours follow the model, never its rank: each model keeps its slot of the validated categorical
palette in every figure. Slots 3-5 are below 3:1 contrast on the surface, so every figure is
paired with a printed table or direct labels.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"tft": SERIES[0], "lstm": SERIES[1], "arima": SERIES[2], "ma_crossover": SERIES[3],
                "base_rate": SERIES[4], "hist_vol": SERIES[5]}
MODEL_ORDER = ["tft", "lstm", "arima", "ma_crossover", "base_rate", "hist_vol"]
BLUE_RAMP = {250: "#86b6ef", 450: "#2a78d6", 650: "#104281"}       # ordinal: horizons 1 / 5 / 10
HORIZON_COLORS = {1: BLUE_RAMP[250], 5: BLUE_RAMP[450], 10: BLUE_RAMP[650]}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.8, "lines.markersize": 6, "figure.dpi": 110, "savefig.dpi": 150,
        "font.size": 10, "text.color": TEXT,
    })


def figures_dir(cfg) -> Path:
    path = cfg.root / "report" / "figures" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig, cfg, name: str) -> Path:
    path = figures_dir(cfg) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def bar_style() -> dict:
    """Thin bars with a surface-coloured gap between neighbours."""
    return {"edgecolor": SURFACE, "linewidth": 1.5}


BLUE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#2a78d6", "#0d366b"])


def dot_compare(ax, table, metric: str, models: list[str], horizons: list[int], low=None, high=None,
                fmt="{:.1%}", reference: float | None = None, value_labels: bool = False):
    """One dot per model per horizon (optionally with an interval), models offset within each horizon.
    A dot plot, not bars, because the axis is zoomed in and bars must start at zero. Values are read
    from the table printed beside each chart; value_labels=True writes them on the dots."""
    models = [m for m in MODEL_ORDER if m in models]
    step = 0.7 / max(len(models), 1)
    for k, m in enumerate(models):
        g = table[table["model"] == m].set_index("horizon_days").reindex(horizons)
        x = range(len(horizons))
        xs = [i - 0.35 + step * (k + 0.5) for i in x]
        if low and high:
            ax.vlines(xs, g[low], g[high], color=MODEL_COLORS[m], linewidth=1.5)
        ax.plot(xs, g[metric], "o", color=MODEL_COLORS[m], markersize=8, label=m,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for xi, v in zip(xs, g[metric]):
            if value_labels and not math.isnan(v):
                ax.annotate(fmt.format(v), (xi, v), xytext=(9, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=7.5, color=TEXT_2)
    if reference is not None:
        ax.axhline(reference, color=TEXT_2, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(horizons)), [f"{h}-day" for h in horizons])
    ax.margins(y=0.1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=len(models))


__all__ = [
    "BLUE_CMAP",
    "HORIZON_COLORS",
    "MODEL_COLORS",
    "MODEL_ORDER",
    "SERIES",
    "TEXT",
    "TEXT_2",
    "apply_style",
    "bar_style",
    "dot_compare",
    "figures_dir",
    "plt",
    "save",
]
