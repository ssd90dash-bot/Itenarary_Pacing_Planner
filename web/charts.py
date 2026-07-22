"""Server-side charts rendered as inline SVG strings.

Reuses the validated `dataviz` palette and the deliberate encoding choices from
`scripts/make_report.py` — a dot strip (not a line) for cost-vs-temperature so
that one-sample-per-cell data does not read as a trend, and a log-log scale
chart. Rendering to SVG server-side keeps the page free of any JS charting
library, so nothing external loads. These are static; the interactive Altair
versions live in the Streamlit app.
"""

from __future__ import annotations

import io
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planner.experiment import CellSummary
from planner.scale import cache_sensitivity

# dataviz reference palette — light categorical slots 1-4 + chrome.
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#52514e"


def _svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    svg = buffer.getvalue()
    # Strip the XML preamble so the fragment embeds directly in HTML.
    start = svg.find("<svg")
    return svg[start:] if start != -1 else svg


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _cap_order(caps: set[str]) -> list[str]:
    return sorted(caps, key=lambda c: (c == "uncapped", c))


def cost_by_temperature(cells: list[CellSummary]) -> str | None:
    """Dot strip: one dot per run, a heavy bar per temperature mean."""
    priced = [c for c in cells if c.avg_cost_usd is not None]
    if not priced:
        return None

    temps = sorted({c.temperature for c in priced})
    caps = _cap_order({c.cap_label for c in priced})
    colour = {cap: SERIES[i % len(SERIES)] for i, cap in enumerate(caps)}

    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    for x, temp in enumerate(temps):
        group = [c for c in priced if c.temperature == temp]
        for c in group:
            ax.scatter(
                x + (caps.index(c.cap_label) - len(caps) / 2) * 0.04,
                c.avg_cost_usd * 1e6,
                s=70,
                color=colour[c.cap_label],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
        mean = statistics.mean(c.avg_cost_usd for c in group) * 1e6
        ax.plot([x - 0.22, x + 0.22], [mean, mean], color=INK, linewidth=2, zorder=4)

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:g}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Cost per run (millionths of $)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour[c], label=f"cap {c}", ms=8)
        for c in caps
    ]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="mean"))
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _style(ax)
    return _svg(fig)


def cost_vs_violations(cells: list[CellSummary]) -> str | None:
    """The headline scatter — bottom-left is cheap and rule-abiding."""
    priced = [c for c in cells if c.avg_cost_usd is not None]
    if not priced:
        return None

    caps = _cap_order({c.cap_label for c in priced})
    colour = {cap: SERIES[i % len(SERIES)] for i, cap in enumerate(caps)}

    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    for c in priced:
        ax.scatter(
            c.avg_violations,
            c.avg_cost_usd * 1e6,
            s=110,
            color=colour[c.cap_label],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            f"{c.temperature:g}",
            (c.avg_violations, c.avg_cost_usd * 1e6),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            color=MUTED,
            va="center",
        )
    ax.set_xlabel("Avg first-draft violations  →  worse")
    ax.set_ylabel("Cost per run (millionths of $)")
    ax.set_xlim(left=-0.4)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour[c], label=f"cap {c}", ms=8)
        for c in caps
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, 1.14))
    _style(ax)
    ax.grid(True, color=GRID, linewidth=1)
    return _svg(fig)


def scale_curve(input_tokens: int, output_tokens: int, model: str) -> str:
    """Monthly cost vs volume, one line per cache assumption, log-log."""
    volumes = [100, 1_000, 10_000, 100_000, 1_000_000]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)

    for i, rate in enumerate((0.0, 0.5, 0.96)):
        costs = [
            cache_sensitivity(input_tokens, output_tokens, v, model, rates=(rate,))[0].monthly_cost_usd
            for v in volumes
        ]
        if any(c is None for c in costs):
            continue
        ax.plot(volumes, costs, color=SERIES[i], linewidth=2, marker="o", markersize=6,
                label=f"{rate:.0%} cached")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Itineraries per month")
    ax.set_ylabel("Monthly cost (USD)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3)
    _style(ax)
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    return _svg(fig)
