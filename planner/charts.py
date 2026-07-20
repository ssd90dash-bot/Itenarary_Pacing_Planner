"""Charts for the Cost Lab report.

Colour values come from the `dataviz` reference palette unchanged — categorical
slots 1-4, which are documented as validating all-pairs in both light and dark
modes. The series count is capped at four for exactly that reason; past four,
the all-pairs colour-blind separation floors cannot be met, so the UI limits the
output-cap selection rather than generating a fifth hue.

Light and dark are separate *selected* step sets from the same ramps, not an
automatic flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import pandas as pd

from .experiment import CellSummary

MAX_SERIES = 4


@dataclass(frozen=True)
class Theme:
    surface: str
    grid: str
    baseline: str
    muted: str
    text: str
    series: tuple[str, ...]
    critical: str


LIGHT = Theme(
    surface="#fcfcfb",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    muted="#898781",
    text="#52514e",
    series=("#2a78d6", "#008300", "#e87ba4", "#eda100"),
    critical="#d03b3b",
)

DARK = Theme(
    surface="#1a1a19",
    grid="#2c2c2a",
    baseline="#383835",
    muted="#898781",
    text="#c3c2b7",
    series=("#3987e5", "#008300", "#d55181", "#c98500"),
    critical="#d03b3b",
)

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def active_theme(is_dark: bool) -> Theme:
    return DARK if is_dark else LIGHT


def _style(chart: alt.Chart, theme: Theme) -> alt.Chart:
    """Recessive chrome: hairline grid, muted axes, no view border."""
    return (
        chart.configure_view(strokeWidth=0, fill=theme.surface)
        .configure_axis(
            gridColor=theme.grid,
            gridWidth=1,
            domainColor=theme.baseline,
            tickColor=theme.baseline,
            labelColor=theme.muted,
            titleColor=theme.text,
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=theme.text,
            titleColor=theme.text,
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=11,
            titleFontSize=11,
            symbolStrokeWidth=2,
        )
        .configure_text(font=FONT)
        .properties(background=theme.surface)
    )


def _frame(cells: list[CellSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "temperature": c.temperature,
                "cap": c.cap_label,
                "cost": c.avg_cost_usd,
                "violations": c.avg_violations,
                "output_tokens": c.avg_output_tokens,
                "samples": c.samples,
                "truncated": c.truncated,
                "runs": c.samples,
            }
            for c in cells
        ]
    )


def _colour(theme: Theme, domain: list[str]) -> alt.Color:
    """Fixed slot order — a series keeps its hue when the filter changes."""
    return alt.Color(
        "cap:N",
        title="Output cap",
        scale=alt.Scale(domain=domain, range=list(theme.series[: len(domain)])),
        legend=alt.Legend(orient="top", direction="horizontal"),
    )


def _series_domain(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["cap"].unique().tolist())[:MAX_SERIES]


def _line_chart(
    frame: pd.DataFrame,
    theme: Theme,
    y_field: str,
    y_title: str,
    y_format: str,
    tooltip_title: str,
) -> alt.Chart:
    domain = _series_domain(frame)
    frame = frame[frame["cap"].isin(domain)]
    colour = _colour(theme, domain)

    base = alt.Chart(frame).encode(
        x=alt.X(
            "temperature:Q",
            title="Temperature",
            scale=alt.Scale(zero=False, nice=False, padding=24),
            axis=alt.Axis(values=sorted(frame["temperature"].unique().tolist())),
        ),
        y=alt.Y(f"{y_field}:Q", title=y_title, axis=alt.Axis(format=y_format)),
        color=colour,
    )

    line = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True))

    # Direct labels on the rightmost point, so identity is never colour-alone.
    last = frame.loc[frame.groupby("cap")["temperature"].idxmax()]
    labels = (
        alt.Chart(last)
        .mark_text(align="left", dx=8, dy=-2, fontSize=11, font=FONT)
        .encode(
            x=alt.X("temperature:Q"),
            y=alt.Y(f"{y_field}:Q"),
            text=alt.Text("cap:N"),
            color=colour,
        )
    )

    hover = base.mark_circle(size=140, opacity=0).encode(
        tooltip=[
            alt.Tooltip("cap:N", title="Output cap"),
            alt.Tooltip("temperature:Q", title="Temperature"),
            alt.Tooltip(f"{y_field}:Q", title=tooltip_title, format=y_format),
            alt.Tooltip("samples:Q", title="Samples"),
        ]
    )

    return _style((line + labels + hover).properties(height=260), theme)


def cost_vs_temperature(cells: list[CellSummary], theme: Theme) -> alt.Chart | None:
    frame = _frame(cells).dropna(subset=["cost"])
    if frame.empty:
        return None
    return _line_chart(
        frame, theme, "cost", "Average cost per run (USD)", "$.5f", "Cost"
    )


def violations_vs_temperature(cells: list[CellSummary], theme: Theme) -> alt.Chart | None:
    frame = _frame(cells)
    if frame.empty:
        return None
    return _line_chart(
        frame,
        theme,
        "violations",
        "Avg first-draft rule violations",
        ".1f",
        "Violations",
    )


def truncation_by_cap(cells: list[CellSummary], theme: Theme) -> alt.Chart | None:
    """Single series — the title names it, so no legend box."""
    frame = _frame(cells)
    if frame.empty:
        return None
    grouped = (
        frame.groupby("cap", as_index=False)[["truncated", "runs"]]
        .sum()
        .assign(rate=lambda d: d["truncated"] / d["runs"])
    )

    bars = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusEnd=4, color=theme.critical)
        .encode(
            x=alt.X("cap:N", title="Output cap", sort=None),
            y=alt.Y(
                "rate:Q",
                title="Runs truncated",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("cap:N", title="Output cap"),
                alt.Tooltip("truncated:Q", title="Truncated runs"),
                alt.Tooltip("runs:Q", title="Total runs"),
                alt.Tooltip("rate:Q", title="Rate", format=".0%"),
            ],
        )
    )
    return _style(bars.properties(height=220), theme)


def cost_vs_violations(cells: list[CellSummary], theme: Theme) -> alt.Chart | None:
    """The headline chart — the sweet spot is the bottom-left corner."""
    frame = _frame(cells).dropna(subset=["cost"])
    if frame.empty:
        return None
    domain = _series_domain(frame)
    frame = frame[frame["cap"].isin(domain)]

    points = (
        alt.Chart(frame)
        .mark_circle(size=140, stroke=theme.surface, strokeWidth=2, opacity=1)
        .encode(
            x=alt.X(
                "violations:Q",
                title="Avg first-draft rule violations  →  worse",
                scale=alt.Scale(zero=True, padding=20),
            ),
            y=alt.Y(
                "cost:Q",
                title="Avg cost per run (USD)  →  pricier",
                axis=alt.Axis(format="$.5f"),
                scale=alt.Scale(zero=True, padding=20),
            ),
            color=_colour(theme, domain),
            tooltip=[
                alt.Tooltip("cap:N", title="Output cap"),
                alt.Tooltip("temperature:Q", title="Temperature"),
                alt.Tooltip("cost:Q", title="Cost", format="$.5f"),
                alt.Tooltip("violations:Q", title="Violations", format=".1f"),
                alt.Tooltip("samples:Q", title="Samples"),
            ],
        )
    )

    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=10, dy=0, fontSize=10, font=FONT, color=theme.muted)
        .encode(
            x=alt.X("violations:Q"),
            y=alt.Y("cost:Q"),
            text=alt.Text("temperature:Q", format=".1f"),
        )
    )

    return _style((points + labels).properties(height=300), theme)
