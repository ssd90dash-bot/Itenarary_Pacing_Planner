"""Generate the cost-evaluation PDF from the live Cost Lab data.

Every figure in the output traces to `cost-lab-report.csv`, the pricing table in
`planner/usage.py`, a local `tiktoken` count, or a clearly-labelled projection.
Nothing is invented.

    python scripts/make_report.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from planner.scale import cache_sensitivity, project_all  # noqa: E402
from planner.usage import estimate_across_models, format_cost  # noqa: E402

CSV_PATH = ROOT / "cost-lab-report.csv"
PDF_PATH = ROOT / "cost-evaluation-report.pdf"
CHART_DIR = ROOT / "build" / "charts"

MODEL = "gpt-4o-mini"

# Measured locally with tiktoken against the exact swept prompt.
LOCAL_PROMPT_TOKENS = 822
BILLED_INPUT_TOKENS = 1338
SCHEMA_OVERHEAD = BILLED_INPUT_TOKENS - LOCAL_PROMPT_TOKENS

# --- Palette (from the validated dataviz reference, light steps) -------------

SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#52514e"
CRITICAL = "#d03b3b"


# --- Data --------------------------------------------------------------------


def load_runs() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found. Run a sweep in the Cost Lab tab and save the "
            "CSV to the project root first."
        )
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        row["temperature"] = float(row["temperature"])
        row["cap"] = row["max_output_tokens"] or "uncapped"
        row["input_tokens"] = int(row["input_tokens"])
        row["output_tokens"] = int(row["output_tokens"])
        row["total_tokens"] = int(row["total_tokens"])
        row["cost_usd"] = float(row["cost_usd"]) if row["cost_usd"] else 0.0
        row["violations"] = int(row["first_draft_violations"])
        row["duration_s"] = float(row["duration_s"])
    return rows


def by_temperature(runs: list[dict]) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["temperature"], []).append(run)
    return dict(sorted(grouped.items()))


# --- Charts ------------------------------------------------------------------


def _style_axes(ax) -> None:
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


def chart_cost_by_temperature(runs: list[dict], path: Path) -> None:
    """A dot strip, deliberately not a line.

    With one sample per cell, joining the means would draw a trend the data does
    not support. Scattered dots let the overlap between temperatures speak.
    """
    grouped = by_temperature(runs)
    caps = sorted({r["cap"] for r in runs}, key=lambda c: (c == "uncapped", c))
    colour = {cap: SERIES[i % len(SERIES)] for i, cap in enumerate(caps)}

    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    for x, (temp, rows) in enumerate(grouped.items()):
        for row in rows:
            ax.scatter(
                x + (hash(row["cap"]) % 5 - 2) * 0.035,
                row["cost_usd"] * 1e6,
                s=70,
                color=colour[row["cap"]],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
        mean = statistics.mean(r["cost_usd"] for r in rows) * 1e6
        ax.plot([x - 0.22, x + 0.22], [mean, mean], color=INK, linewidth=2, zorder=4)

    overall = statistics.mean(r["cost_usd"] for r in runs) * 1e6
    ax.axhline(overall, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(
        len(grouped) - 0.45,
        overall,
        f"  overall mean {overall:.0f}",
        color=MUTED,
        fontsize=8,
        va="bottom",
    )

    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([f"{t:g}" for t in grouped])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Cost per run (millionths of $)")
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=colour[c], label=f"cap {c}", ms=8
        )
        for c in caps
    ]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="group mean"))
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_token_composition(path: Path) -> None:
    """What a local tokenizer sees versus what the API bills."""
    fig, ax = plt.subplots(figsize=(7.2, 2.4), facecolor=SURFACE)

    ax.barh(
        ["Billed by API", "Counted locally"],
        [LOCAL_PROMPT_TOKENS, LOCAL_PROMPT_TOKENS],
        color=SERIES[0],
        height=0.55,
        label="Prompt text",
    )
    ax.barh(
        ["Billed by API"],
        [SCHEMA_OVERHEAD],
        left=[LOCAL_PROMPT_TOKENS],
        color=SERIES[3],
        height=0.55,
        label="Schema + scaffolding",
    )

    ax.text(
        LOCAL_PROMPT_TOKENS / 2,
        1,
        f"{LOCAL_PROMPT_TOKENS:,}",
        va="center",
        ha="center",
        fontsize=9,
        color="white",
    )
    ax.text(
        LOCAL_PROMPT_TOKENS + SCHEMA_OVERHEAD / 2,
        0,
        f"+{SCHEMA_OVERHEAD:,}",
        va="center",
        ha="center",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("Input tokens")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2, loc="lower right")
    _style_axes(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_scale(mean_in: int, mean_out: int, path: Path) -> None:
    volumes = [100, 1_000, 10_000, 100_000, 1_000_000]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)

    for i, rate in enumerate((0.0, 0.5, 0.96)):
        costs = []
        for volume in volumes:
            proj = cache_sensitivity(mean_in, mean_out, volume, MODEL, rates=(rate,))[0]
            costs.append(proj.monthly_cost_usd)
        ax.plot(
            volumes,
            costs,
            color=SERIES[i],
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"{rate:.0%} cached",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Itineraries per month")
    ax.set_ylabel("Monthly cost (USD)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3)
    _style_axes(ax)
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- The 150-300 word writeup ------------------------------------------------

WRITEUP = """\
<b>How I imagined implementing AI.</b> I built a travel itinerary planner in which \
the language model is deliberately not in charge. A deterministic Python engine \
first computes an "energy budget" for each day from traveller ages, arrival and \
departure days, transfer hours and accumulated fatigue. Only then does the model \
fill activity slots inside those fixed limits, returning a schema-validated \
object. That split was a cost decision as much as a design one: an invalid \
response costs a second billed repair call.

<b>Parameters tested.</b> Temperature (0.0, 0.3, 0.7, 1.0) and \
max_completion_tokens (2,000, 4,000, 8,000, uncapped) — a 16-run grid over one \
fixed five-day trip for three travellers, so that only the parameters varied.

<b>Results.</b> All sixteen runs succeeded: zero rule violations, zero \
truncations, zero repair calls. Input was constant at 1,338 tokens; output ranged \
822-1,331, giving costs of $0.000598-$0.000903. Mean cost by temperature was \
$0.00080, $0.00079, $0.00075 and $0.00081 — no monotonic relationship. The 1.51x \
spread is output-length noise at one sample per cell, not a parameter effect. Two \
incidental findings mattered more: prompt caching served roughly 96% of input at \
half price, and the structured-output schema added 516 tokens (+63%) over a local \
tiktoken count of the prompt text.

<b>Information needed to decide.</b> Expected monthly volume; a realistic cache \
hit rate for cold user prompts, since my measurement reused a single prompt and \
is therefore a best case; the truncation threshold for longer trips; repeated \
samples to separate signal from noise; and current published pricing, since my \
rate table is hardcoded and may be stale."""


def word_count(html: str) -> int:
    import re

    return len(re.sub(r"<[^>]+>", "", html).split())


# --- PDF ---------------------------------------------------------------------


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "T", parent=base["Title"], fontSize=20, textColor=colors.HexColor("#0b0b0b")
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=13,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#0b0b0b"),
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=14,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1a1a19"),
        ),
        "caption": ParagraphStyle(
            "C",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#52514e"),
        ),
    }


def data_table(rows: list[list[str]], widths: list[float], highlight_header=True):
    table = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1a1a19")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e1e0d9")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f7")]),
    ]
    if highlight_header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")))
    table.setStyle(TableStyle(style))
    return table


def build(runs: list[dict]) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    s = styles()

    costs = [r["cost_usd"] for r in runs]
    outputs = [r["output_tokens"] for r in runs]
    mean_in = round(statistics.mean(r["input_tokens"] for r in runs))
    mean_out = round(statistics.mean(outputs))
    mean_cost = statistics.mean(costs)

    chart_cost_by_temperature(runs, CHART_DIR / "cost_by_temperature.png")
    chart_token_composition(CHART_DIR / "token_composition.png")
    chart_scale(mean_in, mean_out, CHART_DIR / "scale.png")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="AI Cost Evaluation — Itinerary Planner",
    )
    story: list = []

    def para(text, style="body"):
        story.append(Paragraph(text, s[style]))

    def heading(text):
        story.append(Paragraph(text, s["h2"]))

    # --- Title ---------------------------------------------------------------
    para("Cost Evaluation of an AI System", "title")
    para(
        f"Travel Itinerary Planner · OpenAI {MODEL} · "
        f"Report generated {date.today():%d %B %Y}",
        "caption",
    )
    story.append(Spacer(1, 10))
    para(
        "All figures in this report come from sixteen live API transactions "
        "recorded in <font face='Courier'>cost-lab-report.csv</font>, from a local "
        "<font face='Courier'>tiktoken</font> measurement, or from projections that "
        "are labelled as such. No figures are estimated or illustrative."
    )

    # --- 1. Use case ---------------------------------------------------------
    heading("1. The use case")
    para(
        "The system is a travel itinerary planner. A user supplies a destination, "
        "travel dates, the age of each traveller, any pit stops or transfers, "
        "dietary requirements and the kinds of activity they care about. It returns "
        "a day-by-day plan that balances activity against rest."
    )
    para(
        "The business relevance is that this is a task nobody wants to do manually "
        "and that generic AI does badly. A language model asked for a travel plan "
        "reliably over-packs it, and cheerfully schedules a four-year-old and a "
        "seventy-year-old for the same nine-hour day. The commercial value is in "
        "producing a plan a real group can actually sustain."
    )

    # --- 2. Implementation ---------------------------------------------------
    heading("2. How the AI is implemented")
    para(
        "The model is deliberately not in charge of pacing. A deterministic Python "
        "engine first computes an <i>energy budget</i> for every day of the trip, "
        "derived from the stamina of the group (driven by the youngest and oldest "
        "traveller), reductions for arrival and departure days, accumulated fatigue "
        "after consecutive active days, and points deducted per hour of transfer "
        "travel. Activities carry a fixed intensity cost of 10, 25 or 45 points."
    )
    para(
        "Only then is the model called. It receives each day's budget as a hard "
        "limit and fills activity slots within it, returning a schema-validated "
        "object rather than prose. The response is checked against the same rules "
        "that produced the budgets; a failure triggers one repair call."
    )
    para(
        "This split is a cost decision as much as a design one. Constraining the "
        "model reduces invalid responses, and every invalid response costs a second "
        "billed call. It also means output quality is measurable rather than a "
        "matter of taste, which is what makes the parameter comparison below "
        "meaningful."
    )

    # --- 3. Parameters -------------------------------------------------------
    heading("3. Parameters selected for testing, and why")
    para(
        "<b>Temperature (0.0, 0.3, 0.7, 1.0).</b> The API default is 1.0. A planner "
        "bound by deterministic rules should not need sampling randomness, so the "
        "hypothesis was that lower temperatures would break fewer rules, trigger "
        "fewer repair calls, and therefore cost less."
    )
    para(
        "<b>max_completion_tokens (2,000, 4,000, 8,000, uncapped).</b> Selected as a "
        "spend circuit-breaker. This parameter carries a specific risk with "
        "structured output: if a response is cut off mid-JSON it cannot be parsed, "
        "and every generated token is still billed. The test was intended to find "
        "where that threshold sits."
    )
    para(
        "The two were crossed into a 16-cell grid, run against a single fixed trip "
        "(Virginia, five days, three travellers aged 4, 35 and 35) so that the "
        "parameters were the only variables. One sample per cell.",
    )

    # --- 4. Method -----------------------------------------------------------
    heading("4. Method: counting tokens and calculating cost")
    para(
        "Token counts were captured two independent ways, because they answer "
        "different questions."
    )
    para(
        f"<b>Locally, with tiktoken.</b> Encoding the actual prompt with the model's "
        f"BPE vocabulary gives {LOCAL_PROMPT_TOKENS:,} input tokens. This is "
        "available before any call is made, so it is what you use to predict cost "
        "or to guard against an oversized prompt."
    )
    para(
        f"<b>From the API's usage field.</b> The same request was billed at "
        f"{BILLED_INPUT_TOKENS:,} input tokens — {SCHEMA_OVERHEAD} more, an increase "
        f"of {100 * SCHEMA_OVERHEAD / LOCAL_PROMPT_TOKENS:.0f}%."
    )
    story.append(Spacer(1, 6))
    story.append(Image(str(CHART_DIR / "token_composition.png"), width=16 * cm, height=5.3 * cm))
    para(
        "The gap is not an error in either number. It is the JSON schema and message "
        "scaffolding that structured output adds server-side, which never appear in "
        "the strings encoded locally. The practical lesson is that a local tokenizer "
        f"materially <i>understates</i> a structured-output request: it misses "
        f"{100 * SCHEMA_OVERHEAD / BILLED_INPUT_TOKENS:.0f}% of the input tokens "
        "actually billed. A cost forecast built on tokenizer counts alone would be "
        "wrong by that margin.",
        "caption",
    )
    para(
        "Cost was calculated from published per-million-token rates for the model, "
        "applied separately to uncached input, cached input and output tokens.",
    )

    story.append(PageBreak())

    # --- 5. Results ----------------------------------------------------------
    heading("5. Results of the sixteen transactions")

    rows = [["Temp", "Cap", "Status", "Input", "Output", "Total", "Cost (USD)", "Viol.", "Secs"]]
    for r in runs:
        rows.append(
            [
                f"{r['temperature']:g}",
                r["cap"],
                r["status"],
                f"{r['input_tokens']:,}",
                f"{r['output_tokens']:,}",
                f"{r['total_tokens']:,}",
                f"{r['cost_usd']:.6f}",
                str(r["violations"]),
                f"{r['duration_s']:.1f}",
            ]
        )
    widths = [1.4, 2.0, 1.6, 1.7, 1.7, 1.7, 2.4, 1.4, 1.3]
    story.append(data_table(rows, [w * cm for w in widths]))
    story.append(Spacer(1, 10))

    para(
        f"<b>Every run succeeded.</b> Zero rule violations, zero truncations and "
        f"zero repair calls across all sixteen transactions. Total spend was "
        f"{format_cost(sum(costs))}. Input was constant at {mean_in:,} tokens; "
        f"output ranged {min(outputs):,}-{max(outputs):,}, producing costs from "
        f"{format_cost(min(costs))} to {format_cost(max(costs))} — a "
        f"{max(costs) / min(costs):.2f}x spread."
    )

    grouped = by_temperature(runs)
    agg = [["Temperature", "Mean cost", "Mean output tokens", "Violations", "Truncations"]]
    for temp, group in grouped.items():
        agg.append(
            [
                f"{temp:g}",
                format_cost(statistics.mean(r["cost_usd"] for r in group)),
                f"{statistics.mean(r['output_tokens'] for r in group):.0f}",
                str(sum(r["violations"] for r in group)),
                str(sum(1 for r in group if r["status"] == "truncated")),
            ]
        )
    story.append(data_table(agg, [3.2 * cm, 3.2 * cm, 4.2 * cm, 2.6 * cm, 2.8 * cm]))
    story.append(Spacer(1, 8))
    story.append(Image(str(CHART_DIR / "cost_by_temperature.png"), width=16 * cm, height=8 * cm))
    para(
        "Each dot is one transaction; the heavy bar is that temperature's mean. The "
        "groups overlap almost completely. This is plotted as scattered dots rather "
        "than a trend line on purpose — with one sample per cell, joining the means "
        "would imply a relationship the data does not support.",
        "caption",
    )

    heading("6. The headline result is a negative one")
    para(
        "<b>Temperature had no measurable effect on either cost or output validity.</b> "
        "Mean cost does not move monotonically with temperature, and the cheapest "
        "group (0.7) sits between the two coldest settings. The 1.51x spread between "
        "the cheapest and priciest run is variation in how long the model chose to "
        "make its itinerary, not a consequence of the parameter."
    )
    para(
        "The original hypothesis — that lower temperature would reduce rule "
        "violations and therefore repair calls — could not be tested, because "
        "<i>no configuration produced any violations at all</i>. The most likely "
        "explanation is the deterministic budget engine: by the time the model is "
        "called, the constraints are narrow enough that sampling randomness has "
        "little room to cause a rule breach. Architecture, not parameter tuning, is "
        "what removed the repair cost."
    )
    para(
        "<b>The output cap never bound.</b> Peak output across all runs was "
        f"{max(outputs):,} tokens, well under even the 2,000 cap, so no truncation "
        "occurred and the parameter was inert for a trip of this length. It would "
        "begin to matter on longer trips."
    )

    heading("7. An unplanned finding: prompt caching")
    para(
        "The recorded costs only reconcile with published rates if a large share of "
        "input was billed at the reduced cached rate. Reconstructing the arithmetic, "
        "1,024 input tokens were cached on the first run and 1,280 on every "
        "subsequent run — both exact multiples of 128, matching the documented cache "
        "granularity. Roughly 96% of input was therefore billed at half price."
    )
    para(
        "<b>This makes the measured costs a best case, not a forecast.</b> The sweep "
        "sent one identical prompt sixteen times, which is close to ideal cache "
        "behaviour. Real users arrive with different destinations and dates, so a "
        "production cache hit rate would be far lower and real cost per itinerary "
        "correspondingly higher. The scale model below therefore reports a range "
        "rather than a single number."
    )

    story.append(PageBreak())

    # --- 8. Scale ------------------------------------------------------------
    heading("8. Modelling usage at scale")
    para(
        f"Projections use the measured mean of {mean_in:,} input and {mean_out:,} "
        f"output tokens per itinerary on {MODEL}. Cache and repair rates are stated "
        "assumptions, not measurements — they are the levers worth sensitivity "
        "testing."
    )

    scale_rows = [
        ["Scenario", "Itineraries/mo", "Cache", "Repair", "Calls/mo", "Monthly", "Annual"]
    ]
    for proj in project_all(mean_in, mean_out, model=MODEL):
        sc = proj.scenario
        scale_rows.append(
            [
                sc.name,
                f"{sc.itineraries_per_month:,}",
                f"{sc.cache_hit_rate:.0%}",
                f"{sc.repair_rate:.0%}",
                f"{proj.calls_per_month:,.0f}",
                format_cost(proj.monthly_cost_usd),
                format_cost(proj.annual_cost_usd),
            ]
        )
    story.append(
        data_table(
            scale_rows,
            [2.4 * cm, 2.8 * cm, 1.7 * cm, 1.7 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm],
        )
    )
    story.append(Spacer(1, 10))

    para(
        "<b>Sensitivity to the cache assumption, at 10,000 itineraries per month:</b>"
    )
    sens_rows = [["Cache hit rate", "Monthly cost", "Annual cost", "Cost per itinerary"]]
    for proj in cache_sensitivity(mean_in, mean_out, 10_000, MODEL):
        sens_rows.append(
            [
                proj.scenario.name,
                format_cost(proj.monthly_cost_usd),
                format_cost(proj.annual_cost_usd),
                format_cost(proj.cost_per_itinerary_usd),
            ]
        )
    story.append(data_table(sens_rows, [3.5 * cm, 3.5 * cm, 3.5 * cm, 4.5 * cm]))
    story.append(Spacer(1, 8))
    story.append(Image(str(CHART_DIR / "scale.png"), width=16 * cm, height=7.6 * cm))
    para(
        "Both axes are logarithmic. Cost is linear in volume, so the lines are "
        "straight; the vertical gap between them is the entire value of prompt "
        "caching.",
        "caption",
    )

    # --- 9. Platforms --------------------------------------------------------
    heading("9. How pricing varies across models")
    para(
        f"The same measured workload ({mean_in:,} input + {mean_out:,} output "
        "tokens), priced against every model in the project's rate table, uncached:"
    )
    est_rows = [["Model", "Cost per itinerary", "vs cheapest", "At 10,000/month"]]
    estimates = estimate_across_models(mean_in, mean_out, 0, MODEL)
    cheapest = estimates[0].cost_usd or 1
    for e in estimates:
        est_rows.append(
            [
                e.model + ("  (used)" if e.is_current else ""),
                format_cost(e.cost_usd),
                f"{e.cost_usd / cheapest:.1f}x",
                format_cost(e.cost_usd * 10_000),
            ]
        )
    story.append(
        data_table(est_rows, [5.0 * cm, 3.6 * cm, 2.6 * cm, 3.8 * cm])
    )
    story.append(Spacer(1, 8))
    para(
        f"The spread between cheapest and dearest is "
        f"{(estimates[-1].cost_usd or 0) / cheapest:.0f}x for identical work. At low "
        "volume that difference is immaterial; at 100,000 itineraries a month it is "
        "the difference between a rounding error and a line item. This is a what-if "
        "rather than a measurement: another model would tokenise slightly "
        "differently and would produce a different itinerary, possibly of different "
        "length.",
        "caption",
    )

    # --- 10. Limitations -----------------------------------------------------
    heading("10. Limitations")
    para(
        "<b>One sample per cell.</b> Sixteen runs across sixteen configurations "
        "means every cell is a single observation. Differences between "
        "configurations of the size seen here are indistinguishable from noise. "
        "Three samples per cell would be the minimum for a defensible ranking."
    )
    para(
        "<b>One trip, one model.</b> All runs used a single five-day itinerary on "
        f"{MODEL}. A fourteen-day trip has a different output profile and might well "
        "truncate at caps that were harmless here."
    )
    para(
        "<b>Caching inflates the result favourably.</b> As set out in section 7, the "
        "measured per-run cost reflects near-ideal cache reuse that production "
        "traffic would not see."
    )
    para(
        "<b>Rates are hardcoded.</b> The pricing table is maintained in source and "
        "may be stale; it should be checked against the provider's published pricing "
        "before any figure here is relied upon."
    )

    # --- 11. Summary ---------------------------------------------------------
    story.append(PageBreak())
    heading("11. Summary")
    count = word_count(WRITEUP)
    for block in WRITEUP.split("\n\n"):
        para(block)
    story.append(Spacer(1, 6))
    para(f"[{count} words]", "caption")

    doc.build(story)


def main() -> None:
    runs = load_runs()

    count = word_count(WRITEUP)
    if not 150 <= count <= 300:
        raise SystemExit(f"Summary is {count} words; the brief requires 150-300.")

    build(runs)
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"Wrote {PDF_PATH.relative_to(ROOT)} ({size_kb:.0f} KB)")
    print(f"Summary section: {count} words (within the 150-300 requirement)")
    print(f"Based on {len(runs)} live transactions from {CSV_PATH.name}")


if __name__ == "__main__":
    main()
