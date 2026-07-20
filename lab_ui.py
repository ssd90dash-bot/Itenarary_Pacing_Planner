"""Cost Lab tab — run a parameter sweep and report on it.

Kept out of `app.py` because it is a self-contained feature, and out of
`planner/` because it is presentation, not domain logic.
"""

from __future__ import annotations

import streamlit as st

from planner import charts
from planner.experiment import (
    MAX_SWEEP_RUNS,
    RunResult,
    SweepConfig,
    SweepTooLarge,
    run_sweep,
    summarise,
    to_csv,
)
from planner.llm import DEFAULT_MODEL
from planner.models import TripRequest
from planner.usage import format_cost

TEMPERATURE_OPTIONS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3]
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

CAP_OPTIONS: list[int | None] = [500, 1000, 2000, 4000, 8000, 16000, None]
DEFAULT_CAPS: list[int | None] = [2000, 4000, 8000, None]


def _cap_label(cap: int | None) -> str:
    return f"{cap:,}" if cap is not None else "default (no cap)"


def _is_dark() -> bool:
    """Theme detection, guarded — older Streamlit builds lack `st.context.theme`."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def render_cost_lab(request: TripRequest) -> None:
    st.subheader("🧪 Cost Lab")
    st.caption(
        "Runs the **same trip** across a grid of `temperature` × "
        "`max_completion_tokens` values and measures what each combination costs "
        "— and whether it still produces a valid itinerary."
    )

    transactions = st.session_state["transactions"]
    if not transactions:
        st.info(
            "**Generate one itinerary first.** The Lab estimates a sweep's cost by "
            "extrapolating from a real run — without one it would be guessing, and "
            "guessing about spend is not good enough."
        )
        return

    baseline = transactions[-1]["usage"]
    per_run_cost = baseline.cost_usd

    _render_setup(request, per_run_cost)
    _render_report()


def _render_setup(request: TripRequest, per_run_cost: float | None) -> None:
    with st.container(border=True):
        st.markdown("**Grid**")
        col_a, col_b = st.columns(2)

        temperatures = col_a.multiselect(
            "Temperatures",
            TEMPERATURE_OPTIONS,
            default=DEFAULT_TEMPERATURES,
            help="0.0 is near-deterministic; 1.0 is the API default.",
        )
        caps = col_b.multiselect(
            "Output caps (max_completion_tokens)",
            CAP_OPTIONS,
            default=DEFAULT_CAPS,
            format_func=_cap_label,
            max_selections=charts.MAX_SERIES,
            help=(
                f"Capped at {charts.MAX_SERIES} so the charts stay colour-blind "
                "safe — past four series the separation floors cannot be met."
            ),
        )

        col_c, col_d = st.columns(2)
        samples = col_c.slider(
            "Samples per combination",
            1,
            3,
            1,
            help="Repeats reduce noise but multiply cost. See the caveat below.",
        )
        repair = col_d.checkbox(
            "Include the repair pass",
            value=False,
            help=(
                "Off by default so each run measures the *first* call cleanly. "
                "Turn on to measure the true end-to-end cost per itinerary."
            ),
        )

        config = SweepConfig(
            temperatures=sorted(temperatures),
            max_output_tokens=sorted(caps, key=lambda c: c or 10**9),
            samples=samples,
            repair=repair,
        )

        try:
            config.validate()
        except SweepTooLarge as exc:
            st.error(str(exc))
            return

        estimate = per_run_cost * config.total_runs if per_run_cost else None
        if repair and estimate:
            estimate *= 1.5  # a repair pass roughly adds another call

        st.markdown("**Before you run this**")
        cols = st.columns(3)
        cols[0].metric("Runs", config.total_runs)
        cols[1].metric("Est. spend", format_cost(estimate))
        cols[2].metric("Cap", f"{MAX_SWEEP_RUNS} runs")

        st.caption(
            "Estimated from your most recent generation. Real token counts vary "
            "per run, and runs that truncate still cost money — treat this as a "
            "ballpark, not a quote. The sweep runs to completion once started: "
            "Streamlit cannot process a stop button mid-loop, so the run cap is "
            "the real safety net."
        )

        confirmed = st.checkbox(
            f"I understand this will make **{config.total_runs} real API calls** "
            f"costing roughly **{format_cost(estimate)}**."
        )
        if st.button(
            "Run sweep", type="primary", disabled=not confirmed, key="run_sweep"
        ):
            _execute(request, config)


def _execute(request: TripRequest, config: SweepConfig) -> None:
    progress = st.progress(0.0, text="Starting sweep…")
    live = st.empty()
    runs: list[RunResult] = []

    try:
        for run in run_sweep(request, config):
            runs.append(run)
            spent = sum(r.cost_usd or 0.0 for r in runs)
            progress.progress(
                len(runs) / config.total_runs,
                text=(
                    f"Run {len(runs)}/{config.total_runs} — "
                    f"T={run.temperature}, cap={run.cap_label}, {run.status} — "
                    f"spent so far {format_cost(spent)}"
                ),
            )
            live.dataframe(
                _runs_table(runs), hide_index=True, use_container_width=True
            )
    except SweepTooLarge as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 — keep whatever was paid for
        st.error(f"Sweep stopped early: {type(exc).__name__}: {exc}")

    progress.empty()
    live.empty()

    # Sweep spend counts toward the session tally like any other call.
    for run in runs:
        st.session_state["session_usage"].extend(run.usage.calls)

    st.session_state["sweep_runs"] = runs
    st.session_state["sweep_repair"] = config.repair


def _runs_table(runs: list[RunResult]) -> list[dict]:
    return [
        {
            "T": r.temperature,
            "Cap": r.cap_label,
            "#": r.sample_index + 1,
            "Status": r.status,
            "In": r.usage.input_tokens,
            "Out": r.usage.output_tokens,
            "Cost": format_cost(r.cost_usd),
            "Violations": r.first_draft_violations if r.status == "ok" else "—",
            "Repairs": r.repair_calls,
            "Secs": round(r.duration_s, 1),
            "Error": (r.error or "")[:80],
        }
        for r in runs
    ]


def _render_report() -> None:
    runs: list[RunResult] = st.session_state.get("sweep_runs", [])
    if not runs:
        return

    report = summarise(runs)
    theme = charts.active_theme(_is_dark())

    st.divider()
    st.subheader("Report")

    cols = st.columns(4)
    cols[0].metric("Runs", len(runs))
    cols[1].metric("Total spend", format_cost(report.total_cost_usd))
    cols[2].metric("Total tokens", f"{report.total_tokens:,}")
    cols[3].metric(
        "Wasted tokens",
        f"{report.wasted_tokens:,}",
        help="Tokens billed on runs that truncated or errored and returned nothing.",
    )

    # --- Recommendation ------------------------------------------------------

    best = report.recommendation
    if best is None:
        st.error(report.recommendation_reason)
    else:
        st.success(
            f"**Recommended: temperature {best.temperature}, output cap "
            f"{best.cap_label}.**\n\n{report.recommendation_reason}"
        )
        cap_line = (
            f"OPENAI_MAX_OUTPUT_TOKENS={best.max_output_tokens}"
            if best.max_output_tokens
            else "# OPENAI_MAX_OUTPUT_TOKENS left unset (no cap)"
        )
        st.code(f"OPENAI_TEMPERATURE={best.temperature}\n{cap_line}", language="bash")

    if report.min_samples < 2:
        st.warning(
            "**These results are indicative, not conclusive.** Each combination "
            "was run once, and the same parameters produce a different itinerary "
            "every time. Raise samples to 2–3 before acting on a small difference."
        )

    st.caption(
        f"Specific to this trip and to `{DEFAULT_MODEL}`. A longer trip has a "
        "different output profile and may truncate at caps that were fine here. "
        + (
            "Repair pass included, so these are end-to-end costs."
            if st.session_state.get("sweep_repair")
            else "Repair pass excluded — a real itinerary may cost more when a "
            "draft needs fixing."
        )
    )

    # --- Charts --------------------------------------------------------------

    left, right = st.columns(2)
    with left:
        st.markdown("**Cost per run vs temperature**")
        chart = charts.cost_vs_temperature(report.cells, theme)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No priced successful runs to plot.")

    with right:
        st.markdown("**Rule violations vs temperature**")
        st.caption("The quality signal — lower is better.")
        chart = charts.violations_vs_temperature(report.cells, theme)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

    st.markdown("**Cost vs quality**")
    st.caption(
        "Each point is one configuration, labelled with its temperature. "
        "The sweet spot is the bottom-left: cheap and rule-abiding."
    )
    chart = charts.cost_vs_violations(report.cells, theme)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No priced successful runs to plot.")

    if any(c.truncated for c in report.cells):
        st.markdown("**Truncation rate by output cap**")
        st.caption("Where the output ceiling starts destroying runs you paid for.")
        chart = charts.truncation_by_cap(report.cells, theme)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

    # --- Tables --------------------------------------------------------------

    with st.expander("Per-combination summary", expanded=True):
        st.dataframe(
            [
                {
                    "Temperature": c.temperature,
                    "Cap": c.cap_label,
                    "Samples": c.samples,
                    "OK": c.ok,
                    "Truncated": c.truncated,
                    "Errors": c.errored,
                    "Avg cost": format_cost(c.avg_cost_usd),
                    "Avg in": round(c.avg_input_tokens),
                    "Avg out": round(c.avg_output_tokens),
                    "Avg violations": round(c.avg_violations, 2),
                    "Avg secs": round(c.avg_duration_s, 1),
                }
                for c in report.cells
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Every run"):
        st.dataframe(_runs_table(runs), hide_index=True, use_container_width=True)

    st.download_button(
        "Download report (CSV)",
        data=to_csv(runs),
        file_name="cost-lab-report.csv",
        mime="text/csv",
    )

    if st.button("Clear results"):
        st.session_state.pop("sweep_runs", None)
        st.rerun()
