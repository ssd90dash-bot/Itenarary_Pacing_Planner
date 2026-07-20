"""Cost-at-scale tab — project measured token usage out to a monthly volume."""

from __future__ import annotations

import streamlit as st

from planner.llm import DEFAULT_MODEL
from planner.scale import (
    DEFAULT_SCENARIOS,
    ScaleScenario,
    cache_sensitivity,
    project,
    project_all,
)
from planner.usage import PRICING, format_cost

# Measured mean from the project's own 16-run live sweep.
MEASURED_INPUT = 1338
MEASURED_OUTPUT = 1136


def render_scale_tab() -> None:
    st.subheader("📈 Cost at scale")
    st.caption(
        "A fraction of a cent per itinerary means little on its own. This projects "
        "a measured transaction out to a monthly volume."
    )

    transactions = st.session_state.get("transactions", [])
    if transactions:
        latest = transactions[-1]["usage"]
        default_in, default_out = latest.input_tokens, latest.output_tokens
        source = f"your last generation ({transactions[-1]['destination']})"
    else:
        default_in, default_out = MEASURED_INPUT, MEASURED_OUTPUT
        source = "the recorded 16-run sweep"

    with st.container(border=True):
        st.markdown(f"**Per-itinerary baseline** — from {source}")
        cols = st.columns(3)
        input_tokens = cols[0].number_input(
            "Input tokens", min_value=1, value=int(default_in), step=100
        )
        output_tokens = cols[1].number_input(
            "Output tokens", min_value=1, value=int(default_out), step=100
        )
        model = cols[2].selectbox(
            "Model",
            list(PRICING),
            index=(
                list(PRICING).index(DEFAULT_MODEL) if DEFAULT_MODEL in PRICING else 0
            ),
        )

    # --- Defined scenarios ---------------------------------------------------

    st.markdown("**Defined scenarios**")
    st.caption(
        "Cache and repair rates are *assumptions*, not measurements — they are the "
        "levers worth sensitivity-testing."
    )
    st.dataframe(
        [
            {
                "Scenario": p.scenario.name,
                "Itineraries/mo": f"{p.scenario.itineraries_per_month:,}",
                "Cache": f"{p.scenario.cache_hit_rate:.0%}",
                "Repair": f"{p.scenario.repair_rate:.0%}",
                "Calls/mo": f"{p.calls_per_month:,.0f}",
                "Tokens/mo": f"{p.total_tokens_per_month:,.0f}",
                "Monthly": format_cost(p.monthly_cost_usd),
                "Annual": format_cost(p.annual_cost_usd),
                "Note": p.scenario.note,
            }
            for p in project_all(input_tokens, output_tokens, DEFAULT_SCENARIOS, model)
        ],
        hide_index=True,
        use_container_width=True,
    )

    # --- Custom volume -------------------------------------------------------

    st.markdown("**Your own scenario**")
    cols = st.columns(3)
    volume = cols[0].number_input(
        "Itineraries per month", min_value=0, value=10_000, step=1_000
    )
    cache_rate = cols[1].slider("Cache hit rate", 0.0, 1.0, 0.5, 0.05)
    repair_rate = cols[2].slider("Repair rate", 0.0, 1.0, 0.1, 0.05)

    custom = project(
        input_tokens,
        output_tokens,
        ScaleScenario(
            name="Custom",
            itineraries_per_month=int(volume),
            cache_hit_rate=cache_rate,
            repair_rate=repair_rate,
        ),
        model,
    )
    cols = st.columns(4)
    cols[0].metric("Calls/month", f"{custom.calls_per_month:,.0f}")
    cols[1].metric("Per itinerary", format_cost(custom.cost_per_itinerary_usd))
    cols[2].metric("Monthly", format_cost(custom.monthly_cost_usd))
    cols[3].metric("Annual", format_cost(custom.annual_cost_usd))

    # --- Sensitivity ---------------------------------------------------------

    with st.expander("How much does the cache assumption matter?", expanded=True):
        st.caption(
            "The 96% figure is what this project's own sweep achieved by sending one "
            "identical prompt repeatedly — a best case, not a forecast. Real users "
            "arrive with different destinations and dates, so the honest answer sits "
            "toward the top of this range."
        )
        st.dataframe(
            [
                {
                    "Cache hit rate": p.scenario.name,
                    "Monthly": format_cost(p.monthly_cost_usd),
                    "Annual": format_cost(p.annual_cost_usd),
                    "Per itinerary": format_cost(p.cost_per_itinerary_usd),
                }
                for p in cache_sensitivity(
                    input_tokens, output_tokens, int(volume) or 1, model
                )
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Cost is linear in volume, so these scale proportionally. Rates come from "
        "the hardcoded table in `planner/usage.py` and may be stale."
    )
