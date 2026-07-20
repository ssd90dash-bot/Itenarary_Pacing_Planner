"""Streamlit front end for the itinerary planner.

Inputs and tab wiring only — the tab bodies live in `plan_ui.py` and `lab_ui.py`.
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

import plan_ui
from lab_ui import render_cost_lab
from scale_ui import render_scale_tab
from planner.models import Interest, Pace, PitStop, Traveller, TripRequest
from planner.usage import UsageTotals, format_cost

FOOD_OPTIONS = [
    "Vegetarian",
    "Vegan",
    "Halal",
    "Kosher",
    "Gluten-free",
    "Nut allergy",
    "Dairy-free",
    "Seafood allergy",
]

st.set_page_config(page_title="Itinerary Planner", page_icon="🧭", layout="wide")
st.title("🧭 Itinerary Planner")
st.caption(
    "Plans around your group's actual energy — ages, travel days and accumulated "
    "fatigue set each day's budget before any activity is chosen."
)

# Streamlit reruns this script on every widget interaction, so anything that
# must survive a rerun (the itinerary, the running token tally) lives here.
st.session_state.setdefault("session_usage", UsageTotals())
st.session_state.setdefault("transactions", [])  # one entry per generation
st.session_state.setdefault("result", None)
st.session_state.setdefault("request", None)


# --- Inputs ------------------------------------------------------------------

with st.sidebar:
    st.header("Trip")
    destination = st.text_input("Destination", placeholder="Lisbon, Portugal")

    today = date.today()
    date_range = st.date_input(
        "Dates",
        value=(today + timedelta(days=30), today + timedelta(days=34)),
        min_value=today,
    )
    start_date, end_date = (
        date_range
        if isinstance(date_range, tuple) and len(date_range) == 2
        else (date_range, date_range)
    )
    num_days = (end_date - start_date).days + 1
    st.caption(f"Duration: **{num_days} day(s)**")

    st.header("Travellers")
    traveller_count = st.number_input("How many?", min_value=1, max_value=12, value=2)
    ages: list[int] = []
    for i in range(int(traveller_count)):
        cols = st.columns([1, 2])
        cols[0].markdown(
            f"<div style='padding-top:8px'>#{i + 1}</div>", unsafe_allow_html=True
        )
        ages.append(
            cols[1].number_input(
                "Age",
                min_value=0,
                max_value=110,
                value=35,
                key=f"age_{i}",
                label_visibility="collapsed",
            )
        )

    st.header("Pit stops")
    st.caption("Day trips or transfers between bases — these cost energy.")
    stop_count = st.number_input(
        "How many?", min_value=0, max_value=10, value=0, key="stops"
    )
    pit_stops: list[PitStop] = []
    for i in range(int(stop_count)):
        place = st.text_input("Place", key=f"stop_place_{i}", placeholder="Sintra")
        cols = st.columns(2)
        day_no = cols[0].number_input(
            "On day",
            min_value=1,
            max_value=max(num_days, 1),
            value=min(i + 2, num_days),
            key=f"stop_day_{i}",
        )
        hours = cols[1].number_input(
            "Travel hrs",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
            key=f"stop_hours_{i}",
        )
        if place:
            pit_stops.append(
                PitStop(place=place, day_index=int(day_no) - 1, travel_hours=hours)
            )

    st.header("Food")
    food_prefs = st.multiselect("Dietary requirements", FOOD_OPTIONS)
    food_notes = st.text_input(
        "Anything else?", placeholder="loves street food, no spice"
    )

    st.header("Interests")
    st.caption("Pick what matters, then weight each one.")
    chosen = st.multiselect(
        "Areas of interest",
        list(Interest),
        default=[Interest.history, Interest.experiences],
        format_func=lambda i: i.value.title(),
    )
    interests: dict[Interest, int] = {}
    for interest in chosen:
        interests[interest] = st.slider(
            interest.value.title(), 1, 5, 3, key=f"interest_{interest.value}"
        )

    pace = st.select_slider(
        "Overall pace",
        options=list(Pace),
        value=Pace.balanced,
        format_func=lambda p: p.value.title(),
    )

    go = st.button("Plan my trip", type="primary", use_container_width=True)

    session_usage: UsageTotals = st.session_state["session_usage"]
    if session_usage.call_count:
        st.divider()
        st.header("Session tokens")
        cols = st.columns(2)
        cols[0].metric("Input", f"{session_usage.input_tokens:,}")
        cols[1].metric("Output", f"{session_usage.output_tokens:,}")
        st.metric(
            "Estimated spend",
            format_cost(session_usage.cost_usd),
            help=(
                f"{session_usage.total_tokens:,} tokens across "
                f"{session_usage.call_count} API call(s) in "
                f"{len(st.session_state['transactions'])} generation(s)."
            ),
        )
        if st.button("Reset counter", use_container_width=True):
            st.session_state["session_usage"] = UsageTotals()
            st.session_state["transactions"] = []
            st.rerun()


def current_request() -> TripRequest:
    return TripRequest(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        travellers=[Traveller(age=a) for a in ages],
        pit_stops=pit_stops,
        food_preferences=food_prefs,
        food_notes=food_notes,
        interests=interests,
        pace=pace,
    )


# --- Tabs --------------------------------------------------------------------

plan_tab, lab_tab, scale_tab = st.tabs(
    ["🗓️ Plan a trip", "🧪 Cost Lab", "📈 Cost at scale"]
)

with plan_tab:
    if go:
        if not destination.strip():
            st.error("Please enter a destination.")
            st.stop()
        plan_ui.generate(current_request())

    result = st.session_state["result"]
    request = st.session_state["request"]

    if result is None:
        plan_ui.render_preview(current_request())
    else:
        plan_ui.render_itinerary(request, result)
        plan_ui.render_usage(request, result)

with lab_tab:
    render_cost_lab(current_request())

with scale_tab:
    render_scale_tab()
