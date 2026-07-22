"""Turn posted form fields into the same `TripRequest` the Streamlit app builds.

This mirrors `current_request()` in `app.py` so both frontends construct the
domain object identically — the planner cannot tell which UI it was called from.
Parsing is defensive: an HTML form is untrusted input, whereas Streamlit widgets
are already typed.
"""

from __future__ import annotations

from datetime import date, timedelta

from starlette.datastructures import FormData

from planner.experiment import SweepConfig
from planner.models import Interest, Pace, PitStop, Traveller, TripRequest

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


class FormError(ValueError):
    """A human-readable problem with the submitted form."""


def _int(form: FormData, key: str, default: int) -> int:
    raw = form.get(key)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _float(form: FormData, key: str, default: float) -> float:
    raw = form.get(key)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _date(form: FormData, key: str, default: date) -> date:
    raw = form.get(key)
    if not raw:
        return default
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise FormError(f"'{key}' is not a valid date (expected YYYY-MM-DD).") from exc


def parse_trip_request(form: FormData) -> TripRequest:
    """Build a validated `TripRequest`, raising `FormError` on bad input."""
    destination = str(form.get("destination", "")).strip()
    if not destination:
        raise FormError("Please enter a destination.")

    today = date.today()
    start = _date(form, "start_date", today + timedelta(days=30))
    end = _date(form, "end_date", start + timedelta(days=4))
    if end < start:
        raise FormError("The end date cannot be before the start date.")

    # Ages arrive as repeated `age` fields, one per traveller row.
    ages = [int(a) for a in form.getlist("age") if str(a).strip().isdigit()]
    if not ages:
        raise FormError("Add at least one traveller with an age.")
    travellers = [Traveller(age=a) for a in ages]

    # Pit stops: parallel lists of place / day / hours; keep only named rows.
    places = form.getlist("stop_place")
    days = form.getlist("stop_day")
    hours = form.getlist("stop_hours")
    num_days = (end - start).days + 1
    pit_stops: list[PitStop] = []
    for i, place in enumerate(places):
        name = str(place).strip()
        if not name:
            continue
        day_no = int(days[i]) if i < len(days) and str(days[i]).isdigit() else 1
        day_no = max(1, min(day_no, num_days))
        travel_hours = 0.0
        if i < len(hours):
            try:
                travel_hours = float(hours[i])
            except (TypeError, ValueError):
                travel_hours = 0.0
        pit_stops.append(
            PitStop(place=name, day_index=day_no - 1, travel_hours=travel_hours)
        )

    food_prefs = [f for f in form.getlist("food_pref") if f in FOOD_OPTIONS]
    food_notes = str(form.get("food_notes", "")).strip()

    # Each chosen interest posts both `interest` (the name) and its importance.
    interests: dict[Interest, int] = {}
    for name in form.getlist("interest"):
        try:
            interest = Interest(name)
        except ValueError:
            continue
        interests[interest] = max(1, min(_int(form, f"importance_{name}", 3), 5))

    pace_raw = str(form.get("pace", "balanced"))
    try:
        pace = Pace(pace_raw)
    except ValueError:
        pace = Pace.balanced

    return TripRequest(
        destination=destination,
        start_date=start,
        end_date=end,
        travellers=travellers,
        pit_stops=pit_stops,
        food_preferences=food_prefs,
        food_notes=food_notes,
        interests=interests,
        pace=pace,
    )


def parse_sweep_config(params) -> SweepConfig:
    """Build a `SweepConfig` from query params on the SSE stream URL.

    `SweepConfig.validate()` still enforces the 24-run cap downstream — this only
    parses; it does not decide what is allowed.
    """
    temps = [float(t) for t in params.getlist("temperature")] or [0.0]

    caps: list[int | None] = []
    for c in params.getlist("cap"):
        caps.append(None if c in ("", "none", "uncapped") else int(c))
    if not caps:
        caps = [None]

    samples = 1
    raw_samples = params.get("samples")
    if raw_samples and str(raw_samples).isdigit():
        samples = int(raw_samples)

    repair = params.get("repair") in ("1", "true", "on", "yes")

    return SweepConfig(
        temperatures=sorted(set(temps)),
        max_output_tokens=sorted(set(caps), key=lambda c: c if c is not None else 10**9),
        samples=samples,
        repair=repair,
    )
