"""Tests for the FastAPI HTML frontend. No real API calls — plan_trip and
run_sweep are replaced with fakes, so the whole web layer is exercised for free.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import FormData, QueryParams

from planner.experiment import RunResult
from planner.llm import PlanResult
from planner.models import (
    Activity,
    DayPlan,
    Intensity,
    Interest,
    Itinerary,
    Meal,
    Pace,
    RestBlock,
)
from planner.pacing import build_day_budgets
from planner.usage import CallUsage, UsageTotals
from web import main
from web.forms import parse_sweep_config, parse_trip_request


@pytest.fixture
def client():
    return TestClient(main.app)


# --- Form parsing ------------------------------------------------------------


def _full_form():
    today = date.today()
    return FormData(
        [
            ("destination", "Lisbon"),
            ("start_date", (today + timedelta(days=10)).isoformat()),
            ("end_date", (today + timedelta(days=14)).isoformat()),
            ("age", "35"),
            ("age", "4"),
            ("stop_place", "Sintra"),
            ("stop_day", "2"),
            ("stop_hours", "3"),
            ("food_pref", "Vegetarian"),
            ("food_notes", "mild"),
            ("interest", "history"),
            ("importance_history", "5"),
            ("interest", "outdoor"),
            ("importance_outdoor", "2"),
            ("pace", "relaxed"),
        ]
    )


def test_form_maps_to_a_faithful_trip_request():
    trip = parse_trip_request(_full_form())
    assert trip.destination == "Lisbon"
    assert sorted(t.age for t in trip.travellers) == [4, 35]
    assert trip.num_days == 5
    assert trip.pit_stops[0].place == "Sintra"
    assert trip.pit_stops[0].day_index == 1
    assert trip.pit_stops[0].travel_hours == 3
    assert trip.food_preferences == ["Vegetarian"]
    assert trip.interests[Interest.history] == 5
    assert trip.interests[Interest.outdoor] == 2
    assert trip.pace is Pace.relaxed


def test_missing_destination_is_rejected():
    from web.forms import FormError

    with pytest.raises(FormError):
        parse_trip_request(FormData([("age", "30")]))


def test_end_before_start_is_rejected():
    from web.forms import FormError

    with pytest.raises(FormError):
        parse_trip_request(
            FormData(
                [
                    ("destination", "X"),
                    ("start_date", "2026-09-10"),
                    ("end_date", "2026-09-01"),
                    ("age", "30"),
                ]
            )
        )


def test_blank_pit_stops_are_dropped():
    trip = parse_trip_request(
        FormData(
            [("destination", "X"), ("age", "30"), ("stop_place", ""), ("stop_day", "2")]
        )
    )
    assert trip.pit_stops == []


def test_sweep_config_parsing_handles_uncapped_and_dedupes():
    config = parse_sweep_config(
        QueryParams(
            "temperature=0.0&temperature=0.7&cap=2000&cap=uncapped&samples=2&repair=1"
        )
    )
    assert config.temperatures == [0.0, 0.7]
    assert config.max_output_tokens == [2000, None]
    assert config.samples == 2
    assert config.repair is True


# --- Free routes (no API) ----------------------------------------------------


def test_home_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Plan my trip" in r.text


def test_scale_page_renders_with_projections(client):
    r = client.get("/scale")
    assert r.status_code == 200
    assert "Cost at scale" in r.text
    assert "Pilot" in r.text and "Scale" in r.text  # default scenarios


def test_scale_post_recalculates(client):
    r = client.post("/scale", data={"volume": "50000", "model": "gpt-4o", "cache": "0.0"})
    assert r.status_code == 200
    assert "50,000" in r.text


def test_forecast_fragment_is_free_and_has_cards(client):
    today = date.today()
    r = client.post(
        "/forecast",
        data={
            "destination": "Lisbon",
            "start_date": (today + timedelta(days=5)).isoformat(),
            "end_date": (today + timedelta(days=8)).isoformat(),
            "age": "35",
        },
    )
    assert r.status_code == 200
    assert "Energy forecast" in r.text


def test_lab_page_blocks_without_a_prior_generation(client):
    r = client.get("/lab")
    assert r.status_code == 200
    assert "Generate one itinerary first" in r.text


# --- JSON forecast API (free, no key) ----------------------------------------


def test_api_forecast_returns_budget_json(client):
    r = client.post(
        "/api/forecast",
        json={
            "destination": "Lisbon",
            "start_date": "2026-09-01",
            "end_date": "2026-09-05",
            "travellers": [{"age": 35}, {"age": 4}],
            "interests": {"history": 4, "outdoor": 3},
            "pace": "balanced",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["destination"] == "Lisbon"
    assert body["num_days"] == 5
    assert len(body["days"]) == 5
    first = body["days"][0]
    assert first["is_arrival"] is True
    assert first["date"] == "2026-09-01"
    # A group with a 4-year-old needs a protected midday rest.
    assert first["requires_midday_rest"] is True


def test_api_forecast_reflects_pit_stops(client):
    r = client.post(
        "/api/forecast",
        json={
            "destination": "Lisbon",
            "start_date": "2026-09-01",
            "end_date": "2026-09-04",
            "travellers": [{"age": 30}],
            "pit_stops": [{"place": "Sintra", "day_index": 1, "travel_hours": 3}],
        },
    )
    assert r.status_code == 200
    day2 = r.json()["days"][1]
    assert day2["pit_stop"] == "Sintra"
    assert day2["travel_hours"] == 3


def test_api_forecast_validates_input(client):
    # Missing required `travellers` -> FastAPI 422, not a 500.
    r = client.post("/api/forecast", json={"destination": "X"})
    assert r.status_code == 422


# --- Paid routes, with fakes -------------------------------------------------


def _fake_plan_result():
    usage = UsageTotals()
    usage.add(
        CallUsage(label="draft", model="gpt-4o-mini", input_tokens=1338, output_tokens=1100)
    )
    from datetime import date as _date

    trip_days = 3
    itinerary = Itinerary(
        days=[
            DayPlan(
                day_index=i,
                summary="A gentle day.",
                activities=[
                    Activity(
                        name="Museum",
                        start_time="10:00",
                        end_time="12:00",
                        intensity=Intensity.low,
                        interest=Interest.history,
                        description="…",
                        why="Fits the group.",
                    )
                ],
                meals=[Meal(slot="lunch", time="13:00", suggestion="Café", dietary_note="veg")],
                rest_blocks=[RestBlock(start_time="13:00", end_time="15:00", note="rest")],
            )
            for i in range(trip_days)
        ],
        overall_notes="A balanced trip.",
    )
    return usage, itinerary


def test_plan_route_renders_itinerary_without_spending(client, monkeypatch):
    def fake_plan_trip(trip, **kwargs):
        usage, itinerary = _fake_plan_result()
        return PlanResult(
            itinerary=itinerary,
            budgets=build_day_budgets(trip),
            usage=usage,
            first_draft_violations=[],
        )

    monkeypatch.setattr(main, "_planner", fake_plan_trip)

    today = date.today()
    r = client.post(
        "/plan",
        data={
            "destination": "Lisbon",
            "start_date": (today + timedelta(days=10)).isoformat(),
            "end_date": (today + timedelta(days=12)).isoformat(),
            "age": "35",
            "interest": "history",
            "importance_history": "3",
            "pace": "balanced",
        },
    )
    assert r.status_code == 200
    assert "A balanced trip." in r.text
    assert "Token usage for this itinerary" in r.text
    assert "Tokenizer check" in r.text  # tiktoken comparison surfaced
    assert "1,338" in r.text  # billed input tokens


def test_plan_route_shows_the_error_on_bad_input(client):
    r = client.post("/plan", data={"destination": "", "age": "35"})
    assert r.status_code == 200
    assert "destination" in r.text.lower()


def test_lab_stream_runs_a_fake_sweep_end_to_end(client, monkeypatch):
    # 1) A prior generation, so the guard passes and a trip is stored.
    def fake_plan_trip(trip, **kwargs):
        usage, itinerary = _fake_plan_result()
        return PlanResult(
            itinerary=itinerary, budgets=build_day_budgets(trip), usage=usage
        )

    monkeypatch.setattr(main, "_planner", fake_plan_trip)
    today = date.today()
    client.post(
        "/plan",
        data={
            "destination": "Lisbon",
            "start_date": (today + timedelta(days=10)).isoformat(),
            "end_date": (today + timedelta(days=12)).isoformat(),
            "age": "35",
        },
    )

    # 2) A fake sweeper yielding two runs, so no API is touched.
    def fake_sweeper(trip, config, planner_fn=None, should_stop=None):
        for i, (temp, cap, sample) in enumerate(config.cells()):
            usage = UsageTotals()
            usage.add(
                CallUsage(label="d", model="gpt-4o-mini", input_tokens=1000, output_tokens=500)
            )
            yield RunResult(
                temperature=temp,
                max_output_tokens=cap,
                sample_index=sample,
                status="ok",
                usage=usage,
                first_draft_violations=0,
            )

    monkeypatch.setattr(main, "_sweeper", fake_sweeper)

    r = client.get("/lab/stream?temperature=0.0&temperature=1.0&cap=uncapped&samples=1")
    assert r.status_code == 200
    body = r.text
    assert body.count("event: run") == 2
    assert "event: done" in body
    assert "recommendation" in body


def test_lab_stream_refuses_without_a_prior_generation(client):
    fresh = TestClient(main.app)
    r = fresh.get("/lab/stream?temperature=0.0&cap=uncapped&samples=1")
    assert "event: error" in r.text
    assert "Generate one itinerary first" in r.text


def test_lab_stream_enforces_the_run_cap(client, monkeypatch):
    def fake_plan_trip(trip, **kwargs):
        usage, itinerary = _fake_plan_result()
        return PlanResult(
            itinerary=itinerary, budgets=build_day_budgets(trip), usage=usage
        )

    called = []

    def fake_sweeper(trip, config, **kwargs):
        called.append(1)
        yield from ()

    monkeypatch.setattr(main, "_planner", fake_plan_trip)
    monkeypatch.setattr(main, "_sweeper", fake_sweeper)

    today = date.today()
    client.post(
        "/plan",
        data={
            "destination": "Lisbon",
            "start_date": (today + timedelta(days=10)).isoformat(),
            "end_date": (today + timedelta(days=12)).isoformat(),
            "age": "35",
        },
    )

    # 6 temps x 5 caps x 1 = 30 runs, over the 24 cap.
    q = "&".join(f"temperature={t}" for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    q += "&" + "&".join(f"cap={c}" for c in (500, 1000, 2000, 4000, "uncapped"))
    r = client.get(f"/lab/stream?{q}")
    assert "event: error" in r.text
    assert not called, "the cap must trip before the sweeper is invoked"
