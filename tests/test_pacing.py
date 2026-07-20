"""Unit tests for the energy-budget engine. No API calls, no network."""

from datetime import date, timedelta

import pytest

from planner.models import (
    Activity,
    DayPlan,
    Intensity,
    Interest,
    Itinerary,
    Meal,
    Pace,
    PitStop,
    RestBlock,
    Traveller,
    TripRequest,
)
from planner.pacing import (
    INTENSITY_COST,
    build_day_budgets,
    group_stamina,
    stamina_factor,
    trim_to_budget,
    validate_itinerary,
)


def make_request(*, ages, days=5, pit_stops=None, pace=Pace.balanced):
    start = date(2026, 9, 1)
    return TripRequest(
        destination="Lisbon",
        start_date=start,
        end_date=start + timedelta(days=days - 1),
        travellers=[Traveller(age=a) for a in ages],
        pit_stops=pit_stops or [],
        pace=pace,
    )


# --- Stamina ----------------------------------------------------------------


@pytest.mark.parametrize(
    "age,expected", [(2, 0.45), (8, 0.75), (16, 1.0), (35, 1.0), (65, 0.8), (80, 0.6)]
)
def test_stamina_bands(age, expected):
    assert stamina_factor(age) == expected


def test_group_stamina_sits_between_min_and_mean():
    ages = [35, 35, 3]
    factors = [1.0, 1.0, 0.45]
    assert min(factors) < group_stamina(ages) < sum(factors) / len(factors)


def test_all_adults_is_full_stamina():
    assert group_stamina([30, 45]) == 1.0


# --- Budgets ----------------------------------------------------------------


def test_toddler_and_senior_group_gets_a_much_lower_budget():
    mixed = build_day_budgets(make_request(ages=[35, 3, 72]))
    adults = build_day_budgets(make_request(ages=[35, 38]))
    # Compare a mid-trip day to avoid arrival/departure multipliers.
    assert mixed[2].points < adults[2].points * 0.7


def test_arrival_and_departure_days_are_lighter():
    budgets = build_day_budgets(make_request(ages=[30, 32]))
    assert budgets[0].points < budgets[2].points
    assert budgets[-1].points < budgets[2].points
    assert budgets[0].is_arrival and budgets[-1].is_departure


def test_pit_stop_hours_are_deducted():
    plain = build_day_budgets(make_request(ages=[30, 32]))
    with_stop = build_day_budgets(
        make_request(
            ages=[30, 32], pit_stops=[PitStop(place="Sintra", day_index=1, travel_hours=4)]
        )
    )
    assert with_stop[1].points == pytest.approx(plain[1].points - 32, abs=0.5)
    assert with_stop[1].pit_stop == "Sintra"


def test_fatigue_kicks_in_after_three_active_days():
    budgets = build_day_budgets(make_request(ages=[30, 32], days=8))
    # Days 0, 1 and 2 are all active, so day 3 is the first to carry the penalty.
    assert budgets[3].points < budgets[2].points
    assert any("fatigue" in r.lower() for r in budgets[3].reasons)


def test_low_budget_day_is_promoted_to_recovery():
    budgets = build_day_budgets(
        make_request(ages=[70, 2], days=6, pace=Pace.relaxed)
    )
    assert any(b.is_recovery for b in budgets)
    assert all(not b.high_allowed for b in budgets if b.is_recovery)


def test_high_intensity_days_are_never_adjacent():
    budgets = build_day_budgets(make_request(ages=[30, 32], days=10))
    high_days = [b.day_index for b in budgets if b.high_allowed]
    assert high_days, "an adult group should get some big days"
    assert all(b - a > 1 for a, b in zip(high_days, high_days[1:]))


def test_midday_rest_required_only_for_young_or_senior_groups():
    assert not build_day_budgets(make_request(ages=[30, 32]))[0].requires_midday_rest
    assert build_day_budgets(make_request(ages=[30, 4]))[0].requires_midday_rest
    assert build_day_budgets(make_request(ages=[30, 78]))[0].requires_midday_rest


def test_pace_override_scales_the_budget():
    relaxed = build_day_budgets(make_request(ages=[30, 32], pace=Pace.relaxed))
    packed = build_day_budgets(make_request(ages=[30, 32], pace=Pace.packed))
    assert packed[2].points > relaxed[2].points


# --- Validation -------------------------------------------------------------


def activity(name, intensity):
    return Activity(
        name=name,
        start_time="10:00",
        end_time="12:00",
        intensity=intensity,
        interest=Interest.history,
        description="…",
        why="…",
    )


def day(index, activities, rests=None):
    return DayPlan(
        day_index=index,
        summary="…",
        activities=activities,
        meals=[Meal(slot="lunch", time="13:00", suggestion="…", dietary_note="…")],
        rest_blocks=rests
        if rests is not None
        else [RestBlock(start_time="13:00", end_time="15:00", note="…")],
    )


def test_over_budget_day_is_flagged():
    budgets = build_day_budgets(make_request(ages=[30, 32], days=3))
    packed = [activity(f"stop {i}", Intensity.high) for i in range(4)]
    itinerary = Itinerary(
        days=[day(0, packed), day(1, []), day(2, [])], overall_notes=""
    )
    rules = {v.rule for v in validate_itinerary(itinerary, budgets)}
    assert "over_budget" in rules
    assert "multiple_high" in rules


def test_missing_rest_block_is_flagged():
    budgets = build_day_budgets(make_request(ages=[30, 32], days=1))
    itinerary = Itinerary(
        days=[day(0, [activity("museum", Intensity.low)], rests=[])], overall_notes=""
    )
    assert any(v.rule == "no_rest" for v in validate_itinerary(itinerary, budgets))


def test_clean_itinerary_has_no_violations():
    budgets = build_day_budgets(make_request(ages=[30, 32], days=3))
    itinerary = Itinerary(
        days=[day(i, [activity("walk", Intensity.low)]) for i in range(3)],
        overall_notes="",
    )
    assert validate_itinerary(itinerary, budgets) == []


def test_trim_brings_an_over_packed_day_back_within_budget():
    budgets = build_day_budgets(make_request(ages=[30, 32], days=3))
    itinerary = Itinerary(
        days=[
            day(0, [activity(f"stop {i}", Intensity.high) for i in range(4)]),
            day(1, []),
            day(2, []),
        ],
        overall_notes="",
    )
    notes = trim_to_budget(itinerary, budgets)
    assert notes
    cost = sum(INTENSITY_COST[a.intensity] for a in itinerary.days[0].activities)
    assert cost <= budgets[0].points
