"""Deterministic energy-budget engine.

This module is the actual product logic: it decides how much a group can do on
each day of a trip, *before* any model is asked to name places. It is pure
Python over plain data — no network, no LLM, no Streamlit — so its behaviour is
unit-testable and identical on every run.

The LLM's job downstream is only to fill the slots this engine authorises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .models import Activity, DayPlan, Intensity, Itinerary, Pace, TripRequest

# --- Constants ---------------------------------------------------------------

BASE_POINTS = 100.0

INTENSITY_COST: dict[Intensity, int] = {
    Intensity.low: 10,
    Intensity.moderate: 25,
    Intensity.high: 45,
}

PACE_MULTIPLIER: dict[Pace, float] = {
    Pace.relaxed: 0.8,
    Pace.balanced: 1.0,
    Pace.packed: 1.2,
}

ARRIVAL_MULTIPLIER = 0.6
DEPARTURE_MULTIPLIER = 0.5
FATIGUE_MULTIPLIER = 0.85
FATIGUE_AFTER_DAYS = 3
RECOVERY_THRESHOLD = 45.0
RECOVERY_FLOOR = 30.0
TRAVEL_COST_PER_HOUR = 8.0
MIN_REST_AFTER_HIGH_MIN = 90

YOUNG_CHILD_AGE = 6
SENIOR_AGE = 70


# --- Stamina -----------------------------------------------------------------


def stamina_factor(age: int) -> float:
    """Per-traveller stamina, 0..1. See README for the rationale per band."""
    if age <= 4:
        return 0.45
    if age <= 12:
        return 0.75
    if age <= 59:
        return 1.0
    if age <= 74:
        return 0.8
    return 0.6


def group_stamina(ages: list[int]) -> float:
    """The group moves near its slowest member, but not exactly at their pace.

    Pure `min` punishes a large group for one toddler far too hard, while `mean`
    ignores that the toddler still has to come along. Softening the minimum a
    quarter of the way toward the mean lands between the two.
    """
    if not ages:
        return 1.0
    factors = [stamina_factor(a) for a in ages]
    lo = min(factors)
    mean = sum(factors) / len(factors)
    return lo + 0.25 * (mean - lo)


# --- Day budgets -------------------------------------------------------------


@dataclass
class DayBudget:
    day_index: int
    day_date: date
    points: float
    is_arrival: bool = False
    is_departure: bool = False
    is_recovery: bool = False
    high_allowed: bool = False
    requires_midday_rest: bool = False
    travel_hours: float = 0.0
    pit_stop: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def max_activities(self) -> int:
        """Rough slot count, used to keep the model from listing 9 tiny stops."""
        return max(1, int(self.points // INTENSITY_COST[Intensity.low]) // 2)


def build_day_budgets(req: TripRequest) -> list[DayBudget]:
    """Compute the per-day energy budget for a whole trip."""
    ages = [t.age for t in req.travellers]
    stamina = group_stamina(ages)
    pace_mult = PACE_MULTIPLIER[req.pace]
    needs_midday_rest = any(a < YOUNG_CHILD_AGE or a > SENIOR_AGE for a in ages)

    stops_by_day = {p.day_index: p for p in req.pit_stops}
    n = req.num_days
    budgets: list[DayBudget] = []
    consecutive_active = 0

    for i in range(n):
        reasons: list[str] = []
        points = BASE_POINTS * stamina
        if stamina < 0.95:
            youngest, oldest = min(ages), max(ages)
            reasons.append(
                f"Group stamina {stamina:.2f} (youngest {youngest}, oldest {oldest})"
            )

        if i == 0:
            points *= ARRIVAL_MULTIPLIER
            reasons.append("Arrival day — travel fatigue, kept light")
        elif i == n - 1 and n > 1:
            points *= DEPARTURE_MULTIPLIER
            reasons.append("Departure day — packing and transit")

        if consecutive_active >= FATIGUE_AFTER_DAYS:
            points *= FATIGUE_MULTIPLIER
            reasons.append(
                f"{consecutive_active} active days in a row — accumulated fatigue"
            )

        stop = stops_by_day.get(i)
        travel_hours = stop.travel_hours if stop else 0.0
        if stop:
            points -= travel_hours * TRAVEL_COST_PER_HOUR
            reasons.append(
                f"Transfer to {stop.place} ({travel_hours:g}h) costs "
                f"{travel_hours * TRAVEL_COST_PER_HOUR:.0f} points"
            )

        points *= pace_mult
        if req.pace is not Pace.balanced:
            reasons.append(f"'{req.pace.value}' pace requested (x{pace_mult:g})")

        is_recovery = points < RECOVERY_THRESHOLD
        if is_recovery:
            points = max(points, RECOVERY_FLOOR)
            reasons.append("Recovery day — downtime deliberately protected")
            consecutive_active = 0
        else:
            consecutive_active += 1

        budgets.append(
            DayBudget(
                day_index=i,
                day_date=req.start_date + timedelta(days=i),
                points=round(points, 1),
                is_arrival=i == 0,
                is_departure=i == n - 1 and n > 1,
                is_recovery=is_recovery,
                requires_midday_rest=needs_midday_rest,
                travel_hours=travel_hours,
                pit_stop=stop.place if stop else None,
                reasons=reasons,
            )
        )

    _assign_high_days(budgets)
    return budgets


def _assign_high_days(budgets: list[DayBudget]) -> None:
    """Mark which days may carry the single high-intensity activity.

    Deciding this up front (rather than validating after the fact) means the
    "never two big days back to back" rule can be stated to the model as a
    concrete per-day permission, which it follows far more reliably than a
    general instruction.
    """
    previous_was_high = False
    for b in budgets:
        eligible = (
            not b.is_recovery
            and not b.is_departure
            and b.points >= INTENSITY_COST[Intensity.high] + INTENSITY_COST[Intensity.low]
            and not previous_was_high
        )
        b.high_allowed = eligible
        previous_was_high = eligible


# --- Costing and validation --------------------------------------------------


def day_cost(day: DayPlan) -> int:
    return sum(INTENSITY_COST[a.intensity] for a in day.activities)


@dataclass
class Violation:
    day_index: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"Day {self.day_index + 1}: {self.detail}"


def _to_minutes(hhmm: str) -> int | None:
    try:
        h, m = hhmm.strip().split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def validate_itinerary(
    itinerary: Itinerary, budgets: list[DayBudget]
) -> list[Violation]:
    """Check a model-produced itinerary against the rules the engine promised."""
    violations: list[Violation] = []
    by_index = {b.day_index: b for b in budgets}

    if len(itinerary.days) != len(budgets):
        violations.append(
            Violation(
                0,
                "day_count",
                f"Expected {len(budgets)} days, got {len(itinerary.days)}",
            )
        )

    for day in itinerary.days:
        budget = by_index.get(day.day_index)
        if budget is None:
            violations.append(
                Violation(day.day_index, "unknown_day", "Day is outside the trip")
            )
            continue

        cost = day_cost(day)
        if cost > budget.points:
            violations.append(
                Violation(
                    day.day_index,
                    "over_budget",
                    f"activities cost {cost} points but the budget is "
                    f"{budget.points:g}. Drop or downgrade an activity.",
                )
            )

        highs = [a for a in day.activities if a.intensity is Intensity.high]
        if len(highs) > 1:
            violations.append(
                Violation(
                    day.day_index,
                    "multiple_high",
                    f"{len(highs)} high-intensity activities; at most one is allowed.",
                )
            )
        if highs and not budget.high_allowed:
            violations.append(
                Violation(
                    day.day_index,
                    "high_not_allowed",
                    "no high-intensity activity is permitted on this day "
                    "(recovery, departure, or the day after a big day).",
                )
            )

        if not day.rest_blocks:
            violations.append(
                Violation(day.day_index, "no_rest", "needs at least one rest block.")
            )

        if highs and day.rest_blocks:
            longest = max(
                (
                    (_to_minutes(r.end_time) or 0) - (_to_minutes(r.start_time) or 0)
                    for r in day.rest_blocks
                ),
                default=0,
            )
            if longest < MIN_REST_AFTER_HIGH_MIN:
                violations.append(
                    Violation(
                        day.day_index,
                        "short_rest",
                        f"a high-intensity day needs a rest block of at least "
                        f"{MIN_REST_AFTER_HIGH_MIN} minutes (longest is {longest}).",
                    )
                )

        if budget.requires_midday_rest:
            has_midday = any(
                (_to_minutes(r.start_time) or 0) < 15 * 60
                and (_to_minutes(r.end_time) or 0) > 12 * 60
                for r in day.rest_blocks
            )
            if not has_midday:
                violations.append(
                    Violation(
                        day.day_index,
                        "no_midday_rest",
                        "the group includes a young child or a senior traveller, "
                        "so a rest overlapping 12:00-15:00 is required.",
                    )
                )

    return violations


def trim_to_budget(itinerary: Itinerary, budgets: list[DayBudget]) -> list[str]:
    """Last-resort repair: drop activities until every day fits its budget.

    Only runs if the model failed twice. Returns human-readable notes so the UI
    can be honest about the fact that it edited the plan.
    """
    notes: list[str] = []
    by_index = {b.day_index: b for b in budgets}

    for day in itinerary.days:
        budget = by_index.get(day.day_index)
        if budget is None:
            continue

        # Enforce the one-high rule first, then shave until the cost fits.
        max_high = 1 if budget.high_allowed else 0
        highs = [a for a in day.activities if a.intensity is Intensity.high]
        for extra in highs[max_high:]:
            day.activities.remove(extra)
            notes.append(f"Day {day.day_index + 1}: removed '{extra.name}' (too intense)")

        while day_cost(day) > budget.points and day.activities:
            costliest: Activity = max(
                day.activities, key=lambda a: INTENSITY_COST[a.intensity]
            )
            day.activities.remove(costliest)
            notes.append(
                f"Day {day.day_index + 1}: removed '{costliest.name}' to stay "
                "within the day's energy budget"
            )

    return notes
