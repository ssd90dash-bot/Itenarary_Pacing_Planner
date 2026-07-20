"""Prompt construction.

The model is never asked to pace the trip — `pacing.py` has already decided
that. These prompts hand it a fixed budget per day and ask only that it choose
good places inside those limits.
"""

from __future__ import annotations

from .models import TripRequest
from .pacing import INTENSITY_COST, DayBudget, Violation

SYSTEM_PROMPT = """\
You are an expert local travel planner. You do NOT decide how busy each day is \
— an energy-budget engine has already done that, and its limits are absolute.

Each day comes with:
  - an ENERGY BUDGET in points
  - whether one HIGH-intensity activity is permitted that day
  - any rest blocks that are mandatory

Activity costs are fixed: low = 10 points, moderate = 25, high = 45.
The total cost of a day's activities MUST NOT exceed that day's budget. It is \
correct and expected to leave headroom — an under-filled day is a good day.

Rules you must follow:
1. Never exceed a day's budget, and never place a high-intensity activity on a \
   day where it is not permitted. At most one high-intensity activity per day.
2. Every day needs at least one genuine rest block. On a high-intensity day, \
   include a rest block of at least 90 minutes.
3. Meals belong in `meals`, not `activities`, and must respect the stated \
   dietary requirements. State in `dietary_note` how each suggestion complies.
4. Name real, specific, well-known places in the destination. Never invent \
   venues. If unsure a place exists, choose a more famous one.
5. Group each day geographically so travellers are not criss-crossing the city.
6. In each activity's `why`, refer to these specific travellers — their ages, \
   their interests, their energy on that day.
7. Use 24-hour HH:MM times. Days must be returned in order, with `day_index` \
   matching the briefing exactly.
"""


def _traveller_line(req: TripRequest) -> str:
    ages = sorted(t.age for t in req.travellers)
    parts = [f"{len(ages)} traveller(s), ages {', '.join(str(a) for a in ages)}"]
    if any(a < 6 for a in ages):
        parts.append("includes a young child — naps and stroller-friendly pacing matter")
    if any(a > 70 for a in ages):
        parts.append("includes a senior traveller — seating, shade and short walks matter")
    return "; ".join(parts)


def _interest_targets(req: TripRequest, total_slots: int) -> str:
    if not req.interests:
        return "No specific interests stated — offer a balanced mix."
    weight_sum = sum(req.interests.values()) or 1
    lines = []
    for interest, weight in sorted(
        req.interests.items(), key=lambda kv: -kv[1]
    ):
        target = round(total_slots * weight / weight_sum)
        lines.append(f"  - {interest.value}: importance {weight}/5 → about {target} slots")
    return "\n".join(lines)


def _day_briefing(b: DayBudget) -> str:
    flags = []
    if b.is_arrival:
        flags.append("ARRIVAL DAY")
    if b.is_departure:
        flags.append("DEPARTURE DAY")
    if b.is_recovery:
        flags.append("RECOVERY DAY — low-intensity only, generous downtime")
    if b.pit_stop:
        flags.append(f"TRANSFER to {b.pit_stop} ({b.travel_hours:g}h travel)")

    lines = [
        f"Day {b.day_index + 1} ({b.day_date:%a %d %b}) — day_index={b.day_index}",
        f"  Energy budget: {b.points:g} points (~{b.max_activities} activity slots)",
        f"  High-intensity activity permitted: {'YES (at most one)' if b.high_allowed else 'NO'}",
    ]
    if b.requires_midday_rest:
        lines.append("  MANDATORY: a rest block overlapping 12:00-15:00")
    if flags:
        lines.append(f"  Notes: {' | '.join(flags)}")
    return "\n".join(lines)


def build_user_prompt(req: TripRequest, budgets: list[DayBudget]) -> str:
    total_slots = sum(b.max_activities for b in budgets)
    food = ", ".join(req.food_preferences) if req.food_preferences else "no restrictions"
    if req.food_notes:
        food += f" (also: {req.food_notes})"

    return f"""\
Plan a trip to {req.destination}.

TRAVELLERS: {_traveller_line(req)}
DATES: {req.start_date:%d %b %Y} to {req.end_date:%d %b %Y} ({req.num_days} days)
FOOD REQUIREMENTS: {food}
REQUESTED PACE: {req.pace.value}

INTEREST TARGETS (across roughly {total_slots} activity slots for the whole trip):
{_interest_targets(req, total_slots)}

PER-DAY BUDGETS — these are hard limits:

{chr(10).join(_day_briefing(b) for b in budgets)}

Return exactly {req.num_days} days, in order.
"""


def build_repair_prompt(violations: list[Violation]) -> str:
    listed = "\n".join(f"  - {v}" for v in violations)
    costs = ", ".join(f"{k.value}={v}" for k, v in INTENSITY_COST.items())
    return f"""\
Your itinerary broke the pacing rules:

{listed}

Fix ONLY these problems and return the complete corrected itinerary. Keep every
day that was already valid exactly as it was. Remember the activity costs
({costs}) and that leaving a day under-filled is perfectly acceptable — prefer
removing an activity over shortening rest.
"""
