"""Rendering helpers shared by the UI and the markdown export."""

from __future__ import annotations

from .models import Intensity, TripRequest
from .pacing import DayBudget, day_cost
from .llm import PlanResult
from .usage import format_cost

INTENSITY_BADGE = {
    Intensity.low: "🟢 low",
    Intensity.moderate: "🟡 moderate",
    Intensity.high: "🔴 high",
}


def day_headline(budget: DayBudget) -> str:
    label = f"Day {budget.day_index + 1} · {budget.day_date:%a %d %b}"
    if budget.is_recovery:
        label += "  ·  🛌 recovery day"
    elif budget.pit_stop:
        label += f"  ·  🚗 transfer to {budget.pit_stop}"
    elif budget.is_arrival:
        label += "  ·  ✈️ arrival"
    elif budget.is_departure:
        label += "  ·  ✈️ departure"
    return label


def to_markdown(req: TripRequest, result: PlanResult) -> str:
    """Full itinerary as markdown, for the download button."""
    lines = [
        f"# {req.destination}",
        f"_{req.start_date:%d %b %Y} – {req.end_date:%d %b %Y} · "
        f"{len(req.travellers)} traveller(s) · {req.pace.value} pace_",
        "",
        result.itinerary.overall_notes,
        "",
    ]

    budgets = {b.day_index: b for b in result.budgets}
    for day in result.itinerary.days:
        budget = budgets.get(day.day_index)
        lines.append(f"## {day_headline(budget) if budget else f'Day {day.day_index + 1}'}")
        if budget:
            lines.append(
                f"_Energy used {day_cost(day)} / {budget.points:g} points_"
            )
        lines += ["", f"{day.summary}", ""]

        for activity in day.activities:
            lines.append(
                f"- **{activity.start_time}–{activity.end_time} · {activity.name}** "
                f"({INTENSITY_BADGE[activity.intensity]}, {activity.interest.value})"
            )
            lines.append(f"  - {activity.description}")
            lines.append(f"  - _Why:_ {activity.why}")

        if day.meals:
            lines += ["", "**Meals**"]
            for meal in day.meals:
                lines.append(
                    f"- {meal.time} {meal.slot}: {meal.suggestion} — {meal.dietary_note}"
                )

        if day.rest_blocks:
            lines += ["", "**Rest**"]
            for rest in day.rest_blocks:
                lines.append(f"- {rest.start_time}–{rest.end_time}: {rest.note}")

        if budget and budget.reasons:
            lines += ["", "**Why this pace**"]
            lines += [f"- {reason}" for reason in budget.reasons]

        lines.append("")

    if result.trim_notes:
        lines += ["---", "", "**Adjusted automatically to respect the pacing rules:**"]
        lines += [f"- {note}" for note in result.trim_notes]

    usage = result.usage
    if usage.call_count:
        lines += [
            "---",
            "",
            "**Token usage**",
            "",
            f"- Input: {usage.input_tokens:,} tokens",
            f"- Output: {usage.output_tokens:,} tokens",
            f"- Total: {usage.total_tokens:,} tokens across {usage.call_count} API call(s)",
            f"- Estimated cost: {format_cost(usage.cost_usd)}",
        ]

    return "\n".join(lines)
