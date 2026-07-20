"""Modelling cost at volume.

A per-request cost of a fraction of a cent tells you almost nothing on its own.
The business question is what happens at 10,000 requests a month, and the answer
depends on two things people routinely forget to model:

  * **Cache hit rate.** A repeated prompt prefix bills at a reduced rate. A
    measurement taken by hammering one identical prompt (as a parameter sweep
    does) is the *optimistic* extreme and will understate real cost, where users
    arrive with cold, differing prompts.
  * **Repair rate.** A response that fails validation costs a second call. At
    scale that is a multiplier on the whole bill, not a rounding error.
"""

from __future__ import annotations

from dataclasses import dataclass

from .usage import CallUsage, price_for

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class ScaleScenario:
    """One defined API-call scenario."""

    name: str
    itineraries_per_month: int
    cache_hit_rate: float = 0.0
    repair_rate: float = 0.0
    note: str = ""


# Cache/repair rates here are assumptions, not measurements — they are the
# levers to sensitivity-test, and the report labels them as such.
DEFAULT_SCENARIOS = [
    ScaleScenario("Pilot", 100, 0.0, 0.0, "Internal trial, cold prompts"),
    ScaleScenario("Launch", 1_000, 0.3, 0.1, "Early users, some prompt reuse"),
    ScaleScenario("Growth", 10_000, 0.5, 0.1, "Steady traffic"),
    ScaleScenario("Scale", 100_000, 0.7, 0.1, "High reuse of a stable prompt"),
]


@dataclass
class ScaleProjection:
    scenario: ScaleScenario
    model: str
    calls_per_month: float
    input_tokens_per_month: float
    output_tokens_per_month: float
    cached_input_tokens_per_month: float
    monthly_cost_usd: float | None

    @property
    def annual_cost_usd(self) -> float | None:
        if self.monthly_cost_usd is None:
            return None
        return self.monthly_cost_usd * MONTHS_PER_YEAR

    @property
    def cost_per_itinerary_usd(self) -> float | None:
        if self.monthly_cost_usd is None or not self.scenario.itineraries_per_month:
            return None
        return self.monthly_cost_usd / self.scenario.itineraries_per_month

    @property
    def total_tokens_per_month(self) -> float:
        return self.input_tokens_per_month + self.output_tokens_per_month


def project(
    input_tokens: int,
    output_tokens: int,
    scenario: ScaleScenario,
    model: str = "gpt-4o-mini",
) -> ScaleProjection:
    """Project one itinerary's measured token usage out to a monthly volume.

    A repair is modelled as a full extra call: it resends the conversation plus
    the rejected itinerary, so its input is at least as large as the original.
    Treating it as a whole call is the conservative reading.
    """
    volume = scenario.itineraries_per_month
    calls = volume * (1 + scenario.repair_rate)

    total_input = input_tokens * calls
    total_output = output_tokens * calls
    cached_input = total_input * scenario.cache_hit_rate

    pricing = price_for(model)
    if pricing is None:
        monthly = None
    else:
        # Reuse the single-call cost calculation rather than duplicating rates.
        monthly = CallUsage(
            label=scenario.name,
            model=model,
            input_tokens=int(total_input),
            output_tokens=int(total_output),
            cached_input_tokens=int(cached_input),
        ).cost_usd

    return ScaleProjection(
        scenario=scenario,
        model=model,
        calls_per_month=calls,
        input_tokens_per_month=total_input,
        output_tokens_per_month=total_output,
        cached_input_tokens_per_month=cached_input,
        monthly_cost_usd=monthly,
    )


def project_all(
    input_tokens: int,
    output_tokens: int,
    scenarios: list[ScaleScenario] | None = None,
    model: str = "gpt-4o-mini",
) -> list[ScaleProjection]:
    return [
        project(input_tokens, output_tokens, scenario, model)
        for scenario in (scenarios or DEFAULT_SCENARIOS)
    ]


def cache_sensitivity(
    input_tokens: int,
    output_tokens: int,
    volume: int,
    model: str = "gpt-4o-mini",
    rates: tuple[float, ...] = (0.0, 0.5, 0.96),
) -> list[ScaleProjection]:
    """Same volume, different cache assumptions — the honest answer is a range.

    0.96 is the rate observed in the project's own sweep, which reused one
    identical prompt and is therefore a best case rather than a forecast.
    """
    return [
        project(
            input_tokens,
            output_tokens,
            ScaleScenario(
                name=f"{rate:.0%} cached",
                itineraries_per_month=volume,
                cache_hit_rate=rate,
            ),
            model,
        )
        for rate in rates
    ]
