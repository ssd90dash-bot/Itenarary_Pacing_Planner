"""Token accounting for LLM calls.

One "transaction" (a single Plan my trip click) can span several API calls —
the initial draft plus any repair passes — so usage is recorded per call and
aggregated, never assumed to be one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Pricing -----------------------------------------------------------------
# USD per 1,000,000 tokens. These are hardcoded and WILL go stale — OpenAI
# changes prices and ships new models. Verify against
# https://openai.com/api/pricing/ before trusting the cost figures.
# An unknown model is not an error: token counts still work, cost shows as None.


@dataclass(frozen=True)
class ModelPricing:
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None


PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4.1-nano": ModelPricing(0.10, 0.40, 0.025),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60, 0.10),
    "gpt-4.1": ModelPricing(2.00, 8.00, 0.50),
}


def price_for(model: str) -> ModelPricing | None:
    """Look up pricing, tolerating dated model ids like `gpt-4o-mini-2024-07-18`.

    Longest prefix wins so `gpt-4.1-mini` never resolves to `gpt-4.1`.
    """
    matches = [name for name in PRICING if model.startswith(name)]
    if not matches:
        return None
    return PRICING[max(matches, key=len)]


# --- Usage records -----------------------------------------------------------


@dataclass
class CallUsage:
    """Token usage for a single API call."""

    label: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def billable_input_tokens(self) -> int:
        """Uncached input — cached tokens are billed at the discounted rate."""
        return max(self.input_tokens - self.cached_input_tokens, 0)

    @property
    def cost_usd(self) -> float | None:
        pricing = price_for(self.model)
        if pricing is None:
            return None
        cached_rate = (
            pricing.cached_input_per_m
            if pricing.cached_input_per_m is not None
            else pricing.input_per_m
        )
        return (
            self.billable_input_tokens * pricing.input_per_m
            + self.cached_input_tokens * cached_rate
            + self.output_tokens * pricing.output_per_m
        ) / 1_000_000


@dataclass
class UsageTotals:
    """Aggregate across several calls — one transaction, or a whole session."""

    calls: list[CallUsage] = field(default_factory=list)

    def add(self, call: CallUsage) -> None:
        self.calls.append(call)

    def extend(self, calls: list[CallUsage]) -> None:
        self.calls.extend(calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def cached_input_tokens(self) -> int:
        return sum(c.cached_input_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float | None:
        """None if *any* call's model is unpriced — a partial total would mislead."""
        costs = [c.cost_usd for c in self.calls]
        if not costs or any(c is None for c in costs):
            return None
        return sum(costs)  # type: ignore[arg-type]


def usage_from_response(response, label: str, model: str) -> CallUsage:
    """Extract usage from an OpenAI response, tolerating a missing block.

    `usage` is normally present, but a malformed or streamed response can omit
    it. Reporting zeros is better than crashing a finished itinerary.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return CallUsage(label=label, model=model)

    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0

    return CallUsage(
        label=label,
        model=model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=cached,
    )


# --- Cross-model comparison --------------------------------------------------


@dataclass
class ModelCostEstimate:
    """What a given token count would have cost on one model."""

    model: str
    pricing: ModelPricing
    cost_usd: float
    is_current: bool = False


def estimate_across_models(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    current_model: str = "",
) -> list[ModelCostEstimate]:
    """Price one workload against every model in `PRICING`, cheapest first.

    This is a what-if, not a measurement: the same prompt tokenises slightly
    differently per model and a different model would produce a different
    itinerary, so treat these as order-of-magnitude comparisons.
    """
    current = max(
        (name for name in PRICING if current_model.startswith(name)),
        key=len,
        default=None,
    )

    estimates = [
        ModelCostEstimate(
            model=name,
            pricing=pricing,
            cost_usd=CallUsage(
                label="estimate",
                model=name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            ).cost_usd
            or 0.0,
            is_current=name == current,
        )
        for name, pricing in PRICING.items()
    ]
    return sorted(estimates, key=lambda e: e.cost_usd)


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "n/a"
    if cost < 0.01:
        return f"${cost:.5f}"
    return f"${cost:.4f}"
