"""Unit tests for token accounting. No API calls."""

from types import SimpleNamespace

import pytest

from planner.usage import (
    PRICING,
    CallUsage,
    UsageTotals,
    estimate_across_models,
    format_cost,
    price_for,
    usage_from_response,
)


def call(label="draft", model="gpt-4o-mini", inp=1000, out=500, cached=0):
    return CallUsage(
        label=label,
        model=model,
        input_tokens=inp,
        output_tokens=out,
        cached_input_tokens=cached,
    )


# --- Pricing lookup ----------------------------------------------------------


def test_exact_model_name_resolves():
    assert price_for("gpt-4o-mini") is PRICING["gpt-4o-mini"]


def test_dated_model_id_resolves_to_its_base():
    assert price_for("gpt-4o-mini-2024-07-18") is PRICING["gpt-4o-mini"]


def test_longest_prefix_wins():
    # `gpt-4.1-mini` also starts with `gpt-4.1`; the more specific one must win.
    assert price_for("gpt-4.1-mini") is PRICING["gpt-4.1-mini"]
    assert price_for("gpt-4.1-nano") is PRICING["gpt-4.1-nano"]


def test_unknown_model_has_no_pricing():
    assert price_for("some-future-model") is None


# --- Cost --------------------------------------------------------------------


def test_cost_matches_published_rates():
    # 1M input + 1M output on gpt-4o-mini = $0.15 + $0.60
    usage = call(inp=1_000_000, out=1_000_000)
    assert usage.cost_usd == pytest.approx(0.75)


def test_cached_tokens_are_billed_at_the_discounted_rate():
    plain = call(inp=1_000_000, out=0)
    cached = call(inp=1_000_000, out=0, cached=1_000_000)
    assert cached.cost_usd < plain.cost_usd
    assert cached.cost_usd == pytest.approx(0.075)


def test_billable_input_excludes_cached_and_never_goes_negative():
    assert call(inp=1000, cached=400).billable_input_tokens == 600
    assert call(inp=100, cached=500).billable_input_tokens == 0


def test_unknown_model_reports_tokens_but_no_cost():
    usage = call(model="mystery-model")
    assert usage.total_tokens == 1500
    assert usage.cost_usd is None


# --- Aggregation -------------------------------------------------------------


def test_totals_sum_across_calls():
    totals = UsageTotals()
    totals.add(call(inp=1000, out=500))
    totals.add(call(label="repair", inp=2000, out=300))
    assert totals.input_tokens == 3000
    assert totals.output_tokens == 800
    assert totals.total_tokens == 3800
    assert totals.call_count == 2
    assert totals.cost_usd == pytest.approx(
        (3000 * 0.15 + 800 * 0.60) / 1_000_000
    )


def test_one_unpriced_call_makes_the_whole_total_unpriced():
    """A partial total would silently understate spend, so report None instead."""
    totals = UsageTotals()
    totals.add(call())
    totals.add(call(model="mystery-model"))
    assert totals.total_tokens > 0
    assert totals.cost_usd is None


def test_empty_totals_are_safe():
    totals = UsageTotals()
    assert totals.total_tokens == 0
    assert totals.call_count == 0
    assert totals.cost_usd is None


# --- Extraction from an API response ----------------------------------------


def test_usage_is_read_from_a_response():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1234,
            completion_tokens=567,
            prompt_tokens_details=SimpleNamespace(cached_tokens=128),
        )
    )
    usage = usage_from_response(response, label="draft", model="gpt-4o-mini")
    assert (usage.input_tokens, usage.output_tokens) == (1234, 567)
    assert usage.cached_input_tokens == 128
    assert usage.label == "draft"


def test_missing_usage_block_reports_zeros_rather_than_crashing():
    usage = usage_from_response(SimpleNamespace(), label="draft", model="gpt-4o-mini")
    assert usage.total_tokens == 0
    assert usage.cost_usd == 0


def test_missing_cached_details_defaults_to_zero():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    )
    usage = usage_from_response(response, label="draft", model="gpt-4o-mini")
    assert usage.cached_input_tokens == 0
    assert usage.input_tokens == 100


# --- Cross-model comparison --------------------------------------------------


def test_every_priced_model_is_estimated():
    estimates = estimate_across_models(1000, 500)
    assert {e.model for e in estimates} == set(PRICING)


def test_estimates_are_sorted_cheapest_first():
    costs = [e.cost_usd for e in estimate_across_models(100_000, 50_000)]
    assert costs == sorted(costs)


def test_estimate_matches_the_single_call_calculation():
    """The comparison view must not drift from the real cost calculation."""
    estimates = estimate_across_models(1000, 500, current_model="gpt-4o-mini")
    mini = next(e for e in estimates if e.model == "gpt-4o-mini")
    actual = CallUsage(
        label="x", model="gpt-4o-mini", input_tokens=1000, output_tokens=500
    )
    assert mini.cost_usd == pytest.approx(actual.cost_usd)


def test_current_model_is_flagged_including_dated_ids():
    estimates = estimate_across_models(100, 50, current_model="gpt-4o-mini-2024-07-18")
    flagged = [e.model for e in estimates if e.is_current]
    assert flagged == ["gpt-4o-mini"]


def test_no_model_is_flagged_when_the_current_one_is_unpriced():
    estimates = estimate_across_models(100, 50, current_model="mystery-model")
    assert not any(e.is_current for e in estimates)


def test_cached_tokens_lower_every_estimate():
    plain = {e.model: e.cost_usd for e in estimate_across_models(100_000, 1000)}
    cached = {
        e.model: e.cost_usd
        for e in estimate_across_models(100_000, 1000, cached_input_tokens=100_000)
    }
    assert all(cached[m] < plain[m] for m in plain)


# --- Formatting --------------------------------------------------------------


def test_small_costs_keep_enough_precision_to_be_useful():
    assert format_cost(0.00042) == "$0.00042"
    assert format_cost(1.2345) == "$1.2345"
    assert format_cost(None) == "n/a"
