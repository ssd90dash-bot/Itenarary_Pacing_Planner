"""Tests for usage-at-scale projection. No API calls."""

import pytest

from planner.scale import (
    DEFAULT_SCENARIOS,
    ScaleScenario,
    cache_sensitivity,
    project,
    project_all,
)

# The measured baseline from the project's own live sweep.
BASELINE_INPUT = 1338
BASELINE_OUTPUT = 1136


def scenario(volume=1000, cache=0.0, repair=0.0):
    return ScaleScenario(
        name="test",
        itineraries_per_month=volume,
        cache_hit_rate=cache,
        repair_rate=repair,
    )


# --- Linearity ---------------------------------------------------------------


def test_cost_scales_linearly_with_volume():
    small = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(volume=1_000))
    large = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(volume=10_000))
    assert large.monthly_cost_usd == pytest.approx(small.monthly_cost_usd * 10, rel=1e-6)


def test_annual_is_twelve_months():
    p = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario())
    assert p.annual_cost_usd == pytest.approx(p.monthly_cost_usd * 12)


def test_zero_volume_costs_nothing():
    p = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(volume=0))
    assert p.monthly_cost_usd == 0
    assert p.cost_per_itinerary_usd is None


# --- The two levers ----------------------------------------------------------


def test_caching_reduces_cost():
    cold = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(cache=0.0))
    warm = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(cache=0.96))
    assert warm.monthly_cost_usd < cold.monthly_cost_usd


def test_repairs_add_calls_and_cost():
    clean = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(repair=0.0))
    messy = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(repair=0.5))
    assert messy.calls_per_month == clean.calls_per_month * 1.5
    assert messy.monthly_cost_usd > clean.monthly_cost_usd


def test_repair_rate_multiplies_tokens_too():
    p = project(1000, 500, scenario(volume=100, repair=0.2))
    assert p.calls_per_month == 120
    assert p.input_tokens_per_month == 120_000
    assert p.output_tokens_per_month == 60_000


def test_cost_per_itinerary_accounts_for_repairs():
    """Repairs make each *itinerary* pricier even though tokens per call don't change."""
    clean = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(repair=0.0))
    messy = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(repair=0.5))
    assert messy.cost_per_itinerary_usd > clean.cost_per_itinerary_usd


# --- Models ------------------------------------------------------------------


def test_cheaper_model_costs_less_at_equal_volume():
    mini = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(), model="gpt-4o-mini")
    full = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(), model="gpt-4o")
    assert mini.monthly_cost_usd < full.monthly_cost_usd


def test_unpriced_model_yields_no_cost_but_still_counts_tokens():
    p = project(BASELINE_INPUT, BASELINE_OUTPUT, scenario(), model="mystery-model")
    assert p.monthly_cost_usd is None
    assert p.annual_cost_usd is None
    assert p.total_tokens_per_month > 0


# --- Scenario sets -----------------------------------------------------------


def test_project_all_covers_every_default_scenario():
    projections = project_all(BASELINE_INPUT, BASELINE_OUTPUT)
    assert len(projections) == len(DEFAULT_SCENARIOS)
    costs = [p.monthly_cost_usd for p in projections]
    assert costs == sorted(costs), "higher volume tiers must not cost less"


def test_cache_sensitivity_spans_the_assumptions():
    projections = cache_sensitivity(BASELINE_INPUT, BASELINE_OUTPUT, volume=10_000)
    costs = [p.monthly_cost_usd for p in projections]
    assert costs == sorted(costs, reverse=True), "more caching must cost less"
    assert len(projections) == 3
