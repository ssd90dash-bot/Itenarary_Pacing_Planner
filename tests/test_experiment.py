"""Tests for the Cost Lab sweep engine.

Every test injects a fake planner, so the full sweep → aggregate → report path
is exercised with zero API calls and zero spend.
"""

from datetime import date

import pytest

from planner.experiment import (
    MAX_SWEEP_RUNS,
    RunResult,
    SweepConfig,
    SweepTooLarge,
    run_sweep,
    summarise,
    to_csv,
)
from planner.llm import PlanResult, TruncatedOutputError
from planner.models import Itinerary, Pace, Traveller, TripRequest
from planner.pacing import Violation, build_day_budgets
from planner.usage import CallUsage, UsageTotals


@pytest.fixture
def request_():
    return TripRequest(
        destination="Lisbon",
        start_date=date(2026, 9, 1),
        end_date=date(2026, 9, 3),
        travellers=[Traveller(age=35)],
        pace=Pace.balanced,
    )


def fake_result(*, violations=0, input_tokens=1000, output_tokens=500, repairs=0):
    usage = UsageTotals()
    usage.add(
        CallUsage(
            label="draft",
            model="gpt-4o-mini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    )
    for i in range(repairs):
        usage.add(
            CallUsage(
                label=f"repair {i}",
                model="gpt-4o-mini",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
    return PlanResult(
        itinerary=Itinerary(days=[], overall_notes=""),
        budgets=[],
        usage=usage,
        first_draft_violations=[
            Violation(0, "over_budget", "…") for _ in range(violations)
        ],
        duration_s=1.0,
    )


def planner_that_returns(**kwargs):
    def _planner(req, *, temperature, max_output_tokens, repair):
        return fake_result(**kwargs)

    return _planner


# --- Grid expansion ----------------------------------------------------------


def test_grid_expands_to_temps_times_caps_times_samples(request_):
    config = SweepConfig(
        temperatures=[0.0, 0.5], max_output_tokens=[1000, None], samples=2
    )
    assert config.total_runs == 8
    runs = list(run_sweep(request_, config, planner_fn=planner_that_returns()))
    assert len(runs) == 8


def test_every_combination_appears_exactly_once(request_):
    config = SweepConfig(temperatures=[0.0, 1.0], max_output_tokens=[500, None])
    runs = list(run_sweep(request_, config, planner_fn=planner_that_returns()))
    assert set(r.cell for r in runs) == {
        (0.0, 500),
        (0.0, None),
        (1.0, 500),
        (1.0, None),
    }


def test_parameters_reach_the_planner(request_):
    seen = []

    def _planner(req, *, temperature, max_output_tokens, repair):
        seen.append((temperature, max_output_tokens, repair))
        return fake_result()

    config = SweepConfig(temperatures=[0.4], max_output_tokens=[1234], repair=False)
    list(run_sweep(request_, config, planner_fn=_planner))
    assert seen == [(0.4, 1234, False)]


# --- Safety cap --------------------------------------------------------------


def test_run_cap_raises_before_any_call_is_made(request_):
    called = []

    def _planner(req, **kwargs):
        called.append(1)
        return fake_result()

    config = SweepConfig(
        temperatures=[0.0, 0.2, 0.4, 0.6, 0.8],
        max_output_tokens=[500, 1000, 2000, 4000, None],
        samples=2,
    )
    assert config.total_runs > MAX_SWEEP_RUNS
    with pytest.raises(SweepTooLarge):
        list(run_sweep(request_, config, planner_fn=_planner))
    assert not called, "the cap must be enforced before spending anything"


def test_empty_grid_is_rejected(request_):
    with pytest.raises(SweepTooLarge):
        list(run_sweep(request_, SweepConfig(temperatures=[], max_output_tokens=[100])))


def test_should_stop_halts_the_sweep(request_):
    calls = []

    def _planner(req, **kwargs):
        calls.append(1)
        return fake_result()

    config = SweepConfig(temperatures=[0.0, 0.5, 1.0], max_output_tokens=[100])
    runs = list(
        run_sweep(
            request_, config, planner_fn=_planner, should_stop=lambda: len(calls) >= 2
        )
    )
    assert len(runs) == 2


# --- Failure handling --------------------------------------------------------


def test_truncation_is_recorded_with_its_tokens_and_the_sweep_continues(request_):
    usage = CallUsage(
        label="draft", model="gpt-4o-mini", input_tokens=900, output_tokens=100
    )

    def _planner(req, *, temperature, max_output_tokens, repair):
        if max_output_tokens == 100:
            raise TruncatedOutputError("hit the cap", usage=usage, finish_reason="length")
        return fake_result()

    config = SweepConfig(temperatures=[0.0], max_output_tokens=[100, None])
    runs = list(run_sweep(request_, config, planner_fn=_planner))

    assert len(runs) == 2, "a truncated cell must not abort the sweep"
    truncated = next(r for r in runs if r.status == "truncated")
    assert truncated.usage.total_tokens == 1000, "billed tokens must be preserved"
    assert next(r for r in runs if r.status == "ok")


def test_arbitrary_errors_are_recorded_without_aborting(request_):
    def _planner(req, *, temperature, max_output_tokens, repair):
        if temperature == 0.0:
            raise ValueError("boom")
        return fake_result()

    config = SweepConfig(temperatures=[0.0, 1.0], max_output_tokens=[None])
    runs = list(run_sweep(request_, config, planner_fn=_planner))

    assert [r.status for r in runs] == ["error", "ok"]
    assert "boom" in runs[0].error


# --- Aggregation -------------------------------------------------------------


def make_run(temp, cap, *, status="ok", violations=0, cost_tokens=(1000, 500), sample=0):
    usage = UsageTotals()
    usage.add(
        CallUsage(
            label="draft",
            model="gpt-4o-mini",
            input_tokens=cost_tokens[0],
            output_tokens=cost_tokens[1],
        )
    )
    return RunResult(
        temperature=temp,
        max_output_tokens=cap,
        sample_index=sample,
        status=status,
        usage=usage,
        first_draft_violations=violations,
    )


def test_samples_are_averaged_not_overwritten():
    runs = [
        make_run(0.0, None, violations=0, sample=0),
        make_run(0.0, None, violations=4, sample=1),
    ]
    cell = summarise(runs).cells[0]
    assert cell.samples == 2
    assert cell.avg_violations == 2.0, "must average, not take the last value"


def test_recommendation_picks_the_cheapest_clean_cell():
    runs = [
        # Clean but expensive.
        make_run(0.0, None, violations=0, cost_tokens=(5000, 3000)),
        # Clean and cheap — should win.
        make_run(0.3, None, violations=0, cost_tokens=(1000, 500)),
        # Cheapest overall, but breaks rules.
        make_run(1.0, None, violations=3, cost_tokens=(100, 50)),
    ]
    report = summarise(runs)
    assert report.recommendation.temperature == 0.3
    assert "valid itinerary" in report.recommendation_reason


def test_recommendation_falls_back_to_fewest_violations_when_none_are_clean():
    runs = [
        make_run(0.0, None, violations=2, cost_tokens=(1000, 500)),
        make_run(1.0, None, violations=5, cost_tokens=(100, 50)),
    ]
    report = summarise(runs)
    assert report.recommendation.temperature == 0.0
    assert "No configuration was violation-free" in report.recommendation_reason


def test_a_cell_with_any_truncated_sample_is_not_clean():
    runs = [
        make_run(0.0, 500, violations=0, sample=0),
        make_run(0.0, 500, status="truncated", sample=1),
    ]
    assert not summarise(runs).cells[0].is_clean


def test_wasted_tokens_counts_only_failed_runs():
    runs = [
        make_run(0.0, None, cost_tokens=(1000, 500)),
        make_run(0.0, 100, status="truncated", cost_tokens=(900, 100)),
    ]
    report = summarise(runs)
    assert report.wasted_tokens == 1000
    assert report.total_tokens == 2500


def test_report_with_no_successful_runs_has_no_recommendation():
    report = summarise([make_run(0.0, 100, status="error")])
    assert report.recommendation is None
    assert "No run produced" in report.recommendation_reason


# --- Export ------------------------------------------------------------------


def test_csv_has_a_header_and_one_row_per_run():
    runs = [make_run(0.0, None), make_run(1.0, 500, status="truncated")]
    lines = to_csv(runs).strip().splitlines()
    assert lines[0].startswith("temperature,max_output_tokens")
    assert len(lines) == 3


def test_csv_survives_an_error_containing_newlines():
    run = make_run(0.0, None, status="error")
    run.error = "line one\nline two"
    assert len(to_csv([run]).strip().splitlines()) == 2
