"""Cost Lab — sweep `temperature` × `max_output_tokens` and measure the effect.

Each sweep cell yields two axes that can be compared objectively:

  * **cost** — real token counts from the API's own `usage` block
  * **quality** — how many pacing rules the model's first draft broke, per
    `validate_itinerary()`

That turns "which parameters should we use?" into a computable question rather
than a matter of taste.

No Streamlit imports here — this is pure orchestration, and `planner_fn` is
injected so the whole path is testable without spending a penny.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal

from .llm import PlanResult, TruncatedOutputError, plan_trip
from .models import TripRequest
from .usage import UsageTotals, format_cost

# A sweep is N real API calls. This ceiling is enforced here, not just in the
# UI, so no caller can accidentally start a 200-call run.
MAX_SWEEP_RUNS = 24

DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
DEFAULT_MAX_OUTPUT_TOKENS: list[int | None] = [2000, 4000, 8000, None]

Status = Literal["ok", "truncated", "error"]


class SweepTooLarge(ValueError):
    pass


@dataclass
class SweepConfig:
    temperatures: list[float] = field(default_factory=lambda: list(DEFAULT_TEMPERATURES))
    max_output_tokens: list[int | None] = field(
        default_factory=lambda: list(DEFAULT_MAX_OUTPUT_TOKENS)
    )
    samples: int = 1
    repair: bool = False

    @property
    def total_runs(self) -> int:
        return len(self.temperatures) * len(self.max_output_tokens) * self.samples

    def cells(self) -> list[tuple[float, int | None, int]]:
        return [
            (temperature, cap, sample)
            for temperature in self.temperatures
            for cap in self.max_output_tokens
            for sample in range(self.samples)
        ]

    def validate(self) -> None:
        if not self.temperatures or not self.max_output_tokens:
            raise SweepTooLarge("Pick at least one temperature and one output cap.")
        if self.samples < 1:
            raise SweepTooLarge("Samples must be at least 1.")
        if self.total_runs > MAX_SWEEP_RUNS:
            raise SweepTooLarge(
                f"{self.total_runs} runs exceeds the {MAX_SWEEP_RUNS}-run safety "
                "cap. Reduce the grid or the sample count."
            )


@dataclass
class RunResult:
    temperature: float
    max_output_tokens: int | None
    sample_index: int
    status: Status
    usage: UsageTotals = field(default_factory=UsageTotals)
    first_draft_violations: int = 0
    repair_calls: int = 0
    trimmed_activities: int = 0
    duration_s: float = 0.0
    error: str | None = None

    @property
    def cell(self) -> tuple[float, int | None]:
        return (self.temperature, self.max_output_tokens)

    @property
    def cap_label(self) -> str:
        return str(self.max_output_tokens) if self.max_output_tokens else "default"

    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd


def run_sweep(
    req: TripRequest,
    config: SweepConfig,
    planner_fn: Callable[..., PlanResult] = plan_trip,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[RunResult]:
    """Run the grid, yielding each result as it lands.

    A generator so the UI can render progressively instead of freezing behind a
    single spinner for several minutes.

    Failures are recorded, never raised: a truncated or errored run is a data
    point about those parameters, and aborting would waste every call already
    paid for.
    """
    config.validate()

    for temperature, cap, sample in config.cells():
        if should_stop is not None and should_stop():
            return

        started = time.monotonic()
        try:
            result = planner_fn(
                req,
                temperature=temperature,
                max_output_tokens=cap,
                repair=config.repair,
            )
        except TruncatedOutputError as exc:
            # Expected at low caps. The tokens were still billed, so keep them.
            usage = UsageTotals()
            usage.add(exc.usage)
            yield RunResult(
                temperature=temperature,
                max_output_tokens=cap,
                sample_index=sample,
                status="truncated",
                usage=usage,
                duration_s=time.monotonic() - started,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the sweep
            yield RunResult(
                temperature=temperature,
                max_output_tokens=cap,
                sample_index=sample,
                status="error",
                duration_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            yield RunResult(
                temperature=temperature,
                max_output_tokens=cap,
                sample_index=sample,
                status="ok",
                usage=result.usage,
                first_draft_violations=len(result.first_draft_violations),
                repair_calls=result.repair_calls,
                trimmed_activities=len(result.trim_notes),
                duration_s=result.duration_s or (time.monotonic() - started),
            )


# --- Aggregation -------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class CellSummary:
    """Averaged results for one (temperature, cap) pair across its samples."""

    temperature: float
    max_output_tokens: int | None
    samples: int
    ok: int
    truncated: int
    errored: int
    avg_cost_usd: float | None
    avg_input_tokens: float
    avg_output_tokens: float
    avg_violations: float
    avg_repair_calls: float
    avg_duration_s: float
    wasted_tokens: int

    @property
    def cap_label(self) -> str:
        return str(self.max_output_tokens) if self.max_output_tokens else "default"

    @property
    def is_clean(self) -> bool:
        """Every sample produced a valid itinerary with no rule violations."""
        return (
            self.ok == self.samples
            and self.truncated == 0
            and self.errored == 0
            and self.avg_violations == 0
        )


@dataclass
class SweepReport:
    runs: list[RunResult]
    cells: list[CellSummary]
    recommendation: CellSummary | None
    recommendation_reason: str

    @property
    def total_cost_usd(self) -> float | None:
        costs = [r.cost_usd for r in self.runs]
        if not costs or any(c is None for c in costs):
            return None
        return sum(costs)  # type: ignore[arg-type]

    @property
    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.runs)

    @property
    def wasted_tokens(self) -> int:
        """Tokens paid for on runs that produced nothing usable."""
        return sum(
            r.usage.total_tokens for r in self.runs if r.status in ("truncated", "error")
        )

    @property
    def min_samples(self) -> int:
        return min((c.samples for c in self.cells), default=0)


def summarise(runs: list[RunResult]) -> SweepReport:
    """Aggregate runs per cell and pick a recommended configuration."""
    grouped: dict[tuple[float, int | None], list[RunResult]] = {}
    for run in runs:
        grouped.setdefault(run.cell, []).append(run)

    cells: list[CellSummary] = []
    for (temperature, cap), cell_runs in grouped.items():
        ok_runs = [r for r in cell_runs if r.status == "ok"]
        costs = [r.cost_usd for r in ok_runs if r.cost_usd is not None]
        cells.append(
            CellSummary(
                temperature=temperature,
                max_output_tokens=cap,
                samples=len(cell_runs),
                ok=len(ok_runs),
                truncated=sum(1 for r in cell_runs if r.status == "truncated"),
                errored=sum(1 for r in cell_runs if r.status == "error"),
                avg_cost_usd=_mean(costs) if costs else None,
                avg_input_tokens=_mean([r.usage.input_tokens for r in ok_runs]),
                avg_output_tokens=_mean([r.usage.output_tokens for r in ok_runs]),
                avg_violations=_mean([float(r.first_draft_violations) for r in ok_runs]),
                avg_repair_calls=_mean([float(r.repair_calls) for r in ok_runs]),
                avg_duration_s=_mean([r.duration_s for r in cell_runs]),
                wasted_tokens=sum(
                    r.usage.total_tokens
                    for r in cell_runs
                    if r.status in ("truncated", "error")
                ),
            )
        )

    cells.sort(key=lambda c: (c.temperature, c.max_output_tokens or 10**9))
    recommendation, reason = _recommend(cells)
    return SweepReport(
        runs=runs,
        cells=cells,
        recommendation=recommendation,
        recommendation_reason=reason,
    )


def _recommend(cells: list[CellSummary]) -> tuple[CellSummary | None, str]:
    """Cheapest configuration that reliably produced a valid itinerary.

    Falls back to fewest violations when nothing is clean, because "cheapest"
    alone would happily recommend a config that never produces usable output.
    """
    priced = [c for c in cells if c.avg_cost_usd is not None]
    if not priced:
        return None, "No run produced a priced, successful itinerary."

    clean = [c for c in priced if c.is_clean]
    if clean:
        best = min(clean, key=lambda c: c.avg_cost_usd or 0.0)
        return best, (
            f"Cheapest configuration that produced a valid itinerary in every "
            f"sample, at {format_cost(best.avg_cost_usd)} per run."
        )

    best = min(priced, key=lambda c: (c.avg_violations, c.avg_cost_usd or 0.0))
    return best, (
        "No configuration was violation-free, so this is the one with the fewest "
        f"first-draft violations ({best.avg_violations:.1f} on average) at "
        f"{format_cost(best.avg_cost_usd)} per run. Keep the repair pass enabled."
    )


# --- Export ------------------------------------------------------------------

CSV_COLUMNS = [
    "temperature",
    "max_output_tokens",
    "sample",
    "status",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "first_draft_violations",
    "repair_calls",
    "trimmed_activities",
    "duration_s",
    "error",
]


def to_csv(runs: list[RunResult]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for r in runs:
        writer.writerow(
            [
                r.temperature,
                r.max_output_tokens if r.max_output_tokens is not None else "",
                r.sample_index,
                r.status,
                r.usage.input_tokens,
                r.usage.output_tokens,
                r.usage.total_tokens,
                f"{r.cost_usd:.6f}" if r.cost_usd is not None else "",
                r.first_draft_violations,
                r.repair_calls,
                r.trimmed_activities,
                f"{r.duration_s:.2f}",
                (r.error or "").replace("\n", " "),
            ]
        )
    return buffer.getvalue()
