"""FastAPI application: routes that adapt HTTP to the `planner/` package."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from planner.experiment import SweepConfig, SweepTooLarge, run_sweep, summarise
from planner.llm import (
    DEFAULT_MODEL,
    MissingAPIKey,
    TruncatedOutputError,
    plan_trip,
)
from planner.models import Interest, Pace, TripRequest
from planner.pacing import build_day_budgets, day_cost
from planner.prompts import SYSTEM_PROMPT, build_user_prompt
from planner.render import INTENSITY_BADGE, day_headline, to_markdown
from planner.scale import (
    DEFAULT_SCENARIOS,
    ScaleScenario,
    cache_sensitivity,
    project,
    project_all,
)
from planner.tokenizer import compare, count_messages
from planner.usage import PRICING, estimate_across_models, format_cost, price_for

from . import charts, session
from .forms import FOOD_OPTIONS, FormError, parse_sweep_config, parse_trip_request

BASE_DIR = Path(__file__).resolve().parent

# `plan_trip` and `run_sweep` are module-level so tests can monkeypatch them to
# fakes and exercise every route without spending money.
_planner = plan_trip
_sweeper = run_sweep

app = FastAPI(title="Itinerary Planner")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("WEB_SECRET_KEY", "dev-only-not-secret-change-me"),
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["cost"] = format_cost
templates.env.filters["commas"] = lambda n: f"{int(n):,}"


def _base_context(request: Request) -> dict:
    """Values every page needs: the session token bar and nav state."""
    usage = session.read_totals(request)
    return {
        "request": request,
        "model": DEFAULT_MODEL,
        "session_usage": usage,
        "session_cost": format_cost(usage.cost_usd),
        "has_key": bool(os.getenv("OPENAI_API_KEY")),
    }


# --- Plan a trip -------------------------------------------------------------


def _form_defaults() -> dict:
    today = date.today()
    return {
        "destination": "",
        "start_date": (today + timedelta(days=30)).isoformat(),
        "end_date": (today + timedelta(days=34)).isoformat(),
        "ages": [35, 35],
        "food_options": FOOD_OPTIONS,
        "food_prefs": [],
        "food_notes": "",
        "interests": list(Interest),
        "chosen_interests": {Interest.history.value: 3, Interest.experiences.value: 3},
        "paces": list(Pace),
        "pace": Pace.balanced.value,
    }


def _budget_cards(request_obj) -> list[dict]:
    return [
        {
            "day": b.day_index + 1,
            "date": f"{b.day_date:%a %d %b}",
            "points": b.points,
            "is_recovery": b.is_recovery,
            "high_allowed": b.high_allowed,
            "reasons": b.reasons,
        }
        for b in build_day_budgets(request_obj)
    ]


@app.get("/", response_class=HTMLResponse)
def plan_form(request: Request):
    ctx = _base_context(request)
    ctx.update({"nav": "plan", "form": _form_defaults(), "result": None})
    return templates.TemplateResponse(request, "plan.html", ctx)


@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request):
    """Free budget preview — no API call. Used by the form's live estimate."""
    form = await request.form()
    try:
        trip = parse_trip_request(form)
    except FormError as exc:
        return HTMLResponse(f'<p class="text-rose-600 text-sm">{exc}</p>')
    ctx = {"request": request, "cards": _budget_cards(trip)}
    return templates.TemplateResponse(request, "_forecast.html", ctx)


# --- JSON API ----------------------------------------------------------------
# A JSON-in / JSON-out sibling of /forecast, for Postman and automated tests.
# FastAPI validates the TripRequest body automatically (422 on bad input) and
# documents it at /docs. Pure computation — free, no API call, no key needed.


class BudgetDay(BaseModel):
    day_index: int
    date: str
    points: float
    is_arrival: bool
    is_departure: bool
    is_recovery: bool
    high_allowed: bool
    requires_midday_rest: bool
    pit_stop: str | None
    travel_hours: float
    reasons: list[str]


class ForecastResponse(BaseModel):
    destination: str
    num_days: int
    days: list[BudgetDay]


@app.post("/api/forecast", response_model=ForecastResponse)
def api_forecast(trip: TripRequest) -> ForecastResponse:
    """Return the per-day energy budget as JSON (no itinerary, no cost)."""
    budgets = build_day_budgets(trip)
    return ForecastResponse(
        destination=trip.destination,
        num_days=trip.num_days,
        days=[
            BudgetDay(
                day_index=b.day_index,
                date=b.day_date.isoformat(),
                points=b.points,
                is_arrival=b.is_arrival,
                is_departure=b.is_departure,
                is_recovery=b.is_recovery,
                high_allowed=b.high_allowed,
                requires_midday_rest=b.requires_midday_rest,
                pit_stop=b.pit_stop,
                travel_hours=b.travel_hours,
                reasons=b.reasons,
            )
            for b in budgets
        ],
    )


@app.post("/plan", response_class=HTMLResponse)
async def do_plan(request: Request):
    form = await request.form()
    ctx = _base_context(request)
    ctx.update({"nav": "plan", "form": _form_defaults()})

    try:
        trip = parse_trip_request(form)
    except FormError as exc:
        ctx.update({"result": None, "error": str(exc)})
        return templates.TemplateResponse(request, "plan.html", ctx)

    try:
        result = _planner(trip)
    except MissingAPIKey as exc:
        ctx.update({"result": None, "error": str(exc)})
        return templates.TemplateResponse(request, "plan.html", ctx)
    except Exception as exc:  # surface the real error, not a blank page
        ctx.update({"result": None, "error": f"{type(exc).__name__}: {exc}"})
        return templates.TemplateResponse(request, "plan.html", ctx)

    session.add_usage(request, result.usage, DEFAULT_MODEL)
    session.set_last_trip(request, trip)  # the Cost Lab sweeps this trip

    ctx.update(_base_context(request))  # refresh the token bar post-generation
    ctx["nav"] = "plan"
    ctx["form"] = _form_defaults()
    ctx["result"] = _render_result(trip, result)
    return templates.TemplateResponse(request, "plan.html", ctx)


def _render_result(trip, result) -> dict:
    """Flatten a PlanResult into template-friendly data."""
    budgets = {b.day_index: b for b in result.budgets}
    days = []
    for day in result.itinerary.days:
        budget = budgets.get(day.day_index)
        days.append(
            {
                "headline": day_headline(budget) if budget else f"Day {day.day_index + 1}",
                "summary": day.summary,
                "used": day_cost(day),
                "points": budget.points if budget else 0,
                "pct": min(day_cost(day) / budget.points, 1.0) if budget and budget.points else 0,
                "reasons": budget.reasons if budget else [],
                "activities": [
                    {
                        "time": f"{a.start_time}–{a.end_time}",
                        "name": a.name,
                        "badge": INTENSITY_BADGE[a.intensity],
                        "intensity": a.intensity.value,
                        "interest": a.interest.value,
                        "description": a.description,
                        "why": a.why,
                    }
                    for a in day.activities
                ],
                "meals": [
                    {"time": m.time, "slot": m.slot, "suggestion": m.suggestion, "note": m.dietary_note}
                    for m in day.meals
                ],
                "rests": [
                    {"time": f"{r.start_time}–{r.end_time}", "note": r.note}
                    for r in day.rest_blocks
                ],
            }
        )

    usage = result.usage
    # Tokenizer check: local count of this exact prompt vs what was billed.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(trip, result.budgets)},
    ]
    local = count_messages(messages, DEFAULT_MODEL)
    tok = compare(local, usage.calls[0].input_tokens) if usage.calls else None

    estimates = estimate_across_models(
        usage.input_tokens, usage.output_tokens, usage.cached_input_tokens, DEFAULT_MODEL
    )
    cheapest = estimates[0].cost_usd if estimates else 0.0

    return {
        "notes": result.itinerary.overall_notes,
        "trim_notes": result.trim_notes,
        "violations": [str(v) for v in result.violations],
        "days": days,
        "usage": usage,
        "usage_cost": format_cost(usage.cost_usd),
        "markdown": to_markdown(trip, result),
        "tokenizer": tok,
        "model_costs": [
            {
                "model": e.model + ("  (in use)" if e.is_current else ""),
                "cost": format_cost(e.cost_usd),
                "ratio": "—" if not cheapest else f"{e.cost_usd / cheapest:.1f}x",
            }
            for e in estimates
        ],
        "unpriced": price_for(DEFAULT_MODEL) is None,
    }


@app.post("/download", response_class=PlainTextResponse)
async def download(request: Request):
    """Re-plan is expensive; instead the page posts back the markdown it holds."""
    form = await request.form()
    md = str(form.get("markdown", ""))
    name = str(form.get("name", "itinerary")).split(",")[0].strip().lower() or "itinerary"
    return PlainTextResponse(
        md,
        headers={"Content-Disposition": f'attachment; filename="{name}-itinerary.md"'},
    )


# --- Cost at scale -----------------------------------------------------------

MEASURED_INPUT = 1338
MEASURED_OUTPUT = 1136


@app.get("/scale", response_class=HTMLResponse)
def scale_page(request: Request):
    return _render_scale(request, MEASURED_INPUT, MEASURED_OUTPUT, DEFAULT_MODEL, 10_000, 0.5, 0.1)


@app.post("/scale", response_class=HTMLResponse)
async def scale_post(request: Request):
    form = await request.form()

    def num(key, default):
        raw = form.get(key)
        try:
            return type(default)(raw)
        except (TypeError, ValueError):
            return default

    model = str(form.get("model", DEFAULT_MODEL))
    if model not in PRICING:
        model = DEFAULT_MODEL
    return _render_scale(
        request,
        int(num("input_tokens", MEASURED_INPUT)),
        int(num("output_tokens", MEASURED_OUTPUT)),
        model,
        int(num("volume", 10_000)),
        float(num("cache", 0.5)),
        float(num("repair", 0.1)),
    )


def _render_scale(request, inp, out, model, volume, cache, repair):
    usage = session.read_totals(request)
    baseline_source = None
    ctx = _base_context(request)

    scenarios = [
        {
            "name": p.scenario.name,
            "volume": p.scenario.itineraries_per_month,
            "cache": p.scenario.cache_hit_rate,
            "repair": p.scenario.repair_rate,
            "calls": p.calls_per_month,
            "monthly": format_cost(p.monthly_cost_usd),
            "annual": format_cost(p.annual_cost_usd),
            "note": p.scenario.note,
        }
        for p in project_all(inp, out, DEFAULT_SCENARIOS, model)
    ]

    custom = project(
        inp, out,
        ScaleScenario("Custom", max(volume, 0), cache, repair),
        model,
    )
    sensitivity = [
        {
            "label": p.scenario.name,
            "monthly": format_cost(p.monthly_cost_usd),
            "annual": format_cost(p.annual_cost_usd),
            "per": format_cost(p.cost_per_itinerary_usd),
        }
        for p in cache_sensitivity(inp, out, max(volume, 1), model)
    ]

    ctx.update(
        {
            "nav": "scale",
            "inp": inp,
            "out": out,
            "model": model,
            "models": list(PRICING),
            "volume": volume,
            "cache": cache,
            "repair": repair,
            "scenarios": scenarios,
            "custom": {
                "calls": custom.calls_per_month,
                "per": format_cost(custom.cost_per_itinerary_usd),
                "monthly": format_cost(custom.monthly_cost_usd),
                "annual": format_cost(custom.annual_cost_usd),
            },
            "sensitivity": sensitivity,
            "chart": charts.scale_curve(inp, out, model),
            "app_model": DEFAULT_MODEL,
        }
    )
    return templates.TemplateResponse(request, "scale.html", ctx)


# --- Cost Lab ----------------------------------------------------------------

TEMPERATURE_OPTIONS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3]
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
CAP_OPTIONS = ["2000", "4000", "8000", "uncapped"]
DEFAULT_CAPS = ["2000", "4000", "8000", "uncapped"]


@app.get("/lab", response_class=HTMLResponse)
def lab_page(request: Request):
    ctx = _base_context(request)
    usage = session.read_totals(request)
    trip = session.get_last_trip(request)
    # Per-run estimate extrapolated from real usage, as in the Streamlit lab.
    per_run = usage.cost_usd if usage.calls else None
    ctx.update(
        {
            "nav": "lab",
            "temperatures": TEMPERATURE_OPTIONS,
            "default_temperatures": DEFAULT_TEMPERATURES,
            "caps": CAP_OPTIONS,
            "default_caps": DEFAULT_CAPS,
            "max_series": len(charts.SERIES),
            "has_baseline": usage.calls > 0,
            "trip_destination": trip.destination if trip else None,
            "trip_days": trip.num_days if trip else None,
            "per_run_cost": format_cost(per_run),
            "per_run_raw": per_run or 0.0,
        }
    )
    return templates.TemplateResponse(request, "lab.html", ctx)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.get("/lab/stream")
def lab_stream(request: Request):
    """Stream a sweep over SSE, one event per run then a final report event.

    The blocking generator runs in a threadpool (FastAPI iterates a sync
    generator off the event loop), so long API calls don't block the server.
    """
    params = request.query_params

    trip = session.get_last_trip(request)
    if session.read_totals(request).calls == 0 or trip is None:
        def refuse():
            yield _sse(
                "error",
                {"message": "Generate one itinerary first — the sweep runs that trip "
                            "and estimates cost from the real run rather than guessing."},
            )
        return StreamingResponse(refuse(), media_type="text/event-stream")

    try:
        config = parse_sweep_config(params)
        config.validate()
    except (FormError, SweepTooLarge) as exc:
        message = str(exc)  # bind now; `exc` is cleared when the block exits

        def refuse_bad():
            yield _sse("error", {"message": message})

        return StreamingResponse(refuse_bad(), media_type="text/event-stream")

    def generate():
        runs = []
        try:
            for run in _sweeper(trip, config, planner_fn=_planner):
                runs.append(run)
                yield _sse(
                    "run",
                    {
                        "index": len(runs),
                        "total": config.total_runs,
                        "temperature": run.temperature,
                        "cap": run.cap_label,
                        "status": run.status,
                        "input": run.usage.input_tokens,
                        "output": run.usage.output_tokens,
                        "cost": format_cost(run.cost_usd),
                        "violations": run.first_draft_violations if run.status == "ok" else None,
                        "error": (run.error or "")[:120],
                    },
                )
        except Exception as exc:  # noqa: BLE001 — report, keep what was paid for
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

        for run in runs:
            session.add_usage(request, run.usage, DEFAULT_MODEL)

        report = summarise(runs)
        best = report.recommendation
        yield _sse(
            "done",
            {
                "total_cost": format_cost(report.total_cost_usd),
                "total_tokens": report.total_tokens,
                "wasted_tokens": report.wasted_tokens,
                "min_samples": report.min_samples,
                "recommendation": (
                    None
                    if best is None
                    else {
                        "temperature": best.temperature,
                        "cap": best.cap_label,
                        "reason": report.recommendation_reason,
                        "env": _env_lines(best),
                    }
                ),
                "recommendation_reason": report.recommendation_reason,
                "charts": {
                    "cost_temp": charts.cost_by_temperature(report.cells) or "",
                    "cost_viol": charts.cost_vs_violations(report.cells) or "",
                },
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


def _env_lines(cell) -> str:
    cap = (
        f"OPENAI_MAX_OUTPUT_TOKENS={cell.max_output_tokens}"
        if cell.max_output_tokens
        else "# OPENAI_MAX_OUTPUT_TOKENS left unset (no cap)"
    )
    return f"OPENAI_TEMPERATURE={cell.temperature}\n{cap}"


@app.post("/reset")
async def reset_usage(request: Request):
    session.reset(request)
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/", status_code=303)
