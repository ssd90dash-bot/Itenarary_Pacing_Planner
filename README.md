# 🧭 Itinerary Planner

Plans a trip that balances activity against rest, instead of cramming every
daylight hour. You give it dates, travellers and their ages, pit stops, food
requirements and interests; it returns a day-by-day plan sized to what the group
can actually sustain.

## The idea

Ask an LLM for a travel itinerary and it will happily schedule a 72-year-old and
a 4-year-old for a nine-hour day. So the model does not decide the pacing here.

A deterministic engine in [`planner/pacing.py`](planner/pacing.py) computes an
**energy budget** for each day first. Only then is the model asked to fill those
slots, with the budget stated as a hard limit. The result is checked against the
same rules that produced it, and sent back for repair if it drifts.

```
inputs → pacing engine → per-day budgets → LLM fills slots → validate → repair
```

Because the engine is pure Python, the interesting logic is unit-tested without
spending a single token.

## How a day's budget is computed

Starting from a baseline of 100 points:

| Factor | Effect |
|---|---|
| Group stamina | Scaled by age band — the group moves near its slowest member (0.45 for under-5s up to 1.0 for adults), softened a quarter of the way toward the group mean |
| Arrival day | ×0.6 |
| Departure day | ×0.5 |
| 3+ active days in a row | ×0.85 — fatigue accumulates |
| Pit stop / transfer | −8 points per travel hour |
| Pace override | ×0.8 relaxed, ×1.0 balanced, ×1.2 packed |

Activities cost **10** (low), **25** (moderate) or **45** (high) points. A day
that lands under 45 points becomes an explicit **recovery day**.

Structural rules enforced in code, not left to the prompt:

- At most one high-intensity activity per day, never on consecutive days
- ≥90 minutes of rest on any high-intensity day
- A midday rest whenever the group includes anyone under 6 or over 70
- At least one downtime block every day, regardless of headroom
- Meals are separate scheduled slots that always respect dietary requirements

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Copy [`.env.example`](.env.example) to `.env` and fill in your key. It documents
the optional `OPENAI_MODEL`, `OPENAI_TEMPERATURE` and `OPENAI_MAX_OUTPUT_TOKENS`
settings — including why an output cap is a spend circuit-breaker rather than a
way to shorten itineraries.

`.env` is gitignored — keep it that way.

## Run

```bash
.venv/bin/streamlit run app.py
```

The sidebar shows an **energy forecast** as soon as you enter travellers and
dates. That preview is computed locally and costs nothing — only "Plan my trip"
calls the API.

## Token counting

Every API call's usage is recorded in [`planner/usage.py`](planner/usage.py) and
surfaced in the UI at three levels:

- **Per transaction** — input, output, total and estimated cost for the
  itinerary you just generated
- **Per call** — a breakdown table, because one generation is not always one
  call: a failed validation triggers a repair pass, and that costs extra tokens
- **Per session** — a running tally in the sidebar, with a reset button
- **Across models** — what the same token counts would cost on every model in
  `PRICING`, cheapest first, with a multiplier against the cheapest option and
  the model in use flagged. A rate card is also available before you generate
  anything, so you can compare without spending.

Counts come from the `usage` block on the API response, so they are the
provider's own numbers rather than a local estimate. Cached input tokens are
tracked separately and billed at the reduced rate.

> ⚠️ Prices in `PRICING` are **hardcoded and will go stale**. Check them against
> <https://openai.com/api/pricing/>. A model with no entry is handled cleanly:
> token counts stay exact and the cost shows as `n/a` rather than a wrong number.
> If any call in a total is unpriced, the whole total reports `n/a` rather than
> silently understating spend.

## Cost Lab

A second tab that runs the **same trip** across a grid of `temperature` ×
`max_completion_tokens` and reports what each combination costs.

The point is that both axes are measured, not guessed:

- **Cost** — real token counts from the API's `usage` block
- **Quality** — how many pacing rules the model's *first draft* broke, per
  `validate_itinerary()`. Having a deterministic rule engine means "did this
  configuration work?" is computable rather than a matter of taste.

It reports a recommendation — *cheapest configuration that produced a valid
itinerary in every sample* — as literal `.env` lines to paste.

**A sweep is N real API calls, so cost control is built in:**

- The estimate is extrapolated from your last real generation; with no prior
  generation the Lab refuses to run rather than guess
- An explicit confirmation checkbox states the run count and estimated spend
- A hard **24-run cap**, enforced in `run_sweep()` rather than only in the UI
- Tokens burned on truncated or errored runs are reported separately as waste

Once started, a sweep runs to completion — Streamlit cannot process a stop
button mid-loop, so there is no stop control rather than a dead one. The run cap
is the real safety net.

Caveats the report states in the UI: one sample per cell is indicative, not
conclusive; results are specific to that trip and model; and by default the
repair pass is excluded, so a real itinerary can cost more.

## Cost evaluation report

`cost-evaluation-report.pdf` is generated from the 16 live transactions in
`cost-lab-report.csv`:

```bash
.venv/bin/python scripts/make_report.py
```

Every figure traces to that CSV, the pricing table, a local `tiktoken` count, or
a labelled projection. The script refuses to build if the summary section falls
outside 150–300 words.

**Two findings worth knowing:**

*A local tokenizer understates a structured-output request.* `tiktoken` counts
the prompt at **822** tokens; the API billed **1,338**. The 516-token gap is the
JSON schema and message scaffolding added server-side — 39% of the input you
actually pay for, invisible to local counting. See the "Tokenizer check"
expander in the Plan tab.

*Prompt caching made the sweep look cheaper than production will be.* The costs
only reconcile if ~96% of input was billed at the cached rate, which is what
happens when you send one identical prompt 16 times. Real traffic won't do that,
so the **Cost at scale** tab reports a range across cache assumptions rather than
one number.

## Tests

```bash
.venv/bin/pytest              # whole suite
.venv/bin/pytest tests/test_usage.py   # one file
```

Run them through **pytest**, not `python tests/test_usage.py` — a test file
executed directly imports nothing useful and runs no tests. `pyproject.toml`
puts the project root on `sys.path`, so any pytest invocation works from the
repo root.

Covers the stamina bands, arrival/departure and fatigue multipliers, pit-stop
deduction, recovery-day promotion, the no-adjacent-big-days rule, and the
validation and trimming paths. No network access required.

## Layout

| Path | Role |
|---|---|
| [`planner/pacing.py`](planner/pacing.py) | Energy-budget engine — pure, testable, no I/O |
| [`planner/models.py`](planner/models.py) | Pydantic models; doubles as the structured-output schema |
| [`planner/prompts.py`](planner/prompts.py) | Prompt construction from computed budgets |
| [`planner/llm.py`](planner/llm.py) | OpenAI call plus the validate-and-repair loop |
| [`planner/usage.py`](planner/usage.py) | Token accounting and cost estimation |
| [`planner/tokenizer.py`](planner/tokenizer.py) | Local `tiktoken` counting vs billed counts |
| [`planner/scale.py`](planner/scale.py) | Usage-at-scale projections |
| [`scripts/make_report.py`](scripts/make_report.py) | Builds the PDF cost evaluation |
| [`planner/experiment.py`](planner/experiment.py) | Cost Lab sweep engine and aggregation |
| [`planner/charts.py`](planner/charts.py) | Report charts (no Streamlit import, so testable) |
| [`plan_ui.py`](plan_ui.py) / [`lab_ui.py`](lab_ui.py) | The two tab bodies |
| [`planner/render.py`](planner/render.py) | Markdown export and shared display helpers |
| [`app.py`](app.py) | Streamlit UI |

## Not included

Bookings, live pricing, flight/hotel search, weather, maps and routing,
accounts, or persistence between sessions.
