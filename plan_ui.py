"""Plan-a-trip tab — energy forecast, itinerary rendering and token usage."""

from __future__ import annotations

import streamlit as st

from planner.llm import DEFAULT_MODEL, MissingAPIKey, PlanResult, plan_trip
from planner.models import TripRequest
from planner.pacing import build_day_budgets, day_cost
from planner.prompts import SYSTEM_PROMPT, build_user_prompt
from planner.render import INTENSITY_BADGE, day_headline, to_markdown
from planner.tokenizer import compare, count_messages
from planner.usage import PRICING, estimate_across_models, format_cost, price_for


def generate(request: TripRequest) -> None:
    """Run the planner and store the result so it survives Streamlit reruns."""
    status = st.status("Working out your day budgets…", expanded=True)
    try:
        result = plan_trip(request, on_progress=lambda msg: status.update(label=msg))
    except MissingAPIKey as exc:
        status.update(label="Missing API key", state="error")
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # surface the real error rather than a blank page
        status.update(label="Planning failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    status.update(label="Itinerary ready", state="complete", expanded=False)

    # Bill the session once, here — not on every rerun that redraws the result.
    st.session_state["session_usage"].extend(result.usage.calls)
    st.session_state["transactions"].append(
        {
            "destination": request.destination,
            "days": request.num_days,
            "usage": result.usage,
        }
    )
    st.session_state["result"] = result
    st.session_state["request"] = request


# --- Budget preview (free, no API call) --------------------------------------


def render_preview(request: TripRequest) -> None:
    st.subheader("Energy forecast")
    st.caption("This is computed locally from your inputs — no itinerary generated yet.")

    preview = build_day_budgets(request)
    cols = st.columns(min(len(preview), 7) or 1)
    for i, budget in enumerate(preview):
        with cols[i % len(cols)]:
            st.metric(
                f"Day {budget.day_index + 1}",
                f"{budget.points:g}",
                delta=(
                    "recovery"
                    if budget.is_recovery
                    else ("big day OK" if budget.high_allowed else None)
                ),
                delta_color="off",
            )

    with st.expander("Why these numbers?"):
        for budget in preview:
            if budget.reasons:
                st.markdown(
                    f"**Day {budget.day_index + 1}** — " + "; ".join(budget.reasons)
                )

    with st.expander("Model pricing reference"):
        st.caption(
            "USD per 1,000,000 tokens. Actual cost for your itinerary is shown "
            "after generating one."
        )
        st.dataframe(
            [
                {
                    "Model": (
                        f"{name}  ←  in use" if DEFAULT_MODEL.startswith(name) else name
                    ),
                    "$/1M input": f"${p.input_per_m:g}",
                    "$/1M cached input": (
                        f"${p.cached_input_per_m:g}"
                        if p.cached_input_per_m is not None
                        else "—"
                    ),
                    "$/1M output": f"${p.output_per_m:g}",
                }
                for name, p in sorted(PRICING.items(), key=lambda kv: kv[1].input_per_m)
            ],
            hide_index=True,
            use_container_width=True,
        )
        if price_for(DEFAULT_MODEL) is None:
            st.caption(
                f"⚠️ `{DEFAULT_MODEL}` is configured but has no pricing entry — "
                "costs will show as n/a."
            )


# --- Itinerary ---------------------------------------------------------------


def render_itinerary(request: TripRequest, result: PlanResult) -> None:
    st.success(result.itinerary.overall_notes)
    if result.trim_notes:
        st.warning(
            "Some activities were removed to keep the plan within its energy "
            "budgets:\n\n" + "\n".join(f"- {n}" for n in result.trim_notes)
        )
    if result.violations:
        st.info(
            "Remaining pacing caveats:\n\n"
            + "\n".join(f"- {v}" for v in result.violations)
        )

    budgets = {b.day_index: b for b in result.budgets}
    for day in result.itinerary.days:
        budget = budgets.get(day.day_index)
        with st.expander(
            day_headline(budget) if budget else f"Day {day.day_index + 1}",
            expanded=True,
        ):
            if budget:
                used = day_cost(day)
                st.progress(
                    min(used / budget.points, 1.0) if budget.points else 0.0,
                    text=f"Energy {used} / {budget.points:g} points",
                )
            st.markdown(f"_{day.summary}_")

            left, right = st.columns([3, 2])

            with left:
                for activity in day.activities:
                    st.markdown(
                        f"**{activity.start_time}–{activity.end_time} · "
                        f"{activity.name}**  \n"
                        f"{INTENSITY_BADGE[activity.intensity]} · "
                        f"{activity.interest.value}"
                    )
                    st.markdown(activity.description)
                    st.caption(f"Why this fits you: {activity.why}")
                if not day.activities:
                    st.markdown("_Nothing scheduled — this day is yours._")

            with right:
                if day.rest_blocks:
                    st.markdown("**🛌 Rest**")
                    for rest in day.rest_blocks:
                        st.markdown(f"- {rest.start_time}–{rest.end_time} · {rest.note}")
                if day.meals:
                    st.markdown("**🍽️ Meals**")
                    for meal in day.meals:
                        st.markdown(f"- **{meal.time} {meal.slot}** — {meal.suggestion}")
                        st.caption(meal.dietary_note)
                if budget and budget.reasons:
                    st.markdown("**⚖️ Why this pace**")
                    for reason in budget.reasons:
                        st.caption(f"• {reason}")

    st.download_button(
        "Download itinerary (markdown)",
        data=to_markdown(request, result),
        file_name=f"{request.destination.split(',')[0].strip().lower()}-itinerary.md",
        mime="text/markdown",
    )


# --- Token usage -------------------------------------------------------------


def render_usage(request: TripRequest, result: PlanResult) -> None:
    st.divider()
    st.subheader("🎟️ Token usage for this itinerary")

    usage = result.usage
    cols = st.columns(4)
    cols[0].metric("Input tokens", f"{usage.input_tokens:,}")
    cols[1].metric("Output tokens", f"{usage.output_tokens:,}")
    cols[2].metric("Total", f"{usage.total_tokens:,}")
    cols[3].metric(
        "Estimated cost",
        format_cost(usage.cost_usd),
        help=(
            f"Based on published {DEFAULT_MODEL} pricing. Verify before relying on it."
        ),
    )

    if usage.call_count > 1:
        st.caption(
            f"{usage.call_count} API calls — the repair loop ran because the first "
            "draft broke a pacing rule."
        )
    if usage.cached_input_tokens:
        st.caption(
            f"{usage.cached_input_tokens:,} input tokens were served from cache "
            "at a reduced rate."
        )
    if price_for(DEFAULT_MODEL) is None:
        st.caption(
            f"No pricing on file for `{DEFAULT_MODEL}` — token counts are exact, "
            "cost is unavailable."
        )

    with st.expander("Tokenizer check — local count vs what you were billed"):
        st.caption(
            "`tiktoken` encodes the prompt locally with the model's own vocabulary. "
            "It costs nothing and works before a call is made, so it is what you "
            "use to predict spend. It will not match the bill, and the gap is the "
            "interesting part."
        )
        budgets = build_day_budgets(request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(request, budgets)},
        ]
        local = count_messages(messages, DEFAULT_MODEL)
        first_call = usage.calls[0]
        check = compare(local, first_call.input_tokens)

        cols = st.columns(3)
        cols[0].metric("Counted locally", f"{check.local_estimate:,}")
        cols[1].metric("Billed by API", f"{check.api_reported:,}")
        cols[2].metric(
            "Difference",
            f"{check.delta:+,}",
            delta=f"{check.delta_pct:+.0f}%" if check.delta_pct is not None else None,
            delta_color="off",
        )
        st.caption(check.explanation)

    with st.expander("Per-call breakdown"):
        st.dataframe(
            [
                {
                    "Call": call.label,
                    "Model": call.model,
                    "Input": call.input_tokens,
                    "Cached": call.cached_input_tokens,
                    "Output": call.output_tokens,
                    "Total": call.total_tokens,
                    "Cost": format_cost(call.cost_usd),
                }
                for call in usage.calls
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Cost on every model"):
        st.caption(
            "What **this itinerary's token counts** would cost on each model "
            "priced in `planner/usage.py`. A what-if, not a measurement: another "
            "model would tokenise slightly differently and produce a different "
            "itinerary, so read these as order-of-magnitude comparisons."
        )
        estimates = estimate_across_models(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            current_model=DEFAULT_MODEL,
        )
        cheapest = estimates[0].cost_usd if estimates else 0.0
        st.dataframe(
            [
                {
                    "Model": f"{e.model}  ←  in use" if e.is_current else e.model,
                    "Cost": format_cost(e.cost_usd),
                    "vs cheapest": (
                        "—" if not cheapest else f"{e.cost_usd / cheapest:.1f}x"
                    ),
                    "$/1M in": f"${e.pricing.input_per_m:g}",
                    "$/1M cached in": (
                        f"${e.pricing.cached_input_per_m:g}"
                        if e.pricing.cached_input_per_m is not None
                        else "—"
                    ),
                    "$/1M out": f"${e.pricing.output_per_m:g}",
                }
                for e in estimates
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Priced on {usage.input_tokens:,} input + {usage.output_tokens:,} "
            "output tokens. Rates are hardcoded and may be stale — verify at "
            "https://openai.com/api/pricing/"
        )

    transactions = st.session_state["transactions"]
    if len(transactions) > 1:
        with st.expander(f"All {len(transactions)} generations this session"):
            st.dataframe(
                [
                    {
                        "#": i + 1,
                        "Trip": f"{t['destination']} ({t['days']}d)",
                        "Calls": t["usage"].call_count,
                        "Input": t["usage"].input_tokens,
                        "Output": t["usage"].output_tokens,
                        "Total": t["usage"].total_tokens,
                        "Cost": format_cost(t["usage"].cost_usd),
                    }
                    for i, t in enumerate(transactions)
                ],
                hide_index=True,
                use_container_width=True,
            )

    st.caption(f"Generated with {DEFAULT_MODEL}")
