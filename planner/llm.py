"""OpenAI integration: structured output plus a validate-and-repair loop."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

from .models import Itinerary, TripRequest
from .pacing import DayBudget, Violation, build_day_budgets, trim_to_budget, validate_itinerary
from .prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .usage import CallUsage, UsageTotals, usage_from_response

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_REPAIR_ATTEMPTS = 1


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else None


# `None` means "don't send the parameter at all" — the API default applies.
# Sending an explicit null is not the same thing and is rejected.
DEFAULT_TEMPERATURE = _env_float("OPENAI_TEMPERATURE")
DEFAULT_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS")


class MissingAPIKey(RuntimeError):
    pass


class TruncatedOutputError(RuntimeError):
    """The model hit its output ceiling before completing valid JSON.

    Expected when `max_output_tokens` is set low, so callers that are
    deliberately probing limits (the Cost Lab) can record it as a data point
    rather than a crash. The tokens were still billed — `usage` carries them.
    """

    def __init__(self, message: str, usage: CallUsage, finish_reason: str | None):
        super().__init__(message)
        self.usage = usage
        self.finish_reason = finish_reason


@dataclass
class PlanResult:
    itinerary: Itinerary
    budgets: list[DayBudget]
    violations: list[Violation] = field(default_factory=list)
    trim_notes: list[str] = field(default_factory=list)
    usage: UsageTotals = field(default_factory=UsageTotals)
    # Violations in the model's *first* draft, before any repair. This is the
    # quality signal the Cost Lab measures parameters against.
    first_draft_violations: list[Violation] = field(default_factory=list)
    finish_reasons: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def was_repaired_by_code(self) -> bool:
        return bool(self.trim_notes)

    @property
    def repair_calls(self) -> int:
        return max(self.usage.call_count - 1, 0)


def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKey(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root."
        )
    return OpenAI(api_key=key)


def _parse(
    client: OpenAI,
    messages: list[dict],
    model: str,
    label: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> tuple[Itinerary, CallUsage, str | None]:
    """Call the structured-output endpoint, tolerating either SDK location.

    `parse` graduated out of `client.beta` in newer openai releases; supporting
    both keeps this working across the range pinned in requirements.txt.

    Returns the parsed itinerary, this call's token usage, and the finish reason.
    """
    endpoint = getattr(client.chat.completions, "parse", None)
    if endpoint is None:
        endpoint = client.beta.chat.completions.parse

    # Only send parameters that were actually configured — an omitted key gets
    # the API default, whereas an explicit None is a different (invalid) request.
    optional: dict = {}
    if temperature is not None:
        optional["temperature"] = temperature
    if max_output_tokens is not None:
        optional["max_completion_tokens"] = max_output_tokens

    completion = endpoint(
        model=model,
        messages=messages,
        response_format=Itinerary,
        **optional,
    )
    call_usage = usage_from_response(completion, label=label, model=model)
    finish_reason = completion.choices[0].finish_reason

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        message = (
            "The model returned no parseable itinerary "
            f"(finish_reason={finish_reason}). "
            f"Tokens still billed: {call_usage.total_tokens}."
        )
        if finish_reason == "length":
            raise TruncatedOutputError(
                message
                + f" The output hit the {max_output_tokens}-token cap — raise "
                "OPENAI_MAX_OUTPUT_TOKENS or plan a shorter trip.",
                usage=call_usage,
                finish_reason=finish_reason,
            )
        raise RuntimeError(message)
    return parsed, call_usage, finish_reason


def plan_trip(
    req: TripRequest,
    model: str = DEFAULT_MODEL,
    on_progress: Callable[[str], None] | None = None,
    *,
    temperature: float | None = DEFAULT_TEMPERATURE,
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
    repair: bool = True,
) -> PlanResult:
    """Build day budgets, ask the model to fill them, then hold it to them.

    `repair=False` skips the repair pass, so a caller measuring first-call
    quality (the Cost Lab) is not billed for a second call it does not want.
    """

    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    started = time.monotonic()
    budgets = build_day_budgets(req)
    client = _client()
    usage = UsageTotals()
    finish_reasons: list[str] = []

    progress("Drafting the itinerary…")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(req, budgets)},
    ]
    itinerary, call_usage, finish_reason = _parse(
        client,
        messages,
        model,
        label="Initial draft",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    usage.add(call_usage)
    if finish_reason:
        finish_reasons.append(finish_reason)

    violations = validate_itinerary(itinerary, budgets)
    first_draft_violations = list(violations)

    attempts = MAX_REPAIR_ATTEMPTS if repair else 0
    for attempt in range(attempts):
        if not violations:
            break
        progress(f"Fixing {len(violations)} pacing issue(s)…")
        messages.append(
            {"role": "assistant", "content": itinerary.model_dump_json()}
        )
        messages.append({"role": "user", "content": build_repair_prompt(violations)})
        itinerary, call_usage, finish_reason = _parse(
            client,
            messages,
            model,
            label=f"Repair pass {attempt + 1}",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        usage.add(call_usage)
        if finish_reason:
            finish_reasons.append(finish_reason)
        violations = validate_itinerary(itinerary, budgets)

    trim_notes: list[str] = []
    if violations:
        progress("Trimming over-packed days…")
        trim_notes = trim_to_budget(itinerary, budgets)
        violations = validate_itinerary(itinerary, budgets)

    return PlanResult(
        itinerary=itinerary,
        budgets=budgets,
        violations=violations,
        trim_notes=trim_notes,
        usage=usage,
        first_draft_violations=first_draft_violations,
        finish_reasons=finish_reasons,
        duration_s=time.monotonic() - started,
    )
