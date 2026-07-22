"""Cumulative token tally, held in a signed session cookie.

Only a few integers are stored — input tokens, output tokens, cached tokens and
call count — which fit comfortably in a cookie. Full itineraries are never
persisted; they are re-rendered inline on each POST. This mirrors the
session-token bar in the Streamlit sidebar without needing a server-side store.
"""

from __future__ import annotations

from dataclasses import dataclass

from starlette.requests import Request

from planner.models import TripRequest
from planner.usage import CallUsage

_KEY = "usage"
_TRIP_KEY = "last_trip"


@dataclass
class SessionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    calls: int = 0
    model: str = "gpt-4o-mini"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float | None:
        """Reuse the single-call cost calc rather than duplicating pricing."""
        return CallUsage(
            label="session",
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
        ).cost_usd


def read_totals(request: Request) -> SessionUsage:
    data = request.session.get(_KEY)
    if not data:
        return SessionUsage()
    return SessionUsage(
        input_tokens=int(data.get("input", 0)),
        output_tokens=int(data.get("output", 0)),
        cached_input_tokens=int(data.get("cached", 0)),
        calls=int(data.get("calls", 0)),
        model=data.get("model", "gpt-4o-mini"),
    )


def add_usage(request: Request, usage, model: str) -> None:
    """Fold a generation's `UsageTotals` into the running cookie total."""
    current = read_totals(request)
    request.session[_KEY] = {
        "input": current.input_tokens + usage.input_tokens,
        "output": current.output_tokens + usage.output_tokens,
        "cached": current.cached_input_tokens + usage.cached_input_tokens,
        "calls": current.calls + usage.call_count,
        "model": model,
    }


def set_last_trip(request: Request, trip: TripRequest) -> None:
    """Remember the trip so the Cost Lab can sweep it without a form of its own."""
    request.session[_TRIP_KEY] = trip.model_dump(mode="json")


def get_last_trip(request: Request) -> TripRequest | None:
    data = request.session.get(_TRIP_KEY)
    return TripRequest.model_validate(data) if data else None


def reset(request: Request) -> None:
    request.session.pop(_KEY, None)
    request.session.pop(_TRIP_KEY, None)
