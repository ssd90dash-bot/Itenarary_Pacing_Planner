"""Data models shared by the pacing engine, the LLM contract and the UI.

The `Itinerary` tree doubles as the OpenAI structured-output schema, so every
field here must stay JSON-schema friendly: no bare dicts, no unions beyond
Optional, and Enums rather than free strings wherever the value is closed.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

# --- Inputs -----------------------------------------------------------------


class Intensity(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"


class Interest(str, Enum):
    indoor = "indoor"
    outdoor = "outdoor"
    history = "history"
    experiences = "experiences"
    relaxation = "relaxation"


class Pace(str, Enum):
    relaxed = "relaxed"
    balanced = "balanced"
    packed = "packed"


class Traveller(BaseModel):
    label: str = "Traveller"
    age: int = Field(ge=0, le=120)


class PitStop(BaseModel):
    """A transfer between bases, which eats into that day's energy budget."""

    place: str
    day_index: int = Field(ge=0, description="0-based day of the trip")
    travel_hours: float = Field(ge=0, le=24)


class TripRequest(BaseModel):
    destination: str
    start_date: date
    end_date: date
    travellers: list[Traveller]
    pit_stops: list[PitStop] = []
    food_preferences: list[str] = []
    food_notes: str = ""
    # interest -> importance 1..5; drives how many slots each category gets
    interests: dict[Interest, int] = {}
    pace: Pace = Pace.balanced

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


# --- Outputs (the LLM fills these in) ---------------------------------------


class Activity(BaseModel):
    name: str
    start_time: str = Field(description="24h HH:MM")
    end_time: str = Field(description="24h HH:MM")
    intensity: Intensity
    interest: Interest
    description: str
    why: str = Field(description="Why this fits these travellers specifically")


class Meal(BaseModel):
    slot: str = Field(description="breakfast | lunch | dinner")
    time: str = Field(description="24h HH:MM")
    suggestion: str
    dietary_note: str = Field(description="How this respects the stated food preferences")


class RestBlock(BaseModel):
    start_time: str
    end_time: str
    note: str


class DayPlan(BaseModel):
    day_index: int
    summary: str
    activities: list[Activity]
    meals: list[Meal]
    rest_blocks: list[RestBlock]


class Itinerary(BaseModel):
    """Top-level structured-output target."""

    days: list[DayPlan]
    overall_notes: str
