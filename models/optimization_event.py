"""OptimizationEvent — a record of one optimization action taken by the system.

Written to the `optimization_events` collection by any engine that takes an
action (compress, reroute, throttle, …) and surfaced in the Incident Feed UI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class OptimizationType(str, Enum):
    REROUTE = "reroute"
    COMPRESS = "compress"
    CACHE_HIT = "cache_hit"
    THROTTLE = "throttle"
    PAUSE = "pause"
    BUDGET_ENFORCE = "budget_enforce"
    ALERT = "alert"


class OptimizationEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    type: OptimizationType
    agent_id: str | None = None
    description: str = ""

    # Before/after snapshot for UI transparency (e.g. old model -> new model).
    before: dict = Field(default_factory=dict)
    after: dict = Field(default_factory=dict)

    tokens_saved: int = 0
    cost_saved_usd: float = 0.0

    triggered_by: str = "system"  # "system" | "agent" | "simulator"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": True}
