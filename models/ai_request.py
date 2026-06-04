"""AIRequest — one inbound AI inference request flowing through the system."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AIRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    agent_id: str
    prompt: str
    priority: Priority = Priority.MEDIUM

    # Model actually used for this request (may differ from requested after the
    # TrafficRouter applies quota-based downgrades).
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    # De-duplication: SHA-256 of the normalized prompt, set by DuplicateDetector.
    prompt_hash: str | None = None
    cache_hit: bool = False

    category: str = "general"
    created_at: datetime = Field(default_factory=_utcnow)

    # Serialize enums to their string values (clean JSON + Mongo docs).
    model_config = {"use_enum_values": True}
