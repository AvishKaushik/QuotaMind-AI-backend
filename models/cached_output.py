"""CachedOutput — a stored AI response keyed by prompt hash for de-duplication.

DuplicateDetector writes one of these on a cache miss and reads it back on a
hit, avoiding a paid model call for an identical prompt.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CachedOutput(BaseModel):
    prompt_hash: str          # SHA-256 of the normalized prompt (unique key)
    prompt: str
    response: str
    model: str

    prompt_tokens: int = 0
    completion_tokens: int = 0

    hit_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    last_accessed_at: datetime = Field(default_factory=_utcnow)
