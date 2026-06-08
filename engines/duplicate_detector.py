"""Duplicate prompt detection and response cache.

Uses a stable SHA-256 hash of normalized prompts, checks Redis first for speed,
falls back to MongoDB, and records cache-hit savings as optimization events.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.models import compute_cost
from models.cached_output import CachedOutput
from models.optimization_event import OptimizationEvent, OptimizationType

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - keeps local imports usable before deps install.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator

CacheSource = Literal["redis", "mongo", "none"]


class DuplicateCheckResult(BaseModel):
    prompt_hash: str
    is_duplicate: bool
    source: CacheSource = "none"
    cached_output: CachedOutput | None = None
    response: str | None = None
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0


class DuplicateDetectorInput(BaseModel):
    prompt: str = Field(..., description="Prompt text to hash and check.")
    model: str = Field(default="gemini-2.5-flash", description="Model that would run on a miss.")
    agent_id: str | None = Field(default=None, description="Agent requesting the check.")


class DuplicateDetector:
    """Hash prompts, detect exact duplicates, and manage cached outputs."""

    def __init__(self, redis_ttl_seconds: int = 60 * 60 * 24) -> None:
        self.redis_ttl_seconds = redis_ttl_seconds

    def normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt text before hashing."""
        normalized = re.sub(r"\s+", " ", prompt or "")
        return normalized.strip().lower()

    def hash_prompt(self, prompt: str) -> str:
        """Return SHA-256 hash of the normalized prompt."""
        normalized = self.normalize_prompt(prompt)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def check(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        agent_id: str | None = None,
    ) -> DuplicateCheckResult:
        """Check Redis, then Mongo, for a cached response for this prompt."""
        prompt_hash = self.hash_prompt(prompt)
        cached, source = await self._get_with_source(prompt_hash)

        if cached is None:
            await self._increment_counter("cache:misses")
            return DuplicateCheckResult(prompt_hash=prompt_hash, is_duplicate=False)

        await self._increment_counter("cache:hits")
        cached.hit_count += 1
        cached.last_accessed_at = datetime.now(timezone.utc)
        await self._touch_cache_hit(cached)

        tokens_saved = cached.prompt_tokens + cached.completion_tokens
        cost_saved = compute_cost(
            cached.model or model,
            cached.prompt_tokens,
            cached.completion_tokens,
        )
        result = DuplicateCheckResult(
            prompt_hash=prompt_hash,
            is_duplicate=True,
            source=source,
            cached_output=cached,
            response=cached.response,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved,
        )
        await self._log_cache_hit(result, agent_id, model)
        return result

    async def get(self, prompt_hash: str) -> CachedOutput | None:
        """Return a cached output by hash, if present."""
        cached, _source = await self._get_with_source(prompt_hash)
        return cached

    async def _get_with_source(self, prompt_hash: str) -> tuple[CachedOutput | None, CacheSource]:
        """Return cached output plus the backing store that served it."""
        redis_value = await self._get_from_redis(prompt_hash)
        if redis_value is not None:
            return redis_value, "redis"

        mongo_value = await self._get_from_mongo(prompt_hash)
        if mongo_value is not None:
            await self._set_redis(prompt_hash, mongo_value)
            return mongo_value, "mongo"
        return None, "none"

    async def cache(
        self,
        prompt: str,
        response: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> CachedOutput:
        """Persist a model response so future duplicate prompts can reuse it."""
        prompt_hash = self.hash_prompt(prompt)
        cached = CachedOutput(
            prompt_hash=prompt_hash,
            prompt=prompt,
            response=response,
            model=model,
            prompt_tokens=max(prompt_tokens, 0),
            completion_tokens=max(completion_tokens, 0),
        )
        await self._set_redis(prompt_hash, cached)
        await self._upsert_mongo(cached)
        return cached

    async def check_duplicate(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        agent_id: str | None = None,
    ) -> DuplicateCheckResult:
        """Compatibility alias for callers that use README wording."""
        return await self.check(prompt=prompt, model=model, agent_id=agent_id)

    async def get_cached_output(self, prompt_hash: str) -> CachedOutput | None:
        """Compatibility alias for explicit cache reads."""
        return await self.get(prompt_hash)

    async def store_output(
        self,
        prompt: str,
        response: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> CachedOutput:
        """Compatibility alias for explicit cache writes."""
        return await self.cache(prompt, response, model, prompt_tokens, completion_tokens)

    async def _get_from_redis(self, prompt_hash: str) -> CachedOutput | None:
        try:
            from integrations import cache

            raw = await cache.get_cached(self._redis_key(prompt_hash))
            return CachedOutput(**raw) if raw else None
        except Exception:
            return None

    async def _set_redis(self, prompt_hash: str, cached: CachedOutput) -> None:
        try:
            from integrations import cache

            await cache.set_cached(
                self._redis_key(prompt_hash),
                cached.model_dump(mode="json"),
                ttl_seconds=self.redis_ttl_seconds,
            )
        except Exception:
            pass

    async def _get_from_mongo(self, prompt_hash: str) -> CachedOutput | None:
        try:
            from integrations import db

            doc = await db.cached_outputs().find_one({"prompt_hash": prompt_hash})
            if not doc:
                return None
            doc.pop("_id", None)
            return CachedOutput(**doc)
        except Exception:
            return None

    async def _upsert_mongo(self, cached: CachedOutput) -> None:
        try:
            from integrations import db

            await db.cached_outputs().update_one(
                {"prompt_hash": cached.prompt_hash},
                {"$set": cached.model_dump(mode="json")},
                upsert=True,
            )
        except Exception:
            pass

    async def _touch_cache_hit(self, cached: CachedOutput) -> None:
        await self._set_redis(cached.prompt_hash, cached)
        try:
            from integrations import db

            await db.cached_outputs().update_one(
                {"prompt_hash": cached.prompt_hash},
                {
                    "$inc": {"hit_count": 1},
                    "$set": {"last_accessed_at": cached.last_accessed_at},
                },
            )
        except Exception:
            pass

    async def _increment_counter(self, key: str) -> None:
        try:
            from integrations import cache

            await cache.increment_counter(key)
        except Exception:
            pass

    async def _log_cache_hit(
        self,
        result: DuplicateCheckResult,
        agent_id: str | None,
        requested_model: str,
    ) -> None:
        event = OptimizationEvent(
            type=OptimizationType.CACHE_HIT,
            agent_id=agent_id,
            description="Reused cached response for duplicate prompt.",
            before={"model": requested_model, "prompt_hash": result.prompt_hash},
            after={"source": result.source, "prompt_hash": result.prompt_hash},
            tokens_saved=result.tokens_saved,
            cost_saved_usd=result.cost_saved_usd,
        ).model_dump(mode="json")

        try:
            from integrations import db

            await db.optimization_events().insert_one(event)
        except Exception:
            pass

        try:
            from integrations import cache

            await cache.publish_event(event)
        except Exception:
            pass

    @staticmethod
    def _redis_key(prompt_hash: str) -> str:
        return f"cache:{prompt_hash}"


duplicate_detector = DuplicateDetector()


@tool("duplicate_detector", args_schema=DuplicateDetectorInput)
async def duplicate_detector_tool(**kwargs: Any) -> dict[str, Any]:
    """Check whether a prompt has an existing cached response."""
    result = await duplicate_detector.check(**kwargs)
    return result.model_dump()
