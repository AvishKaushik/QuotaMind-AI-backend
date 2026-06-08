"""Dashboard metrics API.

Aggregates request volume, quota/budget usage, cache effectiveness, and savings
from MongoDB plus the Redis counters maintained by request-time engines.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from config.env import get_float_env, get_int_env

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


class MetricsSummary(BaseModel):
    window_minutes: int
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_used_today: int = 0
    daily_token_limit: int
    quota_percent_used: float = 0.0
    cost_usd: float = 0.0
    cost_usd_today: float = 0.0
    daily_budget_usd: float
    budget_percent_used: float = 0.0
    cache_hit_rate: float = 0.0
    models: dict[str, int] = Field(default_factory=dict)
    agents: dict[str, dict[str, float | int | str | None]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CacheStats(BaseModel):
    hits: int = 0
    misses: int = 0
    total_lookups: int = 0
    hit_rate: float = 0.0
    cached_outputs: int = 0
    cached_output_hits: int = 0
    estimated_tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SavingsSummary(BaseModel):
    window_minutes: int
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    event_count: int = 0
    by_type: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary(
    window_minutes: int = Query(default=24 * 60, ge=5, le=7 * 24 * 60),
) -> MetricsSummary:
    """Return token, cost, quota, budget, model, and agent metrics."""
    request_metrics = await _request_metrics(window_minutes)
    tokens_used_today = await _tokens_used_today(request_metrics["total_tokens"])
    cost_usd_today = await _cost_usd_today(request_metrics["cost_usd"])
    daily_token_limit = await _daily_token_limit()
    daily_budget = await _daily_budget_usd()

    return MetricsSummary(
        window_minutes=window_minutes,
        request_count=request_metrics["request_count"],
        prompt_tokens=request_metrics["prompt_tokens"],
        completion_tokens=request_metrics["completion_tokens"],
        total_tokens=request_metrics["total_tokens"],
        tokens_used_today=tokens_used_today,
        daily_token_limit=daily_token_limit,
        quota_percent_used=round(_ratio(tokens_used_today, daily_token_limit), 4),
        cost_usd=request_metrics["cost_usd"],
        cost_usd_today=cost_usd_today,
        daily_budget_usd=daily_budget,
        budget_percent_used=round(_ratio(cost_usd_today, daily_budget), 4),
        cache_hit_rate=request_metrics["cache_hit_rate"],
        models=request_metrics["models"],
        agents=request_metrics["agents"],
    )


@router.get("/cache-stats", response_model=CacheStats)
async def get_cache_stats() -> CacheStats:
    """Return cache hit/miss counters and cached-output savings estimates."""
    hits, misses = await _cache_counters()
    cached_outputs = await _cached_output_metrics()
    total = hits + misses
    return CacheStats(
        hits=hits,
        misses=misses,
        total_lookups=total,
        hit_rate=round(hits / total, 4) if total else 0.0,
        cached_outputs=cached_outputs["cached_outputs"],
        cached_output_hits=cached_outputs["cached_output_hits"],
        estimated_tokens_saved=cached_outputs["estimated_tokens_saved"],
        estimated_cost_saved_usd=cached_outputs["estimated_cost_saved_usd"],
    )


@router.get("/savings", response_model=SavingsSummary)
async def get_savings(
    window_minutes: int = Query(default=24 * 60, ge=5, le=30 * 24 * 60),
) -> SavingsSummary:
    """Return optimization savings from cache hits, reroutes, and compression."""
    return await _savings_metrics(window_minutes)


async def _request_metrics(window_minutes: int) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    empty = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cache_hit_rate": 0.0,
        "models": {},
        "agents": {},
    }
    try:
        from integrations import db

        cursor = db.ai_requests().aggregate(
            [
                {"$match": _created_after_match(since)},
                {
                    "$facet": {
                        "totals": [
                            {
                                "$group": {
                                    "_id": None,
                                    "request_count": {"$sum": 1},
                                    "prompt_tokens": {"$sum": {"$ifNull": ["$prompt_tokens", 0]}},
                                    "completion_tokens": {
                                        "$sum": {"$ifNull": ["$completion_tokens", 0]}
                                    },
                                    "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
                                    "cache_hits": {
                                        "$sum": {"$cond": [{"$eq": ["$cache_hit", True]}, 1, 0]}
                                    },
                                }
                            }
                        ],
                        "models": [{"$group": {"_id": "$model", "count": {"$sum": 1}}}],
                        "agents": [
                            {"$sort": {"created_at": -1}},
                            {
                                "$group": {
                                    "_id": "$agent_id",
                                    "count": {"$sum": 1},
                                    "tokens": {
                                        "$sum": {
                                            "$add": [
                                                {"$ifNull": ["$prompt_tokens", 0]},
                                                {"$ifNull": ["$completion_tokens", 0]},
                                            ]
                                        }
                                    },
                                    "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
                                    "latest_model": {"$first": "$model"},
                                    "last_seen_at": {"$first": "$created_at"},
                                }
                            },
                        ],
                    }
                },
            ]
        )
        rows = await cursor.to_list(length=1)
        if not rows:
            return empty

        row = rows[0]
        totals = (row.get("totals") or [{}])[0]
        request_count = int(totals.get("request_count", 0) or 0)
        prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
        completion_tokens = int(totals.get("completion_tokens", 0) or 0)
        cache_hits = int(totals.get("cache_hits", 0) or 0)

        return {
            "request_count": request_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(float(totals.get("cost_usd", 0.0) or 0.0), 6),
            "cache_hit_rate": round(cache_hits / request_count, 4) if request_count else 0.0,
            "models": _count_map(row.get("models", []), "unknown"),
            "agents": {
                str(item.get("_id") or "unknown"): {
                    "count": int(item.get("count", 0) or 0),
                    "tokens": int(item.get("tokens", 0) or 0),
                    "cost_usd": round(float(item.get("cost_usd", 0.0) or 0.0), 6),
                    "latest_model": item.get("latest_model"),
                    "last_seen_at": _serialize_datetime(item.get("last_seen_at")),
                }
                for item in row.get("agents", [])
            },
        }
    except Exception:
        return empty


async def _cache_counters() -> tuple[int, int]:
    try:
        from integrations import cache

        client = cache.get_client()
        hits = int(float(await client.get("cache:hits") or 0))
        misses = int(float(await client.get("cache:misses") or 0))
        return hits, misses
    except Exception:
        return 0, 0


async def _cached_output_metrics() -> dict[str, Any]:
    try:
        from integrations import db

        cursor = db.cached_outputs().aggregate(
            [
                {
                    "$group": {
                        "_id": None,
                        "cached_outputs": {"$sum": 1},
                        "cached_output_hits": {"$sum": {"$ifNull": ["$hit_count", 0]}},
                        "estimated_tokens_saved": {
                            "$sum": {
                                "$multiply": [
                                    {"$ifNull": ["$hit_count", 0]},
                                    {
                                        "$add": [
                                            {"$ifNull": ["$prompt_tokens", 0]},
                                            {"$ifNull": ["$completion_tokens", 0]},
                                        ]
                                    },
                                ]
                            }
                        },
                    }
                }
            ]
        )
        rows = await cursor.to_list(length=1)
        row = rows[0] if rows else {}
        savings = await _savings_metrics(window_minutes=30 * 24 * 60)
        return {
            "cached_outputs": int(row.get("cached_outputs", 0) or 0),
            "cached_output_hits": int(row.get("cached_output_hits", 0) or 0),
            "estimated_tokens_saved": int(row.get("estimated_tokens_saved", 0) or 0),
            "estimated_cost_saved_usd": savings.by_type.get("cache_hit", {}).get(
                "cost_saved_usd",
                0.0,
            ),
        }
    except Exception:
        return {
            "cached_outputs": 0,
            "cached_output_hits": 0,
            "estimated_tokens_saved": 0,
            "estimated_cost_saved_usd": 0.0,
        }


async def _savings_metrics(window_minutes: int) -> SavingsSummary:
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    try:
        from integrations import db

        cursor = db.optimization_events().aggregate(
            [
                {"$match": _created_after_match(since)},
                {
                    "$group": {
                        "_id": "$type",
                        "event_count": {"$sum": 1},
                        "tokens_saved": {"$sum": {"$ifNull": ["$tokens_saved", 0]}},
                        "cost_saved_usd": {"$sum": {"$ifNull": ["$cost_saved_usd", 0]}},
                    }
                },
            ]
        )
        by_type: dict[str, dict[str, float | int]] = {}
        event_count = 0
        tokens_saved = 0
        cost_saved = 0.0
        async for row in cursor:
            event_type = str(row.get("_id") or "unknown")
            row_event_count = int(row.get("event_count", 0) or 0)
            row_tokens = int(row.get("tokens_saved", 0) or 0)
            row_cost = round(float(row.get("cost_saved_usd", 0.0) or 0.0), 6)
            by_type[event_type] = {
                "event_count": row_event_count,
                "tokens_saved": row_tokens,
                "cost_saved_usd": row_cost,
            }
            event_count += row_event_count
            tokens_saved += row_tokens
            cost_saved += row_cost

        return SavingsSummary(
            window_minutes=window_minutes,
            event_count=event_count,
            tokens_saved=tokens_saved,
            cost_saved_usd=round(cost_saved, 6),
            by_type=by_type,
        )
    except Exception:
        return SavingsSummary(window_minutes=window_minutes)


async def _tokens_used_today(fallback: int) -> int:
    try:
        from integrations import cache

        raw = await cache.get_client().get("quota:tokens_used_today")
        if raw is not None:
            return max(int(float(raw)), 0)
    except Exception:
        pass
    return fallback


async def _cost_usd_today(fallback: float) -> float:
    try:
        from integrations import cache

        raw = await cache.get_client().get("quota:cost_usd_today")
        if raw is not None:
            return round(max(float(raw), 0.0), 6)
    except Exception:
        pass
    return fallback


async def _daily_token_limit() -> int:
    try:
        from integrations import cache

        raw = await cache.get_client().get("quota:daily_token_limit")
        if raw is None:
            raw = await cache.get_client().get("quota:daily_limit")
        if raw is not None:
            return max(int(float(raw)), 1)
    except Exception:
        pass
    return max(get_int_env("DAILY_QUOTA_TOKEN_LIMIT", 10_000_000), 1)


async def _daily_budget_usd() -> float:
    try:
        from integrations import cache

        raw = await cache.get_client().get("quota:daily_budget_usd")
        if raw is None:
            raw = await cache.get_client().get("quota:daily_limit")
        if raw is not None:
            return max(float(raw), 0.01)
    except Exception:
        pass
    return max(get_float_env("DAILY_BUDGET_USD", 500.0), 0.01)


def _created_after_match(since: datetime) -> dict[str, Any]:
    return {
        "$or": [
            {"created_at": {"$gte": since}},
            {"created_at": {"$gte": since.isoformat()}},
        ]
    }


def _count_map(items: list[dict[str, Any]], default: str) -> dict[str, int]:
    return {
        str(item.get("_id") or default): int(item.get("count", 0) or 0)
        for item in items
    }


def _serialize_datetime(value: Any) -> str | None:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value) if value else None


def _ratio(value: int | float, limit: int | float) -> float:
    if limit <= 0:
        return 1.0 if value > 0 else 0.0
    return max(float(value) / float(limit), 0.0)
