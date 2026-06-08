"""Agent control and optimization insight API.

This router powers the dashboard's agent lane: health checks, manual overrides,
quota forecasting, daily recommendations, and budget configuration.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from config.env import get_float_env, get_int_env
from engines.budget_allocator import DEFAULT_CATEGORY_LIMITS, budget_allocator
from engines.quota_forecaster import quota_forecaster
from engines.recommendations_engine import recommendations_engine
from engines.runaway_detector import runaway_detector
from models.optimization_event import OptimizationEvent, OptimizationType

router = APIRouter(tags=["agents"])


class AgentStatus(BaseModel):
    agent_id: str
    status: str
    current_model: str | None = None
    latest_model: str | None = None
    request_count_today: int = 0
    tokens_today: int = 0
    cost_usd_today: float = 0.0
    cache_hits_today: int = 0
    last_seen_at: datetime | None = None
    health: dict[str, Any] | None = None


class AgentsStatusResponse(BaseModel):
    agents: list[AgentStatus]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PauseAgentRequest(BaseModel):
    paused: bool = True
    reason: str = "Manual dashboard override."
    ttl_seconds: int = Field(default=15 * 60, ge=60, le=24 * 60 * 60)


class PauseAgentResponse(BaseModel):
    agent_id: str
    paused: bool
    reason: str
    ttl_seconds: int | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BudgetConfigRequest(BaseModel):
    daily_budget_usd: float | None = Field(default=None, gt=0)
    daily_token_limit: int | None = Field(default=None, gt=0)
    category_limits: dict[str, float] | None = Field(
        default=None,
        description="Category shares, e.g. {'support': 0.3, 'analytics': 0.25}.",
    )


class BudgetConfigResponse(BaseModel):
    daily_budget_usd: float
    daily_token_limit: int
    category_limits: dict[str, float]
    budget_status: dict[str, Any]
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@router.get("/api/agents/status", response_model=AgentsStatusResponse)
async def get_agents_status() -> AgentsStatusResponse:
    """Return current status for agents seen today."""
    agent_rows = await _agent_usage_today()
    cache_snapshot = await _agent_cache_snapshot()

    agent_ids = set(agent_rows) | set(cache_snapshot)
    agents = [
        _build_agent_status(agent_id, agent_rows.get(agent_id, {}), cache_snapshot.get(agent_id, {}))
        for agent_id in sorted(agent_ids)
    ]
    return AgentsStatusResponse(agents=agents)


@router.get("/api/agents/{agent_id}/health")
async def get_agent_health(
    agent_id: str,
    window_minutes: int = Query(default=5, ge=1, le=120),
) -> dict[str, Any]:
    """Analyze recent traffic for runaway, duplicate, and token-spike risk."""
    if not agent_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="agent_id is required.")
    result = await runaway_detector.analyze_agent(agent_id=agent_id, window_minutes=window_minutes)
    return result.model_dump(mode="json")


@router.post("/api/agents/{agent_id}/pause", response_model=PauseAgentResponse)
async def pause_agent(agent_id: str, payload: PauseAgentRequest) -> PauseAgentResponse:
    """Set or clear a manual pause override for an agent."""
    normalized_agent_id = agent_id.strip()
    if not normalized_agent_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="agent_id is required.")

    await _set_pause_override(normalized_agent_id, payload)
    response = PauseAgentResponse(
        agent_id=normalized_agent_id,
        paused=payload.paused,
        reason=payload.reason,
        ttl_seconds=payload.ttl_seconds if payload.paused else None,
    )
    await _log_manual_pause(response)
    return response


@router.get("/api/forecast")
async def get_forecast(
    lookback_minutes: int = Query(default=60, ge=5, le=24 * 60),
) -> dict[str, Any]:
    """Return quota burn-rate forecast and exhaustion projection."""
    result = await quota_forecaster.forecast_exhaustion(lookback_minutes=lookback_minutes)
    return result.model_dump(mode="json")


@router.get("/api/recommendations")
async def get_recommendations(
    max_recommendations: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    """Generate daily optimization recommendations for the dashboard."""
    report = await recommendations_engine.generate_daily_report(
        max_recommendations=max_recommendations
    )
    return report.model_dump(mode="json")


@router.get("/api/budget/status")
async def get_budget_status() -> dict[str, Any]:
    """Return daily and category budget health."""
    category_limits = await _get_category_limits()
    result = await budget_allocator.check_budget_status(category_limits=category_limits)
    return result.model_dump(mode="json")


@router.post("/api/config/budget", response_model=BudgetConfigResponse)
async def update_budget_config(payload: BudgetConfigRequest) -> BudgetConfigResponse:
    """Update runtime quota and budget config used by routing/enforcement engines."""
    current_category_limits = await _get_category_limits()
    category_limits = (
        _normalize_category_limits(payload.category_limits)
        if payload.category_limits is not None
        else current_category_limits or _normalize_category_limits(DEFAULT_CATEGORY_LIMITS)
    )
    daily_budget_usd = (
        payload.daily_budget_usd
        if payload.daily_budget_usd is not None
        else await _get_daily_budget_usd()
    )
    daily_token_limit = (
        payload.daily_token_limit
        if payload.daily_token_limit is not None
        else await _get_daily_token_limit()
    )

    await _persist_budget_config(daily_budget_usd, daily_token_limit, category_limits)
    budget_status = await budget_allocator.check_budget_status(category_limits=category_limits)

    return BudgetConfigResponse(
        daily_budget_usd=round(daily_budget_usd, 4),
        daily_token_limit=daily_token_limit,
        category_limits=category_limits,
        budget_status=budget_status.model_dump(mode="json"),
    )


async def _agent_usage_today() -> dict[str, dict[str, Any]]:
    start = datetime.combine(datetime.now(timezone.utc).date(), time.min, timezone.utc)
    try:
        from integrations import db

        cursor = db.ai_requests().aggregate(
            [
                {"$match": {"created_at": {"$gte": start}}},
                {"$sort": {"created_at": -1}},
                {
                    "$group": {
                        "_id": "$agent_id",
                        "request_count_today": {"$sum": 1},
                        "tokens_today": {
                            "$sum": {"$add": ["$prompt_tokens", "$completion_tokens"]}
                        },
                        "cost_usd_today": {"$sum": "$cost_usd"},
                        "cache_hits_today": {
                            "$sum": {"$cond": [{"$eq": ["$cache_hit", True]}, 1, 0]}
                        },
                        "latest_model": {"$first": "$model"},
                        "last_seen_at": {"$first": "$created_at"},
                    }
                },
            ]
        )
        rows: dict[str, dict[str, Any]] = {}
        async for row in cursor:
            agent_id = str(row.get("_id") or "unknown")
            rows[agent_id] = {
                "request_count_today": int(row.get("request_count_today", 0) or 0),
                "tokens_today": int(row.get("tokens_today", 0) or 0),
                "cost_usd_today": round(float(row.get("cost_usd_today", 0.0) or 0.0), 6),
                "cache_hits_today": int(row.get("cache_hits_today", 0) or 0),
                "latest_model": row.get("latest_model"),
                "last_seen_at": row.get("last_seen_at"),
            }
        return rows
    except Exception:
        return {}


async def _agent_cache_snapshot() -> dict[str, dict[str, Any]]:
    try:
        from integrations import cache

        client = cache.get_client()
        keys: set[str] = set()
        for pattern in ("agent_route:*", "agent_health:*", "agent_paused:*", "agent_throttle:*"):
            async for key in client.scan_iter(match=pattern):
                keys.add(str(key))

        snapshot: dict[str, dict[str, Any]] = {}
        for key in keys:
            prefix, agent_id = key.split(":", 1)
            record = snapshot.setdefault(agent_id, {})
            if prefix == "agent_route":
                record["route"] = await cache.get_cached(key)
            elif prefix == "agent_health":
                record["health"] = await cache.get_cached(key)
            elif prefix == "agent_paused":
                record["paused"] = True
            elif prefix == "agent_throttle":
                record["throttled"] = True
        return snapshot
    except Exception:
        return {}


def _build_agent_status(
    agent_id: str,
    row: dict[str, Any],
    cached: dict[str, Any],
) -> AgentStatus:
    route = cached.get("route") if isinstance(cached.get("route"), dict) else {}
    health = cached.get("health") if isinstance(cached.get("health"), dict) else None
    status_name = "paused" if cached.get("paused") else "throttled" if cached.get("throttled") else "active"

    return AgentStatus(
        agent_id=agent_id,
        status=status_name,
        current_model=route.get("model"),
        latest_model=row.get("latest_model"),
        request_count_today=row.get("request_count_today", 0),
        tokens_today=row.get("tokens_today", 0),
        cost_usd_today=row.get("cost_usd_today", 0.0),
        cache_hits_today=row.get("cache_hits_today", 0),
        last_seen_at=row.get("last_seen_at"),
        health=health,
    )


async def _set_pause_override(agent_id: str, payload: PauseAgentRequest) -> None:
    try:
        from integrations import cache

        client = cache.get_client()
        key = f"agent_paused:{agent_id}"
        if payload.paused:
            await cache.set_cached(
                key,
                {"reason": payload.reason, "created_at": datetime.now(timezone.utc).isoformat()},
                ttl_seconds=payload.ttl_seconds,
            )
        else:
            await client.delete(key)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to update pause override: {type(exc).__name__}",
        ) from exc


async def _log_manual_pause(response: PauseAgentResponse) -> None:
    event = OptimizationEvent(
        type=OptimizationType.PAUSE if response.paused else OptimizationType.ALERT,
        agent_id=response.agent_id,
        description=response.reason if response.paused else "Manual pause override cleared.",
        after={
            "paused": response.paused,
            "ttl_seconds": response.ttl_seconds,
        },
        triggered_by="manual",
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


async def _get_category_limits() -> dict[str, float] | None:
    try:
        from integrations import cache

        value = await cache.get_cached("budget:category_limits")
        return _normalize_category_limits(value) if isinstance(value, dict) else None
    except Exception:
        return None


async def _get_daily_budget_usd() -> float:
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


async def _get_daily_token_limit() -> int:
    try:
        from integrations import cache

        raw = await cache.get_client().get("quota:daily_token_limit")
        if raw is not None:
            return max(int(float(raw)), 1)
    except Exception:
        pass
    return max(get_int_env("DAILY_QUOTA_TOKEN_LIMIT", 10_000_000), 1)


async def _persist_budget_config(
    daily_budget_usd: float,
    daily_token_limit: int,
    category_limits: dict[str, float] | None,
) -> None:
    try:
        from integrations import cache

        client = cache.get_client()
        await client.set("quota:daily_budget_usd", daily_budget_usd)
        await client.set("quota:daily_token_limit", daily_token_limit)
        if category_limits is not None:
            await cache.set_cached("budget:category_limits", category_limits)
        await cache.publish_event(
            {
                "type": "budget_config_updated",
                "daily_budget_usd": daily_budget_usd,
                "daily_token_limit": daily_token_limit,
                "category_limits": category_limits,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to persist budget config: {type(exc).__name__}",
        ) from exc


def _normalize_category_limits(value: dict[str, Any] | None) -> dict[str, float] | None:
    if value is None:
        return None

    cleaned = {
        (str(name).strip().lower() or "general"): max(float(share), 0.0)
        for name, share in value.items()
    }
    total = sum(cleaned.values())
    if total <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="category_limits must contain at least one positive share.",
        )
    return {name: round(share / total, 6) for name, share in cleaned.items()}
