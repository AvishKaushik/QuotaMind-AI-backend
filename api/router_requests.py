"""Request ingestion API.

The ingest route is the synchronous optimization gate in front of Gemini:
duplicate cache -> routing -> budget enforcement -> optional compression ->
Gemini -> persistence/counters/events.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config.models import DEFAULT_MODEL, compute_cost
from engines.budget_allocator import budget_allocator
from engines.duplicate_detector import duplicate_detector
from engines.prompt_compressor import prompt_compressor
from engines.traffic_router import traffic_router
from integrations.gemini_client import gemini_client
from models.ai_request import AIRequest, Priority

router = APIRouter(prefix="/api/requests", tags=["requests"])
logger = logging.getLogger("quotamind.requests")


class IngestRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    priority: Priority = Priority.MEDIUM
    model: str | None = Field(default=None, description="Requested model before routing.")
    category: str = "general"
    force_compress: bool = False


class IngestResponse(BaseModel):
    request_id: str
    agent_id: str
    response: str
    cache_hit: bool
    prompt_hash: str
    requested_model: str
    model: str
    priority: str
    category: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    optimizations: dict[str, Any]


class RequestSummary(BaseModel):
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    models: dict[str, int] = Field(default_factory=dict)
    priorities: dict[str, int] = Field(default_factory=dict)
    categories: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    agents: dict[str, dict[str, float | int | str | None]] = Field(default_factory=dict)
    window_minutes: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@router.get("/summary", response_model=RequestSummary)
async def get_requests_summary(
    window_minutes: int = Query(default=24 * 60, ge=5, le=7 * 24 * 60),
) -> RequestSummary:
    """Return aggregated request traffic stats for the dashboard."""
    return await _request_summary(window_minutes)


@router.get("/live")
async def live_requests() -> EventSourceResponse:
    """Stream live request ingest/cache events over SSE."""
    return EventSourceResponse(_request_event_stream())


@router.post("/ingest", response_model=IngestResponse)
async def ingest_request(payload: IngestRequest) -> IngestResponse:
    """Ingest one AI request and run all request-time optimization tools."""
    requested_model = payload.model or DEFAULT_MODEL
    priority = payload.priority.value
    original_prompt = payload.prompt.strip()
    original_prompt_tokens = gemini_client.count_tokens(original_prompt)
    estimated_completion_tokens = _estimate_completion_tokens(original_prompt_tokens, priority)
    logger.info(
        "ingest.start agent_id=%s priority=%s requested_model=%s category=%s prompt_tokens=%s",
        payload.agent_id,
        priority,
        requested_model,
        payload.category,
        original_prompt_tokens,
    )

    duplicate = await duplicate_detector.check(
        prompt=original_prompt,
        model=requested_model,
        agent_id=payload.agent_id,
    )
    if duplicate.is_duplicate and duplicate.response is not None:
        logger.info(
            "ingest.cache_hit agent_id=%s prompt_hash=%s source=%s cost_saved_usd=%.6f",
            payload.agent_id,
            duplicate.prompt_hash,
            duplicate.source,
            duplicate.cost_saved_usd,
        )
        request_doc = AIRequest(
            agent_id=payload.agent_id,
            prompt=original_prompt,
            priority=priority,
            model=duplicate.cached_output.model if duplicate.cached_output else requested_model,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            prompt_hash=duplicate.prompt_hash,
            cache_hit=True,
            category=payload.category,
        )
        await _persist_request(request_doc, extra={"served_from": duplicate.source})
        await _publish_request_event(
            "request_cache_hit",
            request_doc,
            {"source": duplicate.source, "cost_saved_usd": duplicate.cost_saved_usd},
        )
        return IngestResponse(
            request_id=request_doc.id,
            agent_id=payload.agent_id,
            response=duplicate.response,
            cache_hit=True,
            prompt_hash=duplicate.prompt_hash,
            requested_model=requested_model,
            model=request_doc.model,
            priority=priority,
            category=payload.category,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            optimizations={
                "duplicate": duplicate.model_dump(mode="json", exclude={"cached_output"}),
                "routing": None,
                "budget": None,
                "compression": None,
            },
        )

    routing = await traffic_router.reroute_agent(
        agent_id=payload.agent_id,
        priority=priority,
        requested_model=requested_model,
        prompt_tokens=original_prompt_tokens,
        completion_tokens=estimated_completion_tokens,
    )
    logger.info(
        "ingest.routing agent_id=%s requested_model=%s selected_model=%s rerouted=%s reason=%s",
        payload.agent_id,
        routing.original_model,
        routing.selected_model,
        routing.rerouted,
        routing.reason,
    )
    projected_cost = compute_cost(
        routing.selected_model,
        original_prompt_tokens,
        estimated_completion_tokens,
    )
    budget = await budget_allocator.enforce_budget_limits(
        agent_id=payload.agent_id,
        category=payload.category,
        priority=priority,
        projected_cost_usd=projected_cost,
    )
    logger.info(
        "ingest.budget agent_id=%s action=%s current_spend_usd=%.6f projected_spend_usd=%.6f",
        payload.agent_id,
        budget.action,
        budget.current_spend_usd,
        budget.projected_spend_usd,
    )
    if budget.action in {"pause", "throttle"}:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": budget.reason,
                "action": budget.action,
                "budget": budget.model_dump(mode="json"),
                "routing": routing.model_dump(mode="json"),
            },
        )

    final_prompt = original_prompt
    compression_result = None
    should_compress = (
        payload.force_compress
        or budget.action == "compress"
        or original_prompt_tokens >= 2_000
    )
    if should_compress:
        compression_result = await prompt_compressor.compress(
            prompt=original_prompt,
            agent_id=payload.agent_id,
            model=routing.selected_model,
            target_ratio=0.65,
        )
        final_prompt = compression_result.compressed_prompt

    try:
        logger.info(
            "ingest.gemini_start agent_id=%s model=%s final_prompt_tokens=%s",
            payload.agent_id,
            routing.selected_model,
            gemini_client.count_tokens(final_prompt),
        )
        generation = await gemini_client.generate(
            prompt=final_prompt,
            model=routing.selected_model,
        )
    except Exception as exc:
        logger.exception(
            "ingest.gemini_failed agent_id=%s requested_model=%s selected_model=%s",
            payload.agent_id,
            requested_model,
            routing.selected_model,
        )
        raise HTTPException(
            status_code=_generation_error_status(exc),
            detail={
                "message": "Gemini generation failed before the request could be recorded as spend.",
                "error": str(exc),
                "requested_model": requested_model,
                "selected_model": routing.selected_model,
                "routing": routing.model_dump(mode="json"),
                "hint": (
                    "If this is a model quota issue, retry with a priority/model that routes to "
                    "an available Gemini model, or update your Gemini quota/billing."
                ),
            },
        ) from exc
    actual_cost = compute_cost(
        generation.model,
        generation.prompt_tokens,
        generation.completion_tokens,
    )
    logger.info(
        "ingest.gemini_success agent_id=%s model=%s prompt_tokens=%s completion_tokens=%s cost_usd=%.6f",
        payload.agent_id,
        generation.model,
        generation.prompt_tokens,
        generation.completion_tokens,
        actual_cost,
    )

    cached = await duplicate_detector.cache(
        prompt=original_prompt,
        response=generation.text,
        model=generation.model,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
    )
    request_doc = AIRequest(
        agent_id=payload.agent_id,
        prompt=original_prompt,
        priority=priority,
        model=generation.model,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        cost_usd=actual_cost,
        prompt_hash=cached.prompt_hash,
        cache_hit=False,
        category=payload.category,
    )

    optimizations = {
        "duplicate": duplicate.model_dump(mode="json"),
        "routing": routing.model_dump(mode="json"),
        "budget": budget.model_dump(mode="json"),
        "compression": compression_result.model_dump(mode="json") if compression_result else None,
    }
    await _persist_request(request_doc, extra={"optimizations": optimizations})
    await _increment_usage_counters(generation.prompt_tokens + generation.completion_tokens, actual_cost)
    await _publish_request_event("request_ingested", request_doc, optimizations)
    logger.info(
        "ingest.complete request_id=%s agent_id=%s cost_usd=%.6f total_tokens=%s",
        request_doc.id,
        payload.agent_id,
        actual_cost,
        generation.prompt_tokens + generation.completion_tokens,
    )

    return IngestResponse(
        request_id=request_doc.id,
        agent_id=payload.agent_id,
        response=generation.text,
        cache_hit=False,
        prompt_hash=cached.prompt_hash,
        requested_model=requested_model,
        model=generation.model,
        priority=priority,
        category=payload.category,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        cost_usd=actual_cost,
        optimizations=optimizations,
    )


async def _request_summary(window_minutes: int) -> RequestSummary:
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
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
                                    "prompt_tokens": {"$sum": "$prompt_tokens"},
                                    "completion_tokens": {"$sum": "$completion_tokens"},
                                    "cost_usd": {"$sum": "$cost_usd"},
                                    "cache_hits": {
                                        "$sum": {"$cond": [{"$eq": ["$cache_hit", True]}, 1, 0]}
                                    },
                                }
                            }
                        ],
                        "models": [{"$group": {"_id": "$model", "count": {"$sum": 1}}}],
                        "priorities": [{"$group": {"_id": "$priority", "count": {"$sum": 1}}}],
                        "categories": [
                            {
                                "$group": {
                                    "_id": "$category",
                                    "count": {"$sum": 1},
                                    "cost_usd": {"$sum": "$cost_usd"},
                                }
                            }
                        ],
                        "agents": [
                            {"$sort": {"created_at": -1}},
                            {
                                "$group": {
                                    "_id": "$agent_id",
                                    "count": {"$sum": 1},
                                    "tokens": {
                                        "$sum": {"$add": ["$prompt_tokens", "$completion_tokens"]}
                                    },
                                    "cost_usd": {"$sum": "$cost_usd"},
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
            return RequestSummary(window_minutes=window_minutes)

        row = rows[0]
        totals = (row.get("totals") or [{}])[0]
        prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
        completion_tokens = int(totals.get("completion_tokens", 0) or 0)
        request_count = int(totals.get("request_count", 0) or 0)
        cache_hits = int(totals.get("cache_hits", 0) or 0)

        return RequestSummary(
            request_count=request_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(float(totals.get("cost_usd", 0.0) or 0.0), 6),
            cache_hits=cache_hits,
            cache_hit_rate=round(cache_hits / request_count, 4) if request_count else 0.0,
            models=_count_map(row.get("models", []), "unknown"),
            priorities=_count_map(row.get("priorities", []), "unknown"),
            categories={
                str(item.get("_id") or "general"): {
                    "count": int(item.get("count", 0) or 0),
                    "cost_usd": round(float(item.get("cost_usd", 0.0) or 0.0), 6),
                }
                for item in row.get("categories", [])
            },
            agents={
                str(item.get("_id") or "unknown"): {
                    "count": int(item.get("count", 0) or 0),
                    "tokens": int(item.get("tokens", 0) or 0),
                    "cost_usd": round(float(item.get("cost_usd", 0.0) or 0.0), 6),
                    "latest_model": item.get("latest_model"),
                    "last_seen_at": (
                        item["last_seen_at"].isoformat()
                        if hasattr(item.get("last_seen_at"), "isoformat")
                        else None
                    ),
                }
                for item in row.get("agents", [])
            },
            window_minutes=window_minutes,
        )
    except Exception:
        return RequestSummary(window_minutes=window_minutes)


async def _request_event_stream() -> AsyncIterator[dict[str, str]]:
    try:
        from integrations import cache

        async for event in cache.subscribe():
            event_type = str(event.get("type") or "")
            if event_type not in {"request_ingested", "request_cache_hit"}:
                continue
            yield {"event": event_type, "data": json.dumps(event)}
    except Exception as exc:
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "type": "request_stream_error",
                    "message": f"Request event stream unavailable: {type(exc).__name__}",
                }
            ),
        }


async def _persist_request(request_doc: AIRequest, extra: dict[str, Any] | None = None) -> None:
    doc = request_doc.model_dump(mode="python")
    if extra:
        doc.update(extra)
    try:
        from integrations import db

        await db.ai_requests().insert_one(doc)
        logger.info(
            "ingest.persisted request_id=%s agent_id=%s cost_usd=%.6f",
            request_doc.id,
            request_doc.agent_id,
            request_doc.cost_usd,
        )
    except Exception:
        logger.exception(
            "ingest.persist_failed request_id=%s agent_id=%s",
            request_doc.id,
            request_doc.agent_id,
        )


def _count_map(items: list[dict[str, Any]], default: str) -> dict[str, int]:
    return {
        str(item.get("_id") or default): int(item.get("count", 0) or 0)
        for item in items
    }


def _created_after_match(since: datetime) -> dict[str, Any]:
    return {
        "$or": [
            {"created_at": {"$gte": since}},
            {"created_at": {"$gte": since.isoformat()}},
        ]
    }


def _estimate_completion_tokens(prompt_tokens: int, priority: str) -> int:
    """Internal pre-call estimate; actual completion tokens come from Gemini output."""
    priority_floor = {
        "critical": 1024,
        "high": 768,
        "medium": 512,
        "low": 256,
    }.get(priority, 512)
    prompt_based = int(prompt_tokens * 0.35)
    return max(priority_floor, min(prompt_based, 4_096))


def _generation_error_status(exc: Exception) -> int:
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in {400, 401, 403, 404, 408, 409, 429}:
        return int(status_code)
    text = str(exc).lower()
    if "resource_exhausted" in text or "quota" in text or "429" in text:
        return status.HTTP_429_TOO_MANY_REQUESTS
    return status.HTTP_502_BAD_GATEWAY


async def _increment_usage_counters(tokens: int, cost_usd: float) -> None:
    try:
        from integrations import cache

        client = cache.get_client()
        tokens_total = await client.incrby("quota:tokens_used_today", tokens)
        cost_total = await client.incrbyfloat("quota:cost_usd_today", cost_usd)
        logger.info(
            "ingest.counters_updated tokens_added=%s cost_added_usd=%.6f tokens_used_today=%s cost_usd_today=%.6f",
            tokens,
            cost_usd,
            tokens_total,
            float(cost_total),
        )
    except Exception:
        logger.exception(
            "ingest.counters_failed tokens_added=%s cost_added_usd=%.6f",
            tokens,
            cost_usd,
        )


async def _publish_request_event(
    event_type: str,
    request_doc: AIRequest,
    optimizations: dict[str, Any],
) -> None:
    event = {
        "type": event_type,
        "request_id": request_doc.id,
        "agent_id": request_doc.agent_id,
        "model": request_doc.model,
        "priority": request_doc.priority
        if isinstance(request_doc.priority, str)
        else request_doc.priority.value,
        "category": request_doc.category,
        "cost_usd": request_doc.cost_usd,
        "total_tokens": request_doc.prompt_tokens + request_doc.completion_tokens,
        "cache_hit": request_doc.cache_hit,
        "prompt_hash": request_doc.prompt_hash,
        "optimizations": optimizations,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from integrations import cache

        await cache.publish_event(event)
        logger.info(
            "ingest.event_published type=%s request_id=%s",
            event_type,
            request_doc.id,
        )
    except Exception:
        logger.exception(
            "ingest.event_publish_failed type=%s request_id=%s",
            event_type,
            request_doc.id,
        )
