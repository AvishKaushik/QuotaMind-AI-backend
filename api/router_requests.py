"""Request ingestion API.

The ingest route is the synchronous optimization gate in front of Gemini:
duplicate cache -> routing -> budget enforcement -> optional compression ->
Gemini -> persistence/counters/events.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from config.models import DEFAULT_MODEL, compute_cost
from engines.budget_allocator import budget_allocator
from engines.duplicate_detector import duplicate_detector
from engines.prompt_compressor import prompt_compressor
from engines.traffic_router import traffic_router
from integrations.gemini_client import gemini_client
from models.ai_request import AIRequest, Priority

router = APIRouter(prefix="/api/requests", tags=["requests"])


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


@router.post("/ingest", response_model=IngestResponse)
async def ingest_request(payload: IngestRequest) -> IngestResponse:
    """Ingest one AI request and run all request-time optimization tools."""
    requested_model = payload.model or DEFAULT_MODEL
    priority = payload.priority.value
    original_prompt = payload.prompt.strip()
    original_prompt_tokens = gemini_client.count_tokens(original_prompt)
    estimated_completion_tokens = _estimate_completion_tokens(original_prompt_tokens, priority)

    duplicate = await duplicate_detector.check(
        prompt=original_prompt,
        model=requested_model,
        agent_id=payload.agent_id,
    )
    if duplicate.is_duplicate and duplicate.response is not None:
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

    generation = await gemini_client.generate(
        prompt=final_prompt,
        model=routing.selected_model,
    )
    actual_cost = compute_cost(
        generation.model,
        generation.prompt_tokens,
        generation.completion_tokens,
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


async def _persist_request(request_doc: AIRequest, extra: dict[str, Any] | None = None) -> None:
    doc = request_doc.model_dump(mode="json")
    if extra:
        doc.update(extra)
    try:
        from integrations import db

        await db.ai_requests().insert_one(doc)
    except Exception:
        pass


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


async def _increment_usage_counters(tokens: int, cost_usd: float) -> None:
    try:
        from integrations import cache

        client = cache.get_client()
        await client.incrby("quota:tokens_used_today", tokens)
        await client.incrbyfloat("quota:cost_usd_today", cost_usd)
    except Exception:
        pass


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
        "cache_hit": request_doc.cache_hit,
        "prompt_hash": request_doc.prompt_hash,
        "optimizations": optimizations,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from integrations import cache

        await cache.publish_event(event)
    except Exception:
        pass
