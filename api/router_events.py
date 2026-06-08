"""Real-time event stream API.

Bridges the Redis pub/sub event bus to the frontend through Server-Sent Events.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from config.env import get_env

router = APIRouter(prefix="/api/events", tags=["events"])
logger = logging.getLogger("quotamind.events")


@router.get("/stream")
async def stream_events(request: Request) -> EventSourceResponse:
    """Stream every Redis event on the QuotaMind event bus."""
    return EventSourceResponse(_event_stream(request))


async def _event_stream(request: Request) -> AsyncIterator[dict[str, str]]:
    channel = get_env("EVENTS_CHANNEL", "quotamind:events")
    yield _sse("connected", {"type": "connected", "channel": channel, "created_at": _now()})

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    subscriber = asyncio.create_task(_subscribe_to_redis(queue, channel))
    heartbeat = asyncio.create_task(_heartbeat(queue))

    try:
        while not await request.is_disconnected():
            event = await queue.get()
            event_type = str(event.get("type") or "message")
            yield _sse(event_type, event)
    finally:
        subscriber.cancel()
        heartbeat.cancel()
        await asyncio.gather(subscriber, heartbeat, return_exceptions=True)
        logger.info("events.stream_disconnected channel=%s", channel)


async def _subscribe_to_redis(queue: asyncio.Queue[dict[str, Any]], channel: str) -> None:
    try:
        from integrations import cache

        logger.info("events.stream_subscribed channel=%s", channel)
        async for event in cache.subscribe(channel=channel):
            await queue.put(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("events.stream_failed channel=%s", channel)
        await queue.put(
            {
                "type": "stream_error",
                "message": f"Event stream unavailable: {type(exc).__name__}",
                "created_at": _now(),
            }
        )


async def _heartbeat(queue: asyncio.Queue[dict[str, Any]]) -> None:
    try:
        while True:
            await asyncio.sleep(15)
            await queue.put({"type": "heartbeat", "created_at": _now()})
    except asyncio.CancelledError:
        raise


def _sse(event_type: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": event_type, "data": json.dumps(payload)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
