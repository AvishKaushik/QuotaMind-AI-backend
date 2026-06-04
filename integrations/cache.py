"""Redis (asyncio) client — lifecycle, KV/counter helpers, and the event bus.

The pub/sub channel (`settings.events_channel`) is the ONLY way the two lanes
talk to each other in real time: optimization engines `publish_event(...)`, and
the SSE router `subscribe()`s to stream events to the frontend.

Usage:
    from integrations import cache
    await cache.connect()                  # on app startup
    await cache.set_cached("k", {...}, ttl_seconds=3600)
    await cache.publish_event({"type": "request_ingested", ...})
    await cache.close()                    # on app shutdown
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import redis.asyncio as redis

from config.settings import settings

_client: redis.Redis | None = None


async def connect() -> None:
    """Open the Redis connection and verify with PING. Idempotent."""
    global _client
    if _client is not None:
        return
    _client = redis.from_url(settings.redis_url, decode_responses=True)
    await _client.ping()


async def close() -> None:
    """Close the Redis connection. Call on app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


def get_client() -> redis.Redis:
    if _client is None:
        raise RuntimeError("Redis not connected — call cache.connect() on startup.")
    return _client


# ── Key/value + counter helpers ────────────────────────────────────
async def get_cached(key: str) -> Any | None:
    """JSON-decode a stored value, or None if the key is absent."""
    raw = await get_client().get(key)
    return json.loads(raw) if raw is not None else None


async def set_cached(key: str, value: Any, ttl_seconds: int | None = None) -> None:
    """JSON-encode and store a value, optionally with a TTL in seconds."""
    await get_client().set(key, json.dumps(value), ex=ttl_seconds)


async def increment_counter(key: str, amount: int = 1) -> int:
    """Atomically increment a counter (e.g. tokens used today). Returns new value."""
    return await get_client().incrby(key, amount)


# ── Pub/Sub event bus (cross-lane channel) ─────────────────────────
async def publish_event(event: dict, channel: str | None = None) -> None:
    """Publish a JSON event to the shared channel."""
    await get_client().publish(channel or settings.events_channel, json.dumps(event))


async def subscribe(channel: str | None = None) -> AsyncIterator[dict]:
    """Async-iterate JSON events from the channel. Drives the SSE endpoint."""
    chan = channel or settings.events_channel
    pubsub = get_client().pubsub()
    await pubsub.subscribe(chan)
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                yield json.loads(message["data"])
            except (ValueError, TypeError):
                continue
    finally:
        await pubsub.unsubscribe(chan)
        await pubsub.aclose()
