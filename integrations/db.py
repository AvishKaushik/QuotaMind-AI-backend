"""MongoDB (Motor) async client — lifecycle, collection accessors, and indexes.

Usage:
    from integrations import db
    await db.connect()            # on app startup
    await db.ai_requests().insert_one(doc)
    await db.close()              # on app shutdown
"""
from __future__ import annotations

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from config.settings import settings

# Collection names — single source of truth for both lanes.
COLLECTION_AI_REQUESTS = "ai_requests"
COLLECTION_OPTIMIZATION_EVENTS = "optimization_events"
COLLECTION_AGENT_LOGS = "agent_logs"
COLLECTION_CACHED_OUTPUTS = "cached_outputs"

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect() -> None:
    """Open the Mongo connection, ping to verify, and build indexes. Idempotent."""
    global _client, _db
    if _client is not None:
        return
    _client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    _db = _client[settings.mongodb_db_name]
    await _client.admin.command("ping")
    await ensure_indexes()


async def close() -> None:
    """Close the Mongo connection. Call on app shutdown."""
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not connected — call db.connect() on startup.")
    return _db


def _coll(name: str) -> AsyncIOMotorCollection:
    return get_db()[name]


def ai_requests() -> AsyncIOMotorCollection:
    return _coll(COLLECTION_AI_REQUESTS)


def optimization_events() -> AsyncIOMotorCollection:
    return _coll(COLLECTION_OPTIMIZATION_EVENTS)


def agent_logs() -> AsyncIOMotorCollection:
    return _coll(COLLECTION_AGENT_LOGS)


def cached_outputs() -> AsyncIOMotorCollection:
    return _coll(COLLECTION_CACHED_OUTPUTS)


async def ensure_indexes() -> None:
    """Create the indexes both lanes rely on. Safe to call repeatedly."""
    await ai_requests().create_index("created_at")
    await ai_requests().create_index("agent_id")
    await ai_requests().create_index("prompt_hash")

    await optimization_events().create_index("created_at")
    await optimization_events().create_index("type")

    await agent_logs().create_index("created_at")
    await agent_logs().create_index("cycle")

    await cached_outputs().create_index("prompt_hash", unique=True)
