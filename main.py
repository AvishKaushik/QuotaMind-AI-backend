"""QuotaMind AI — FastAPI application entrypoint.

Foundation skeleton: wires settings, CORS, the MongoDB + Redis connections, and
a /health endpoint. Lane routers (requests, events, metrics, agent, simulator)
are mounted here as each lane delivers them — uncomment at the bottom.

Run locally:
    .venv/bin/uvicorn main:app --reload --port 8080
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.orchestrator import orchestrator
from api import (
    router_agent,
    router_events,
    router_metrics,
    router_requests,
    router_simulator,
)
from config.env import get_env
from config.settings import settings
from integrations import cache, db
from integrations.dynatrace_mcp import dynatrace_mcp

logging.basicConfig(
    level=get_env("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s:%(name)s:%(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    await cache.connect()
    # Connect the Dynatrace MCP server in the background so a slow npx cold start
    # never blocks boot; reads fall back to REST until the session is ready.
    mcp_task = asyncio.create_task(dynatrace_mcp.connect_mcp())
    # Start the autonomous optimization loop unless explicitly disabled.
    if get_env("ORCHESTRATOR_AUTOSTART", "true").lower() in {"1", "true", "yes"}:
        await orchestrator.start()
    yield
    # Shutdown
    await orchestrator.stop()
    mcp_task.cancel()
    await dynatrace_mcp.close_mcp()
    await dynatrace_mcp.close()
    await cache.close()
    await db.close()


app = FastAPI(title="QuotaMind AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Liveness + dependency probe. Reports per-dependency status."""
    status = {"app": "ok", "mongodb": "unknown", "redis": "unknown"}
    try:
        await db.get_db().command("ping")
        status["mongodb"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["mongodb"] = f"error: {type(e).__name__}"
    try:
        await cache.get_client().ping()
        status["redis"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["redis"] = f"error: {type(e).__name__}"
    status["dynatrace"] = (
        "mcp" if dynatrace_mcp.mcp_available
        else "rest" if dynatrace_mcp.configured
        else "unconfigured"
    )
    return status


# ── Lane routers ──────────────────────────────────────────────────
app.include_router(router_requests.router)
app.include_router(router_agent.router)
app.include_router(router_metrics.router)
app.include_router(router_events.router)
app.include_router(router_simulator.router)
