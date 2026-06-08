"""QuotaMind AI — FastAPI application entrypoint.

Foundation skeleton: wires settings, CORS, the MongoDB + Redis connections, and
a /health endpoint. Lane routers (requests, events, metrics, agent, simulator)
are mounted here as each lane delivers them — uncomment at the bottom.

Run locally:
    .venv/bin/uvicorn main:app --reload --port 8080
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import router_requests # include everything later
from config.env import get_list_env
from integrations import cache, db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    await cache.connect()
    yield
    # Shutdown
    await cache.close()
    await db.close()


app = FastAPI(title="QuotaMind AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_list_env("CORS_ORIGINS", "http://localhost:3000"),
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
    return status


# ── Lane routers ──────────────────────────────────────────────────
app.include_router(router_requests.router)

# ── Lane routers — uncomment each as it lands ──────────────────────
# from api import (
#     router_requests, router_events, router_metrics, router_agent, router_simulator,
# )
# app.include_router(router_requests.router)
# app.include_router(router_events.router)
# app.include_router(router_metrics.router)
# app.include_router(router_agent.router)
# app.include_router(router_simulator.router)
