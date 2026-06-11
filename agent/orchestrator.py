"""Agent orchestrator — the autonomous 30-second optimization loop.

Each cycle it: collects metrics (quota forecast, budget, per-agent runaway
analysis) + Dynatrace observability signals, hands the snapshot to the Agent
Builder via `agent_runner`, which reasons and executes optimization actions, and
logs/publishes the result. Registered as a FastAPI background task in main.py.

The loop is defensive: any single-cycle failure is logged and the loop keeps
running, so the demo's autonomous agent never silently dies.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.agent_runner import agent_runner
from config.env import get_int_env
from engines.budget_allocator import budget_allocator
from engines.quota_forecaster import quota_forecaster
from engines.runaway_detector import runaway_detector
from integrations.dynatrace_mcp import dynatrace_mcp
from models.agent_log import AgentLog

logger = logging.getLogger("quotamind.orchestrator")


class AgentOrchestrator:
    """Runs the periodic collect → reason → act → log cycle."""

    def __init__(self) -> None:
        self.interval_seconds = get_int_env("ORCHESTRATOR_INTERVAL_SECONDS", 30)
        self._task: asyncio.Task | None = None
        self.cycle = 0
        self.last_run_at: datetime | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background loop (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info("orchestrator.started interval=%ss", self.interval_seconds)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("orchestrator.stopped")

    async def _run_loop(self) -> None:
        try:
            while True:
                try:
                    await self.run_once()
                except Exception:
                    logger.exception("orchestrator.cycle_failed cycle=%s", self.cycle)
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            raise

    # ── One cycle ─────────────────────────────────────────────────────

    async def run_once(self) -> AgentLog:
        """Collect context, run the agent, return the resulting AgentLog."""
        self.cycle += 1
        self.last_run_at = datetime.now(timezone.utc)
        context = await self._build_context()
        log = await agent_runner.run_cycle(context, cycle=self.cycle)
        return log

    async def _build_context(self) -> dict[str, Any]:
        forecast = await self._safe(quota_forecaster.forecast_exhaustion(), {})
        budget = await self._safe(budget_allocator.check_budget_status(), {})
        agents = await self._agent_snapshots()
        dynatrace_metrics = await self._safe(dynatrace_mcp.get_metrics(), {})
        dynatrace_briefing = await self._safe(dynatrace_mcp.get_observability_briefing(), "")

        return {
            "cycle": self.cycle,
            "generated_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "forecast": _dump(forecast),
            "budget": _dump(budget),
            "agents": agents,
            "dynatrace": {**_dump(dynatrace_metrics), "briefing": dynatrace_briefing},
        }

    async def _agent_snapshots(self) -> list[dict[str, Any]]:
        """Per-agent runaway analysis + current model for recently active agents."""
        snapshots: list[dict[str, Any]] = []
        for agent_id, meta in (await self._recent_agents()).items():
            try:
                analysis = await runaway_detector.analyze_agent(agent_id, window_minutes=5)
                current_model = await self._current_model(agent_id)
                snapshots.append(
                    {
                        "agent_id": agent_id,
                        "priority": meta.get("priority", "medium"),
                        "current_model": current_model,
                        "request_count": analysis.request_count,
                        "total_tokens": analysis.total_tokens,
                        "risk_score": analysis.risk_score,
                        "severity": analysis.severity,
                        "recommended_action": analysis.recommended_action,
                    }
                )
            except Exception:
                logger.exception("orchestrator.agent_snapshot_failed agent=%s", agent_id)
        return snapshots

    async def _recent_agents(self) -> dict[str, dict[str, Any]]:
        """Agents seen in the last 10 minutes, with their latest priority."""
        since = datetime.now(timezone.utc) - timedelta(minutes=10)
        try:
            from integrations import db

            cursor = db.ai_requests().aggregate(
                [
                    {"$match": {"created_at": {"$gte": since}}},
                    {"$sort": {"created_at": -1}},
                    {
                        "$group": {
                            "_id": "$agent_id",
                            "priority": {"$first": "$priority"},
                            "count": {"$sum": 1},
                        }
                    },
                ]
            )
            out: dict[str, dict[str, Any]] = {}
            async for row in cursor:
                agent_id = str(row.get("_id") or "").strip()
                if agent_id:
                    out[agent_id] = {
                        "priority": str(row.get("priority") or "medium"),
                        "count": int(row.get("count", 0) or 0),
                    }
            return out
        except Exception:
            return {}

    async def _current_model(self, agent_id: str) -> str | None:
        try:
            from integrations import cache

            route = await cache.get_cached(f"agent_route:{agent_id}")
            return route.get("model") if isinstance(route, dict) else None
        except Exception:
            return None

    @staticmethod
    async def _safe(awaitable, default):
        try:
            return await awaitable
        except Exception:
            return default

    def status(self) -> dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "cycle": self.cycle,
            "interval_seconds": self.interval_seconds,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
        }


def _dump(obj: Any) -> dict:
    """Coerce a pydantic result or dict into a plain JSON-able dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj if isinstance(obj, dict) else {}


# Module-level singleton.
orchestrator = AgentOrchestrator()
