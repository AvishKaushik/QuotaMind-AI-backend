"""Simulator control API — the demo's crisis buttons and traffic toggle.

Powers the dashboard's floating Simulator Panel: a steady-traffic start/stop
toggle (requests/min) plus the four one-click crisis scenarios and a reset.
Each scenario produces real data the optimization engines and agent react to.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from simulator.workload_simulator import workload_simulator

router = APIRouter(prefix="/api/simulator", tags=["simulator"])


@router.get("/status")
async def simulator_status() -> dict[str, Any]:
    """Current simulator state (running, rate, totals) for the panel."""
    return workload_simulator.status()


@router.post("/start")
async def simulator_start(
    rate_per_min: int = Query(default=60, ge=1, le=600),
) -> dict[str, Any]:
    """Start steady background traffic at the given requests/min."""
    return await workload_simulator.start(rate_per_min=rate_per_min)


@router.post("/stop")
async def simulator_stop() -> dict[str, Any]:
    """Stop background traffic."""
    return await workload_simulator.stop()


@router.post("/spike")
async def simulator_spike(
    count: int = Query(default=40, ge=1, le=300),
) -> dict[str, Any]:
    """Crisis 1 — sudden traffic burst (token-burn spike)."""
    return await workload_simulator.trigger_spike(count=count)


@router.post("/runaway")
async def simulator_runaway(
    agent_id: str = Query(default="summarizer-1"),
    count: int = Query(default=35, ge=1, le=200),
) -> dict[str, Any]:
    """Crisis 2 — a single agent loops on duplicate prompts (runaway)."""
    return await workload_simulator.trigger_runaway(agent_id=agent_id, count=count)


@router.post("/budget-overflow")
async def simulator_budget_overflow() -> dict[str, Any]:
    """Crisis 3 — drive spend toward the daily budget ceiling."""
    return await workload_simulator.trigger_budget_overflow()


@router.post("/dynatrace-anomaly")
async def simulator_dynatrace_anomaly() -> dict[str, Any]:
    """Crisis 4 — surface a Dynatrace latency anomaly (pushes a real custom event)."""
    return await workload_simulator.trigger_dynatrace_anomaly()


@router.post("/reset")
async def simulator_reset() -> dict[str, Any]:
    """Reset — clear counters, agent state, and all simulated data."""
    return await workload_simulator.reset()
