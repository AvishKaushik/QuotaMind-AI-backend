"""Dynatrace integration — observability signals for the Agent Orchestrator.

This is the seam the orchestrator pulls real-world signals from every 30s:
active problems, latency spikes, and a compact metrics summary it embeds in the
Gemini reasoning context. It can also push QuotaMind's own optimization actions
back into Dynatrace as custom events (closing the observability loop).

Two transports, hidden behind the public methods:
  * REST (`/api/v2`, classic `Api-Token`) — structured signals + event push.
  * The official @dynatrace-oss/dynatrace-mcp-server over stdio (platform token)
    — used for the agent-facing `get_observability_briefing()` (the genuine MCP
    usage in the loop). Reads route through MCP; writes stay on REST because the
    MCP server gates write tools behind interactive human approval, which an
    autonomous 30s loop can't satisfy.

All methods degrade gracefully: if credentials are missing, MCP fails to start,
or Dynatrace is unreachable, they log and fall back to REST / return empty-safe
structures so a blip never crashes the 30-second agent loop or the live demo.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from config.env import get_env, get_float_env, get_int_env

try:
    import httpx
except ImportError:  # pragma: no cover - dependency is installed in app env.
    httpx = None

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - dependency is installed in app env.
    ClientSession = StdioServerParameters = stdio_client = None

logger = logging.getLogger(__name__)

# Dynatrace reports service response time in MICROseconds.
_MICROS_PER_MS = 1000.0


class DynatraceMCPClient:
    """Async client for Dynatrace observability signals.

    Public methods (the stable interface both the orchestrator and engines use):
        get_active_problems()        -> list[dict]   (REST, structured)
        get_latency_spike_events()   -> list[dict]   (REST, structured)
        get_metrics()                -> dict         (REST, structured)
        push_custom_event(...)       -> dict         (REST write)
        get_observability_briefing() -> str          (MCP, agent-facing; REST fallback)

    Lifecycle (wired into the FastAPI lifespan in main.py):
        connect_mcp()  — start the persistent MCP session at startup
        close_mcp()    — tear it down at shutdown
    """

    def __init__(self) -> None:
        self.env_url = get_env("DYNATRACE_ENV_URL").rstrip("/")
        self.api_token = get_env("DYNATRACE_API_TOKEN")
        self.timeout = get_float_env("DYNATRACE_TIMEOUT_SECONDS", 10.0)
        # p95 latency (ms) above which we treat a service as "spiking".
        self.latency_spike_ms = get_int_env("DYNATRACE_LATENCY_SPIKE_MS", 2000)
        self._client: Any = None

        # MCP transport (official Dynatrace MCP server over stdio).
        self.platform_token = get_env("DT_PLATFORM_TOKEN")
        # The MCP server wants the PLATFORM host (…apps.dynatrace.com), not the
        # REST …live… host. Prefer the explicit DT_ENVIRONMENT, else derive it.
        self.dt_environment = get_env("DT_ENVIRONMENT") or self.env_url.replace(
            ".live.", ".apps."
        )
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_session: Any = None
        self._mcp_lock = asyncio.Lock()
        self.mcp_available = False

    @property
    def configured(self) -> bool:
        return bool(self.env_url and self.api_token and httpx is not None)

    @property
    def mcp_configured(self) -> bool:
        return bool(self.platform_token and self.dt_environment and stdio_client is not None)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.env_url,
                headers={"Authorization": f"Api-Token {self.api_token}"},
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        """GET an /api/v2 path; return parsed JSON or None on any failure."""
        if not self.configured:
            logger.warning("Dynatrace not configured; returning empty signal for %s", path)
            return None
        try:
            resp = await self._get_client().get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - never let a blip crash the loop
            logger.warning("Dynatrace GET %s failed: %s: %s", path, type(e).__name__, e)
            return None

    # ── Public interface ──────────────────────────────────────────────

    async def get_active_problems(self) -> list[dict]:
        """Open problems from Dynatrace, normalized to the orchestrator's shape.

        Returns a list of:
            {problemId, title, severity, status, affectedEntities, startTime}
        """
        data = await self._get(
            "/api/v2/problems",
            params={"problemSelector": 'status("OPEN")', "pageSize": 50},
        )
        if not data:
            return []
        problems: list[dict] = []
        for p in data.get("problems", []):
            problems.append(
                {
                    "problemId": p.get("problemId") or p.get("displayId", ""),
                    "title": p.get("title", ""),
                    "severity": p.get("severityLevel", ""),
                    "status": p.get("status", ""),
                    "affectedEntities": [
                        e.get("name", e.get("entityId", {}).get("id", ""))
                        for e in p.get("affectedEntities", [])
                    ],
                    "startTime": p.get("startTime", ""),
                }
            )
        return problems

    async def get_latency_spike_events(self) -> list[dict]:
        """Services whose p95 response time exceeds the spike threshold.

        Returns a list of:
            {entity, latency_p95_ms, threshold_ms, metric}
        """
        latencies = await self._service_p95_latencies()
        spikes: list[dict] = []
        for entity, p95_ms in latencies.items():
            if p95_ms >= self.latency_spike_ms:
                spikes.append(
                    {
                        "entity": entity,
                        "latency_p95_ms": round(p95_ms, 1),
                        "threshold_ms": self.latency_spike_ms,
                        "metric": "builtin:service.response.time",
                    }
                )
        return spikes

    async def get_metrics(self) -> dict:
        """Compact observability summary embedded in the Gemini reasoning context.

        Returns:
            {active_problems: int, latency_p95_ms: float|None, anomalies: list[str]}
        """
        problems = await self.get_active_problems()
        latencies = await self._service_p95_latencies()
        peak_p95 = max(latencies.values(), default=None)

        anomalies: list[str] = [p["title"] for p in problems if p["title"]]
        for entity, p95_ms in latencies.items():
            if p95_ms >= self.latency_spike_ms:
                anomalies.append(f"Response time spike on {entity} ({round(p95_ms)}ms p95)")

        return {
            "active_problems": len(problems),
            "latency_p95_ms": round(peak_p95, 1) if peak_p95 is not None else None,
            "anomalies": anomalies,
        }

    async def push_custom_event(
        self,
        title: str,
        description: str = "",
        *,
        event_type: str = "CUSTOM_INFO",
        properties: dict | None = None,
    ) -> dict:
        """Push a QuotaMind optimization action into Dynatrace as a custom event.

        Closes the loop: agent decisions become visible in the Dynatrace timeline.
        Returns {"ok": bool, ...}; never raises.
        """
        if not self.configured:
            logger.warning("Dynatrace not configured; skipping push_custom_event %r", title)
            return {"ok": False, "reason": "not_configured"}
        payload = {
            "eventType": event_type,
            "title": title,
            "properties": {"description": description, **(properties or {})},
        }
        try:
            resp = await self._get_client().post("/api/v2/events/ingest", json=payload)
            resp.raise_for_status()
            return {"ok": True, "response": resp.json() if resp.content else {}}
        except Exception as e:  # noqa: BLE001
            logger.warning("Dynatrace push_custom_event failed: %s: %s", type(e).__name__, e)
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    # ── MCP transport (agent-facing reads) ────────────────────────────

    async def connect_mcp(self) -> bool:
        """Start a persistent session to the official Dynatrace MCP server (stdio).

        Call once at app startup. On any failure the REST transport remains the
        automatic fallback, so this never blocks boot. Returns mcp_available.
        """
        if not self.mcp_configured:
            logger.info(
                "Dynatrace MCP not configured (need DT_PLATFORM_TOKEN + DT_ENVIRONMENT); "
                "using REST transport only."
            )
            return False
        async with self._mcp_lock:
            if self.mcp_available:
                return True
            try:
                env = {
                    **os.environ,
                    "DT_ENVIRONMENT": self.dt_environment,
                    "DT_PLATFORM_TOKEN": self.platform_token,
                    "DT_MCP_DISABLE_TELEMETRY": "true",
                }
                # Pinned to the version pre-installed in the Docker image so npx
                # resolves locally instead of hitting the registry on every cold start.
                package = os.environ.get(
                    "DT_MCP_PACKAGE", "@dynatrace-oss/dynatrace-mcp-server@1.8.7"
                )
                params = StdioServerParameters(
                    command="npx",
                    args=["-y", package],
                    env=env,
                )
                stack = AsyncExitStack()
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._mcp_stack = stack
                self._mcp_session = session
                self.mcp_available = True
                logger.info("Dynatrace MCP server connected (%s).", self.dt_environment)
            except Exception as e:  # noqa: BLE001 - fall back to REST, never crash boot
                logger.warning(
                    "Dynatrace MCP connect failed (%s: %s); falling back to REST.",
                    type(e).__name__,
                    e,
                )
                self.mcp_available = False
            return self.mcp_available

    async def close_mcp(self) -> None:
        async with self._mcp_lock:
            if self._mcp_stack is not None:
                try:
                    await self._mcp_stack.aclose()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Dynatrace MCP close error: %s", e)
            self._mcp_stack = None
            self._mcp_session = None
            self.mcp_available = False

    async def _call_mcp_text(self, tool: str, arguments: dict) -> str | None:
        """Call an MCP tool; return its concatenated text content, or None on failure."""
        if not self.mcp_available or self._mcp_session is None:
            return None
        try:
            async with self._mcp_lock:
                result = await self._mcp_session.call_tool(tool, arguments)
        except Exception as e:  # noqa: BLE001
            logger.warning("Dynatrace MCP tool %s failed: %s: %s", tool, type(e).__name__, e)
            return None
        parts = [getattr(b, "text", "") for b in result.content if getattr(b, "text", None)]
        text = "\n".join(parts).strip()
        if getattr(result, "isError", False):
            logger.warning("Dynatrace MCP tool %s returned error: %s", tool, text[:200])
            return None
        return text or None

    async def get_observability_briefing(self) -> str:
        """Natural-language observability briefing for the Gemini agent context.

        Primary path: the Dynatrace MCP server's `list_problems` (LLM-ready text)
        — the genuine MCP usage inside the agent loop. Falls back to a summary
        built from the REST problems feed if MCP is unavailable.
        """
        text = await self._call_mcp_text(
            "list_problems", {"timeframe": "24h", "status": "ACTIVE"}
        )
        if text:
            return f"[Dynatrace MCP] {text}"
        problems = await self.get_active_problems()
        if not problems:
            return "[Dynatrace REST] No active problems."
        lines = [
            f"- {p['title']} (severity {p['severity']}; "
            f"entities: {', '.join(p['affectedEntities']) or 'n/a'})"
            for p in problems
        ]
        return "[Dynatrace REST] Active problems:\n" + "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────────

    async def _service_p95_latencies(self) -> dict[str, float]:
        """Per-service p95 response time in ms, keyed by entity display name."""
        data = await self._get(
            "/api/v2/metrics/query",
            params={
                "metricSelector": "builtin:service.response.time:percentile(95)",
                "from": "now-5m",
                "resolution": "Inf",
            },
        )
        if not data:
            return {}
        out: dict[str, float] = {}
        for result in data.get("result", []):
            for series in result.get("data", []):
                dims = series.get("dimensionMap", {}) or {}
                name = (
                    dims.get("dt.entity.service.name")
                    or next(iter(dims.values()), None)
                    or "unknown-service"
                )
                values = [v for v in series.get("values", []) if v is not None]
                if values:
                    out[name] = values[-1] / _MICROS_PER_MS
        return out


# Module-level singleton (matches gemini_client / db / cache conventions).
dynatrace_mcp = DynatraceMCPClient()
