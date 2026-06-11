"""Agent runner — calls Google Cloud Agent Builder and executes its decisions.

Integration pattern (Option A, per the live Dialogflow CX playbook): the agent
OUTPUTS a structured response — a `REASONING:` bullet list followed by an
`ACTIONS:` JSON array of {"tool", "parameters"} — and THIS module parses it,
executes each tool against the optimization engines / Redis routing state, logs
the full trace to `agent_logs`, and publishes an `agent_decision` SSE event for
the dashboard's reasoning panel.

If Agent Builder is unreachable or unconfigured, `_fallback_decision()` derives a
safe, deterministic action set from the context so the orchestrator loop — and
the demo — always produces a visible decision.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from config.env import get_env
from models.agent_log import AgentDecision, AgentLog
from models.optimization_event import OptimizationEvent, OptimizationType

logger = logging.getLogger("quotamind.agent")

try:
    from google.cloud import dialogflowcx_v3 as dialogflow
except ImportError:  # pragma: no cover - dependency installed in app env.
    dialogflow = None

# Map the playbook's tool names (and common synonyms) to our internal handlers.
_TOOL_ALIASES = {
    "reroute_workload": "reroute",
    "reroute": "reroute",
    "compress_prompt_for_agent": "compress",
    "compress": "compress",
    "throttle_agent": "throttle",
    "throttle": "throttle",
    "pause_agent": "pause",
    "pause": "pause",
    "cache_response": "cache",
    "cache": "cache",
    "send_alert": "alert",
    "alert": "alert",
}


class AgentRunner:
    """Calls the Agent Builder playbook and executes the returned actions."""

    def __init__(self) -> None:
        self.project_id = get_env("GCP_PROJECT_ID")
        self.location = get_env("GCP_LOCATION", "us-central1")
        self.agent_id = get_env("AGENT_BUILDER_AGENT_ID")
        self.language_code = "en"

    @property
    def configured(self) -> bool:
        return bool(self.project_id and self.agent_id and dialogflow is not None)

    async def run_cycle(self, context: dict, cycle: int = 0) -> AgentLog:
        """One reasoning+execution cycle: call agent, parse, execute, log."""
        prompt = self._build_prompt(context)
        raw_response: str | None = None
        try:
            if self.configured:
                raw_response = await self._call_agent_builder(prompt)
        except Exception as e:  # noqa: BLE001 - fall back, never crash the loop
            logger.warning("Agent Builder call failed (%s: %s); using fallback.",
                           type(e).__name__, e)

        if raw_response:
            reasoning = self._parse_reasoning(raw_response)
            decisions = self._parse_actions(raw_response)
            source = "agent_builder"
        else:
            reasoning, decisions = self._fallback_decision(context)
            source = "fallback"

        if not reasoning:
            reasoning = ["No anomalies requiring action this cycle."]

        for decision in decisions:
            await self._execute_decision(decision)

        log = AgentLog(
            cycle=cycle,
            context=context,
            reasoning=reasoning,
            decisions=decisions,
            raw_response=raw_response,
        )
        await self._persist_log(log, source)
        await self._publish_decision_event(log, source)
        logger.info("agent.cycle=%s source=%s decisions=%s", cycle, source, len(decisions))
        return log

    # ── Agent Builder (Dialogflow CX) ─────────────────────────────────

    async def _call_agent_builder(self, prompt: str) -> str:
        """Send the context prompt to the playbook; return concatenated text."""
        # Regional CX agents (e.g. us-central1) require a regional API endpoint;
        # the default global endpoint returns 400 for non-global locations.
        client_options = None
        if self.location and self.location != "global":
            from google.api_core.client_options import ClientOptions

            client_options = ClientOptions(
                api_endpoint=f"{self.location}-dialogflow.googleapis.com"
            )
        client = dialogflow.SessionsAsyncClient(client_options=client_options)
        session = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/agents/{self.agent_id}/sessions/quotamind-orchestrator"
        )
        text_input = dialogflow.TextInput(text=prompt[:4000])
        query_input = dialogflow.QueryInput(text=text_input, language_code=self.language_code)
        request = dialogflow.DetectIntentRequest(session=session, query_input=query_input)
        response = await client.detect_intent(request=request)

        parts: list[str] = []
        for message in response.query_result.response_messages:
            if message.text and message.text.text:
                parts.extend(message.text.text)
        return "\n".join(parts).strip()

    def _build_prompt(self, context: dict) -> str:
        """Render the metrics snapshot into the playbook's expected input."""
        return (
            "You are QuotaMind's optimization agent. Analyze this snapshot and "
            "respond with a REASONING: bullet list and an ACTIONS: JSON array of "
            '{"tool","parameters"} using tools reroute_workload, '
            "compress_prompt_for_agent, throttle_agent, pause_agent, cache_response, "
            "send_alert. Never throttle or pause critical agents.\n\n"
            f"CONTEXT:\n{json.dumps(context, default=str, indent=2)}"
        )

    # ── Response parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse_reasoning(text: str) -> list[str]:
        section = text
        upper = text.upper()
        if "REASONING:" in upper:
            start = upper.index("REASONING:") + len("REASONING:")
            end = upper.index("ACTIONS:") if "ACTIONS:" in upper else len(text)
            section = text[start:end]
        bullets: list[str] = []
        for line in section.splitlines():
            cleaned = line.strip().lstrip("-*•0123456789. ").strip()
            if cleaned:
                bullets.append(cleaned)
        return bullets

    @staticmethod
    def _parse_actions(text: str) -> list[AgentDecision]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            raw = json.loads(match.group(0))
        except (ValueError, TypeError):
            return []
        decisions: list[AgentDecision] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            if not tool:
                continue
            params = item.get("parameters") or item.get("params") or {}
            decisions.append(
                AgentDecision(tool=tool, parameters=params if isinstance(params, dict) else {})
            )
        return decisions

    # ── Deterministic fallback (Agent Builder unavailable) ────────────

    @staticmethod
    def _fallback_decision(context: dict) -> tuple[list[str], list[AgentDecision]]:
        reasoning: list[str] = []
        decisions: list[AgentDecision] = []

        forecast = context.get("forecast") or {}
        if forecast.get("status") in {"at_risk", "exhausted"}:
            reasoning.append(
                f"Quota {round(forecast.get('percent_used', 0) * 100)}% used; "
                f"status {forecast.get('status')}."
            )

        for agent in context.get("agents", []):
            agent_id = agent.get("agent_id")
            priority = (agent.get("priority") or "medium").lower()
            severity = agent.get("severity")
            action = agent.get("recommended_action")
            if priority == "critical":
                continue  # never throttle/pause critical agents
            if severity in {"high", "critical"} and action in {"throttle", "pause"}:
                reasoning.append(
                    f"{agent_id} shows runaway risk ({severity}); {action} recommended."
                )
                tool = "throttle_agent" if action == "throttle" else "pause_agent"
                decisions.append(AgentDecision(tool=tool, parameters={"agent_id": agent_id}))
            elif agent.get("current_model") == "gemini-2.5-pro" and priority in {"medium", "low"}:
                reasoning.append(
                    f"{agent_id} ({priority}) is on gemini-2.5-pro unnecessarily; reroute to flash."
                )
                decisions.append(
                    AgentDecision(
                        tool="reroute_workload",
                        parameters={"agent_id": agent_id, "target_model": "gemini-2.5-flash"},
                    )
                )

        dynatrace = context.get("dynatrace") or {}
        if dynatrace.get("active_problems"):
            reasoning.append(
                f"Dynatrace reports {dynatrace['active_problems']} active problem(s)."
            )
            decisions.append(
                AgentDecision(
                    tool="send_alert",
                    parameters={
                        "severity": "warning",
                        "message": f"Dynatrace anomalies: {', '.join(dynatrace.get('anomalies', [])) or 'see Dynatrace'}.",
                    },
                )
            )
        return reasoning, decisions

    # ── Tool execution ────────────────────────────────────────────────

    async def _execute_decision(self, decision: AgentDecision) -> None:
        action = _TOOL_ALIASES.get(decision.tool.strip().lower())
        params = decision.parameters or {}
        agent_id = params.get("agent_id")
        try:
            if action == "reroute":
                model = params.get("target_model") or params.get("model") or "gemini-2.5-flash"
                await self._set_state(f"agent_route:{agent_id}", {"model": model})
                await self._log_event(
                    OptimizationType.REROUTE, agent_id,
                    f"Rerouted {agent_id} → {model}.", after={"model": model},
                )
                decision.result = f"rerouted to {model}"
            elif action == "compress":
                await self._set_state(f"agent_compress:{agent_id}", True, ttl=15 * 60)
                await self._log_event(
                    OptimizationType.COMPRESS, agent_id, f"Compression enabled for {agent_id}."
                )
                decision.result = "compression enabled"
            elif action == "throttle":
                await self._set_state(f"agent_throttle:{agent_id}", True, ttl=10 * 60)
                await self._log_event(
                    OptimizationType.THROTTLE, agent_id, f"Throttled {agent_id}."
                )
                decision.result = "throttled"
            elif action == "pause":
                await self._set_state(
                    f"agent_paused:{agent_id}",
                    {"reason": "agent decision", "created_at": _now_iso()},
                    ttl=15 * 60,
                )
                await self._log_event(OptimizationType.PAUSE, agent_id, f"Paused {agent_id}.")
                decision.result = "paused"
            elif action == "cache":
                await self._log_event(
                    OptimizationType.CACHE_HIT, agent_id, f"Caching encouraged for {agent_id}."
                )
                decision.result = "cache encouraged"
            elif action == "alert":
                message = params.get("message", "Operational alert.")
                severity = params.get("severity", "info")
                await self._log_event(
                    OptimizationType.ALERT, agent_id, message, after={"severity": severity}
                )
                decision.result = "alert raised"
            else:
                decision.result = f"unknown tool: {decision.tool}"
                decision.executed = False
                return

            decision.executed = True
            # Mirror mutating actions into Dynatrace (closes the observability loop).
            if action in {"reroute", "throttle", "pause"}:
                await self._push_to_dynatrace(action, agent_id, decision.result)
        except Exception as e:  # noqa: BLE001
            logger.warning("agent.execute_failed tool=%s: %s", decision.tool, e)
            decision.executed = False
            decision.result = f"error: {type(e).__name__}"

    async def _set_state(self, key: str, value, ttl: int | None = None) -> None:
        from integrations import cache

        await cache.set_cached(key, value, ttl_seconds=ttl)

    async def _log_event(
        self,
        type_: OptimizationType,
        agent_id: str | None,
        description: str,
        after: dict | None = None,
    ) -> None:
        event = OptimizationEvent(
            type=type_, agent_id=agent_id, description=description,
            after=after or {}, triggered_by="agent",
        ).model_dump(mode="json")
        try:
            from integrations import db

            await db.optimization_events().insert_one(dict(event))
        except Exception:
            pass
        try:
            from integrations import cache

            await cache.publish_event(event)
        except Exception:
            pass

    async def _push_to_dynatrace(self, action: str, agent_id: str | None, result: str) -> None:
        try:
            from integrations.dynatrace_mcp import dynatrace_mcp

            await dynatrace_mcp.push_custom_event(
                f"QuotaMind: {action} {agent_id}",
                f"Agent action: {result}.",
                properties={"agent_id": str(agent_id), "action": action},
            )
        except Exception:
            pass

    async def _persist_log(self, log: AgentLog, source: str) -> None:
        doc = log.model_dump(mode="json")
        doc["source"] = source
        try:
            from integrations import db

            await db.agent_logs().insert_one(doc)
        except Exception:
            logger.exception("agent.persist_log_failed cycle=%s", log.cycle)

    async def _publish_decision_event(self, log: AgentLog, source: str) -> None:
        event = {
            "type": "agent_decision",
            "cycle": log.cycle,
            "source": source,
            "reasoning": log.reasoning,
            "decisions": [d.model_dump() for d in log.decisions],
            "created_at": _now_iso(),
        }
        try:
            from integrations import cache

            await cache.publish_event(event)
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Module-level singleton.
agent_runner = AgentRunner()
