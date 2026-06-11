"""Workload simulator — plays the role of a company's AI tools for the demo.

It runs TWO lanes of traffic:

1. LIVE lane (the real thing): every ~12s one request is sent through the real
   `/api/requests/ingest` pipeline — the duplicate cache, traffic router, budget
   guard, prompt compressor, and a real Gemini call all execute. The four agents
   behave like real company tools (a support bot answering the same FAQs all day,
   a summarizer that over-uses Pro, ...), so every engine genuinely fires and the
   savings on the dashboard are real numbers.

2. SYNTHETIC lane (volume): writes realistic `AIRequest` records straight to
   MongoDB and bumps the Redis counters, exactly the shape the real pipeline
   emits. High-rate crisis scenarios (spike / runaway / budget overflow) stay on
   this lane on purpose so a burst of hundreds of requests can't exhaust the
   free-tier Gemini quota mid-demo.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.env import get_env, get_int_env
from config.models import compute_cost
from models.ai_request import AIRequest

logger = logging.getLogger("quotamind.simulator")


@dataclass
class AgentProfile:
    agent_id: str
    priority: str
    category: str
    model: str
    mean_prompt_tokens: int
    mean_completion_tokens: int
    prompts: list[str] = field(default_factory=list)
    # Live-lane behavior: a small fixed pool of "questions people really ask"
    # (repeats → genuine cache hits) + templates that render unique prompts
    # (misses → real Gemini calls). Weight controls how often this agent speaks.
    live_fixed_prompts: list[str] = field(default_factory=list)
    live_templates: list[str] = field(default_factory=list)
    live_weight: float = 1.0
    force_compress: bool = False


# The four demo agents. Token means are chosen so the optimization story is
# visible: summarizer-1 over-uses gemini-2.5-pro for medium work (reroute bait),
# support-bot-1 is critical and must be protected, analytics is cheap/low.
AGENT_PROFILES: list[AgentProfile] = [
    AgentProfile(
        agent_id="support-bot-1",
        priority="critical",
        category="support",
        model="gemini-2.5-pro",
        mean_prompt_tokens=1200,
        mean_completion_tokens=600,
        prompts=[
            "Customer cannot reset their password after the latest app update.",
            "Summarize this angry support ticket and suggest a resolution.",
            "How do I issue a refund for a duplicate subscription charge?",
        ],
        # Customers ask the same questions all day → real cache hits.
        live_fixed_prompts=[
            "How do I reset my password?",
            "How do I cancel my subscription?",
            "How can I get a refund for a duplicate charge?",
            "How do I update the credit card on my account?",
            "How do I enable two-factor authentication?",
            "How do I export my account data?",
        ],
        live_templates=[
            "A customer on the {plan} plan reports that {issue}. Draft a short, empathetic reply with next steps.",
        ],
        live_weight=0.40,
    ),
    AgentProfile(
        agent_id="summarizer-1",
        priority="medium",
        category="summarization",
        model="gemini-2.5-pro",
        mean_prompt_tokens=3000,
        mean_completion_tokens=800,
        prompts=[
            "Summarize the attached 12-page quarterly earnings report.",
            "Condense this meeting transcript into five bullet points.",
            "Produce an executive summary of the product requirements doc.",
        ],
        # Long wordy prompts on the expensive model → compressor + reroute bait.
        live_fixed_prompts=[
            "Summarize the following team update into three bullet points. "
            "This week the platform team focused primarily on stabilizing the deployment "
            "pipeline after the incident on Monday, in which a misconfigured health check "
            "caused rolling restarts across the staging cluster. The team also made progress "
            "on the database migration plan: the proposal to move the analytics tables to a "
            "separate read replica was approved, and the first dry run is scheduled for next "
            "Thursday. On the hiring side, two backend candidates moved to the final round. "
            "Finally, please note that the office will be closed on Friday for maintenance, "
            "and everyone should plan to work remotely that day.",
        ],
        live_templates=[
            "Summarize this meeting note in two sentences. Attendees discussed the {topic} "
            "initiative at length: the main blocker remains {blocker}, the owner is the {team} "
            "team, and the agreed next step is to {next_step} before the end of the sprint. "
            "Several attendees raised concerns about timeline risk, but the overall mood was "
            "constructive, and a follow-up review was scheduled for {day}.",
        ],
        live_weight=0.15,
        force_compress=True,
    ),
    AgentProfile(
        agent_id="analytics-agent-1",
        priority="low",
        category="analytics",
        model="gemini-2.5-flash",
        mean_prompt_tokens=800,
        mean_completion_tokens=300,
        prompts=[
            "Classify these 50 reviews as positive, negative, or neutral.",
            "Extract the top keywords from yesterday's search logs.",
            "Tag this batch of events by funnel stage.",
        ],
        # Scheduled reports re-run the same queries → cache hits on flash.
        live_fixed_prompts=[
            "List three KPIs a SaaS startup should track weekly and one sentence on why.",
            "What is a good north-star metric for a B2B analytics product? Answer in two sentences.",
            "Explain the difference between churn rate and retention rate in one sentence each.",
        ],
        live_templates=[
            "Classify the sentiment of this product review as positive, negative, or neutral, "
            "and give a one-line justification: \"{review}\"",
        ],
        live_weight=0.25,
    ),
    AgentProfile(
        agent_id="workflow-agent-1",
        priority="high",
        category="workflow",
        model="gemini-2.5-pro",
        mean_prompt_tokens=2000,
        mean_completion_tokens=700,
        prompts=[
            "Draft the next step in the customer onboarding workflow.",
            "Decide which approval branch this purchase order should follow.",
            "Generate the follow-up tasks for this closed deal.",
        ],
        # Workflow decisions are always about a new PO/customer → always unique
        # prompts on an expensive model: pure traffic-router bait, no cache hits.
        live_templates=[
            "A purchase order for ${amount} from the {team} team is awaiting approval. "
            "Company policy: under $500 auto-approve, $500-$5000 needs a manager, above "
            "$5000 needs a director. State which approval branch applies and why, in two sentences.",
            "A new customer just signed up for the {plan} plan. List the three onboarding "
            "tasks the success team should do in week one, one line each.",
        ],
        live_weight=0.20,
    ),
]
_PROFILE_BY_ID = {p.agent_id: p for p in AGENT_PROFILES}

DUPLICATE_RATE = 0.15  # ~15% of requests reuse a recent prompt (cache-hit bait).

# Fill-in values for live_templates. Combinations are numerous enough that a
# rendered prompt is effectively always new → guaranteed cache miss → real call.
_TEMPLATE_VALUES: dict[str, list[str]] = {
    "plan": ["Starter", "Pro", "Business", "Enterprise"],
    "issue": [
        "exports keep timing out on large files",
        "the mobile app logs them out every few minutes",
        "invoices are not arriving by email",
        "the dashboard shows stale numbers since yesterday",
        "API keys stopped working after rotation",
    ],
    "topic": ["search relevance", "billing migration", "mobile redesign", "data retention"],
    "blocker": [
        "a pending security review",
        "missing load-test results",
        "an unresolved dependency upgrade",
        "unclear ownership of the rollout",
    ],
    "team": ["platform", "growth", "infra", "payments", "data"],
    "next_step": [
        "ship the feature flag to 10% of users",
        "finish the runbook draft",
        "schedule the load test",
        "close out the open audit items",
    ],
    "day": ["Monday", "Wednesday", "Friday"],
    "review": [
        "Setup took five minutes and the dashboards are gorgeous. Support answered in an hour.",
        "The app crashes every time I open settings. Two weeks and no fix.",
        "Does what it says. Nothing special, but no complaints either.",
        "Pricing doubled with one email of notice. The product is fine; the trust is gone.",
        "Honestly the best tool we adopted this year — the alerts alone paid for it.",
    ],
    "number": ["7", "12", "23", "48"],
    "amount": ["180", "1200", "4800", "9500", "25000"],
}


def _render_template(template: str) -> str:
    values = {key: random.choice(options) for key, options in _TEMPLATE_VALUES.items()}
    return template.format(**values)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()


def _sample_tokens(mean: int) -> int:
    """Realistic per-request token count: gaussian around the mean, clamped."""
    value = int(random.gauss(mean, mean * 0.25))
    return max(int(mean * 0.3), value)


class WorkloadSimulator:
    """Generates synthetic-but-realistic AI traffic and crisis scenarios."""

    def __init__(self) -> None:
        self._recent_prompts: dict[str, list[str]] = {p.agent_id: [] for p in AGENT_PROFILES}
        self._task: asyncio.Task | None = None
        self._rate_per_min: int = 0
        self.total_generated: int = 0
        self.last_scenario: str | None = None
        # LIVE lane: real /api/requests/ingest traffic interleaved with synthetic.
        self._live_enabled = get_env("SIMULATOR_LIVE_INGEST", "true").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self._live_interval = max(
            5, get_int_env("SIMULATOR_LIVE_INGEST_INTERVAL_SECONDS", 12)
        )
        self._last_live_at: float = 0.0
        self.live_requests: int = 0
        self.live_cache_hits: int = 0
        self.live_blocked: int = 0
        self.live_errors: int = 0

    # ── Core generation ───────────────────────────────────────────────

    async def generate_request(
        self,
        agent_id: str | None = None,
        *,
        force_duplicate: bool = False,
        model_override: str | None = None,
        token_multiplier: float = 1.0,
    ) -> dict:
        """Create one realistic AIRequest, persist it, bump counters, emit event."""
        profile = (
            _PROFILE_BY_ID.get(agent_id) if agent_id else random.choice(AGENT_PROFILES)
        )
        if profile is None:  # unknown agent_id → synthesize a generic profile
            profile = AgentProfile(
                agent_id=agent_id or "agent",
                priority="medium",
                category="general",
                model="gemini-2.5-flash",
                mean_prompt_tokens=1000,
                mean_completion_tokens=400,
                prompts=["Generic request."],
            )

        is_duplicate = force_duplicate or (
            bool(self._recent_prompts.get(profile.agent_id))
            and random.random() < DUPLICATE_RATE
        )
        if is_duplicate and self._recent_prompts.get(profile.agent_id):
            prompt = random.choice(self._recent_prompts[profile.agent_id])
        else:
            prompt = random.choice(profile.prompts)
            self._remember_prompt(profile.agent_id, prompt)

        model = model_override or profile.model
        prompt_hash = _prompt_hash(prompt)

        if is_duplicate:
            # Cache hit: served from store, no model spend (mirrors real pipeline).
            request = AIRequest(
                agent_id=profile.agent_id, prompt=prompt, priority=profile.priority,
                model=model, prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                prompt_hash=prompt_hash, cache_hit=True, category=profile.category,
            )
            await self._persist(request)
            await self._set_agent_route(profile.agent_id, model)
            await self._publish_request_event("request_cache_hit", request, tokens=0)
            self.total_generated += 1
            return request.model_dump(mode="json")

        prompt_tokens = int(_sample_tokens(profile.mean_prompt_tokens) * token_multiplier)
        completion_tokens = int(_sample_tokens(profile.mean_completion_tokens) * token_multiplier)
        cost = compute_cost(model, prompt_tokens, completion_tokens)

        request = AIRequest(
            agent_id=profile.agent_id, prompt=prompt, priority=profile.priority,
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost, prompt_hash=prompt_hash, cache_hit=False, category=profile.category,
        )
        await self._persist(request)
        await self._set_agent_route(profile.agent_id, model)
        await self._increment_counters(prompt_tokens + completion_tokens, cost)
        await self._publish_request_event(
            "request_ingested", request, tokens=prompt_tokens + completion_tokens
        )
        self.total_generated += 1
        return request.model_dump(mode="json")

    async def generate_batch(self, count: int, **kwargs) -> int:
        for _ in range(max(count, 0)):
            await self.generate_request(**kwargs)
        return count

    # ── LIVE lane: real requests through /api/requests/ingest ─────────

    async def send_live_request(self) -> dict | None:
        """Send one request through the real ingest pipeline (all engines run).

        65% of the time an agent re-asks one of its fixed prompts — after the
        first occurrence those are genuine duplicate-cache hits ($0, no Gemini
        call). The rest render a unique prompt from a template → a real Gemini
        Flash call. Net quota use stays around two real calls per minute.
        """
        # Imported here, not at module top: api.router_requests imports the
        # engine stack, which must not load as a side effect of the simulator.
        from fastapi import HTTPException

        from api.router_requests import IngestRequest, ingest_request

        live_profiles = [
            p for p in AGENT_PROFILES if p.live_fixed_prompts or p.live_templates
        ]
        if not live_profiles:
            return None
        profile = random.choices(
            live_profiles, weights=[p.live_weight for p in live_profiles], k=1
        )[0]

        use_fixed = profile.live_fixed_prompts and (
            not profile.live_templates or random.random() < 0.65
        )
        if use_fixed:
            prompt = random.choice(profile.live_fixed_prompts)
        else:
            prompt = _render_template(random.choice(profile.live_templates))

        payload = IngestRequest(
            agent_id=profile.agent_id,
            prompt=prompt,
            priority=profile.priority,
            model=profile.model,
            category=profile.category,
            force_compress=profile.force_compress,
        )
        try:
            response = await ingest_request(payload)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            if exc.status_code == 429:
                self.live_blocked += 1
                await self._publish_blocked_event(profile.agent_id, detail)
                logger.info(
                    "simulator.live blocked agent=%s reason=%s",
                    profile.agent_id, detail.get("message"),
                )
            else:
                self.live_errors += 1
                logger.warning(
                    "simulator.live ingest failed agent=%s status=%s detail=%s",
                    profile.agent_id, exc.status_code, detail.get("message"),
                )
            return None
        except Exception:
            self.live_errors += 1
            logger.warning("simulator.live ingest crashed", exc_info=True)
            return None

        self.live_requests += 1
        if response.cache_hit:
            self.live_cache_hits += 1
        logger.info(
            "simulator.live sent agent=%s cache_hit=%s model=%s cost=%.6f",
            profile.agent_id, response.cache_hit, response.model, response.cost_usd,
        )
        return response.model_dump(mode="json")

    async def _send_live_request_safe(self) -> None:
        try:
            await self.send_live_request()
        except Exception:
            self.live_errors += 1
            logger.warning("simulator.live lane error", exc_info=True)

    async def _publish_blocked_event(self, agent_id: str, detail: dict) -> None:
        try:
            from integrations import cache

            await cache.publish_event(
                {
                    "type": "request_blocked",
                    "agent_id": agent_id,
                    "reason": detail.get("message") or "Budget limit reached",
                    "action": detail.get("action"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:
            pass

    # ── Crisis scenarios (wired to the 5 simulator endpoints) ─────────

    async def trigger_spike(self, count: int = 40) -> dict:
        """A sudden burst of traffic across all agents — token burn spike."""
        self.last_scenario = "spike"
        await self.generate_batch(count)
        logger.info("simulator.spike generated=%s", count)
        return {"scenario": "spike", "requests_generated": count}

    async def trigger_runaway(self, agent_id: str = "summarizer-1", count: int = 35) -> dict:
        """One agent loops on near-duplicate prompts — trips the runaway detector."""
        self.last_scenario = "runaway"
        # Seed one prompt then hammer duplicates of it at high token size.
        await self.generate_request(agent_id=agent_id)
        for _ in range(max(count - 1, 0)):
            await self.generate_request(
                agent_id=agent_id, force_duplicate=True, token_multiplier=1.4
            )
        logger.info("simulator.runaway agent=%s generated=%s", agent_id, count)
        return {"scenario": "runaway", "agent_id": agent_id, "requests_generated": count}

    async def trigger_budget_overflow(self, target_fraction: float = 0.95) -> dict:
        """Push spend toward the daily budget so forecaster/budget engines fire.

        Pushes the cost counter directly to ~95% of budget (a handful of requests
        could never approach a $500 ceiling), then adds a little visible pro-model
        traffic so the dashboard shows real activity behind the crisis.
        """
        self.last_scenario = "budget-overflow"
        from integrations import cache

        try:
            client = cache.get_client()
            daily_budget = float(await client.get("quota:daily_budget_usd") or 0) or 500.0
            spent = float(await client.get("quota:cost_usd_today") or 0.0)
        except Exception:
            client, daily_budget, spent = None, 500.0, 0.0

        target = daily_budget * target_fraction
        bumped = 0.0
        if client is not None and spent < target:
            bumped = round(target - spent, 6)
            try:
                await client.incrbyfloat("quota:cost_usd_today", bumped)
                # Keep tokens roughly consistent with the spend bump.
                await client.incrby("quota:tokens_used_today", int(bumped / 0.005 * 1000))
            except Exception:
                bumped = 0.0

        # A few real expensive requests for visible traffic + reroute opportunities.
        for _ in range(8):
            await self.generate_request(
                agent_id="workflow-agent-1",
                model_override="gemini-2.5-pro",
                token_multiplier=1.6,
            )

        logger.info(
            "simulator.budget_overflow bumped=%.2f target=%.2f budget=%.2f",
            bumped, target, daily_budget,
        )
        return {
            "scenario": "budget-overflow",
            "spend_bumped_usd": round(bumped, 2),
            "approx_spend_usd": round(spent + bumped, 2),
            "daily_budget_usd": round(daily_budget, 2),
            "requests_generated": 8,
        }

    async def trigger_dynatrace_anomaly(self) -> dict:
        """Surface a Dynatrace-style anomaly: push a real custom event + emit signal."""
        self.last_scenario = "dynatrace-anomaly"
        from integrations import cache
        from integrations.dynatrace_mcp import dynatrace_mcp

        title = "Response time degradation on support-service"
        push = await dynatrace_mcp.push_custom_event(
            title,
            "Simulated latency spike: p95 2340ms on support-service.",
            event_type="CUSTOM_ALERT",
            properties={"source": "quotamind-simulator", "p95_ms": "2340"},
        )
        event = {
            "type": "dynatrace_anomaly",
            "title": title,
            "severity": "PERFORMANCE",
            "latency_p95_ms": 2340,
            "affected_entity": "support-service",
            "pushed_to_dynatrace": bool(push.get("ok")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await cache.publish_event(event)
        except Exception:
            pass
        logger.info("simulator.dynatrace_anomaly pushed_ok=%s", push.get("ok"))
        return {"scenario": "dynatrace-anomaly", **event}

    async def reset(self) -> dict:
        """Clear counters, agent state, and simulated data — back to a clean slate."""
        self.last_scenario = "reset"
        await self.stop()
        self._recent_prompts = {p.agent_id: [] for p in AGENT_PROFILES}
        self.total_generated = 0

        self.live_requests = 0
        self.live_cache_hits = 0
        self.live_blocked = 0
        self.live_errors = 0
        self._last_live_at = 0.0

        deleted_requests = 0
        deleted_events = 0
        try:
            from integrations import db

            r1 = await db.ai_requests().delete_many({"simulated": True})
            r2 = await db.optimization_events().delete_many({"triggered_by": "simulator"})
            # Live-lane and Try-it requests went through the real pipeline, so
            # they carry no `simulated` flag — match them by agent instead.
            demo_agents = [p.agent_id for p in AGENT_PROFILES] + ["you"]
            r3 = await db.ai_requests().delete_many({"agent_id": {"$in": demo_agents}})
            await db.cached_outputs().delete_many({})
            deleted_requests = r1.deleted_count + r3.deleted_count
            deleted_events = r2.deleted_count
        except Exception:
            logger.exception("simulator.reset db cleanup failed")

        try:
            from integrations import cache

            client = cache.get_client()
            await client.delete("quota:tokens_used_today", "quota:cost_usd_today")
            for pattern in (
                "agent_route:*", "agent_health:*", "agent_paused:*",
                "agent_throttle:*", "agent_compress:*", "cache:*",
            ):
                async for key in client.scan_iter(match=pattern):
                    await client.delete(key)
            await cache.publish_event(
                {"type": "simulator_reset", "created_at": datetime.now(timezone.utc).isoformat()}
            )
        except Exception:
            logger.exception("simulator.reset cache cleanup failed")

        logger.info(
            "simulator.reset deleted_requests=%s deleted_events=%s",
            deleted_requests, deleted_events,
        )
        return {
            "scenario": "reset",
            "deleted_requests": deleted_requests,
            "deleted_events": deleted_events,
        }

    # ── Continuous background traffic (the panel's requests/min) ──────

    async def start(self, rate_per_min: int = 60) -> dict:
        """Start generating steady background traffic at the given rate."""
        await self.stop()
        self._rate_per_min = max(1, min(rate_per_min, 600))
        self._task = asyncio.create_task(self._run_loop())
        logger.info("simulator.start rate_per_min=%s", self._rate_per_min)
        return self.status()

    async def stop(self) -> dict:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._rate_per_min = 0
        return self.status()

    async def _run_loop(self) -> None:
        try:
            while True:
                interval = 60.0 / max(self._rate_per_min, 1)
                await self.generate_request()
                if (
                    self._live_enabled
                    and time.monotonic() - self._last_live_at >= self._live_interval
                ):
                    self._last_live_at = time.monotonic()
                    asyncio.create_task(self._send_live_request_safe())
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("simulator.loop crashed")

    def status(self) -> dict:
        return {
            "running": self._task is not None and not self._task.done(),
            "rate_per_min": self._rate_per_min,
            "total_generated": self.total_generated,
            "last_scenario": self.last_scenario,
            "agents": [p.agent_id for p in AGENT_PROFILES],
            "live_ingest_enabled": self._live_enabled,
            "live_interval_seconds": self._live_interval,
            "live_requests": self.live_requests,
            "live_cache_hits": self.live_cache_hits,
            "live_blocked": self.live_blocked,
            "live_errors": self.live_errors,
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _remember_prompt(self, agent_id: str, prompt: str) -> None:
        pool = self._recent_prompts.setdefault(agent_id, [])
        pool.append(prompt)
        if len(pool) > 10:
            pool.pop(0)

    async def _persist(self, request: AIRequest) -> None:
        doc = request.model_dump(mode="python")
        doc["simulated"] = True
        try:
            from integrations import db

            await db.ai_requests().insert_one(doc)
        except Exception:
            logger.exception("simulator.persist failed agent=%s", request.agent_id)

    async def _increment_counters(self, tokens: int, cost_usd: float) -> None:
        try:
            from integrations import cache

            client = cache.get_client()
            await client.incrby("quota:tokens_used_today", tokens)
            await client.incrbyfloat("quota:cost_usd_today", cost_usd)
        except Exception:
            pass

    async def _set_agent_route(self, agent_id: str, model: str) -> None:
        try:
            from integrations import cache

            await cache.set_cached(f"agent_route:{agent_id}", {"model": model})
        except Exception:
            pass

    async def _publish_request_event(self, event_type: str, request: AIRequest, tokens: int) -> None:
        event = {
            "type": event_type,
            "request_id": request.id,
            "agent_id": request.agent_id,
            "model": request.model,
            "priority": request.priority,
            "category": request.category,
            "cost_usd": request.cost_usd,
            "total_tokens": tokens,
            "cache_hit": request.cache_hit,
            "prompt_hash": request.prompt_hash,
            "simulated": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            from integrations import cache

            await cache.publish_event(event)
        except Exception:
            pass


# Module-level singleton (matches engine/integration conventions).
workload_simulator = WorkloadSimulator()
