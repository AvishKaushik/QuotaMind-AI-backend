"""Traffic routing engine for model-tier optimization.

Selects the cheapest model that still satisfies request priority, while reacting
to quota and budget pressure. Reroutes are logged as optimization events.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config.env import get_float_env, get_int_env
from config.models import (
    COST_PER_1K_TOKENS,
    DEFAULT_MODEL,
    MODEL_TIERS,
    PRIORITY_TO_MIN_TIER,
    compute_cost,
)
from models.optimization_event import OptimizationEvent, OptimizationType

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - keeps local imports usable before deps install.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator


class RoutingDecision(BaseModel):
    original_model: str
    selected_model: str
    priority: str
    min_tier: int
    original_tier: int
    selected_tier: int
    quota_percent_used: float
    budget_percent_used: float
    estimated_cost_usd: float
    estimated_savings_usd: float
    rerouted: bool
    reason: str


class TrafficRouterInput(BaseModel):
    priority: str = Field(default="medium", description="Request priority.")
    requested_model: str | None = Field(default=None, description="Model requested by caller.")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    agent_id: str | None = Field(default=None)


class TrafficRouter:
    """Route requests across Gemini model tiers based on priority and pressure."""

    def __init__(
        self,
        quota_warn_threshold: float = 0.70,
        quota_constrain_threshold: float = 0.90,
    ) -> None:
        self.quota_warn_threshold = quota_warn_threshold
        self.quota_constrain_threshold = quota_constrain_threshold

    async def get_optimal_model(
        self,
        priority: str = "medium",
        requested_model: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        quota_percent_used: float | None = None,
        budget_percent_used: float | None = None,
    ) -> RoutingDecision:
        """Return the best model for a request without mutating agent state."""
        normalized_priority = (priority or "medium").strip().lower()
        original_model = requested_model if requested_model in MODEL_TIERS else DEFAULT_MODEL
        min_tier = PRIORITY_TO_MIN_TIER.get(normalized_priority, PRIORITY_TO_MIN_TIER["medium"])
        original_tier = MODEL_TIERS.get(original_model, MODEL_TIERS[DEFAULT_MODEL])

        quota_pct = (
            quota_percent_used
            if quota_percent_used is not None
            else await self._quota_percent_used()
        )
        budget_pct = (
            budget_percent_used
            if budget_percent_used is not None
            else await self._budget_percent_used()
        )
        pressure = max(quota_pct, budget_pct)

        selected_model, reason = self._select_model(
            original_model=original_model,
            original_tier=original_tier,
            min_tier=min_tier,
            priority=normalized_priority,
            pressure=pressure,
        )
        selected_tier = MODEL_TIERS[selected_model]
        original_cost = compute_cost(original_model, prompt_tokens, completion_tokens)
        selected_cost = compute_cost(selected_model, prompt_tokens, completion_tokens)

        return RoutingDecision(
            original_model=original_model,
            selected_model=selected_model,
            priority=normalized_priority,
            min_tier=min_tier,
            original_tier=original_tier,
            selected_tier=selected_tier,
            quota_percent_used=round(quota_pct, 4),
            budget_percent_used=round(budget_pct, 4),
            estimated_cost_usd=selected_cost,
            estimated_savings_usd=round(max(original_cost - selected_cost, 0.0), 6),
            rerouted=selected_model != original_model,
            reason=reason,
        )

    async def reroute_agent(
        self,
        agent_id: str,
        priority: str = "medium",
        requested_model: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> RoutingDecision:
        """Pick a model for an agent and record the reroute decision if changed."""
        decision = await self.get_optimal_model(
            priority=priority,
            requested_model=requested_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        await self._store_agent_route(agent_id, decision.selected_model)
        if decision.rerouted:
            await self._log_reroute(agent_id, decision)
        return decision

    def _select_model(
        self,
        original_model: str,
        original_tier: int,
        min_tier: int,
        priority: str,
        pressure: float,
    ) -> tuple[str, str]:
        if pressure >= 1.0 and priority not in {"critical", "high"}:
            selected = self._cheapest_model_for_tier(min_tier)
            return selected, "Quota or budget is exhausted; route non-critical traffic to minimum safe tier."

        if pressure >= self.quota_constrain_threshold:
            selected = self._cheapest_model_for_tier(min_tier)
            return selected, "Quota or budget is constrained; route to minimum tier allowed by priority."

        if pressure >= self.quota_warn_threshold:
            target_tier = max(min_tier, min(original_tier, 2))
            selected = self._cheapest_model_for_tier(target_tier)
            return selected, "Quota or budget usage is elevated; avoid premium tiers where safe."

        if original_tier < min_tier:
            selected = self._cheapest_model_for_tier(min_tier)
            return selected, "Requested model is below the priority floor; upgrade to the minimum safe tier."

        selected = self._cheapest_model_for_tier(min_tier)
        if selected != original_model:
            return selected, "Cheapest model selected while preserving the request priority floor."
        return original_model, "Requested model satisfies priority and quota constraints."

    async def _quota_percent_used(self) -> float:
        try:
            from integrations import cache

            raw_pct = await cache.get_client().get("quota:current_pct")
            if raw_pct is not None:
                return self._normalize_percent(float(raw_pct))

            used = await cache.get_client().get("quota:tokens_used_today")
            daily_limit = await cache.get_client().get("quota:daily_limit")
            token_limit = float(daily_limit or get_int_env("DAILY_QUOTA_TOKEN_LIMIT", 10_000_000))
            return self._ratio(float(used or 0), token_limit)
        except Exception:
            return 0.0

    async def _budget_percent_used(self) -> float:
        try:
            from integrations import cache

            spent = await cache.get_client().get("quota:cost_usd_today")
            daily_budget = get_float_env("DAILY_BUDGET_USD", 500.0)
            return self._ratio(float(spent or 0), daily_budget)
        except Exception:
            return 0.0

    async def _store_agent_route(self, agent_id: str, model: str) -> None:
        try:
            from integrations import cache

            await cache.set_cached(f"agent_route:{agent_id}", {"model": model})
        except Exception:
            pass

    async def _log_reroute(self, agent_id: str, decision: RoutingDecision) -> None:
        event = OptimizationEvent(
            type=OptimizationType.REROUTE,
            agent_id=agent_id,
            description=decision.reason,
            before={
                "model": decision.original_model,
                "tier": decision.original_tier,
                "estimated_cost_usd": round(
                    decision.estimated_cost_usd + decision.estimated_savings_usd,
                    6,
                ),
            },
            after={
                "model": decision.selected_model,
                "tier": decision.selected_tier,
                "estimated_cost_usd": decision.estimated_cost_usd,
            },
            cost_saved_usd=decision.estimated_savings_usd,
        ).model_dump(mode="json")

        try:
            from integrations import db

            await db.optimization_events().insert_one(event)
        except Exception:
            pass

        try:
            from integrations import cache

            await cache.publish_event(event)
        except Exception:
            pass

    @staticmethod
    def _cheapest_model_for_tier(tier: int) -> str:
        candidates = [model for model, model_tier in MODEL_TIERS.items() if model_tier == tier]
        if not candidates:
            return DEFAULT_MODEL
        return min(candidates, key=lambda model: COST_PER_1K_TOKENS[model]["input"])

    @staticmethod
    def _ratio(value: float, limit: float) -> float:
        if limit <= 0:
            return 1.0 if value > 0 else 0.0
        return max(value / limit, 0.0)

    @staticmethod
    def _normalize_percent(value: float) -> float:
        return value / 100 if value > 1 else max(value, 0.0)


traffic_router = TrafficRouter()


@tool("traffic_router", args_schema=TrafficRouterInput)
async def traffic_router_tool(**kwargs: Any) -> dict[str, Any]:
    """Choose the optimal model for a request under quota and budget pressure."""
    agent_id = kwargs.pop("agent_id", None)
    if agent_id:
        decision = await traffic_router.reroute_agent(agent_id=agent_id, **kwargs)
    else:
        decision = await traffic_router.get_optimal_model(**kwargs)
    return decision.model_dump()
