"""Budget allocation and enforcement engine.

Tracks daily spend overall and by request category, then recommends or records
enforcement actions when usage crosses configured budget thresholds.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.env import get_float_env
from models.optimization_event import OptimizationEvent, OptimizationType

logger = logging.getLogger("quotamind.budget")

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - lets local logic run before deps are installed.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator

BudgetStatus = Literal["healthy", "watch", "constrained", "exhausted"]
BudgetAction = Literal["allow", "compress", "reroute", "throttle", "pause"]

DEFAULT_CATEGORY_LIMITS: dict[str, float] = {
    "support": 0.30,
    "summarization": 0.20,
    "analytics": 0.25,
    "workflow": 0.15,
    "general": 0.10,
}


class CategoryBudget(BaseModel):
    category: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    percent_used: float
    status: BudgetStatus


class BudgetStatusResult(BaseModel):
    daily_limit_usd: float
    spent_usd: float
    remaining_usd: float
    percent_used: float
    status: BudgetStatus
    projected_end_of_day_usd: float
    categories: list[CategoryBudget]
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BudgetEnforcementResult(BaseModel):
    allowed: bool
    action: BudgetAction
    status: BudgetStatus
    reason: str
    agent_id: str | None = None
    category: str = "general"
    priority: str = "medium"
    current_spend_usd: float
    projected_spend_usd: float
    daily_limit_usd: float
    percent_used: float
    category_percent_used: float | None = None


class BudgetAllocatorInput(BaseModel):
    agent_id: str | None = Field(default=None, description="Agent being evaluated.")
    category: str = Field(default="general", description="Budget category for the request.")
    priority: str = Field(default="medium", description="Request priority.")
    projected_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional cost that would be incurred if allowed.",
    )


class BudgetAllocator:
    """Compute budget status and enforce spend guardrails."""

    def __init__(
        self,
        category_limits: dict[str, float] | None = None,
        warn_threshold: float = 0.70,
        constrain_threshold: float = 0.90,
    ) -> None:
        self.category_limits = category_limits or DEFAULT_CATEGORY_LIMITS
        self.warn_threshold = warn_threshold
        self.constrain_threshold = constrain_threshold

    async def check_budget_status(
        self,
        category_limits: dict[str, float] | None = None,
    ) -> BudgetStatusResult:
        """Return overall and per-category daily budget health."""
        daily_limit = await self._daily_limit_usd()
        category_spend = await self._category_spend_today()
        spent = await self._total_spend_today(category_spend)
        limits = self._normalized_category_limits(category_limits)

        categories = [
            self._category_result(name, daily_limit * share, category_spend.get(name, 0.0))
            for name, share in limits.items()
        ]

        for name, amount in category_spend.items():
            if name not in limits:
                categories.append(self._category_result(name, 0.0, amount))

        percent_used = self._ratio(spent, daily_limit)
        logger.info(
            "budget.status spent_usd=%.6f daily_limit_usd=%.4f percent_used=%.6f categories=%s",
            spent,
            daily_limit,
            percent_used,
            {item.category: item.spent_usd for item in categories},
        )
        return BudgetStatusResult(
            daily_limit_usd=round(daily_limit, 4),
            spent_usd=round(spent, 6),
            remaining_usd=round(max(daily_limit - spent, 0.0), 6),
            percent_used=round(percent_used, 6),
            status=self._status_for(percent_used),
            projected_end_of_day_usd=round(self._project_end_of_day(spent), 6),
            categories=sorted(categories, key=lambda item: item.percent_used, reverse=True),
        )

    async def enforce_budget_limits(
        self,
        agent_id: str | None = None,
        category: str = "general",
        priority: str = "medium",
        projected_cost_usd: float = 0.0,
    ) -> BudgetEnforcementResult:
        """Decide whether a request should be allowed, optimized, throttled, or paused."""
        normalized_category = (category or "general").strip().lower()
        normalized_priority = (priority or "medium").strip().lower()
        status = await self.check_budget_status()

        projected_spend = status.spent_usd + max(projected_cost_usd, 0.0)
        projected_pct = self._ratio(projected_spend, status.daily_limit_usd)
        category_status = self._find_category(status, normalized_category)
        category_pct = category_status.percent_used if category_status else None

        action, reason = self._choose_action(
            normalized_priority,
            projected_pct,
            category_pct,
            projected_cost_usd,
        )

        result = BudgetEnforcementResult(
            allowed=action not in {"throttle", "pause"},
            action=action,
            status=self._status_for(projected_pct),
            reason=reason,
            agent_id=agent_id,
            category=normalized_category,
            priority=normalized_priority,
            current_spend_usd=status.spent_usd,
            projected_spend_usd=round(projected_spend, 6),
            daily_limit_usd=status.daily_limit_usd,
            percent_used=round(projected_pct, 6),
            category_percent_used=category_pct,
        )
        logger.info(
            "budget.enforcement agent_id=%s category=%s priority=%s projected_cost_usd=%.6f action=%s allowed=%s",
            agent_id,
            normalized_category,
            normalized_priority,
            projected_cost_usd,
            result.action,
            result.allowed,
        )

        if action != "allow":
            await self._log_enforcement(result)

        return result

    async def _daily_limit_usd(self) -> float:
        try:
            from integrations import cache

            raw = await cache.get_client().get("quota:daily_budget_usd")
            if raw is None:
                raw = await cache.get_client().get("quota:daily_limit")
            if raw is not None:
                return max(float(raw), 0.01)
        except Exception:
            logger.exception("budget.daily_limit_read_failed")
        return max(get_float_env("DAILY_BUDGET_USD", 500.0), 0.01)

    async def _total_spend_today(self, category_spend: dict[str, float]) -> float:
        try:
            from integrations import cache

            raw = await cache.get_client().get("quota:cost_usd_today")
            if raw is not None:
                return max(float(raw), 0.0)
        except Exception:
            logger.exception("budget.redis_spend_read_failed")
        return max(sum(category_spend.values()), 0.0)

    async def _category_spend_today(self) -> dict[str, float]:
        start = datetime.combine(datetime.now(timezone.utc).date(), time.min, timezone.utc)
        try:
            from integrations import db

            cursor = db.ai_requests().aggregate(
                [
                    {
                        "$match": {
                            "$or": [
                                {"created_at": {"$gte": start}},
                                {"created_at": {"$gte": start.isoformat()}},
                            ]
                        }
                    },
                    {
                        "$group": {
                            "_id": {"$ifNull": ["$category", "general"]},
                            "spent_usd": {"$sum": "$cost_usd"},
                        }
                    },
                ]
            )
            return {
                str(row["_id"]).lower(): round(float(row.get("spent_usd", 0.0)), 6)
                async for row in cursor
            }
        except Exception:
            logger.exception("budget.category_spend_read_failed")
            return {}

    def _normalized_category_limits(self, limits: dict[str, float] | None) -> dict[str, float]:
        raw = limits or self.category_limits
        cleaned = {
            (name or "general").strip().lower(): max(float(share), 0.0)
            for name, share in raw.items()
        }
        total = sum(cleaned.values())
        if total <= 0:
            return {"general": 1.0}
        return {name: share / total for name, share in cleaned.items()}

    def _category_result(self, category: str, limit: float, spent: float) -> CategoryBudget:
        percent = self._ratio(spent, limit) if limit else 1.0 if spent > 0 else 0.0
        return CategoryBudget(
            category=category,
            limit_usd=round(limit, 6),
            spent_usd=round(spent, 6),
            remaining_usd=round(max(limit - spent, 0.0), 6),
            percent_used=round(percent, 6),
            status=self._status_for(percent),
        )

    def _choose_action(
        self,
        priority: str,
        projected_pct: float,
        category_pct: float | None,
        projected_cost_usd: float,
    ) -> tuple[BudgetAction, str]:
        pressure = max(projected_pct, category_pct or 0.0)

        if pressure >= 1.0:
            if priority == "critical":
                return "reroute", "Budget is exhausted; keep critical traffic on the cheapest safe route."
            return "pause", "Budget is exhausted; pause non-critical spend."

        if pressure >= self.constrain_threshold:
            if priority in {"low", "medium"}:
                return "throttle", "Budget is constrained; throttle lower-priority traffic."
            return "compress", "Budget is constrained; compress prompts before inference."

        if pressure >= self.warn_threshold and projected_cost_usd > 0:
            return "compress", "Budget usage is elevated; compress to slow burn rate."

        return "allow", "Budget is within configured limits."

    async def _log_enforcement(self, result: BudgetEnforcementResult) -> None:
        event = OptimizationEvent(
            type=OptimizationType.BUDGET_ENFORCE,
            agent_id=result.agent_id,
            description=result.reason,
            before={
                "spent_usd": result.current_spend_usd,
                "projected_spend_usd": result.projected_spend_usd,
                "daily_limit_usd": result.daily_limit_usd,
                "category": result.category,
                "priority": result.priority,
            },
            after={"action": result.action, "allowed": result.allowed},
            triggered_by="system",
        ).model_dump(mode="json")

        try:
            from integrations import db

            await db.optimization_events().insert_one(event)
        except Exception:
            logger.exception(
                "budget.enforcement_persist_failed agent_id=%s action=%s",
                result.agent_id,
                result.action,
            )

        try:
            from integrations import cache

            await cache.publish_event(event)
        except Exception:
            logger.exception(
                "budget.enforcement_event_publish_failed agent_id=%s action=%s",
                result.agent_id,
                result.action,
            )

    @staticmethod
    def _find_category(status: BudgetStatusResult, category: str) -> CategoryBudget | None:
        return next((item for item in status.categories if item.category == category), None)

    @staticmethod
    def _ratio(value: float, limit: float) -> float:
        if limit <= 0:
            return 1.0 if value > 0 else 0.0
        return max(value / limit, 0.0)

    def _status_for(self, ratio: float) -> BudgetStatus:
        if ratio >= 1.0:
            return "exhausted"
        if ratio >= self.constrain_threshold:
            return "constrained"
        if ratio >= self.warn_threshold:
            return "watch"
        return "healthy"

    @staticmethod
    def _project_end_of_day(spent: float) -> float:
        now = datetime.now(timezone.utc)
        elapsed_seconds = (
            now - datetime.combine(now.date(), time.min, timezone.utc)
        ).total_seconds()
        if elapsed_seconds <= 0:
            return spent
        day_seconds = 24 * 60 * 60
        return max(spent * (day_seconds / elapsed_seconds), spent)


budget_allocator = BudgetAllocator()


@tool("budget_allocator", args_schema=BudgetAllocatorInput)
async def budget_allocator_tool(**kwargs: Any) -> dict[str, Any]:
    """Evaluate request spend against daily and category budgets."""
    result = await budget_allocator.enforce_budget_limits(**kwargs)
    return result.model_dump()
