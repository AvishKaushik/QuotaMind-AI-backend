"""Quota burn-rate forecasting engine."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.env import get_int_env
from models.optimization_event import OptimizationEvent, OptimizationType

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - keeps local imports usable before deps install.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator

ForecastStatus = Literal["healthy", "watch", "at_risk", "exhausted"]


class QuotaForecastResult(BaseModel):
    daily_token_limit: int
    tokens_used_today: int
    tokens_remaining: int
    percent_used: float
    burn_rate_tokens_per_hour: float
    projected_end_of_day_tokens: int
    projected_percent_used: float
    exhaustion_eta: datetime | None = None
    hours_until_exhaustion: float | None = None
    status: ForecastStatus
    reason: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuotaForecasterInput(BaseModel):
    lookback_minutes: int = Field(default=60, ge=5, le=24 * 60)


class QuotaForecaster:
    """Forecast quota exhaustion from current counters and recent request volume."""

    def __init__(self, warn_threshold: float = 0.70, risk_threshold: float = 0.90) -> None:
        self.warn_threshold = warn_threshold
        self.risk_threshold = risk_threshold

    async def forecast_exhaustion(self, lookback_minutes: int = 60) -> QuotaForecastResult:
        """Project token usage through end of day and estimate exhaustion time."""
        daily_limit = await self._daily_token_limit()
        used_today = await self._tokens_used_today()
        recent_tokens = await self._tokens_in_window(lookback_minutes)
        burn_rate = self._burn_rate_per_hour(recent_tokens, lookback_minutes)

        now = datetime.now(timezone.utc)
        day_end = datetime.combine(now.date(), time.max, timezone.utc)
        hours_remaining_today = max((day_end - now).total_seconds() / 3600, 0.0)
        projected = int(round(used_today + burn_rate * hours_remaining_today))
        remaining = max(daily_limit - used_today, 0)

        hours_until_exhaustion = None
        exhaustion_eta = None
        if burn_rate > 0 and remaining > 0:
            hours_until_exhaustion = remaining / burn_rate
            exhaustion_eta = now + timedelta(hours=hours_until_exhaustion)
        elif remaining <= 0:
            hours_until_exhaustion = 0.0
            exhaustion_eta = now

        percent_used = self._ratio(used_today, daily_limit)
        projected_percent = self._ratio(projected, daily_limit)
        status, reason = self._status_and_reason(percent_used, projected_percent, exhaustion_eta)

        result = QuotaForecastResult(
            daily_token_limit=daily_limit,
            tokens_used_today=used_today,
            tokens_remaining=remaining,
            percent_used=round(percent_used, 4),
            burn_rate_tokens_per_hour=round(burn_rate, 2),
            projected_end_of_day_tokens=projected,
            projected_percent_used=round(projected_percent, 4),
            exhaustion_eta=exhaustion_eta,
            hours_until_exhaustion=(
                round(hours_until_exhaustion, 2) if hours_until_exhaustion is not None else None
            ),
            status=status,
            reason=reason,
        )

        if status in {"at_risk", "exhausted"}:
            await self._log_forecast_alert(result)
        return result

    async def _daily_token_limit(self) -> int:
        try:
            from integrations import cache

            raw = await cache.get_client().get("quota:daily_token_limit")
            if raw is None:
                raw = await cache.get_client().get("quota:daily_limit")
            if raw is not None:
                return max(int(float(raw)), 1)
        except Exception:
            pass
        return max(get_int_env("DAILY_QUOTA_TOKEN_LIMIT", 10_000_000), 1)

    async def _tokens_used_today(self) -> int:
        try:
            from integrations import cache

            raw = await cache.get_client().get("quota:tokens_used_today")
            if raw is not None:
                return max(int(float(raw)), 0)
        except Exception:
            pass

        start = datetime.combine(datetime.now(timezone.utc).date(), time.min, timezone.utc)
        return await self._sum_tokens_since(start)

    async def _tokens_in_window(self, lookback_minutes: int) -> int:
        since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        return await self._sum_tokens_since(since)

    async def _sum_tokens_since(self, since: datetime) -> int:
        try:
            from integrations import db

            cursor = db.ai_requests().aggregate(
                [
                    {"$match": {"created_at": {"$gte": since}}},
                    {
                        "$group": {
                            "_id": None,
                            "prompt_tokens": {"$sum": "$prompt_tokens"},
                            "completion_tokens": {"$sum": "$completion_tokens"},
                        }
                    },
                ]
            )
            row = await cursor.to_list(length=1)
            if not row:
                return 0
            doc = row[0]
            return int(doc.get("prompt_tokens", 0) + doc.get("completion_tokens", 0))
        except Exception:
            return 0

    async def _log_forecast_alert(self, result: QuotaForecastResult) -> None:
        event = OptimizationEvent(
            type=OptimizationType.ALERT,
            description=result.reason,
            before={
                "tokens_used_today": result.tokens_used_today,
                "daily_token_limit": result.daily_token_limit,
                "percent_used": result.percent_used,
            },
            after={
                "projected_end_of_day_tokens": result.projected_end_of_day_tokens,
                "projected_percent_used": result.projected_percent_used,
                "exhaustion_eta": result.exhaustion_eta.isoformat()
                if result.exhaustion_eta
                else None,
            },
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

    def _status_and_reason(
        self,
        percent_used: float,
        projected_percent: float,
        exhaustion_eta: datetime | None,
    ) -> tuple[ForecastStatus, str]:
        if percent_used >= 1.0:
            return "exhausted", "Daily token quota is already exhausted."
        if projected_percent >= 1.0:
            eta = f" around {exhaustion_eta.isoformat()}" if exhaustion_eta else ""
            return "at_risk", f"Current burn rate projects quota exhaustion{eta}."
        if percent_used >= self.risk_threshold or projected_percent >= self.risk_threshold:
            return "at_risk", "Token usage is projected to approach the daily limit."
        if percent_used >= self.warn_threshold or projected_percent >= self.warn_threshold:
            return "watch", "Token usage is elevated but still below critical thresholds."
        return "healthy", "Token usage is within expected daily quota."

    @staticmethod
    def _burn_rate_per_hour(tokens: int, lookback_minutes: int) -> float:
        hours = max(lookback_minutes / 60, 1 / 60)
        return tokens / hours

    @staticmethod
    def _ratio(value: int | float, limit: int | float) -> float:
        if limit <= 0:
            return 1.0 if value > 0 else 0.0
        return max(value / limit, 0.0)


quota_forecaster = QuotaForecaster()


@tool("quota_forecaster", args_schema=QuotaForecasterInput)
async def quota_forecaster_tool(**kwargs: Any) -> dict[str, Any]:
    """Forecast daily token quota exhaustion."""
    result = await quota_forecaster.forecast_exhaustion(**kwargs)
    return result.model_dump()
