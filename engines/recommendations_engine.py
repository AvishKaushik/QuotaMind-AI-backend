"""Daily optimization recommendations engine."""
from __future__ import annotations

import json
from datetime import datetime, time, timezone
from typing import Any

from google import genai
from pydantic import BaseModel, Field

from config.env import get_env, get_float_env, get_int_env

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - keeps local imports usable before deps install.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator


class Recommendation(BaseModel):
    title: str
    impact: str
    rationale: str
    action: str
    estimated_savings_usd: float = 0.0


class DailyRecommendationReport(BaseModel):
    date: str
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[Recommendation] = Field(default_factory=list)
    generated_by: str = "local"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecommendationsInput(BaseModel):
    max_recommendations: int = Field(default=5, ge=1, le=10)


class RecommendationsEngine:
    """Create daily optimization insights from request, cache, and event metrics."""

    async def generate_daily_report(
        self,
        max_recommendations: int = 5,
    ) -> DailyRecommendationReport:
        """Return a daily optimization report for the dashboard."""
        metrics = await self._collect_daily_metrics()
        gemini_report = await self._generate_with_gemini(metrics, max_recommendations)
        if gemini_report is not None:
            return gemini_report

        recommendations = self._fallback_recommendations(metrics, max_recommendations)
        summary = self._fallback_summary(metrics, recommendations)
        return DailyRecommendationReport(
            date=datetime.now(timezone.utc).date().isoformat(),
            summary=summary,
            metrics=metrics,
            recommendations=recommendations,
            generated_by="local",
        )

    async def _collect_daily_metrics(self) -> dict[str, Any]:
        start = datetime.combine(datetime.now(timezone.utc).date(), time.min, timezone.utc)
        request_metrics = await self._request_metrics(start)
        event_metrics = await self._event_metrics(start)
        cache_metrics = await self._cache_metrics()

        daily_budget = get_float_env("DAILY_BUDGET_USD", 500.0)
        daily_quota = get_int_env("DAILY_QUOTA_TOKEN_LIMIT", 10_000_000)
        total_tokens = request_metrics["prompt_tokens"] + request_metrics["completion_tokens"]

        return {
            **request_metrics,
            **event_metrics,
            **cache_metrics,
            "daily_budget_usd": daily_budget,
            "daily_quota_token_limit": daily_quota,
            "budget_percent_used": self._ratio(request_metrics["cost_usd"], daily_budget),
            "quota_percent_used": self._ratio(total_tokens, daily_quota),
        }

    async def _request_metrics(self, start: datetime) -> dict[str, Any]:
        empty = {
            "request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "models": {},
            "priorities": {},
            "categories": {},
        }
        try:
            from integrations import db

            cursor = db.ai_requests().aggregate(
                [
                    {"$match": {"created_at": {"$gte": start}}},
                    {
                        "$facet": {
                            "totals": [
                                {
                                    "$group": {
                                        "_id": None,
                                        "request_count": {"$sum": 1},
                                        "prompt_tokens": {"$sum": "$prompt_tokens"},
                                        "completion_tokens": {"$sum": "$completion_tokens"},
                                        "cost_usd": {"$sum": "$cost_usd"},
                                    }
                                }
                            ],
                            "models": [{"$group": {"_id": "$model", "count": {"$sum": 1}}}],
                            "priorities": [{"$group": {"_id": "$priority", "count": {"$sum": 1}}}],
                            "categories": [{"$group": {"_id": "$category", "cost_usd": {"$sum": "$cost_usd"}}}],
                        }
                    },
                ]
            )
            rows = await cursor.to_list(length=1)
            if not rows:
                return empty
            row = rows[0]
            totals = row.get("totals") or [{}]
            total = totals[0] if totals else {}
            return {
                "request_count": int(total.get("request_count", 0) or 0),
                "prompt_tokens": int(total.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(total.get("completion_tokens", 0) or 0),
                "cost_usd": round(float(total.get("cost_usd", 0.0) or 0.0), 6),
                "models": {str(item.get("_id") or "unknown"): item.get("count", 0) for item in row.get("models", [])},
                "priorities": {
                    str(item.get("_id") or "unknown"): item.get("count", 0)
                    for item in row.get("priorities", [])
                },
                "categories": {
                    str(item.get("_id") or "general"): round(float(item.get("cost_usd", 0.0)), 6)
                    for item in row.get("categories", [])
                },
            }
        except Exception:
            return empty

    async def _event_metrics(self, start: datetime) -> dict[str, Any]:
        try:
            from integrations import db

            cursor = db.optimization_events().aggregate(
                [
                    {"$match": {"created_at": {"$gte": start}}},
                    {
                        "$group": {
                            "_id": "$type",
                            "count": {"$sum": 1},
                            "tokens_saved": {"$sum": "$tokens_saved"},
                            "cost_saved_usd": {"$sum": "$cost_saved_usd"},
                        }
                    },
                ]
            )
            by_type: dict[str, Any] = {}
            tokens_saved = 0
            cost_saved = 0.0
            async for row in cursor:
                event_type = str(row.get("_id") or "unknown")
                by_type[event_type] = {
                    "count": int(row.get("count", 0) or 0),
                    "tokens_saved": int(row.get("tokens_saved", 0) or 0),
                    "cost_saved_usd": round(float(row.get("cost_saved_usd", 0.0) or 0.0), 6),
                }
                tokens_saved += by_type[event_type]["tokens_saved"]
                cost_saved += by_type[event_type]["cost_saved_usd"]
            return {
                "optimization_events": by_type,
                "tokens_saved": tokens_saved,
                "cost_saved_usd": round(cost_saved, 6),
            }
        except Exception:
            return {"optimization_events": {}, "tokens_saved": 0, "cost_saved_usd": 0.0}

    async def _cache_metrics(self) -> dict[str, Any]:
        try:
            from integrations import cache

            hits = int(await cache.get_client().get("cache:hits") or 0)
            misses = int(await cache.get_client().get("cache:misses") or 0)
            total = hits + misses
            return {
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_rate": round(hits / total, 4) if total else 0.0,
            }
        except Exception:
            return {"cache_hits": 0, "cache_misses": 0, "cache_hit_rate": 0.0}

    async def _generate_with_gemini(
        self,
        metrics: dict[str, Any],
        max_recommendations: int,
    ) -> DailyRecommendationReport | None:
        api_key = get_env("GEMINI_API_KEY")
        if not api_key:
            return None

        try:
            client = genai.Client(api_key=api_key)
            model_name = get_env("GEMINI_REASONING_MODEL", "gemini-2.5-pro")
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=(
                    "You are QuotaMind AI. Generate concise daily optimization recommendations.\n"
                    "Return strict JSON with keys: summary, recommendations.\n"
                    "Each recommendation must include title, impact, rationale, action, "
                    "estimated_savings_usd.\n"
                    f"Limit recommendations to {max_recommendations}.\n"
                    f"Metrics JSON: {json.dumps(metrics, sort_keys=True)}"
                ),
            )
            payload = self._extract_json(response.text or "")
            if not payload:
                return None
            return DailyRecommendationReport(
                date=datetime.now(timezone.utc).date().isoformat(),
                summary=str(payload.get("summary") or "Daily optimization report generated."),
                metrics=metrics,
                recommendations=[
                    Recommendation(**item)
                    for item in payload.get("recommendations", [])[:max_recommendations]
                ],
                generated_by=model_name,
            )
        except Exception:
            return None

    def _fallback_recommendations(
        self,
        metrics: dict[str, Any],
        max_recommendations: int,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []

        if metrics["cache_hit_rate"] < 0.20 and metrics["request_count"] >= 20:
            recommendations.append(
                Recommendation(
                    title="Increase duplicate-response reuse",
                    impact="medium",
                    rationale="Cache hit rate is low relative to request volume.",
                    action="Normalize prompts before hashing and cache deterministic agent responses.",
                )
            )

        pro_count = metrics["models"].get("gemini-2.5-pro", 0)
        if pro_count > 0:
            recommendations.append(
                Recommendation(
                    title="Route routine traffic away from Pro",
                    impact="high",
                    rationale=f"{pro_count} requests used gemini-2.5-pro today.",
                    action="Use TrafficRouter priority floors so low and medium traffic runs on Flash tiers.",
                )
            )

        if metrics["budget_percent_used"] >= 0.70:
            recommendations.append(
                Recommendation(
                    title="Enable budget pressure controls",
                    impact="high",
                    rationale="Daily budget usage is elevated.",
                    action="Turn on compression and throttling for low-priority categories.",
                )
            )

        if metrics["quota_percent_used"] >= 0.70:
            recommendations.append(
                Recommendation(
                    title="Reduce token burn rate",
                    impact="high",
                    rationale="Daily token quota usage is elevated.",
                    action="Compress long prompts and cap runaway agents until burn rate normalizes.",
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    title="Maintain current optimization posture",
                    impact="low",
                    rationale="Budget, quota, and cache metrics are within expected ranges.",
                    action="Continue monitoring burn rate and cache hit rate throughout the day.",
                )
            )

        return recommendations[:max_recommendations]

    @staticmethod
    def _fallback_summary(metrics: dict[str, Any], recommendations: list[Recommendation]) -> str:
        return (
            f"Processed {metrics['request_count']} requests today with "
            f"${metrics['cost_usd']:.4f} spend and {metrics['cache_hit_rate']:.0%} cache hit rate. "
            f"Generated {len(recommendations)} recommendation(s)."
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _ratio(value: float, limit: float) -> float:
        if limit <= 0:
            return 1.0 if value > 0 else 0.0
        return max(value / limit, 0.0)


recommendations_engine = RecommendationsEngine()


@tool("recommendations_engine", args_schema=RecommendationsInput)
async def recommendations_engine_tool(**kwargs: Any) -> dict[str, Any]:
    """Generate daily optimization recommendations."""
    result = await recommendations_engine.generate_daily_report(**kwargs)
    return result.model_dump()