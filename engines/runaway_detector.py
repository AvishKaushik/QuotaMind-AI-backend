"""Runaway agent detection engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from models.optimization_event import OptimizationEvent, OptimizationType

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - keeps local imports usable before deps install.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator

RunawaySeverity = Literal["normal", "watch", "high", "critical"]
RunawayAction = Literal["allow", "compress", "throttle", "pause"]


class RunawaySignal(BaseModel):
    name: str
    score: float
    detail: str


class RunawayAnalysisResult(BaseModel):
    agent_id: str
    window_minutes: int
    request_count: int
    total_tokens: int
    avg_tokens_per_request: float
    duplicate_prompt_count: int
    risk_score: float
    severity: RunawaySeverity
    recommended_action: RunawayAction
    signals: list[RunawaySignal] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunawayDetectorInput(BaseModel):
    agent_id: str = Field(..., description="Agent to analyze.")
    window_minutes: int = Field(default=5, ge=1, le=120)


class RunawayDetector:
    """Detect runaway loops, request storms, and token spikes per agent."""

    def __init__(
        self,
        request_rate_threshold: int = 30,
        token_threshold: int = 100_000,
        avg_tokens_threshold: int = 8_000,
        duplicate_threshold: int = 8,
    ) -> None:
        self.request_rate_threshold = request_rate_threshold
        self.token_threshold = token_threshold
        self.avg_tokens_threshold = avg_tokens_threshold
        self.duplicate_threshold = duplicate_threshold

    async def analyze_agent(
        self,
        agent_id: str,
        window_minutes: int = 5,
    ) -> RunawayAnalysisResult:
        """Analyze an agent's recent traffic and return a risk score/action."""
        docs = await self._recent_requests(agent_id, window_minutes)
        request_count = len(docs)
        total_tokens = sum(self._tokens(doc) for doc in docs)
        avg_tokens = total_tokens / request_count if request_count else 0.0
        duplicate_count = self._duplicate_prompt_count(docs)

        signals = self._build_signals(
            request_count=request_count,
            total_tokens=total_tokens,
            avg_tokens=avg_tokens,
            duplicate_count=duplicate_count,
            window_minutes=window_minutes,
        )
        risk_score = min(sum(signal.score for signal in signals), 1.0)
        severity, action = self._severity_and_action(risk_score)

        result = RunawayAnalysisResult(
            agent_id=agent_id,
            window_minutes=window_minutes,
            request_count=request_count,
            total_tokens=total_tokens,
            avg_tokens_per_request=round(avg_tokens, 2),
            duplicate_prompt_count=duplicate_count,
            risk_score=round(risk_score, 4),
            severity=severity,
            recommended_action=action,
            signals=signals,
        )

        await self._store_agent_health(result)
        if severity in {"high", "critical"}:
            await self._log_detection(result)
        return result

    async def _recent_requests(self, agent_id: str, window_minutes: int) -> list[dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        try:
            from integrations import db

            cursor = db.ai_requests().find(
                {"agent_id": agent_id, "created_at": {"$gte": since}},
                {
                    "_id": 0,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_hash": 1,
                    "priority": 1,
                    "created_at": 1,
                },
            )
            return await cursor.to_list(length=1000)
        except Exception:
            return []

    def _build_signals(
        self,
        request_count: int,
        total_tokens: int,
        avg_tokens: float,
        duplicate_count: int,
        window_minutes: int,
    ) -> list[RunawaySignal]:
        signals: list[RunawaySignal] = []
        rate_threshold = self.request_rate_threshold * max(window_minutes / 5, 0.2)
        token_threshold = self.token_threshold * max(window_minutes / 5, 0.2)

        if request_count > rate_threshold:
            score = min((request_count / rate_threshold - 1) * 0.35, 0.45)
            signals.append(
                RunawaySignal(
                    name="request_rate_spike",
                    score=round(score, 4),
                    detail=f"{request_count} requests in {window_minutes} minutes.",
                )
            )

        if total_tokens > token_threshold:
            score = min((total_tokens / token_threshold - 1) * 0.30, 0.35)
            signals.append(
                RunawaySignal(
                    name="token_spike",
                    score=round(score, 4),
                    detail=f"{total_tokens} tokens in {window_minutes} minutes.",
                )
            )

        if avg_tokens > self.avg_tokens_threshold:
            score = min((avg_tokens / self.avg_tokens_threshold - 1) * 0.20, 0.25)
            signals.append(
                RunawaySignal(
                    name="large_request_size",
                    score=round(score, 4),
                    detail=f"Average request size is {avg_tokens:.0f} tokens.",
                )
            )

        if duplicate_count >= self.duplicate_threshold:
            score = min(duplicate_count / max(request_count, 1), 0.30)
            signals.append(
                RunawaySignal(
                    name="recursive_duplicate_prompts",
                    score=round(score, 4),
                    detail=f"{duplicate_count} repeated prompt hashes detected.",
                )
            )

        return signals

    async def _store_agent_health(self, result: RunawayAnalysisResult) -> None:
        try:
            from integrations import cache

            await cache.set_cached(
                f"agent_health:{result.agent_id}",
                result.model_dump(mode="json"),
                ttl_seconds=10 * 60,
            )
            if result.recommended_action == "pause":
                await cache.set_cached(f"agent_paused:{result.agent_id}", True, ttl_seconds=15 * 60)
            elif result.recommended_action == "throttle":
                await cache.set_cached(f"agent_throttle:{result.agent_id}", True, ttl_seconds=10 * 60)
        except Exception:
            pass

    async def _log_detection(self, result: RunawayAnalysisResult) -> None:
        event_type = (
            OptimizationType.PAUSE
            if result.recommended_action == "pause"
            else OptimizationType.THROTTLE
        )
        event = OptimizationEvent(
            type=event_type,
            agent_id=result.agent_id,
            description=f"Runaway risk {result.severity}: {result.recommended_action} recommended.",
            before={
                "request_count": result.request_count,
                "total_tokens": result.total_tokens,
                "risk_score": result.risk_score,
            },
            after={
                "recommended_action": result.recommended_action,
                "signals": [signal.model_dump() for signal in result.signals],
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

    @staticmethod
    def _tokens(doc: dict[str, Any]) -> int:
        return int(doc.get("prompt_tokens", 0) or 0) + int(doc.get("completion_tokens", 0) or 0)

    @staticmethod
    def _duplicate_prompt_count(docs: list[dict[str, Any]]) -> int:
        counts: dict[str, int] = {}
        for doc in docs:
            prompt_hash = doc.get("prompt_hash")
            if prompt_hash:
                counts[str(prompt_hash)] = counts.get(str(prompt_hash), 0) + 1
        return sum(count - 1 for count in counts.values() if count > 1)

    @staticmethod
    def _severity_and_action(risk_score: float) -> tuple[RunawaySeverity, RunawayAction]:
        if risk_score >= 0.85:
            return "critical", "pause"
        if risk_score >= 0.55:
            return "high", "throttle"
        if risk_score >= 0.25:
            return "watch", "compress"
        return "normal", "allow"


runaway_detector = RunawayDetector()


@tool("runaway_detector", args_schema=RunawayDetectorInput)
async def runaway_detector_tool(**kwargs: Any) -> dict[str, Any]:
    """Analyze an agent for runaway request or token behavior."""
    result = await runaway_detector.analyze_agent(**kwargs)
    return result.model_dump()
