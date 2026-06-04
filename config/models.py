"""Model pricing, tier rankings, and cost computation.

Single source of truth for "what does a call cost" and "which model is allowed
for which priority". Both lanes import from here so numbers never drift.
"""
from __future__ import annotations

# Cost in USD per 1,000 tokens, split by input (prompt) and output (completion).
# Source: PROJECT_FLOW pricing table.
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":   {"input": 0.00125,  "output": 0.005},
    "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001,   "output": 0.0004},
}

# Tier ranking — higher number = more capable / more expensive.
MODEL_TIERS: dict[str, int] = {
    "gemini-2.0-flash": 1,
    "gemini-2.5-flash": 2,
    "gemini-2.5-pro":   3,
}

# Minimum model tier allowed per request priority. The TrafficRouter may route
# at or above this floor, but must never downgrade a request below it — this is
# what protects `critical` workloads from being throttled to a weak model.
PRIORITY_TO_MIN_TIER: dict[str, int] = {
    "critical": 3,
    "high":     2,
    "medium":   1,
    "low":      1,
}

DEFAULT_MODEL = "gemini-2.5-flash"


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of a single model call. Unknown models return 0.0."""
    pricing = COST_PER_1K_TOKENS.get(model)
    if not pricing:
        return 0.0
    cost = (
        (prompt_tokens / 1000) * pricing["input"]
        + (completion_tokens / 1000) * pricing["output"]
    )
    return round(cost, 6)


def model_for_tier(tier: int) -> str:
    """Cheapest model at exactly the given tier (fallback: DEFAULT_MODEL)."""
    for name, t in MODEL_TIERS.items():
        if t == tier:
            return name
    return DEFAULT_MODEL
