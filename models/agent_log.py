"""AgentLog — one reasoning + decision cycle produced by the orchestrator/agent.

The orchestrator builds a context snapshot, calls the Agent Builder playbook,
parses its `REASONING:` and `ACTIONS:` blocks, executes the tools, and persists
the whole trace here for the dashboard's Agent Reasoning Panel.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    """One parsed entry from the playbook's ACTIONS: JSON array."""
    tool: str
    parameters: dict = Field(default_factory=dict)
    executed: bool = False
    result: str | None = None


class AgentLog(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    cycle: int = 0

    # Snapshot of the metrics fed into the agent this cycle.
    context: dict = Field(default_factory=dict)

    # Human-readable bullets parsed from the playbook "REASONING:" block.
    reasoning: list[str] = Field(default_factory=list)

    # Parsed "ACTIONS:" block → decisions, each with its execution outcome.
    decisions: list[AgentDecision] = Field(default_factory=list)

    raw_response: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
