"""Application settings — loaded from environment / .env via pydantic-settings.

Import the ready-to-use singleton everywhere:

    from config.settings import settings
    settings.mongodb_uri
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────
    app_env: str = "development"
    port: int = 8080
    cors_origins: str = "http://localhost:3000"

    # ── MongoDB Atlas ──────────────────────────────────────────────
    mongodb_uri: str
    mongodb_db_name: str = "quotamind"

    # ── Redis ──────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Google Gemini (AI Studio) ──────────────────────────────────
    gemini_api_key: str = ""
    gemini_reasoning_model: str = "gemini-2.5-pro"
    gemini_fast_model: str = "gemini-2.5-flash"

    # ── Google Cloud Agent Builder (Dialogflow CX) ─────────────────
    gcp_project_id: str = ""
    gcp_location: str = "us-central1"
    agent_builder_agent_id: str = ""
    google_application_credentials: str = "./service-account.json"

    # ── Dynatrace MCP ──────────────────────────────────────────────
    dynatrace_env_url: str = ""
    dynatrace_api_token: str = ""

    # ── Quota / budget defaults ────────────────────────────────────
    daily_quota_token_limit: int = 10_000_000
    daily_budget_usd: float = 500.0
    orchestrator_interval_seconds: int = 30

    # ── Cross-lane event bus ───────────────────────────────────────
    # Redis pub/sub channel both lanes publish to / the SSE router reads from.
    events_channel: str = "quotamind:events"

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS_ORIGINS as a list (comma-separated in the env)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Import this singleton; it's instantiated once at process start.
settings = get_settings()
