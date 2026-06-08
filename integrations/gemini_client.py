"""Gemini API client used by request ingestion and engines."""
from __future__ import annotations

from pydantic import BaseModel
import tiktoken

from config.env import get_env

try:
    from google import genai
except ImportError:  # pragma: no cover - dependency is installed in app env.
    genai = None


class GeminiGeneration(BaseModel):
    model: str
    prompt: str
    text: str
    prompt_tokens: int
    completion_tokens: int


class GeminiClient:
    """Thin async wrapper around google-genai."""

    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding("cl100k_base")

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
    ) -> GeminiGeneration:
        """Generate text from Gemini and return token metadata."""
        model_name = model or get_env("GEMINI_FAST_MODEL", "gemini-2.5-flash")
        api_key = get_env("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env.")
        if genai is None:
            raise RuntimeError("google-genai is not installed.")

        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = (response.text or "").strip()
        return GeminiGeneration(
            model=model_name,
            prompt=prompt,
            text=text,
            prompt_tokens=self.count_tokens(prompt),
            completion_tokens=self.count_tokens(text),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))


gemini_client = GeminiClient()
