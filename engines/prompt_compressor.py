"""Prompt compression engine and LangChain tool.

The compressor uses Gemini Flash when configured. If Gemini is unavailable, it
falls back to light local cleanup so callers still get a safe result.
"""
from __future__ import annotations

import re
from typing import Any

from google import genai
import tiktoken
from pydantic import BaseModel, Field

from config.env import get_env
from config.models import compute_cost
from models.optimization_event import OptimizationEvent, OptimizationType

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - lets local logic run before deps are installed.
    def tool(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator

# What we are outputing!
class CompressionResult(BaseModel):
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    cost_saved_usd: float
    compression_ratio: float
    used_model: str

# schema for the compressor input!
class PromptCompressorInput(BaseModel):
    prompt: str = Field(..., description="Prompt text to compress.")
    agent_id: str | None = Field(default=None, description="Agent requesting compression.")
    model: str = Field(default="gemini-2.5-pro", description="Original target model.")
    target_ratio: float = Field(default=0.6, ge=0.1, le=0.95)


class PromptCompressor:
    """Compress prompts before expensive model calls."""

    def __init__(self, compression_model: str | None = None) -> None:
        self.compression_model = compression_model
        self.encoding = tiktoken.get_encoding("cl100k_base")

    async def compress(
        self,
        prompt: str,
        agent_id: str | None = None,
        model: str = "gemini-2.5-pro",
        target_ratio: float = 0.6,
    ) -> CompressionResult:
        """Return a shorter prompt with token and cost savings metadata."""
        clean_prompt = prompt.strip()
        original_tokens = self._count_tokens(clean_prompt)

        if not clean_prompt:
            return self._build_result("", "", 0, model)

        compressed = await self._compress_with_gemini(clean_prompt, target_ratio)
        if not compressed:
            compressed = self._local_cleanup(clean_prompt)

        if self._count_tokens(compressed) >= original_tokens:
            compressed = clean_prompt

        result = self._build_result(clean_prompt, compressed, original_tokens, model)
        await self._log_savings(result, agent_id, model)
        return result

    async def _compress_with_gemini(self, prompt: str, target_ratio: float) -> str | None:
        try:
            gemini_api_key = get_env("GEMINI_API_KEY")
            if not gemini_api_key:
                return None

            model_name = self.compression_model or get_env("GEMINI_FAST_MODEL", "gemini-2.5-flash")
            client = genai.Client(api_key=gemini_api_key)
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=(
                    "Compress this prompt without losing requirements, constraints, or context.\n"
                    f"Target length: about {int(target_ratio * 100)}% of the original.\n"
                    "Return only the compressed prompt and nothing else.\n\n"
                    f"{prompt}"
                ),
            )
            return (response.text or "").strip()
        except Exception:
            return None

    def _build_result(
        self,
        original_prompt: str,
        compressed_prompt: str,
        original_tokens: int,
        model: str,
    ) -> CompressionResult:
        compressed_tokens = self._count_tokens(compressed_prompt)
        tokens_saved = max(original_tokens - compressed_tokens, 0)
        cost_saved = compute_cost(model, tokens_saved, 0)
        ratio = compressed_tokens / original_tokens if original_tokens else 1.0

        return CompressionResult(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved,
            compression_ratio=round(ratio, 3),
            used_model=model,
        )

    async def _log_savings(
        self,
        result: CompressionResult,
        agent_id: str | None,
        model: str,
    ) -> None:
        if result.tokens_saved <= 0:
            return

        event = OptimizationEvent(
            type=OptimizationType.COMPRESS,
            agent_id=agent_id,
            description="Compressed prompt before inference.",
            before={"model": model, "tokens": result.original_tokens},
            after={"model": model, "tokens": result.compressed_tokens},
            tokens_saved=result.tokens_saved,
            cost_saved_usd=result.cost_saved_usd,
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

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @staticmethod
    def _local_cleanup(prompt: str) -> str:
        prompt = re.sub(r"[ \t]+", " ", prompt)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        return prompt.strip()


prompt_compressor = PromptCompressor()


@tool("prompt_compressor", args_schema=PromptCompressorInput)
async def compress_prompt_tool(**kwargs: Any) -> dict[str, Any]:
    """Compress long prompts and report token/cost savings."""
    result = await prompt_compressor.compress(**kwargs)
    return result.model_dump()


prompt_compressor_tool = compress_prompt_tool