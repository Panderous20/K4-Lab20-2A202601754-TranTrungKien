"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# ── Approximate pricing per 1 K tokens (USD) ── gpt-4o-mini as default ──
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":       (0.000150, 0.000600),   # (input, output) per 1K tok
    "gpt-4o":            (0.002500, 0.010000),
    "gpt-4-turbo":       (0.010000, 0.030000),
    "gpt-3.5-turbo":     (0.000500, 0.001500),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(model: str, input_tok: int, output_tok: int) -> float | None:
    """Return estimated cost in USD, or None if model pricing is unknown."""
    prices = _PRICING.get(model)
    if prices is None:
        return None
    return (input_tok / 1_000) * prices[0] + (output_tok / 1_000) * prices[1]


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        self._model = settings.openai_model
        self._timeout = settings.timeout_seconds
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            timeout=float(self._timeout),
        )
        logger.info("LLMClient initialised  model=%s  timeout=%ss", self._model, self._timeout)

    # Retry on transient OpenAI errors (rate-limit, server error, timeout)
    @retry(
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call OpenAI Chat Completions and return a structured LLMResponse.

        Handles retry, timeout, and token-usage logging as recommended by the
        project skeleton — agents should never call the SDK directly.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError):
            # Let tenacity handle these via the @retry decorator
            raise
        except openai.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise AgentExecutionError(f"OpenAI API error: {exc}") from exc

        # ── Parse response ──────────────────────────────────────────────
        choice = response.choices[0]
        content = choice.message.content or ""

        input_tokens: int | None = None
        output_tokens: int | None = None
        cost: float | None = None

        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = _estimate_cost(self._model, input_tokens, output_tokens)

        logger.info(
            "LLM call  model=%s  in=%s  out=%s  cost=$%s  finish=%s",
            self._model,
            input_tokens,
            output_tokens,
            f"{cost:.6f}" if cost is not None else "N/A",
            choice.finish_reason,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
