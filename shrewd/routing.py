"""Model routing with cost tracking and privacy classification.

Routes chat requests to the appropriate provider based on model name,
tracks token usage and estimated costs, and classifies models as public
or private (local-only) for data privacy compliance.
"""

import logging
import os
from datetime import datetime
from typing import Any

from shrewd.models import ChatMessage, ChatResponse, Provider
from shrewd.providers import BaseProvider, get_provider

logger = logging.getLogger("shrewd")

# Model registry: maps model names to provider and metadata
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # OpenAI
    "gpt-4o": {"provider": Provider.OPENAI, "context_window": 128000, "is_private": False},
    "gpt-4o-mini": {"provider": Provider.OPENAI, "context_window": 128000, "is_private": False},
    "gpt-3.5-turbo": {"provider": Provider.OPENAI, "context_window": 16385, "is_private": False},
    # Anthropic
    "claude-3-opus": {"provider": Provider.ANTHROPIC, "context_window": 200000, "is_private": False},
    "claude-3-sonnet": {"provider": Provider.ANTHROPIC, "context_window": 200000, "is_private": False},
    "claude-3-haiku": {"provider": Provider.ANTHROPIC, "context_window": 200000, "is_private": False},
    # Google
    "gemini-pro": {"provider": Provider.GOOGLE, "context_window": 32768, "is_private": False},
    "gemini-1.5-pro": {"provider": Provider.GOOGLE, "context_window": 1048576, "is_private": False},
    "gemini-1.5-flash": {"provider": Provider.GOOGLE, "context_window": 1048576, "is_private": False},
    # Ollama (local/private)
    "llama3": {"provider": Provider.OLLAMA, "context_window": 8192, "is_private": True},
    "llama3.1": {"provider": Provider.OLLAMA, "context_window": 131072, "is_private": True},
    "llama3.2": {"provider": Provider.OLLAMA, "context_window": 131072, "is_private": True},
    "gemma2": {"provider": Provider.OLLAMA, "context_window": 8192, "is_private": True},
    "gemma4:12b": {"provider": Provider.OLLAMA, "context_window": 8192, "is_private": True},
    "mistral": {"provider": Provider.OLLAMA, "context_window": 8192, "is_private": True},
    "phi3": {"provider": Provider.OLLAMA, "context_window": 4096, "is_private": True},
}

# Estimated costs per 1M tokens (input/output) for public models
COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "gemini-pro": (0.50, 1.50),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
}

# In-memory cost tracker (reset on restart)
_cost_log: list[dict[str, Any]] = []


def resolve_provider(model: str) -> tuple[Provider, dict[str, Any]]:
    """Resolve the provider and metadata for a model name.

    Returns:
        Tuple of (Provider enum, model metadata dict).
    """
    if model in MODEL_REGISTRY:
        meta = MODEL_REGISTRY[model]
        return meta["provider"], meta
    # Default: try Ollama for unknown models (they may be locally loaded)
    return Provider.OLLAMA, {"provider": Provider.OLLAMA, "context_window": 8192, "is_private": True}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the cost of a completion in USD.

    Returns 0.0 for private/local models.
    """
    rates = COST_PER_MILLION.get(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    cost = (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate
    return round(cost, 6)


def record_usage(model: str, provider: Provider, prompt_tokens: int, completion_tokens: int, cost: float) -> dict[str, Any]:
    """Record token usage and cost for tracking."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "provider": provider.value,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost,
    }
    _cost_log.append(entry)
    logger.info(
        f"Usage: model={model} provider={provider.value} "
        f"prompt={prompt_tokens} completion={completion_tokens} cost=${cost:.6f}"
    )
    return entry


def get_cost_summary() -> dict[str, Any]:
    """Get a summary of all tracked costs."""
    if not _cost_log:
        return {"total_cost_usd": 0.0, "total_tokens": 0, "requests": 0, "by_provider": {}, "by_model": {}}

    total_cost = sum(e["cost_usd"] for e in _cost_log)
    total_tokens = sum(e["total_tokens"] for e in _cost_log)

    by_provider: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}

    for entry in _cost_log:
        prov = entry["provider"]
        if prov not in by_provider:
            by_provider[prov] = {"cost_usd": 0.0, "tokens": 0, "requests": 0}
        by_provider[prov]["cost_usd"] += entry["cost_usd"]
        by_provider[prov]["tokens"] += entry["total_tokens"]
        by_provider[prov]["requests"] += 1

        model = entry["model"]
        if model not in by_model:
            by_model[model] = {"cost_usd": 0.0, "tokens": 0, "requests": 0}
        by_model[model]["cost_usd"] += entry["cost_usd"]
        by_model[model]["tokens"] += entry["total_tokens"]
        by_model[model]["requests"] += 1

    # Round costs
    for v in by_provider.values():
        v["cost_usd"] = round(v["cost_usd"], 6)
    for v in by_model.values():
        v["cost_usd"] = round(v["cost_usd"], 6)

    return {
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "requests": len(_cost_log),
        "by_provider": by_provider,
        "by_model": by_model,
    }


async def route_chat(
    messages: list[ChatMessage],
    model: str | None = None,
    provider: Provider | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    require_private: bool = False,
) -> ChatResponse:
    """Route a chat request to the appropriate provider.

    Args:
        messages: List of chat messages.
        model: Model name (e.g., "gpt-4o"). If not specified, uses default.
        provider: Override provider. If specified, model is required.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.
        require_private: If True, only route to private (local) models.

    Raises:
        ValueError: If the model is not private but require_private is True.
        RuntimeError: If the provider is not configured.
    """
    # Determine model and provider
    if model is None:
        # Default model from env or fallback
        model = os.getenv("SHREWD_DEFAULT_MODEL", "llama3")

    resolved_provider, meta = resolve_provider(model)

    # If provider is explicitly specified, override
    if provider is not None:
        resolved_provider = provider

    # Privacy check
    if require_private and not meta.get("is_private", True):
        raise ValueError(
            f"Model '{model}' ({resolved_provider.value}) is not private. "
            f"Use a local model (Ollama) for private data."
        )

    # Check provider availability
    prov_instance: BaseProvider = get_provider(resolved_provider)
    if not prov_instance.check_available():
        raise RuntimeError(
            f"Provider '{resolved_provider.value}' is not configured. "
            f"Set the required API key environment variable."
        )

    # Execute the chat request
    response = await prov_instance.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Track cost
    cost = estimate_cost(model, response.prompt_tokens, response.completion_tokens)
    record_usage(model, resolved_provider, response.prompt_tokens, response.completion_tokens, cost)
    response.cost_usd = cost

    return response
