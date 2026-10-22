"""FastAPI application for Shrewd LLM abstraction service.

Routes chat requests to real LLM providers (OpenAI, Anthropic, Google, Ollama)
with cost tracking and privacy classification.
"""

import os
import hmac
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header

from shrewd.models import Provider, ChatMessage, ChatResponse
from shrewd.providers import get_provider
from shrewd.routing import (
    MODEL_REGISTRY,
    route_chat,
    get_cost_summary,
    resolve_provider,
)

logger = logging.getLogger("shrewd")

app = FastAPI(
    title="Shrewd",
    description="LLM provider abstraction for the Evershade ecosystem",
    version="1.1.0",
)

API_KEY = os.getenv("API_KEY", "Shrewd-dev-key-change-me")


def verify_bearer(authorization: Optional[str] = Header(None)) -> None:
    """Verify Bearer token in Authorization header."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header. Use: Authorization: Bearer <key>")
    if not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "Shrewd"}


@app.post("/api/chat")
async def chat(data: dict[str, Any], authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    """Route a chat request to the appropriate LLM provider.

    Supports real providers: OpenAI, Anthropic, Google Gemini, and Ollama.
    Falls back to simulated response if no provider is configured.
    """
    verify_bearer(authorization)
    model = data.get("model")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1000)
    require_private = data.get("require_private", False)
    messages_data = data.get("messages", [])

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_data]

    try:
        response = await route_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            require_private=require_private,
        )
        return response.to_dict()
    except (ValueError, RuntimeError) as e:
        # If provider is not configured, fall back to simulation
        logger.warning(f"Provider error, falling back to simulation: {e}")
        resolved_provider, _ = resolve_provider(model or "llama3")
        last_msg = messages[-1].content if messages else ""
        response = ChatResponse(
            content=f"[Simulated {resolved_provider.value} response] You said: {last_msg[:100]}",
            provider=resolved_provider,
            model=model or "llama3",
            prompt_tokens=len(last_msg),
            completion_tokens=20,
        )
        return response.to_dict()
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=502, detail=f"Provider error: {str(e)}")


@app.get("/api/models")
async def list_models(provider: Optional[str] = None, authorization: Optional[str] = Header(None)) -> list[dict[str, Any]]:
    """List available models with provider and privacy info."""
    verify_bearer(authorization)
    results = []
    for name, meta in MODEL_REGISTRY.items():
        prov = meta["provider"]
        if provider and prov.value != provider:
            continue
        prov_instance = get_provider(prov)
        results.append({
            "name": name,
            "provider": prov.value,
            "context_window": meta.get("context_window", 0),
            "is_private": meta.get("is_private", True),
            "is_available": prov_instance.check_available(),
        })
    return results


@app.get("/api/providers")
async def list_providers(authorization: Optional[str] = Header(None)) -> list[dict[str, Any]]:
    """List available providers with availability status."""
    verify_bearer(authorization)
    provider_map = {
        Provider.OPENAI: get_provider(Provider.OPENAI),
        Provider.ANTHROPIC: get_provider(Provider.ANTHROPIC),
        Provider.GOOGLE: get_provider(Provider.GOOGLE),
        Provider.OLLAMA: get_provider(Provider.OLLAMA),
    }
    results = []
    for prov, instance in provider_map.items():
        results.append({
            "name": prov.value,
            "is_available": instance.check_available(),
        })
    return results


@app.get("/api/cost/summary")
async def cost_summary(authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    """Get a summary of tracked LLM usage costs."""
    verify_bearer(authorization)
    return get_cost_summary()
