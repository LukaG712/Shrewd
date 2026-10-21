"""FastAPI application for Chauncey LLM abstraction service."""

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query

from chauncey.models import Provider, ChatMessage, ChatResponse


app = FastAPI(
    title="Chauncey",
    description="LLM provider abstraction for the Evershade ecosystem",
    version="1.0.0",
)

API_KEY = os.getenv("API_KEY", "Chauncey-dev-key-change-me")

# Available models registry
_models = {
    "gpt-4o": Provider.OPENAI,
    "gpt-3.5-turbo": Provider.OPENAI,
    "claude-3-opus": Provider.ANTHROPIC,
    "claude-3-sonnet": Provider.ANTHROPIC,
    "gemini-pro": Provider.GOOGLE,
    "llama3": Provider.OLLAMA,
    "gemma4:12b": Provider.OLLAMA,
}


def verify_key(x_key: Optional[str] = Query(None)) -> None:
    if x_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "Chauncey"}


@app.post("/api/chat")
async def chat(data: dict[str, Any], x_key: Optional[str] = Query(None)) -> dict[str, Any]:
    verify_key(x_key)
    model = data.get("model", "gpt-4o")
    provider = _models.get(model, Provider.OLLAMA)
    messages = data.get("messages", [])
    # Simulate response (in production, route to actual provider)
    last_msg = messages[-1]["content"] if messages else ""
    response = ChatResponse(
        content=f"[Simulated {provider.value} response] You said: {last_msg[:100]}",
        provider=provider,
        model=model,
        prompt_tokens=len(last_msg),
        completion_tokens=20,
    )
    return response.to_dict()


@app.get("/api/models")
async def list_models(provider: Optional[str] = None, x_key: Optional[str] = Query(None)) -> list[dict[str, Any]]:
    verify_key(x_key)
    results = []
    for name, prov in _models.items():
        if provider and prov.value != provider:
            continue
        results.append({"name": name, "provider": prov.value, "is_available": True})
    return results


@app.get("/api/providers")
async def list_providers(x_key: Optional[str] = Query(None)) -> list[str]:
    verify_key(x_key)
    return list(set(p.value for p in _models.values()))
