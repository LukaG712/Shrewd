"""Provider implementations for Shrewd LLM abstraction.

Each provider class handles authentication, request formatting, and response
parsing for a specific LLM API. Providers use httpx for async HTTP calls.
"""

import logging
import os
from abc import ABC, abstractmethod

import httpx

from shrewd.models import ChatMessage, ChatResponse, Provider

logger = logging.getLogger("shrewd")


class BaseProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> ChatResponse:
        """Send a chat completion request and return the response."""
        ...

    @abstractmethod
    def check_available(self) -> bool:
        """Check if the provider is configured and available."""
        ...

    def _get_timeout(self) -> float:
        return float(os.getenv("SHREWD_REQUEST_TIMEOUT", "60"))


class OpenAIProvider(BaseProvider):
    """OpenAI API provider (GPT-4o, GPT-3.5-turbo, etc.)."""

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", self.BASE_URL)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResponse(
            content=choice["message"]["content"],
            provider=Provider.OPENAI,
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def check_available(self) -> bool:
        return bool(self.api_key)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider (Claude models)."""

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", self.BASE_URL)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> ChatResponse:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Anthropic requires system message separation
        system_msg = None
        other_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                other_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": model,
            "messages": other_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            payload["system"] = system_msg

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        content_blocks = data.get("content", [])
        content_text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content_text += block["text"]

        return ChatResponse(
            content=content_text,
            provider=Provider.ANTHROPIC,
            model=model,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )

    def check_available(self) -> bool:
        return bool(self.api_key)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.base_url = os.getenv("GOOGLE_BASE_URL", self.BASE_URL)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> ChatResponse:
        url = f"{self.base_url}/models/{model}:generateContent"
        params = {"key": self.api_key}

        # Google Gemini format
        system_msg = None
        other_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                other_messages.append({
                    "role": "user" if m.role == "user" else "model",
                    "parts": [{"text": m.content}],
                })

        payload = {
            "contents": other_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_msg:
            payload["systemInstruction"] = {"parts": [{"text": system_msg}]}

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates", [])
        content_text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                content_text += part.get("text", "")

        usage = data.get("usageMetadata", {})
        return ChatResponse(
            content=content_text,
            provider=Provider.GOOGLE,
            model=model,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
        )

    def check_available(self) -> bool:
        return bool(self.api_key)


class OllamaProvider(BaseProvider):
    """Local Ollama provider for self-hosted models."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> ChatResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        return ChatResponse(
            content=message.get("content", ""),
            provider=Provider.OLLAMA,
            model=model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    def check_available(self) -> bool:
        # Ollama is always "available" if the URL is set; real check happens on first call
        return True


def get_provider(provider: Provider) -> BaseProvider:
    """Get the provider instance for a given Provider enum."""
    providers = {
        Provider.OPENAI: OpenAIProvider(),
        Provider.ANTHROPIC: AnthropicProvider(),
        Provider.GOOGLE: GoogleProvider(),
        Provider.OLLAMA: OllamaProvider(),
    }
    return providers[provider]
