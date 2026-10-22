"""Data models for Shrewd LLM abstraction."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatRequest:
    def __init__(
        self,
        messages: list[ChatMessage],
        provider: Provider | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.messages = messages
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class ChatResponse:
    def __init__(
        self,
        content: str,
        provider: Provider,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        id: UUID | None = None,
        created_at: datetime | None = None,
    ):
        self.id = id or uuid4()
        self.content = content
        self.provider = provider
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at.isoformat(),
        }


class ModelInfo:
    def __init__(
        self,
        name: str,
        provider: Provider,
        context_window: int,
        is_available: bool = True,
    ):
        self.name = name
        self.provider = provider
        self.context_window = context_window
        self.is_available = is_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider.value,
            "context_window": self.context_window,
            "is_available": self.is_available,
        }
