"""Tests for Shrewd LLM abstraction service."""

import os
import pytest
from fastapi.testclient import TestClient

from shrewd.api import app


@pytest.fixture(autouse=True)
def setup_env():
    os.environ["API_KEY"] = "test-key"
    yield


client = TestClient(app)


class TestHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "Shrewd"


class TestChat:
    def test_chat(self):
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4o"},
            params={"x_key": "test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["provider"] == "openai"


class TestModels:
    def test_list_models(self):
        response = client.get("/api/models", params={"x_key": "test-key"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) > 0

    def test_list_providers(self):
        response = client.get("/api/providers", params={"x_key": "test-key"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestAuth:
    def test_invalid_key(self):
        response = client.get("/api/models", params={"x_key": "wrong"})
        assert response.status_code == 401
