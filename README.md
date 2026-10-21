# Shrewd

> *"Oh boy..."*
> ~ **Luigi**, probably

**Shrewd** is the LLM provider abstraction service for the Evershade ecosystem. It provides a unified interface for OpenAI, Anthropic, Google, and Ollama, letting you swap models and manage costs.

**Luigi's Mansion origin:** *Shrewd* ~ one of the Three Sisters (Possessors) from Luigi's Mansion: Dark Moon, known for being cunning and intelligent, perfect for an LLM service.

## Features

- **Multi-provider** ~ OpenAI, Anthropic, Google, Ollama
- **Model routing** ~ Automatic fallback and load balancing
- **Cost tracking** ~ Per-service usage metrics

## Quick Start

```bash
docker compose up -d Shrewd
pip install -e ".[dev]"
python -m uvicorn shrewd.api:app --host 0.0.0.0 --port 8089
```

## API

- **Swagger UI:** `http://localhost:8089/docs`
- **ReDoc:** `http://localhost:8089/redoc`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `Shrewd-dev-key-change-me` | API authentication key |
| `PORT` | `8089` | Server port |
| `OPENAI_API_KEY` | (optional) | OpenAI API key |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama endpoint |

## License

MIT
