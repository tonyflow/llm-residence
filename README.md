# Mini FastAPI Inference Server (Learning Build)

This is a first implementation of a small inference server that demonstrates core concepts:

- Unified API (`/v1/chat/completions`, OpenAI-style shape)
- Model registry (`models.yaml`)
- Lazy model loading + warm model cache + idle eviction
- Runtime adapters (`echo`, `ollama`)
- Non-streaming and streaming (SSE)

## Project Layout

```text
app/
  adapters/
    base.py
    echo.py
    ollama.py
  services/
    chat_service.py
  config.py
  main.py
  model_manager.py
  model_registry.py
  schemas.py
models.yaml
requirements.txt
```

## Quick Start

1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4) Test endpoints

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## Example Requests

### Echo model

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "echo-mini",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

### Ollama-backed model

Make sure Ollama is running and the model exists locally:

```bash
ollama run llama3.2:latest
```

Then call:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-ollama",
    "messages": [{"role": "user", "content": "Explain KV cache in 3 lines"}],
    "temperature": 0.2,
    "max_tokens": 120,
    "stream": false
  }'
```

### Streaming (SSE)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-ollama",
    "messages": [{"role": "user", "content": "Write a short haiku about servers"}],
    "stream": true
  }'
```

## Notes

- `ModelManager` loads adapters on first request and keeps them warm.
- Idle adapters are evicted using `warm_ttl_seconds` in `models.yaml`.
- This implementation is intentionally minimal and meant for learning.
