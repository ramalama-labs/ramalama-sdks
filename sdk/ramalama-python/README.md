# Ramalama Python SDK

Build local-first AI apps on top of the Ramalama CLI. The SDK provisions models in containers and exposes a simple API for on-device inference.

## Capabilities

- **LLM chat** with OpenAI-compatible endpoints for direct requests.
- **Speech-to-Text (STT)** with Whisper (coming soon).

## Requirements

- Ramalama CLI installed (the SDK manages invocation; you do not need to call it directly).
- Docker or Podman running locally.

CLI installation guide: https://docs.ramalama.com/cli/getting-started/installation

## Install

```bash
pip install ramalama-sdk
```

### Verify

```python
import ramalama_sdk

print(f"SDK Version: {ramalama_sdk.__version__}")
```

## Quick Start

```python
from ramalama_sdk import RamalamaModel

with RamalamaModel(model="tinyllama") as model:
    response = model.chat("How tall is Michael Jordan?")
    print(response["content"])
```

## Model initialization

```python
from ramalama_sdk import RamalamaModel

model = RamalamaModel(
    model="tinyllama",
    base_image=None,
    temp=0.7,
    ngl=20,
    max_tokens=256,
    threads=8,
    ctx_size=4096,
    timeout=30,
)
```

### Parameters

    model: Model name or identifier.
    base_image: Container image to use for serving, if different from config.
    temp: Temperature override for sampling.
    ngl: GPU layers override.
    max_tokens: Maximum tokens for completions.
    threads: CPU threads override.
    ctx_size: Context window override.
    timeout: Seconds to wait for server readiness.

## Download models

Use download() to fetch models before serving. The model identifier controls the source:

hf:// for HuggingFace
ollama:// for Ollama
oci:// for OCI artifacts/images
file:// for local files

```python
from ramalama_sdk import RamalamaModel

model = RamalamaModel(model="hf://ggml-org/gpt-oss-20b-GGUF")
model.download()
```

## Documentation

Python SDK: https://docs.ramalama.com/sdk/python
Quick start: https://docs.ramalama.com/sdk/python/quickstart
