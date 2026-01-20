<p align="center">
  <img src="assets/logo.webp" alt="RamaLama Labs Logo" width="140"/>
</p>

<h1 align="center">RamaLama Labs</h1>

<p align="center">
  <strong>Programmable AI on any device.</strong><br/>
  Run LLMs locally on any hardware. If you can build a container you can deploy AI.
</p>

<p align="center">
  <a href="https://github.com/ramalama-labs/ramalama-sdks"><img src="https://img.shields.io/github/stars/ramalama-labs/ramalama-sdks?style=flat-square" alt="GitHub Stars" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/MIT%202.0-blue?style=flat-square" alt="License" /></a>
  <a href="https://discord.gg/cFyDXs9nS9"><img src="https://img.shields.io/badge/Discord-Join-cFyDXs9nS9?style=flat-square&logo=discord&logoColor=white" alt="Discord" /></a>
</p>


---

[RamaLama](https://github.com/containers/ramalama) is an open source container orchestration system which makes working with AI simple, straightforward, and familiar using OCI containers.

Ramalama lets you add AI features to your local application while running entirely on device. It can be used on any device with an applicable container manager like docker or podman with support for most model repositories.

    LLM Chat — HuggingFace, Ollama, ModelScope, OCI, local files, URLs
    Speech-to-Text — Whisper



## SDKs

| Platform | Status | Installation | Documentation |
|----------|--------|--------------|---------------|
| **python** | Active Development | [pypi](#python) | [docs.ramalama.com/sdk/python](https://docs.ramalama.com/sdk/python/introduction) |
| **Typescript** | planned | -- | -- |
| **Go** | planned | -- | -- |
| **Rust** | planned | -- | -- |



## Requirements

- Docker or Podman running locally.
- Python 3.10+ for the Python SDK.



## Quick Start

### Python

**Install from pypi**

```python
pip install ramalama-sdk
```

```python
from ramalama_sdk import RamalamaModel

with RamalamaModel(model="tinyllama") as model:
    response = model.chat("How tall is Michael Jordan?")
    print(response["content"])  # Michael Jordan is 6'6" tall, which is 1.97 meters tall.
```

## Repository Structure

```text
.
├── assets/                # Branding assets
├── sdk/                   # SDK implementations
│   └── ramalama-python/   # Python SDK package
├── LICENSE
└── README.md
```

## Support

- Discord: [https://discord.gg/cFyDXs9nS9](https://discord.gg/cFyDXs9nS90)
- GitHub Issues: [https://github.com/ramalama-labs/ramalama-sdks/issues](https://github.com/ramalama-labs/ramalama-sdks/issues)
- X: [@RamaLamaLabs](https://x.com/RamaLamaLabs)
