import subprocess

import pytest
from ramalama.config import get_config

ramalama_conf = get_config()


def has_container_runtime() -> bool:
    if ramalama_conf.engine is None:
        return False

    # E2E requires a working engine daemon/session, not just a configured binary.
    try:
        result = subprocess.run(
            [str(ramalama_conf.engine), "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False

    return result.returncode == 0


requires_container = pytest.mark.skipif(
    not has_container_runtime(),
    reason="No usable container runtime (docker/podman) available",
)


@pytest.fixture
def small_model():
    """A small model suitable for integration testing."""
    return "hf://ggml-org/SmolVLM-500M-Instruct-GGUF"
