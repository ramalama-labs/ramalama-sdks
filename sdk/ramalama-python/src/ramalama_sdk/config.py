"""SDK-level configuration loaded from environment and overridable at runtime."""

from __future__ import annotations

import os
import socket
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from ramalama.config import get_config

from ramalama_sdk.logger import logger

LOCAL_CONNECT_HOST = "127.0.0.1"
DOCKER_CONNECT_HOST = "host.docker.internal"
PODMAN_CONNECT_HOST = "host.containers.internal"


def is_running_in_container() -> bool:
    """Detect whether the current Python process is running in a container."""
    return bool(os.environ.get("container") or os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"))


@lru_cache(maxsize=1)
def resolve_engine() -> str | None:
    """Resolve the active container engine from SDK env or Ramalama config."""
    if (from_env := os.environ.get("RAMALAMA_SDK_ENGINE")) is not None:
        return os.path.basename(from_env)

    config = get_config()

    if config.engine is None:
        return None
    return config.engine


def normalized_engine_name() -> str | None:
    """Return the normalized engine executable name, if available."""
    if (engine := resolve_engine()) is not None:
        return os.path.basename(engine)
    return engine


def host_resolves(host: str) -> bool:
    """Return whether a hostname can be resolved in the current environment."""
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


def default_connect_host() -> str:
    """Compute the default connect host for host or containerized SDK clients."""
    if not is_running_in_container():
        return LOCAL_CONNECT_HOST

    logger.warning(f"Detected SDK is running inside of a container. ")
    match normalized_engine_name():
        case "docker":
            return DOCKER_CONNECT_HOST
        case "podman":
            return PODMAN_CONNECT_HOST
        case _:
            for host in [DOCKER_CONNECT_HOST, PODMAN_CONNECT_HOST]:
                if host_resolves(host):
                    return host

            logger.warning(f"Could not resolve a connection host. Defaulting to {LOCAL_CONNECT_HOST}")
            return LOCAL_CONNECT_HOST


class BaseRamalamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAMALAMA_SDK_", extra="ignore")


class ConnectionSettings(BaseRamalamaSettings):
    bind_host: str = "127.0.0.1"
    connect_host: str = Field(default_factory=default_connect_host)

    @field_validator("connect_host", mode="after")
    @classmethod
    def populate_connect_host(cls, host: str) -> Any:
        if not host_resolves(host):
            logger.warning(f"Could not resolve connect_host: {host}")
        return host


class ContainerSettings(BaseRamalamaSettings):
    engine: str | None = Field(default_factory=resolve_engine)


class SDKSettings(BaseModel):
    """Global SDK settings for host selection and client connectivity."""

    connection: ConnectionSettings = Field(default_factory=ConnectionSettings)
    container: ContainerSettings = Field(default_factory=ContainerSettings)

    def get_locked(self) -> FrozenSDKSettings:
        payload = self.model_dump()
        return FrozenSDKSettings.model_validate(payload)


class FrozenSDKSettings(SDKSettings):
    model_config = ConfigDict(frozen=True)


settings = SDKSettings()


def get_sdk_config() -> FrozenSDKSettings:
    """Return an immutable snapshot of current global SDK settings."""
    return settings.get_locked()
