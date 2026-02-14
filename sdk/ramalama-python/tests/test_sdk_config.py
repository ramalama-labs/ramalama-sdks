import pytest

from ramalama_sdk.config import (
    DOCKER_CONNECT_HOST,
    LOCAL_CONNECT_HOST,
    PODMAN_CONNECT_HOST,
    ConnectionSettings,
    SDKSettings,
    get_sdk_config,
    settings,
)


@pytest.fixture(autouse=True)
def _restore_settings():
    original = get_sdk_config()
    try:
        yield
    finally:
        settings.connection.bind_host = original.connection.bind_host
        settings.connection.connect_host = original.connection.connect_host
        settings.container.engine = original.container.engine


def test_sdk_config_defaults():
    settings.connection.bind_host = "127.0.0.1"
    settings.connection.connect_host = LOCAL_CONNECT_HOST
    config = get_sdk_config()
    assert config.connection.bind_host == "127.0.0.1"
    assert config.connection.connect_host == LOCAL_CONNECT_HOST


def test_direct_singleton_mutation_updates_snapshot():
    settings.connection.bind_host = "0.0.0.0"
    settings.connection.connect_host = "host.docker.internal"
    config = get_sdk_config()
    assert config.connection.bind_host == "0.0.0.0"
    assert config.connection.connect_host == "host.docker.internal"


def test_sdk_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("RAMALAMA_SDK_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("RAMALAMA_SDK_CONNECT_HOST", "host.docker.internal")
    config = SDKSettings()
    assert config.connection.bind_host == "0.0.0.0"
    assert config.connection.connect_host == "host.docker.internal"


def test_get_sdk_config_returns_frozen_snapshot():
    settings.connection.bind_host = "0.0.0.0"
    settings.connection.connect_host = LOCAL_CONNECT_HOST
    snapshot = get_sdk_config()
    with pytest.raises(Exception):
        snapshot.connection = ConnectionSettings(bind_host="127.0.0.1", connect_host=LOCAL_CONNECT_HOST)


def test_connect_host_defaults_to_local_outside_container(monkeypatch):
    monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: False)
    config = SDKSettings(connection=ConnectionSettings()).get_locked()
    assert config.connection.connect_host == LOCAL_CONNECT_HOST


def test_connect_host_defaults_to_docker_host_inside_container(monkeypatch):
    monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: True)
    monkeypatch.setattr("ramalama_sdk.config.resolve_engine", lambda: "docker")
    config = SDKSettings(connection=ConnectionSettings()).get_locked()
    assert config.connection.connect_host == DOCKER_CONNECT_HOST


def test_connect_host_defaults_to_podman_host_inside_container(monkeypatch):
    monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: True)
    monkeypatch.setattr("ramalama_sdk.config.resolve_engine", lambda: "podman")
    config = SDKSettings(connection=ConnectionSettings()).get_locked()
    assert config.connection.connect_host == PODMAN_CONNECT_HOST


def test_connect_host_explicit_override_wins(monkeypatch):
    monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: True)
    monkeypatch.setattr("ramalama_sdk.config.resolve_engine", lambda: "docker")
    config = SDKSettings(connection=ConnectionSettings(connect_host="10.0.0.55")).get_locked()
    assert config.connection.connect_host == "10.0.0.55"
