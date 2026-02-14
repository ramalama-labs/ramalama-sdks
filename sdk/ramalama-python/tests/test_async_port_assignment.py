import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import ramalama_sdk.main as sdk_main
import ramalama_sdk.utils as sdk_utils


class _FakeTransport:
    def ensure_model_exists(self, _args):
        return None

    def serve_nonblocking(self, _args, _cmd):
        return object()

    def _cleanup_server_process(self, _process):
        return None


class _ExitedProcess:
    def __init__(self, exit_code: int):
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code


def _fake_config():
    return SimpleNamespace(
        engine="podman",
        image="quay.io/ramalama/test",
        store="/tmp/ramalama-test",
        runtime="",
        host="localhost",
        ctx_size=2048,
        threads=1,
        ngl=0,
        temp=0.0,
        max_tokens=128,
        cache_reuse=False,
        thinking=False,
    )


class _FakePortReserver:
    def __init__(self):
        self.lock = threading.Lock()
        self._ports = [41001, 41002, 41003, 41004, 41005]
        self._in_use: set[int] = set()

    def reserve_port(self, host="127.0.0.1"):
        del host
        with self.lock:
            for port in self._ports:
                if port not in self._in_use:
                    self._in_use.add(port)
                    return port
        raise IOError("No available test ports")

    def release_port(self, port, host="127.0.0.1"):
        del host
        with self.lock:
            self._in_use.discard(int(port))


def _mock_runtime(monkeypatch):
    monkeypatch.setattr(sdk_utils, "port_reserver", _FakePortReserver())
    monkeypatch.setattr(sdk_main, "is_healthy", lambda _args, host=None: True)
    monkeypatch.setattr(sdk_main, "New", lambda _model_name, _args: _FakeTransport())
    monkeypatch.setattr(sdk_main, "assemble_command", lambda _args: ["ramalama", "serve"])


@pytest.mark.asyncio
async def test_async_servers_started_together_get_unique_ports(monkeypatch):
    _mock_runtime(monkeypatch)

    models = [sdk_main.AsyncRamalamaModel("test-model", config=_fake_config(), timeout=1) for _ in range(4)]
    await asyncio.gather(*(model.serve() for model in models))
    try:
        ports = [model.args.port for model in models]
        assert len(set(ports)) == len(models)
    finally:
        await asyncio.gather(*(model.stop() for model in models))


@pytest.mark.asyncio
async def test_async_server_cleanup_releases_reserved_ports(monkeypatch):
    _mock_runtime(monkeypatch)

    first_batch = [sdk_main.AsyncRamalamaModel("test-model", config=_fake_config(), timeout=1) for _ in range(2)]
    await asyncio.gather(*(model.serve() for model in first_batch))
    first_ports = {model.args.port for model in first_batch}
    await asyncio.gather(*(model.stop() for model in first_batch))

    second_batch = [sdk_main.AsyncRamalamaModel("test-model", config=_fake_config(), timeout=1) for _ in range(2)]
    await asyncio.gather(*(model.serve() for model in second_batch))
    try:
        second_ports = {model.args.port for model in second_batch}
        assert second_ports == first_ports
    finally:
        await asyncio.gather(*(model.stop() for model in second_batch))


def test_sync_serve_fails_fast_when_process_exits(monkeypatch):
    _mock_runtime(monkeypatch)
    monkeypatch.setattr(sdk_main, "is_healthy", lambda _args, host=None: False)

    class _ProcessExitTransport(_FakeTransport):
        def serve_nonblocking(self, _args, _cmd):
            return _ExitedProcess(exit_code=17)

    monkeypatch.setattr(sdk_main, "New", lambda _model_name, _args: _ProcessExitTransport())

    model = sdk_main.RamalamaModel("test-model", config=_fake_config(), container=False, timeout=5)
    start_time = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 17"):
        model.serve()
    assert time.monotonic() - start_time < 1.0


@pytest.mark.asyncio
async def test_async_serve_fails_fast_when_process_exits(monkeypatch):
    _mock_runtime(monkeypatch)
    monkeypatch.setattr(sdk_main, "is_healthy", lambda _args, host=None: False)

    class _ProcessExitTransport(_FakeTransport):
        def serve_nonblocking(self, _args, _cmd):
            return _ExitedProcess(exit_code=23)

    monkeypatch.setattr(sdk_main, "New", lambda _model_name, _args: _ProcessExitTransport())

    model = sdk_main.AsyncRamalamaModel("test-model", config=_fake_config(), container=False, timeout=5)
    start_time = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 23"):
        await model.serve()
    assert time.monotonic() - start_time < 1.0
