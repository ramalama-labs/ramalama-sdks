import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest
from ramalama.model_store.global_store import ModelFile

import ramalama_sdk.main as sdk_main
from ramalama_sdk.config import ConnectionSettings, get_sdk_config, settings
from ramalama_sdk.main import AsyncRamalamaModel, ModelStore, RamalamaModel, ServerAttributes
from ramalama_sdk.schemas import ChatMessage
from ramalama_sdk.utils import LOCAL_PORT_RESERVATION_HOST, make_chat_request


class TestServerAttributes:
    def test_initial_state(self):
        attrs = ServerAttributes(host="localhost")
        assert attrs.port is None
        assert attrs.ready is False

    def test_open(self, monkeypatch):
        attrs = ServerAttributes(host="localhost")
        monkeypatch.setattr("ramalama_sdk.utils.port_reserver.reserve_port", lambda host: 8080)
        attrs.open()
        assert attrs.port == 8080
        assert attrs.ready is True
        assert attrs.url == "http://localhost:8080"

    def test_close(self, monkeypatch):
        attrs = ServerAttributes(host="localhost")
        attrs.port = 8080
        attrs.ready = True

        released: list[tuple[int | str, str]] = []

        def _release(port, host):
            released.append((port, host))

        monkeypatch.setattr("ramalama_sdk.utils.port_reserver.release_port", _release)
        attrs.close()
        assert released == [(8080, LOCAL_PORT_RESERVATION_HOST)]
        assert attrs.port is None
        assert attrs.ready is False

    def test_open_reserves_port_on_local_loopback(self, monkeypatch):
        seen: dict[str, str] = {}

        def _reserve(host):
            seen["host"] = host
            return 8080

        attrs = ServerAttributes(host="host.docker.internal")
        monkeypatch.setattr("ramalama_sdk.utils.port_reserver.reserve_port", _reserve)
        attrs.open()
        assert seen["host"] == LOCAL_PORT_RESERVATION_HOST

    def test_is_healthy_treats_404_as_healthy(self, monkeypatch):
        attrs = ServerAttributes(host="localhost")
        attrs.port = 8080

        def _urlopen(_request, timeout):
            del timeout
            raise HTTPError(attrs.health_url, 404, "Not Found", hdrs=None, fp=None)

        monkeypatch.setattr("urllib.request.urlopen", _urlopen)
        assert attrs.is_healthy() is True


class TestHostAwareIsHealthy:
    def test_is_healthy_uses_args_host(self, monkeypatch):
        seen: dict[str, str | int | bool] = {}

        class _FakeResponse:
            def __init__(self, status: int, body: bytes = b""):
                self.status = status
                self.reason = ""
                self._body = body

            def read(self):
                return self._body

        class _FakeConnection:
            def __init__(self, host, port, timeout):
                seen["host"] = host
                seen["port"] = port
                seen["timeout"] = timeout
                self._responses = [
                    _FakeResponse(200, b""),
                    _FakeResponse(200, b'{"models":[{"name":"org/test-model"}]}'),
                ]
                self._requests: list[str] = []

            def set_debuglevel(self, level):
                seen["debug_level"] = level

            def request(self, _method, path):
                self._requests.append(path)
                seen["paths"] = list(self._requests)

            def getresponse(self):
                return self._responses.pop(0)

            def close(self):
                seen["closed"] = True

        monkeypatch.setattr(sdk_main, "HTTPConnection", _FakeConnection)
        monkeypatch.setattr(sdk_main, "New", lambda _model, _args: SimpleNamespace(model_alias="org/test-model"))

        args = SimpleNamespace(host="10.88.0.42", port=5111, debug=False, runtime="llama.cpp", MODEL="test-model")
        assert sdk_main.is_healthy(args) is True
        assert seen["host"] == "10.88.0.42"
        assert seen["paths"] == ["/health", "/models"]
        assert seen["closed"] is True


class TestRamalamaModelChat:
    def test_chat_raises_when_server_not_running(self):
        model = RamalamaModel("test-model")
        with pytest.raises(RuntimeError, match="Server is not running"):
            model.chat("Hello")


class MockChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data)

        response = {"choices": [{"message": {"content": f"Response to: {request_body['messages'][-1]['content']}"}}]}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_server():
    try:
        server = HTTPServer(("localhost", 0), MockChatHandler)
    except PermissionError as exc:
        pytest.skip(f"Local HTTP server unavailable in this environment: {exc}")
    port = server.server_address[1]
    thread = Thread(target=server.handle_request)
    thread.start()
    yield port
    thread.join(timeout=1)


class TestMakeChatRequestIntegration:
    def test_make_chat_request(self, mock_server):
        model = MagicMock()
        model.model_name = "test-model"
        model.server_attributes.ready = True
        model.server_attributes.chat_url = f"http://localhost:{mock_server}/v1/chat/completions"
        model.args.temp = None
        model.args.max_tokens = None

        response = make_chat_request(model, "Hello")
        assert response["role"] == "assistant"
        assert "Response to: Hello" in response["content"]

    def test_make_chat_request_with_history(self, mock_server):
        model = MagicMock()
        model.model_name = "test-model"
        model.server_attributes.ready = True
        model.server_attributes.chat_url = f"http://localhost:{mock_server}/v1/chat/completions"
        model.args.temp = None
        model.args.max_tokens = None

        history: list[ChatMessage] = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        response = make_chat_request(model, "How are you?", history=history)
        assert response["role"] == "assistant"
        assert "Response to: How are you?" in response["content"]


class TestAsyncRamalamaModelChat:
    @pytest.mark.asyncio
    async def test_chat_raises_when_server_not_running(self):
        model = AsyncRamalamaModel("test-model")
        with pytest.raises(RuntimeError, match="Server is not running"):
            await model.chat("Hello")

    @pytest.mark.asyncio
    async def test_chat_with_history(self, mock_server):
        model = AsyncRamalamaModel("test-model")
        model.server_attributes.ready = True
        model.server_attributes.port = mock_server
        model.args.temp = None
        model.args.max_tokens = None

        history: list[ChatMessage] = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        response = await model.chat("How are you?", history=history)
        assert response["role"] == "assistant"
        assert "Response to: How are you?" in response["content"]


class TestRuntimeArgs:
    def test_runtime_args_passthrough(self):
        model = RamalamaModel("test-model", runtime_args=["--chat-template", "chatml"])
        assert model.args.runtime_args == ["--chat-template", "chatml"]

    def test_no_jinja_appends_flag(self):
        model = RamalamaModel("test-model", no_jinja=True)
        assert "--no-jinja" in model.args.runtime_args

    def test_no_jinja_does_not_duplicate_flag(self):
        model = RamalamaModel("test-model", runtime_args=["--no-jinja"], no_jinja=True)
        assert model.args.runtime_args.count("--no-jinja") == 1


class TestHostConfiguration:
    def setup_method(self):
        self._original_settings = get_sdk_config()

    def teardown_method(self):
        settings.connection.bind_host = self._original_settings.connection.bind_host
        settings.connection.connect_host = self._original_settings.connection.connect_host
        settings.container.engine = self._original_settings.container.engine

    def test_bind_and_connect_hosts_are_loaded_from_global_sdk_config(self):
        settings.connection.bind_host = "0.0.0.0"
        settings.connection.connect_host = "host.docker.internal"
        model = RamalamaModel("test-model")
        assert model.bind_host == "0.0.0.0"
        assert model.connect_host == "host.docker.internal"

    def test_default_global_sdk_hosts_are_used(self):
        settings.connection.bind_host = "127.0.0.1"
        settings.connection.connect_host = "127.0.0.1"
        model = RamalamaModel("test-model")
        assert model.bind_host == "127.0.0.1"
        assert model.connect_host == "127.0.0.1"

    def test_args_host_uses_bind_host(self):
        settings.connection.bind_host = "0.0.0.0"
        settings.connection.connect_host = "127.0.0.1"
        model = RamalamaModel("test-model")
        assert model.args.host == "0.0.0.0"

    def test_server_attribute_urls_use_connect_host(self):
        settings.connection.bind_host = "0.0.0.0"
        settings.connection.connect_host = "host.docker.internal"
        model = RamalamaModel("test-model")
        model.server_attributes.port = 8080
        assert model.server_attributes.health_url == "http://host.docker.internal:8080/health"
        assert model.server_attributes.chat_url == "http://host.docker.internal:8080/v1/chat/completions"

    def test_async_server_attribute_urls_use_connect_host(self):
        settings.connection.bind_host = "0.0.0.0"
        settings.connection.connect_host = "host.docker.internal"
        model = AsyncRamalamaModel("test-model")
        model.server_attributes.port = 8080
        assert model.server_attributes.health_url == "http://host.docker.internal:8080/health"
        assert model.server_attributes.chat_url == "http://host.docker.internal:8080/v1/chat/completions"

    def test_model_defaults_connect_host_to_local_outside_container(self, monkeypatch):
        monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: False)
        settings.connection = ConnectionSettings(bind_host=settings.connection.bind_host)
        model = RamalamaModel("test-model", config=SimpleNamespace(engine="docker"))
        assert model.connect_host == "127.0.0.1"

    def test_model_defaults_connect_host_to_docker_inside_container(self, monkeypatch):
        monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: True)
        monkeypatch.setattr("ramalama_sdk.config.resolve_engine", lambda: "docker")
        settings.connection = ConnectionSettings(bind_host=settings.connection.bind_host)
        model = RamalamaModel("test-model", config=SimpleNamespace(engine="docker"))
        assert model.connect_host == "host.docker.internal"

    def test_model_defaults_connect_host_to_podman_inside_container(self, monkeypatch):
        monkeypatch.setattr("ramalama_sdk.config.is_running_in_container", lambda: True)
        monkeypatch.setattr("ramalama_sdk.config.resolve_engine", lambda: "podman")
        settings.connection = ConnectionSettings(bind_host=settings.connection.bind_host)
        model = RamalamaModel("test-model", config=SimpleNamespace(engine="podman"))
        assert model.connect_host == "host.containers.internal"


class TestServeReadinessHost:
    def setup_method(self):
        self._original_settings = get_sdk_config()

    def teardown_method(self):
        settings.connection.bind_host = self._original_settings.connection.bind_host
        settings.connection.connect_host = self._original_settings.connection.connect_host
        settings.container.engine = self._original_settings.container.engine

    def test_sync_serve_uses_connect_host_for_readiness(self, monkeypatch):
        seen: dict[str, str | None] = {}

        def _is_healthy(_args, host=None):
            seen["host"] = host
            return True

        settings.connection.connect_host = "host.docker.internal"
        monkeypatch.setattr(sdk_main.RamalamaModel, "start_server", lambda self: None)
        monkeypatch.setattr(sdk_main, "is_healthy", _is_healthy)

        model = RamalamaModel("test-model", timeout=1)
        model.serve()

        assert seen["host"] == "host.docker.internal"

    @pytest.mark.asyncio
    async def test_async_serve_uses_connect_host_for_readiness(self, monkeypatch):
        seen: dict[str, str | None] = {}

        def _is_healthy(_args, host=None):
            seen["host"] = host
            return True

        settings.connection.connect_host = "host.docker.internal"
        monkeypatch.setattr(sdk_main.AsyncRamalamaModel, "start_server", lambda self: None)
        monkeypatch.setattr(sdk_main, "is_healthy", _is_healthy)

        model = AsyncRamalamaModel("test-model", timeout=1)
        await model.serve()

        assert seen["host"] == "host.docker.internal"


class TestListModels:
    def _fake_model_listing(self, *args, **kwargs):
        return {
            "ollama://library/foo:latest": [ModelFile(name="foo.gguf", modified=100.0, size=10, is_partial=False)],
            "ollama://library/bar:latest": [ModelFile(name="bar.gguf", modified=200.0, size=20, is_partial=False)],
            "ollama://library/bad:latest": [ModelFile(name="bad.gguf", modified=300.0, size=30, is_partial=True)],
        }

    def test_list_models_sync(self, monkeypatch):
        monkeypatch.setattr(
            "ramalama.model_store.global_store.GlobalModelStore.list_models",
            self._fake_model_listing,
        )
        store = ModelStore(store_path="/tmp/ramalama-test", engine="podman")
        records = store.list_models()
        assert [record.name for record in records] == [
            "ollama://library/bar:latest",
            "ollama://library/foo:latest",
        ]
        assert records[0].last_modified > records[1].last_modified

    @pytest.mark.asyncio
    async def test_list_models_async(self, monkeypatch):
        monkeypatch.setattr(
            "ramalama.model_store.global_store.GlobalModelStore.list_models",
            self._fake_model_listing,
        )
        store = ModelStore(store_path="/tmp/ramalama-test", engine="podman")
        records = await asyncio.to_thread(store.list_models)
        assert [record.name for record in records] == [
            "ollama://library/bar:latest",
            "ollama://library/foo:latest",
        ]
        assert records[0].last_modified > records[1].last_modified
