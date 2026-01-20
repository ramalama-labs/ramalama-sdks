import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import MagicMock

import pytest

from ramalama_sdk.main import (
    AsyncRamalamaModel,
    RamalamaModel,
    ServerAttributes,
    make_chat_request,
)


class TestServerAttributes:
    def test_initial_state(self):
        attrs = ServerAttributes()
        assert attrs.url is None
        assert attrs.open is False

    def test_start(self):
        attrs = ServerAttributes()
        attrs.start(8080)
        assert attrs.url == "http://localhost:8080/v1"
        assert attrs.open is True

    def test_stop(self):
        attrs = ServerAttributes()
        attrs.start(8080)
        attrs.stop()
        assert attrs.url is None
        assert attrs.open is False


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
    server = HTTPServer(("localhost", 0), MockChatHandler)
    port = server.server_address[1]
    thread = Thread(target=server.handle_request)
    thread.start()
    yield port
    thread.join(timeout=1)


class TestMakeChatRequestIntegration:
    def test_make_chat_request(self, mock_server):
        model = MagicMock()
        model.model_name = "test-model"
        model.server_attributes.open = True
        model.server_attributes.url = f"http://localhost:{mock_server}/v1"
        model.args.temp = None
        model.args.max_tokens = None

        response = make_chat_request(model, "Hello")
        assert response["role"] == "assistant"
        assert "Response to: Hello" in response["content"]

    def test_make_chat_request_with_history(self, mock_server):
        model = MagicMock()
        model.model_name = "test-model"
        model.server_attributes.open = True
        model.server_attributes.url = f"http://localhost:{mock_server}/v1"
        model.args.temp = None
        model.args.max_tokens = None

        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
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
        model.server_attributes.open = True
        model.server_attributes.url = f"http://localhost:{mock_server}/v1"
        model.args.temp = None
        model.args.max_tokens = None

        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        response = await model.chat("How are you?", history=history)
        assert response["role"] == "assistant"
        assert "Response to: How are you?" in response["content"]
