import json
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING

from ramalama.model_store.global_store import GlobalModelStore

from ramalama_sdk.schemas import ChatMessage, ModelRecord

if TYPE_CHECKING:
    from ramalama_sdk.main import RamalamaModelBase


LOCAL_PORT_RESERVATION_HOST = "127.0.0.1"


def pick_free_tcp_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class PortReservationSystem:
    def __init__(self):
        self.lock: Lock = Lock()
        self.reserved_ports: dict[str, set[int]] = dict()

    def reserve_port(self, host: str = "127.0.0.1") -> int:
        """Reserve a serving port for this process to avoid concurrent collisions."""
        self.reserved_ports.setdefault(host, set())
        with self.lock:
            port = None
            while port is None:
                trial_port = pick_free_tcp_port(host)
                if trial_port not in self.reserved_ports[host]:
                    port = trial_port
                    self.reserved_ports[host].add(port)
            return port

    def release_port(self, port: int | str, host: str = "127.0.0.1") -> None:
        if (port_set := self.reserved_ports.get(host, None)) is None:
            return

        with self.lock:
            port_set.discard(int(port))


port_reserver = PortReservationSystem()


@dataclass
class ServerAttributes:
    """Track connection endpoints and readiness for a running model server."""

    host: str
    scheme: str = "http"
    base_path: str = "v1"
    health_path: str | None = "/health"
    chat_path: str | None = "/v1/chat/completions"
    port: int | None = None
    ready: bool = False

    def open(self):
        if self.ready:
            return
        self.port = port_reserver.reserve_port(LOCAL_PORT_RESERVATION_HOST)
        self.ready = True

    def close(self):
        if self.port is None:
            return
        port_reserver.release_port(self.port, LOCAL_PORT_RESERVATION_HOST)
        self.port = None
        self.ready = False

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        if self.health_path is None:
            raise ValueError("No health path for this server")
        return f"{self.url}/{self.health_path.lstrip('/')}"

    @property
    def chat_url(self) -> str:
        if self.chat_path is None:
            raise ValueError("No chat path for this server")
        return f"{self.url}/{self.chat_path.lstrip('/')}"

    def is_healthy(self) -> bool:
        """Check whether the configured health endpoint is responding.

        Returns:
            True when the endpoint returns HTTP 200 or 404.
        """
        if not self.health_path:
            return False

        request = urllib.request.Request(self.health_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                return response.status in (200, 404)
        except urllib.error.HTTPError as exc:
            return exc.code == 404


def list_models(store: GlobalModelStore, engine: str) -> list[ModelRecord]:
    """List locally available models, sorted by last modified time."""
    models = store.list_models(engine=engine, show_container=True)
    local_tz = datetime.now().astimezone().tzinfo

    ret = []
    for model, files in models.items():
        is_partial = False
        size = 0
        last_modified = 0
        for f in files:
            if f.is_partial:
                is_partial = True
                break

            size += f.size
            last_modified = max(last_modified, f.modified)

        if not is_partial:
            model = ModelRecord(
                name=model,
                size=size,
                last_modified=datetime.fromtimestamp(last_modified, tz=local_tz),
            )

            ret.append(model)

    ret.sort(key=lambda m: m.last_modified, reverse=True)
    return ret


def make_chat_request(
    model: "RamalamaModelBase", message: str, history: list[ChatMessage] | None = None
) -> ChatMessage:
    """Send a synchronous chat completion request to the running server.

    Args:
        model: Active model instance with a running server.
        message: User prompt content.
        history: Optional prior conversation messages.

    Returns:
        Assistant message payload.

    Raises:
        RuntimeError: If the server is not running.
    """
    if not model.server_attributes.ready:
        raise RuntimeError("Server is not running. Call serve() first.")

    messages: list[ChatMessage] = list(history) if history else []
    messages.append({"role": "user", "content": message})

    data: dict = {
        "model": model.model_name,
        "messages": messages,
        "stream": False,
    }
    if model.args.temp:
        data["temperature"] = float(model.args.temp)
    if model.args.max_tokens:
        data["max_completion_tokens"] = model.args.max_tokens

    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(
        model.server_attributes.chat_url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    response = urllib.request.urlopen(request)
    response = json.loads(response.read())

    result = ChatMessage(role="assistant", content=response["choices"][0]["message"]["content"])
    return result
