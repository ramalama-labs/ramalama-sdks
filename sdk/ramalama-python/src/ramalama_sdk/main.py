"""Ramalama Python SDK public interface for starting a model server and chatting."""

import asyncio
import copy
import json
import time
import urllib.request
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from http.client import HTTPException
from types import SimpleNamespace
from typing import Literal, TypedDict

from ramalama.command.factory import assemble_command
from ramalama.config import CONFIG as ramalama_conf
from ramalama.engine import stop_container
from ramalama.transports.base import Transport, compute_serving_port
from ramalama.transports.transport_factory import New


class RamalamaNoContainerManagerError(Exception):
    """Raised when no supported container manager (docker/podman) is available."""


class RamalamaServerTimeoutError(Exception):
    """Raised when the model server fails to become healthy in time."""


class ChatMessage(TypedDict):
    """Chat completion message payload.

    Attributes:
        role: Message author role.
        content: Message text content.
    """

    role: Literal["system", "user", "assistant", "developer"]
    content: str


def is_server_healthy(port: int | str) -> bool:
    """Check whether the local server health endpoint is responding.

    Args:
        port: Port for the local server.

    Returns:
        True if the server responds with HTTP 200.
    """
    url = f"http://localhost:{port}/health"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=1) as response:
        return response.status == 200


@dataclass
class ServerAttributes:
    """Track server address and running state."""

    url: str | None = None
    open: bool = False

    def start(self, port: int | str):
        """Mark the server as running and set its base API URL."""
        self.open = True
        self.url = f"http://localhost:{port}/v1"

    def stop(self):
        """Mark the server as stopped and clear its base API URL."""
        self.url = None
        self.open = False


@dataclass
class ModelArgs:
    """Optional model runtime overrides for inference."""

    temp: float | None = None
    ngl: int | None = None
    max_tokens: int | None = None
    threads: int | None = None
    ctx_size: int | None = None


class RamalamaModelBase(ABC):
    """Base class for Ramalama model sessions."""

    def __init__(
        self,
        model: str,
        base_image: str | None = None,
        temp: float | None = None,
        ngl: int | None = None,
        max_tokens: int | None = None,
        threads: int | None = None,
        ctx_size: int | None = None,
        timeout: int = 30,
    ):
        """Initialize a model session.

        Args:
            model: Model name or identifier.
            base_image: Container image to use for serving, if different from config.
            temp: Temperature override for sampling.
            ngl: GPU layers override.
            max_tokens: Maximum tokens for completions.
            threads: CPU threads override.
            ctx_size: Context window override.
            timeout: Seconds to wait for server readiness.
        """
        self.model_name = model
        self.base_image = base_image or ramalama_conf.image
        self.timeout = timeout

        self.transport: Transport | None = None
        self.server_attributes: ServerAttributes = ServerAttributes()
        self.process = None

        self.model_args = ModelArgs(temp=temp, ngl=ngl, max_tokens=max_tokens, threads=threads, ctx_size=ctx_size)

    @cached_property
    def args(self):
        """Create and memoize CLI arguments used to start the server."""
        args = SimpleNamespace(
            engine=ramalama_conf.engine,
            container=True,
            store=ramalama_conf.store,
            runtime=ramalama_conf.runtime,
            subcommand="serve",
            MODEL=self.model_name,
            dryrun=False,
            generate=None,
            noout=False,
            image=self.base_image,
            host=ramalama_conf.host,
            context=self.model_args.ngl or ramalama_conf.ctx_size,
            threads=self.model_args.ngl or ramalama_conf.threads,
            ngl=self.model_args.ngl or ramalama_conf.ngl,
            temp=self.model_args.ngl or ramalama_conf.temp,
            max_tokens=self.model_args.ngl or ramalama_conf.max_tokens,
            cache_reuse=ramalama_conf.cache_reuse,
            webui="off",
            thinking=ramalama_conf.thinking,
            runtime_args=[],
            seed=None,
            debug=False,
            model_draft=None,
            api=None,
            quiet=True,
            name="",
        )
        args.port = compute_serving_port(args)
        args.pull = "always"
        return args


def make_chat_request(model: RamalamaModelBase, message: str, history: list[ChatMessage] | None = None) -> ChatMessage:
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
    if not model.server_attributes.open:
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

    url = f"{model.server_attributes.url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    response = urllib.request.urlopen(request)
    response = json.loads(response.read())

    result = ChatMessage(role="assistant", content=response["choices"][0]["message"]["content"])
    return result


class RamalamaModel(RamalamaModelBase):
    """Synchronous Ramalama model interface."""

    def start_server(self):
        """Start the model server in the background.

        Raises:
            RamalamaNoContainerManagerError: If no container manager is available.
        """
        if ramalama_conf.engine is None:
            raise RamalamaNoContainerManagerError(
                "No detected container manager installed on this system.\n"
                "Please install either docker or podman to proceed"
            )

        if self.transport is None:
            self.transport = New(self.model_name, self.args)

        self.transport.ensure_model_exists(self.args)
        cmd = assemble_command(self.args)
        self.process = self.transport.serve_nonblocking(self.args, cmd)
        self.server_attributes.start(self.args.port)

    def cleanup(self):
        """Stop the server process and clean up any container state."""
        if self.transport and self.process:
            if self.args.container and self.args.name:
                args = copy.copy(self.args)
                args.ignore = True
                stop_container(args, self.args.name, remove=True)
            else:
                self.transport._cleanup_server_process(self.process)
        elif self.args.container and self.args.name:
            args = copy.copy(self.args)
            args.ignore = True
            stop_container(args, self.args.name, remove=True)
        self.server_attributes.stop()
        self.process = None

    def serve(self):
        """Start the server and wait until it reports healthy.

        Raises:
            RamalamaNoContainerManagerError: If no container manager is available.
            RamalamaServerTimeoutError: If the server is not healthy before timeout.
        """
        self.start_server()

        start_time = time.time()

        while True:
            try:
                if is_server_healthy(self.args.port):
                    break
            except (ConnectionError, HTTPException, OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                pass
            if time.time() - start_time > self.timeout:
                self.cleanup()
                raise RamalamaServerTimeoutError(f"Server failed to become healthy within {self.timeout} seconds")
            time.sleep(0.1)

    def stop(self):
        """Stop the server and release resources."""
        self.cleanup()

    def download(self) -> bool:
        """Ensure the model is downloaded locally.

        Returns:
            True once the model is available locally.
        """
        if self.transport is None:
            self.transport = New(self.model_name, self.args)

        if self.transport.exists():
            return True
        self.transport.pull(self.args)
        return True

    def chat(self, message: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """Send a chat completion request.

        Args:
            message: User prompt content.
            history: Optional prior conversation messages.

        Returns:
            Assistant message payload.

        Raises:
            RuntimeError: If the server is not running.
        """
        return make_chat_request(self, message, history)

    def __enter__(self):
        """Context manager entry that starts the server."""
        self.serve()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit that stops the server."""
        self.stop()
        return False


class AsyncRamalamaModel(RamalamaModelBase):
    """Asyncio-friendly Ramalama model interface."""

    def start_server(self):
        """Start the model server in the background.

        Raises:
            RamalamaNoContainerManagerError: If no container manager is available.
        """
        if ramalama_conf.engine is None:
            raise RamalamaNoContainerManagerError(
                "No detected container manager installed on this system.\n"
                "Please install either docker or podman to proceed"
            )

        if self.transport is None:
            self.transport = New(self.model_name, self.args)

        self.transport.ensure_model_exists(self.args)
        cmd = assemble_command(self.args)
        self.process = self.transport.serve_nonblocking(self.args, cmd)
        self.server_attributes.start(self.args.port)

    def cleanup(self):
        """Stop the server process and clean up any container state."""
        if self.transport and self.process:
            if self.args.container and self.args.name:
                args = copy.copy(self.args)
                args.ignore = True
                stop_container(args, self.args.name, remove=True)
            else:
                self.transport._cleanup_server_process(self.process)
        elif self.args.container and self.args.name:
            args = copy.copy(self.args)
            args.ignore = True
            stop_container(args, self.args.name, remove=True)
        self.server_attributes.stop()
        self.process = None

    async def serve(self):
        """Start the server and wait until it reports healthy.

        Raises:
            RamalamaNoContainerManagerError: If no container manager is available.
            RamalamaServerTimeoutError: If the server is not healthy before timeout.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.start_server)

        async def wait_healthy(interval: float = 0.1):
            while True:
                try:
                    if is_server_healthy(self.args.port):
                        break
                except (ConnectionError, HTTPException, OSError, UnicodeDecodeError, json.JSONDecodeError):
                    pass
                await asyncio.sleep(interval)

        try:
            await asyncio.wait_for(wait_healthy(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self.cleanup()
            raise RamalamaServerTimeoutError(f"Server failed to become healthy within {self.timeout} seconds")

    async def stop(self):
        """Stop the server and release resources."""
        self.cleanup()

    async def download(self) -> bool:
        """Ensure the model is downloaded locally.

        Returns:
            True once the model is available locally.
        """
        if self.transport is None:
            self.transport = New(self.model_name, self.args)

        if self.transport.exists():
            return True

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.transport.pull, self.args)
        return True

    async def chat(self, message: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """Send a chat completion request.

        Args:
            message: User prompt content.
            history: Optional prior conversation messages.

        Returns:
            Assistant message payload.

        Raises:
            RuntimeError: If the server is not running.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, make_chat_request, self, message, history)

    async def __aenter__(self):
        """Async context manager entry that starts the server."""
        await self.serve()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit that stops the server."""
        await self.stop()
        return False
