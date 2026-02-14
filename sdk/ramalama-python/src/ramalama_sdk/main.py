"""Ramalama Python SDK public interface for starting a model server and chatting."""

import asyncio
import copy
import json
import time
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from http.client import HTTPConnection, HTTPException
from types import SimpleNamespace

from ramalama.command.factory import assemble_command
from ramalama.config import Config, get_config
from ramalama.engine import inspect as inspect_container
from ramalama.engine import logs as container_logs
from ramalama.engine import stop_container
from ramalama.model_store.global_store import GlobalModelStore
from ramalama.shortnames import Shortnames
from ramalama.transports.api import APITransport
from ramalama.transports.base import Transport
from ramalama.transports.transport_factory import New

from ramalama_sdk.config import get_sdk_config
from ramalama_sdk.errors import RamalamaNoContainerManagerError, RamalamaServerTimeoutError
from ramalama_sdk.schemas import ChatMessage, ModelRecord
from ramalama_sdk.utils import ServerAttributes, list_models, make_chat_request


def is_healthy(args, timeout: float = 3, model_name: str | None = None, host: str | None = None) -> bool:
    """Check server readiness using SDK host semantics.

    The check probes `/health` and `/models` on the selected host. For llama.cpp
    style runtimes, a `404` health response is treated as healthy to match
    upstream behavior.

    Args:
        args: Serve argument namespace used by the transport.
        timeout: Socket timeout in seconds for HTTP requests.
        model_name: Optional model alias to look for in `/models`. When omitted,
            the alias is computed from `args.MODEL`.
        host: Optional host override for connectivity checks.

    Returns:
        True when the server is reachable and advertises the expected model.
    """
    conn = None
    try:
        target_host = host or getattr(args, "host", None) or "127.0.0.1"
        conn = HTTPConnection(target_host, args.port, timeout=timeout)
        if getattr(args, "debug", False):
            conn.set_debuglevel(1)

        runtime = getattr(args, "runtime", None) or get_config().runtime
        if runtime == "vllm":
            conn.request("GET", "/ping")
            vllm_ping_resp = conn.getresponse()
            return vllm_ping_resp.status == 200

        conn.request("GET", "/health")
        health_resp = conn.getresponse()
        health_resp.read()
        if health_resp.status not in (200, 404):
            return False

        conn.request("GET", "/models")
        models_resp = conn.getresponse()
        if models_resp.status != 200:
            return False

        content = models_resp.read()
        if not content:
            return False

        body = json.loads(content)
        if "models" not in body:
            return False

        model_names = [m["name"] for m in body["models"]]
        if not model_name:
            model_name = New(args.MODEL, args).model_alias

        return any(model_name in name for name in model_names)
    finally:
        if conn:
            conn.close()


@dataclass
class ModelArgs:
    """Optional model runtime overrides for inference."""

    temp: float | None = None
    ngl: int | None = None
    max_tokens: int | None = None
    threads: int | None = None
    ctx_size: int | None = None
    container: bool | None = None
    pull: str = "missing"
    runtime_args: list[str] = field(default_factory=list)
    no_jinja: bool = False


class ModelStore:
    def __init__(self, store_path: str | None = None, engine: str | None = None, conf: Config | None = None):
        self.conf = conf or get_config()
        self.model_store = GlobalModelStore(store_path or self.conf.store)

        if (resolved_engine := engine or self.conf.engine) is None:
            message = "No container manager was provided or detected on the system. Please pass an `engine` value"
            raise Exception(message)

        self.engine = resolved_engine

    def list_models(self) -> list[ModelRecord]:
        """List locally available models, sorted by most recently modified first.

        Returns:
            Models available in the local store, newest first.
        """
        return list_models(self.model_store, self.engine)


@lru_cache
def get_shortname_helper() -> Shortnames:
    """Return a cached Ramalama shortname resolver."""
    return Shortnames()


def resolve_shortnames(model: str) -> str:
    """Resolve a model shortname to its canonical model reference."""
    shortnames = get_shortname_helper()
    return shortnames.resolve(model)


class RamalamaModelBase(ABC):
    """Base class for Ramalama model sessions."""

    def __init__(
        self,
        model: str,
        scheme: str = "http",
        health_path: str | None = None,
        chat_path: str | None = None,
        base_image: str | None = None,
        temp: float | None = None,
        ngl: int | None = None,
        max_tokens: int | None = None,
        threads: int | None = None,
        ctx_size: int | None = None,
        timeout: float = 30,
        config: Config | None = None,
        container: bool | None = None,
        pull: str = "missing",
        runtime_args: list[str] | None = None,
        no_jinja: bool = False,
    ):
        """Initialize a model session.

        Host values (`bind_host`/`connect_host`) are sourced from global SDK config.
        Model names are resolved through Ramalama shortnames before transport
        selection.

        Args:
            model: Model name or identifier, including optional shortnames.
            base_image: Container image to use for serving, if different from config.
            temp: Temperature override for sampling.
            ngl: GPU layers override.
            max_tokens: Maximum tokens for completions.
            threads: CPU threads override.
            ctx_size: Context window override.
            timeout: Seconds to wait for server readiness.
            runtime_args: Additional runtime flags passed through to the inference server.
            no_jinja: Disable Jinja chat template parsing (`--no-jinja`) at runtime.
        """
        self.conf = config or get_config()
        sdk_config = get_sdk_config()

        self.model_name = resolve_shortnames(model)
        self.base_image = base_image or "quay.io/ramalama/ramalama:0.16.0"  # self.conf.image
        self.bind_host = sdk_config.connection.bind_host
        self.connect_host = sdk_config.connection.connect_host

        self.timeout = timeout

        self._transport: Transport | None = None

        serve_args = {}
        if health_path is not None:
            serve_args.setdefault("health_path", health_path)
        if chat_path is not None:
            serve_args.setdefault("chat_path", chat_path)
        self.server_attributes: ServerAttributes = ServerAttributes(scheme=scheme, host=self.connect_host, **serve_args)

        self.process = None

        self.model_args = ModelArgs(
            temp=temp,
            ngl=ngl,
            max_tokens=max_tokens,
            threads=threads,
            ctx_size=ctx_size,
            container=container,
            pull=pull,
            runtime_args=list(runtime_args) if runtime_args is not None else [],
            no_jinja=no_jinja,
        )

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            transport = New(self.model_name, self.args)
            if isinstance(transport, APITransport):
                raise NotImplementedError("API transports not yet supported by the SDK")

            self._transport = transport

        return self._transport

    @cached_property
    def args(self):
        """Create and memoize CLI arguments used to start the server."""
        args = SimpleNamespace(
            engine=self.conf.engine,
            container=self.model_args.container if self.model_args.container is not None else True,
            store=self.conf.store,
            runtime=self.conf.runtime,
            subcommand="serve",
            MODEL=self.model_name,
            dryrun=False,
            generate=None,
            noout=True,
            image=self.base_image,
            host=self.bind_host,
            context=self.model_args.ctx_size or self.conf.ctx_size,
            threads=self.model_args.threads or self.conf.threads,
            ngl=self.model_args.ngl or self.conf.ngl,
            temp=self.model_args.temp or self.conf.temp,
            max_tokens=self.model_args.max_tokens or self.conf.max_tokens,
            cache_reuse=self.conf.cache_reuse,
            webui="off",
            thinking=self.conf.thinking,
            runtime_args=list(self.model_args.runtime_args),
            seed=None,
            debug=False,
            model_draft=None,
            api=None,
            quiet=True,
            name="",
        )
        if self.model_args.no_jinja and "--no-jinja" not in args.runtime_args:
            args.runtime_args.append("--no-jinja")
        args.pull = self.model_args.pull
        return args

    def _container_log_tail(self, max_lines: int = 80) -> str:
        if not self.args.name:
            return "Container logs unavailable (container name was not set)."

        try:
            output = container_logs(self.args, self.args.name, ignore_stderr=not self.args.debug).strip()
        except Exception as exc:
            return f"Container logs unavailable ({type(exc).__name__}: {exc})."

        if not output:
            return "Container logs are empty."

        lines = output.splitlines()
        return "\n".join(lines[-max_lines:])

    def _startup_failure(self, start_time: float) -> str | None:
        if self.args.container:
            if not self.args.name:
                return None

            try:
                status = inspect_container(self.args, self.args.name, format="{{ .State.Status }}", ignore_stderr=True)
            except Exception as exc:
                if time.time() - start_time < 0.5:
                    return None

                logs_tail = self._container_log_tail()
                return (
                    f"Container '{self.args.name}' disappeared before becoming healthy. "
                    f"It may have exited and been auto-removed (--rm).\n"
                    f"Inspect error: {type(exc).__name__}: {exc}\n"
                    f"Container logs:\n{logs_tail}"
                )

            status = status.strip()
            if status in {"running", "created", "restarting"}:
                return None

            logs_tail = self._container_log_tail()
            return (
                f"Container '{self.args.name}' entered status '{status}' before becoming healthy.\n"
                f"Container logs:\n{logs_tail}"
            )

        if self.process is None:
            return "Server process was not created."

        if (exit_code := self.process.poll()) is None:
            return None

        return f"Server process exited with code {exit_code} before becoming healthy."


class RamalamaModel(RamalamaModelBase):
    """Synchronous Ramalama model interface."""

    def start_server(self):
        """Start the model server in the background.

        Raises:
            RamalamaNoContainerManagerError: If no container manager is available.
        """
        if self.conf.engine is None:
            raise RamalamaNoContainerManagerError(
                "No detected container manager installed on this system.\n"
                "Please install either docker or podman to proceed"
            )

        try:
            self.transport.ensure_model_exists(self.args)
            self.server_attributes.open()
            self.args.port = self.server_attributes.port
            cmd = assemble_command(self.args)  # type: ignore
            self.process = self.transport.serve_nonblocking(self.args, cmd)

        except Exception:
            self.cleanup()
            raise

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
        self.server_attributes.close()
        try:
            del self.args.port
        except AttributeError:
            pass
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
            if startup_failure := self._startup_failure(start_time):
                self.cleanup()
                raise RuntimeError(startup_failure)

            try:
                if is_healthy(self.args, host=self.connect_host):
                    break
            except (ConnectionError, HTTPException, OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
            except RuntimeError:
                self.cleanup()
                raise
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
        if self.conf.engine is None:
            raise RamalamaNoContainerManagerError(
                "No detected container manager installed on this system.\n"
                "Please install either docker or podman to proceed"
            )

        try:
            self.server_attributes.open()
            self.args.port = self.server_attributes.port
            self.transport.ensure_model_exists(self.args)
            cmd = assemble_command(self.args)  # type: ignore
            self.process = self.transport.serve_nonblocking(self.args, cmd)

        except Exception:
            self.cleanup()
            raise

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
        self.server_attributes.close()
        try:
            del self.args.port
        except AttributeError:
            pass
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
                if startup_failure := self._startup_failure(start_time):
                    raise RuntimeError(startup_failure)

                try:
                    if is_healthy(self.args, host=self.connect_host):
                        break
                except (ConnectionError, HTTPException, OSError, UnicodeDecodeError, json.JSONDecodeError):
                    pass
                await asyncio.sleep(interval)

        start_time = time.time()

        try:
            await asyncio.wait_for(wait_healthy(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self.cleanup()
            raise RamalamaServerTimeoutError(f"Server failed to become healthy within {self.timeout} seconds")
        except RuntimeError:
            self.cleanup()
            raise

    async def stop(self):
        """Stop the server and release resources."""
        self.cleanup()

    async def download(self) -> bool:
        """Ensure the model is downloaded locally.

        Returns:
            True once the model is available locally.
        """
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
