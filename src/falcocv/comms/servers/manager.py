"""Lifecycle manager for configured protocol servers."""

from __future__ import annotations

from typing import Any

from .base import ServerRuntime, ServerRuntimeError, ServerStatus
from .config import (
    ModbusTcpServerConfig,
    OpcUaServerConfig,
    ServerConfigError,
    parse_servers_config,
)
from .modbus import ModbusTcpServerRuntime
from .opcua import OpcUaServerRuntime


class ServerManager:
    """Start and stop all enabled server runtimes for one HMI project."""

    def __init__(self, comms, config: Any, log=None):
        self.comms = comms
        self.config = config
        self._log = log
        self._runtimes: list[ServerRuntime] = []
        self._statuses: list[ServerStatus] = []

    @property
    def statuses(self) -> tuple[ServerStatus, ...]:
        return tuple(self._statuses)

    def _log_message(self, message: str, level: str = "info"):
        if self._log is None:
            return
        try:
            self._log(message, level)
        except Exception:
            pass

    def start(self) -> None:
        if self._runtimes:
            return
        try:
            configs = parse_servers_config(self.config)
        except ServerConfigError as exc:
            raise ServerRuntimeError(str(exc)) from exc
        for config in configs:
            if not config.enabled:
                self._statuses.append(
                    ServerStatus(
                        id=config.id,
                        type=type(config).__name__,
                        enabled=False,
                    )
                )
                continue
            if isinstance(config, ModbusTcpServerConfig):
                runtime = ModbusTcpServerRuntime(
                    config, self.comms, self._log_message
                )
            elif isinstance(config, OpcUaServerConfig):
                runtime = OpcUaServerRuntime(
                    config, self.comms, self._log_message
                )
            else:
                continue
            self._runtimes.append(runtime)
            self._statuses.append(runtime.status)
            try:
                runtime.start()
            except Exception as exc:
                runtime.status.last_error = str(exc)
                self._log_message(
                    f"Server '{config.id}' failed to start: {exc}", "error"
                )
        if self._runtimes:
            self._log_message(
                f"Comms servers started: {len(self._runtimes)} runtime(s)"
            )

    def stop(self) -> None:
        for runtime in reversed(self._runtimes):
            try:
                runtime.stop()
            except Exception as exc:
                self._log_message(
                    f"Server '{runtime.config.id}' stop failed: {exc}",
                    "warning",
                )
        self._runtimes.clear()
