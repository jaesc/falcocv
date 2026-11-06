"""Base classes for server-side communications runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ServerRuntimeError(RuntimeError):
    """A server runtime could not start or is unavailable."""


@dataclass
class ServerStatus:
    id: str
    type: str
    enabled: bool
    running: bool = False
    last_error: str = ""


class ServerRuntime(ABC):
    """A protocol server that exposes tags from a CommsManager."""

    server_type = ""

    def __init__(self, config, manager, log=None):
        self.config = config
        self.manager = manager
        self._log = log
        self.status = ServerStatus(
            id=config.id,
            type=self.server_type,
            enabled=config.enabled,
        )

    def log(self, message: str, level: str = "info"):
        callback = self._log
        if callback is None:
            return
        try:
            callback(message, level)
        except Exception:
            pass

    @abstractmethod
    def start(self) -> None:
        """Start the server."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the server."""

    @property
    def running(self) -> bool:
        return self.status.running
