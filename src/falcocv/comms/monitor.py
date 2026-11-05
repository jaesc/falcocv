"""Synchronous monitor session used by the HMI Designer signal monitor.

The session is deliberately UI-free.  The designer runs ``refresh``/``write``
from a worker thread so a slow PLC cannot freeze the Qt event loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import CommsConfig, CommsError, TagDef, TagValue
from .manager import CommsManager


class CommsMonitorSession:
    """A lightweight read/write facade over one :class:`CommsManager`."""

    def __init__(self, config: CommsConfig, log: Callable[[str, str], None] | None = None):
        self.config = config
        self.manager = CommsManager(config, log=log)
        self.tags: tuple[TagDef, ...] = tuple(
            tag for device in config.devices for tag in device.tags
        )
        self._read_groups: dict[str, list[str]] = {}
        for tag in self.tags:
            if tag.access in ("r", "rw"):
                self._read_groups.setdefault(tag.device, []).append(tag.id)

    def start(self) -> None:
        self.manager.start()

    def stop(self) -> None:
        self.manager.stop()

    def refresh(self) -> dict[str, TagValue]:
        """Read all readable tags and return the current tag-store snapshot."""

        for tag_ids in self._read_groups.values():
            try:
                self.manager.read_many(tag_ids)
            except CommsError:
                # The failed device is marked bad in the store; keep showing
                # the other devices instead of aborting the whole refresh.
                continue
        return self.manager.snapshot()

    def read(self, tag_id: str) -> TagValue:
        return self.manager.read_tag(tag_id)

    def write(self, tag_id: str, value: Any) -> None:
        self.manager.write(tag_id, value)

    def statuses(self):
        return self.manager.statuses()

    def snapshot(self) -> dict[str, TagValue]:
        return self.manager.snapshot()

    def on_change(self, tag_id: str | None, callback):
        return self.manager.on_change(tag_id, callback)
