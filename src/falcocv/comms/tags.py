"""Thread-safe latest-value store for communications tags."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .base import TagValue

TagCallback = Callable[[str, TagValue], None]


class TagStore:
    """Keep the last known value and quality for every tag.

    The store is deliberately independent of the drivers.  Widgets and future
    web endpoints can consume snapshots or subscribe without knowing whether a
    value came from the dummy driver, Modbus or a PLC.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._values: dict[str, TagValue] = {}
        self._listeners: dict[str, list[TagCallback]] = {}
        self._any_listeners: list[TagCallback] = []

    def set(
        self,
        tag_id: str,
        value,
        *,
        quality: str = "good",
        error: str = "",
    ) -> TagValue:
        """Store a value and notify subscribers."""

        tag_value = TagValue(
            value=value,
            quality=quality,
            timestamp=time.time(),
            error=error,
        )
        with self._lock:
            self._values[tag_id] = tag_value
            callbacks = list(self._listeners.get(tag_id, ()))
            callbacks.extend(self._any_listeners)
        for callback in callbacks:
            try:
                callback(tag_id, tag_value)
            except Exception:
                # A broken UI/listener must not break device I/O.
                pass
        return tag_value

    def get(self, tag_id: str) -> TagValue | None:
        with self._lock:
            return self._values.get(tag_id)

    def snapshot(self) -> dict[str, TagValue]:
        with self._lock:
            return dict(self._values)

    def on_change(self, tag_id: str | None, callback: TagCallback):
        """Register a callback.  ``None`` subscribes to every tag.

        Returns an unsubscribe function.
        """

        with self._lock:
            if tag_id is None:
                self._any_listeners.append(callback)
                listeners = self._any_listeners
                target = None
            else:
                listeners = self._listeners.setdefault(tag_id, [])
                target = tag_id

        def unsubscribe():
            with self._lock:
                if target is None:
                    if callback in self._any_listeners:
                        self._any_listeners.remove(callback)
                else:
                    current = self._listeners.get(target)
                    if current and callback in current:
                        current.remove(callback)
                    if current == []:
                        self._listeners.pop(target, None)

        return unsubscribe

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
