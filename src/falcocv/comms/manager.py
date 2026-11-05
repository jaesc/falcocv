"""Runtime manager for communications devices and tags."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .base import (
    CommsConfig,
    CommsError,
    DeviceConfig,
    DeviceDriver,
    DeviceStatus,
    TagDef,
    TagValue,
    coerce_tag_value,
)
from .drivers import create_driver
from .tags import TagStore

LogCallback = Callable[[str, str], None]


@dataclass
class _RuntimeDevice:
    config: DeviceConfig
    driver: DeviceDriver
    status: DeviceStatus
    lock: threading.RLock = field(default_factory=threading.RLock)


class CommsManager:
    """Own the lifecycle and public API of all configured devices.

    The manager is intentionally synchronous from the caller's point of view.
    A background thread only performs optional polling for tags with
    ``poll_ms > 0``.
    """

    def __init__(self, config: CommsConfig, log: LogCallback | None = None):
        self.config = config
        self._log_callback = log
        self._store = TagStore()
        self._lock = threading.RLock()
        self._devices: dict[str, _RuntimeDevice] = {}
        self._statuses: dict[str, DeviceStatus] = {}
        self._tags: dict[str, TagDef] = {}
        self._running = False
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

    @property
    def store(self) -> TagStore:
        return self._store

    @property
    def tags(self) -> tuple[TagDef, ...]:
        """Snapshot of the active tag definitions."""

        with self._lock:
            return tuple(self._tags.values())

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._devices.clear()
            self._statuses.clear()
            self._tags.clear()
            self._store.clear()

            for device_config in self.config.devices:
                status = DeviceStatus(
                    id=device_config.id,
                    driver=device_config.driver,
                )
                self._statuses[device_config.id] = status
                for tag in device_config.tags:
                    self._tags[tag.id] = tag

                if not device_config.enabled:
                    status.last_error = "device disabled"
                    self._store.set(
                        tag.id,
                        tag.default,
                        quality="disabled",
                        error=status.last_error,
                    )
                    continue

                try:
                    driver = create_driver(device_config)
                    driver.start()
                except Exception as exc:
                    status.last_error = str(exc)
                    for tag in device_config.tags:
                        self._store.set(
                            tag.id,
                            tag.default,
                            quality="bad",
                            error=status.last_error,
                        )
                    self._log(
                        f"Comms device '{device_config.id}' failed to start: {exc}",
                        "error",
                    )
                    continue

                status.connected = driver.is_connected()
                if not status.connected:
                    status.last_error = "device is not connected"
                runtime = _RuntimeDevice(
                    config=device_config,
                    driver=driver,
                    status=status,
                )
                self._devices[device_config.id] = runtime
                for tag in device_config.tags:
                    self._store.set(
                        tag.id,
                        tag.default,
                        quality="good" if status.connected else "bad",
                        error="" if status.connected else status.last_error,
                    )
                self._log(
                    f"Comms device '{device_config.id}' "
                    f"({'connected' if status.connected else 'disconnected'})",
                    "info" if status.connected else "warning",
                )

            self._running = True
            self._start_polling()
            self._log(
                f"Comms started: {len(self._devices)} active device(s), "
                f"{len(self._tags)} tag(s)",
                "info",
            )

    def stop(self) -> None:
        with self._lock:
            if not self._running and not self._devices:
                self._store.clear()
                return
            self._running = False
            self._poll_stop.set()

        self._join_poll_thread()

        with self._lock:
            for runtime in self._devices.values():
                try:
                    runtime.driver.stop()
                except Exception as exc:
                    self._log(
                        f"Comms device '{runtime.config.id}' stop failed: {exc}",
                        "warning",
                    )
                runtime.status.connected = False
            self._devices.clear()
            self._tags.clear()
            self._store.clear()
            # Keep statuses available for a final status read, but mark them
            # disconnected until the manager is started again.
            for status in self._statuses.values():
                status.connected = False
            self._log("Comms stopped", "info")

    def _start_polling(self) -> None:
        tags = [
            tag
            for tag in self._tags.values()
            if tag.poll_ms > 0
        ]
        if not tags:
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(tags,),
            name="falcocv-comms-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def _join_poll_thread(self) -> None:
        thread = self._poll_thread
        self._poll_thread = None
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _poll_loop(self, tags: list[TagDef]) -> None:
        next_due = {tag.id: time.monotonic() for tag in tags}
        while not self._poll_stop.is_set():
            now = time.monotonic()
            due_by_device: dict[str, list[TagDef]] = {}
            for tag in tags:
                if now >= next_due.get(tag.id, now):
                    due_by_device.setdefault(tag.device, []).append(tag)
            for device_tags in due_by_device.values():
                if self._poll_stop.is_set():
                    return
                try:
                    self._read_tags(device_tags)
                except CommsError:
                    # The store was already marked bad by the read helper.
                    pass
                for tag in device_tags:
                    interval = max(tag.poll_ms, 1) / 1000.0
                    next_due[tag.id] = time.monotonic() + interval
            sleep_for = 0.02
            if next_due:
                sleep_for = min(
                    sleep_for,
                    max(0.0, min(next_due.values()) - time.monotonic()),
                )
            self._poll_stop.wait(sleep_for or 0.001)

    # ── API ───────────────────────────────────────────────────────────────

    def read(self, tag_id: str) -> Any:
        return self._read_tag_value(tag_id).value

    def read_tag(self, tag_id: str) -> TagValue:
        return self._read_tag_value(tag_id)

    def tag(self, tag_id: str) -> TagValue | None:
        return self._store.get(tag_id)

    def snapshot(self) -> dict[str, TagValue]:
        return self._store.snapshot()

    def on_change(self, tag_id: str | None, callback):
        return self._store.on_change(tag_id, callback)

    def read_many(self, tag_ids: list[str]) -> dict[str, Any]:
        tags: list[TagDef] = []
        seen: set[str] = set()
        for tag_id in tag_ids:
            tag = self._get_tag(tag_id)
            if tag.id not in seen:
                tags.append(tag)
                seen.add(tag.id)

        grouped: dict[str, list[TagDef]] = {}
        for tag in tags:
            grouped.setdefault(tag.device, []).append(tag)

        values: dict[str, Any] = {}
        for device_tags in grouped.values():
            values.update(self._read_tags(device_tags))
        return {tag_id: values[tag_id] for tag_id in tag_ids}

    def write(self, tag_id: str, value: Any) -> None:
        self.write_many({tag_id: value})

    def write_many(self, values: dict[str, Any]) -> None:
        prepared: dict[str, list[tuple[TagDef, Any]]] = {}
        for tag_id, value in values.items():
            tag = self._get_tag(tag_id)
            if tag.access == "r":
                raise CommsError(
                    "tag is read-only",
                    device=tag.device,
                    tag=tag.id,
                )
            prepared.setdefault(tag.device, []).append(
                (tag, coerce_tag_value(tag, value))
            )

        for device_tags in prepared.values():
            tag = device_tags[0][0]
            runtime = self._devices.get(tag.device)
            if runtime is None:
                raise CommsError(
                    "device is not active",
                    device=tag.device,
                    tag=tag.id,
                )
            self._require_running(tag)
            with runtime.lock:
                try:
                    runtime.driver.write_many(device_tags)
                except Exception as exc:
                    message = str(exc)
                    runtime.status.last_error = message
                    for batch_tag, value in device_tags:
                        self._store.set(
                            batch_tag.id,
                            value,
                            quality="bad",
                            error=message,
                        )
                    raise CommsError(
                        message,
                        device=tag.device,
                        tag=tag.id,
                        cause=exc,
                    ) from exc
                runtime.status.last_error = ""
                runtime.status.last_write_ts = time.time()
                for batch_tag, value in device_tags:
                    self._store.set(batch_tag.id, value, quality="good")

    def wait_for(
        self,
        tag_id: str,
        predicate,
        *,
        timeout: float | None = None,
        interval: float = 0.05,
    ) -> TagValue:
        """Wait until a tag value matches ``predicate``.

        ``predicate`` may be a value (compared with ``==``) or a callable
        receiving the current value.
        """

        match = predicate if callable(predicate) else lambda value: value == predicate
        deadline = (
            None if timeout is None else time.monotonic() + max(0.0, timeout)
        )
        last_error: CommsError | None = None
        while True:
            try:
                value = self._read_tag_value(tag_id)
                if match(value.value):
                    return value
                last_error = None
            except CommsError as exc:
                last_error = exc
            if deadline is not None and time.monotonic() >= deadline:
                if last_error is not None:
                    raise last_error
                raise TimeoutError(
                    f"timed out waiting for tag '{tag_id}'"
                )
            time.sleep(max(0.001, interval))

    def status(self, device_id: str | None = None):
        with self._lock:
            if device_id is not None:
                status = self._statuses.get(device_id)
                return replace(status) if status else None
            return [replace(status) for status in self._statuses.values()]

    def statuses(self) -> list[DeviceStatus]:
        return self.status()

    # ── internal helpers ──────────────────────────────────────────────────

    def _read_tag_value(self, tag_id: str) -> TagValue:
        tag = self._get_tag(tag_id)
        self._read_tags([tag])
        value = self._store.get(tag.id)
        if value is None:
            raise CommsError(
                "driver did not return a tag value",
                device=tag.device,
                tag=tag.id,
            )
        return value

    def _read_tags(self, tags: list[TagDef]) -> dict[str, Any]:
        if not tags:
            return {}
        first = tags[0]
        if any(tag.device != first.device for tag in tags):
            raise ValueError("all tags in a batch must belong to one device")
        runtime = self._devices.get(first.device)
        if runtime is None:
            raise CommsError(
                "device is not active",
                device=first.device,
                tag=first.id,
            )
        self._require_running(first)
        with runtime.lock:
            try:
                values = runtime.driver.read_many(tags)
            except Exception as exc:
                message = str(exc)
                runtime.status.last_error = message
                for tag in tags:
                    self._store.set(
                        tag.id,
                        None,
                        quality="bad",
                        error=message,
                    )
                raise CommsError(
                    message,
                    device=first.device,
                    tag=first.id,
                    cause=exc,
                ) from exc

            if not isinstance(values, dict):
                message = "driver read_many must return a dict"
                runtime.status.last_error = message
                for tag in tags:
                    self._store.set(
                        tag.id,
                        None,
                        quality="bad",
                        error=message,
                    )
                raise CommsError(
                    message,
                    device=first.device,
                    tag=first.id,
                )

            missing = [tag for tag in tags if tag.id not in values]
            if missing:
                message = "driver read_many did not return all requested tags"
                runtime.status.last_error = message
                for tag in tags:
                    self._store.set(
                        tag.id,
                        None,
                        quality="bad",
                        error=message,
                    )
                raise CommsError(
                    message,
                    device=first.device,
                    tag=missing[0].id,
                )

            result = {tag.id: values[tag.id] for tag in tags}
            for tag in tags:
                self._store.set(tag.id, result[tag.id], quality="good")

            runtime.status.last_error = ""
            runtime.status.last_read_ts = time.time()
            return result

    def _get_tag(self, tag_id: str) -> TagDef:
        with self._lock:
            tag = self._tags.get(tag_id)
        if tag is None:
            raise CommsError(f"unknown tag '{tag_id}'", tag=tag_id)
        return tag

    def _require_running(self, tag: TagDef) -> None:
        if not self._running:
            raise CommsError(
                "communications manager is not running",
                device=tag.device,
                tag=tag.id,
            )

    def _log(self, message: str, level: str = "info") -> None:
        callback = self._log_callback
        if callback is None:
            return
        try:
            callback(message, level)
        except Exception:
            pass
