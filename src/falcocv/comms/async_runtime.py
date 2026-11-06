"""Small asyncio loop bridge for synchronous communications drivers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRuntime:
    """Run coroutines from a dedicated background event loop.

    OPC UA client and server objects are tied to an event loop.  Each runtime
    owns its own loop so a synchronous server callback cannot deadlock by
    submitting work to the same loop it is already running on.
    """

    def __init__(self, name: str = "falcocv-comms"):
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def start(self):
        with self._lock:
            if self._loop is not None:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop,),
                name=f"{self._name}-loop",
                daemon=True,
            )
            self._loop = loop
            self._thread = thread
            thread.start()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, T], timeout: float = 10.0) -> T:
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("async runtime is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"async communications operation timed out after {timeout:g}s"
            ) from None

    def stop(self):
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=2.0)
        try:
            loop.close()
        except Exception:
            pass
