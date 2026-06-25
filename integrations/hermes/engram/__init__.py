"""Engram memory plugin — Hermes MemoryProvider over engram's HTTP API.

Engram is a bi-temporal memory engine (System-1 fast write / System-2 async
consolidation). This provider is a thin HTTP client: recall is injected into
context each turn (engram already surfaces the supersession chain — "current X
· prev Y until T" — by default), and each turn is written back for
consolidation. Context-only: no tools; memory shows up unasked.

Namespace = bearer token (engram's ENGRAM_OPEN mode: bearer text == namespace),
so one engram server serves many isolated agents/profiles.

Config via $HERMES_HOME/engram/config.json or env:
  ENGRAM_URL        — base URL of the engram server (default http://127.0.0.1:9178)
  ENGRAM_NAMESPACE  — memory namespace / bearer (default: agent_identity, else "hermes")
  ENGRAM_TIMEOUT    — request timeout seconds (default 30)
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:
    from hermes_constants import get_hermes_home
except Exception:  # noqa: BLE001 -- standalone/test import without the Hermes runtime
    def get_hermes_home() -> Path:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://127.0.0.1:9178"
_WRITER_SENTINEL = object()


def _load_config() -> dict:
    """$HERMES_HOME/engram/config.json, then env. Env wins (explicit override)."""
    cfg: dict = {}
    path = get_hermes_home() / "engram" / "config.json"
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            logger.warning("engram: bad config.json (%s); ignoring", e)
    if os.environ.get("ENGRAM_URL"):
        cfg["url"] = os.environ["ENGRAM_URL"]
    if os.environ.get("ENGRAM_NAMESPACE"):
        cfg["namespace"] = os.environ["ENGRAM_NAMESPACE"]
    if os.environ.get("ENGRAM_TIMEOUT"):
        cfg["timeout"] = os.environ["ENGRAM_TIMEOUT"]
    return cfg


class EngramMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._cfg = _load_config()
        self._url = str(self._cfg.get("url", _DEFAULT_URL)).rstrip("/")
        self._namespace = self._cfg.get("namespace", "")
        try:
            self._timeout = float(self._cfg.get("timeout", 30))
        except (TypeError, ValueError):
            self._timeout = 30.0
        self._session_id = ""
        self._write_ok = True  # primary-context only; skipped for cron/subagent
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._writer_q: "queue.Queue" = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._shutting_down = threading.Event()

    @property
    def name(self) -> str:
        return "engram"

    def is_available(self) -> bool:
        # No network here (per ABC). A base URL always resolves (defaulted); selection is already gated by
        # the memory.provider config key, so being configured == ready. Failures degrade gracefully below.
        return bool(self._url)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        # namespace precedence: explicit config/env > per-profile identity > "hermes"
        if not self._namespace:
            self._namespace = kwargs.get("agent_identity") or "hermes"
        # only the primary agent should write — cron/subagent/flush turns would pollute the user's memory
        ctx = kwargs.get("agent_context", "primary")
        self._write_ok = ctx == "primary"
        self._ensure_writer()
        logger.info("engram: ready (url=%s namespace=%s write=%s)", self._url, self._namespace, self._write_ok)

    def system_prompt_block(self) -> str:
        return (
            "# Engram Memory\n"
            f"Active. Namespace: {self._namespace}. Relevant long-term memory is injected into context "
            "automatically — including how facts changed over time (current value and what it replaced)."
        )

    # -- HTTP ----------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self._url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._namespace}"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as r:  # noqa: S310 -- self-hosted URL
            return json.load(r)

    # -- recall (inject) -----------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._shutting_down.is_set() or not query.strip():
            return

        def _run() -> None:
            try:
                ctx = self._post("/v1/recall", {"query": query, "context_only": True}).get("context", "")
                with self._prefetch_lock:
                    self._prefetch_result = ctx or ""
            except Exception as e:  # noqa: BLE001 -- recall must never break the turn
                logger.debug("engram prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="engram-prefetch")
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result, self._prefetch_result = self._prefetch_result, ""
        if not result.strip():
            # no queued result (first turn) -> recall inline so the very first turn still has memory
            try:
                result = self._post("/v1/recall", {"query": query, "context_only": True}).get("context", "")
            except Exception as e:  # noqa: BLE001
                logger.debug("engram inline recall failed: %s", e)
                return ""
        return result.strip()

    # -- write (consolidate) -------------------------------------------------

    def _ensure_writer(self) -> None:
        if self._writer_thread and self._writer_thread.is_alive():
            return
        self._shutting_down.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="engram-writer")
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        while True:
            try:
                job = self._writer_q.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    return
                self._post("/v1/remember", {"content": job, "scope": "auto", "session_id": self._session_id})
            except Exception as e:  # noqa: BLE001 -- a write failure must not kill the writer
                logger.debug("engram remember failed: %s", e)
            finally:
                self._writer_q.task_done()

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        if not self._write_ok or self._shutting_down.is_set() or not user_content.strip():
            return
        # one episode per turn; engram's extractor reads the multi-turn context to resolve coreference
        self._writer_q.put(f"User: {user_content.strip()}\nAssistant: {assistant_content.strip()}")

    # -- tools: none (context-only; memory is injected, not queried) ----------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        self._shutting_down.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_q.put(_WRITER_SENTINEL)
            self._writer_thread.join(timeout=5.0)
