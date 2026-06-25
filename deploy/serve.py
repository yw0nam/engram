"""Container entrypoint: engram HTTP API wired to LOCAL OpenAI-compatible models, reasoning DISABLED.

Why a bootstrap (not just env vars): engram's built-in provider resolution routes chat + embeddings
through one global OpenAI base URL, but here the LLM (:5535) and embedder (:5502) are different ports.
So we build both clients explicitly with stdlib urllib (no litellm / torch in the image) and inject them
into the MemoryService singleton the FastAPI app already uses.

    uvicorn deploy.serve:app
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

from engram.llm.base import LLM
from engram.embed.base import Embedder

LLM_BASE = os.environ.get("ENGRAM_LLM_BASE_URL", "http://192.168.0.50:5535/v1")
LLM_MODEL = os.environ.get("ENGRAM_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
EMB_BASE = os.environ.get("ENGRAM_EMBED_BASE_URL", "http://192.168.0.50:5502/v1")
EMB_MODEL = os.environ.get("ENGRAM_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
EMB_DIM = int(os.environ.get("ENGRAM_EMBED_DIM", "2048"))
# Qwen3 disables CoT via chat_template_kwargs.enable_thinking=False (vLLM). Default off per request.
THINKING = os.environ.get("ENGRAM_LLM_THINKING", "off").lower() in ("1", "on", "true")


def _post(url: str, payload: dict, timeout: float = 180.0, attempts: int = 5) -> dict:
    body = json.dumps(payload).encode()
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e  # transient (model warming up / blip) -> backoff and retry
            time.sleep(min(2 ** i, 15))
    raise last  # type: ignore[misc]


class LocalLLM(LLM):
    """Chat model with reasoning off (Qwen enable_thinking=False)."""
    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": LLM_MODEL, "messages": messages,
            "temperature": kwargs.get("temperature", 0.0), "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if not THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return _post(f"{LLM_BASE}/chat/completions", payload)["choices"][0]["message"]["content"] or ""


class LocalEmbedder(Embedder):
    def __init__(self) -> None:
        self.dim = EMB_DIM  # lazy: avoids a boot-time network dependency; updated on first real embed
    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out = _post(f"{EMB_BASE}/embeddings", {"model": EMB_MODEL, "input": list(texts)})
        vecs = [d["embedding"] for d in out["data"]]
        if vecs:
            self.dim = len(vecs[0])
        return vecs


# Build the app, then swap in the local clients. ENGRAM_EMBEDDER=hashing keeps MemoryService.__init__
# dependency-free (no BGE download); we overwrite the embedder/llm right after.
os.environ.setdefault("ENGRAM_EMBEDDER", "hashing")
os.environ.setdefault("ENGRAM_OPEN", "1")
os.environ.setdefault("ENGRAM_DATA_DIR", "/data/engram")

from engram.server.app import app, svc  # noqa: E402

_service = svc()
_service.embedder = LocalEmbedder()
_service.llm = LocalLLM()
_service.answerer = _service.llm
