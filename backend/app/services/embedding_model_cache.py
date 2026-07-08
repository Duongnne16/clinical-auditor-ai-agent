from __future__ import annotations

import os
from threading import Lock
from typing import Any

from backend.app.core.config import get_settings


_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = Lock()


def get_sentence_transformer(model_name: str) -> Any:
    """Return a process-wide SentenceTransformer instance.

    Loading the same model more than once can crash native torch/transformers code
    on some Windows setups, so retrieval services share one cached instance.
    """

    if get_settings().disable_local_embeddings:
        raise RuntimeError("local embeddings are disabled")

    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "true")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required") from exc

        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        return model
