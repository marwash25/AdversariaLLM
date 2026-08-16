"""Helpers for fingerprint-keyed torch caches."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def tensor_content_hash(t: Tensor) -> str:
    """Stable content hash of a tensor (dtype, shape, and bytes)."""
    arr = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def fingerprint_value(value: Any) -> Any:
    """Convert a value into a JSON-serializable fingerprint fragment.

    Tensors become content hashes; mappings/sequences are converted recursively;
    ints/floats/bools/strs/None are kept (with float/int normalization).
    """
    if isinstance(value, Tensor):
        return tensor_content_hash(value)
    if isinstance(value, Mapping):
        return {str(k): fingerprint_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [fingerprint_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported fingerprint value type: {type(value)!r}")


def make_fingerprint(parts: Mapping[str, Any]) -> dict:
    """Build a JSON-serializable fingerprint dict from named cache-relevant inputs."""
    return {str(k): fingerprint_value(v) for k, v in parts.items()}


def fingerprint_hash8(fingerprint: Mapping[str, Any]) -> str:
    """First 8 hex chars of a stable hash of the fingerprint dict."""
    payload = json.dumps(dict(fingerprint), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def load_torch_cache_if_fingerprint_matches(
    path: Path,
    fingerprint: Mapping[str, Any],
    map_location=None,
) -> dict | None:
    """Load a torch cache if it exists and its stored fingerprint matches.

    Returns the loaded dict on success, otherwise None (missing file or mismatch).
    """
    if not path.exists():
        return None
    cache = torch.load(path, map_location=map_location, weights_only=False)
    if cache.get("fingerprint") != dict(fingerprint):
        logging.warning(f"Cache fingerprint mismatch at {path}; ignoring cache.")
        return None
    return cache
