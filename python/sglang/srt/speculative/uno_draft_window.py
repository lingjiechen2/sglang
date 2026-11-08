"""Opt-in draft-only KV window for linear UNO (env UNO_DRAFT_KV_WINDOW / UNO_DRAFT_KV_WINDOW_LAYERS).

UNO_DRAFT_KV_WINDOW=W (tokens, 0 = off): the draft rows of the draft pass read only the last W keys; the clean seed row and the
verify pass keep the full context. UNO_DRAFT_KV_WINDOW_LAYERS="0,2,4-6" (0-based) restricts the window to those layers, unset = all.
"""

from __future__ import annotations

import os
from typing import Optional

DRAFT_WINDOW_ENV = "UNO_DRAFT_KV_WINDOW"
DRAFT_WINDOW_LAYERS_ENV = "UNO_DRAFT_KV_WINDOW_LAYERS"


def parse_layer_ids(spec: str) -> Optional[frozenset]:
    """Parse "0,2,4-6" into a frozenset of layer ids; empty -> None (all layers)."""

    spec = spec.strip()
    if not spec:
        return None
    ids = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"Empty entry in {DRAFT_WINDOW_LAYERS_ENV}={spec!r}")
        if "-" in item:
            low, high = (int(part) for part in item.split("-"))
            if low > high:
                raise ValueError(f"Descending range {item!r} in {DRAFT_WINDOW_LAYERS_ENV}")
            ids.update(range(low, high + 1))
        else:
            ids.add(int(item))
    return frozenset(ids)


def draft_window_from_env() -> tuple:
    """Return (window_tokens, layer_ids or None); window 0 means off."""

    window = int(os.environ.get(DRAFT_WINDOW_ENV, "0"))
    if window < 0:
        raise ValueError(f"{DRAFT_WINDOW_ENV} must be >= 0, got {window}")
    return window, parse_layer_ids(os.environ.get(DRAFT_WINDOW_LAYERS_ENV, ""))


def full_attention_backend(backend):
    """The FlashAttention backend that carries the window attributes (unwraps hybrid backends)."""

    full = getattr(backend, "full_attn_backend", backend)
    if not hasattr(full, "uno_draft_window"):
        raise RuntimeError(f"{DRAFT_WINDOW_ENV} needs the FlashAttention backend.")
    return full


def check_layers(layers: Optional[frozenset], num_layers: int) -> None:
    if layers is None:
        return
    bad = sorted(i for i in layers if not 0 <= i < num_layers)
    if bad:
        raise ValueError(f"{DRAFT_WINDOW_LAYERS_ENV} has ids {bad} outside [0, {num_layers}).")
