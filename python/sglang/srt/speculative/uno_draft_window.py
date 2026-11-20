"""Draft KV window (context trimming) for UNO: --uno-draft-kv-window / --uno-draft-kv-window-layers.

With a window W > 0, every row of the draft pass -- the seed row included -- reads only the last W keys, so the draft
pass is one attention call that never touches keys outside the window, and the seed row carries the draft adapter.
The verify pass then starts at the seed row and recomputes it with the full context (its K/V are overwritten); the
seed row's token becomes a verified draft instead of a free token. Tree UNO does the same through its seed-root
layout: the seed is the verify tree's root and the clean row its first level (Q >= F + 1).
--uno-draft-kv-window-layers "0,2,4-6" (0-based) restricts the window to those layers; unset = every layer.
"""

from __future__ import annotations

import os
from typing import Optional

from sglang.srt.runtime_context import get_spec

WINDOW_ARG = "--uno-draft-kv-window"
LAYERS_ARG = "--uno-draft-kv-window-layers"
# The environment switches these flags replaced; a stale export must not be silently ignored.
REMOVED_ENV = (
    "UNO_DRAFT_KV_WINDOW",
    "UNO_DRAFT_KV_WINDOW_LAYERS",
    "UNO_DRAFT_KV_WINDOW_SEED",
    "UNO_DRAFT_KV_WINDOW_SEED_LORA",
)


def parse_layer_ids(spec: Optional[str]) -> Optional[frozenset]:
    """Parse "0,2,4-6" into a frozenset of layer ids; empty -> None (all layers)."""

    spec = (spec or "").strip()
    if not spec:
        return None
    ids = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"Empty entry in {LAYERS_ARG}={spec!r}")
        if "-" in item:
            low, high = (int(part) for part in item.split("-"))
            if low > high:
                raise ValueError(f"Descending range {item!r} in {LAYERS_ARG}")
            ids.update(range(low, high + 1))
        else:
            ids.add(int(item))
    return frozenset(ids)


def draft_window() -> tuple:
    """Return (window_tokens, layer_ids or None); window 0 means off."""

    for name in REMOVED_ENV:
        if os.environ.get(name, "").strip() not in ("", "0"):
            raise ValueError(f"{name} was replaced by {WINDOW_ARG} (which also windows the seed row).")
    spec = get_spec()
    window = int(spec.uno_draft_kv_window or 0)
    if window < 0:
        raise ValueError(f"{WINDOW_ARG} must be >= 0, got {window}")
    return window, parse_layer_ids(spec.uno_draft_kv_window_layers)


def full_attention_backend(backend):
    """The FlashAttention backend that carries the window attributes (unwraps hybrid backends)."""

    full = getattr(backend, "full_attn_backend", backend)
    if not hasattr(full, "uno_draft_window"):
        raise RuntimeError(f"{WINDOW_ARG} needs the FlashAttention backend.")
    return full


def check_layers(layers: Optional[frozenset], num_layers: int) -> None:
    if layers is None:
        return
    bad = sorted(i for i in layers if not 0 <= i < num_layers)
    if bad:
        raise ValueError(f"{LAYERS_ARG} has ids {bad} outside [0, {num_layers}).")
