"""LoRA parts of a MoVA UNO adapter that the regular LoRA manager cannot run.

K2 Horizon MoVA keeps q/k/v as separate projections (MoVA layers have no v_proj,
their values come from routed value experts), and the routed experts are
``FusedMoE`` modules that must not receive LoRA. The LoRA manager fuses q/k/v
into ``qkv_proj`` and treats every FusedMoE as a LoRA target once ``gate_up_proj``
and ``down_proj`` are targeted, so it only runs ``o_proj``. Everything else of the
adapter lives in ``mova/lora.safetensors`` and is applied here, on UNO draft rows
only (the seed row and every base-only forward are untouched):

  * ``q_proj`` / ``k_proj`` / ``v_proj`` and the dense / shared-expert MLP
    projections: added to the layer output by a forward hook;
  * ``v_experts``: one LoRA shared by all routed value experts, added to each
    selected expert's projection *before* the value SiLU;
  * ``mlp.experts.{gate,up,down}_proj``: one LoRA per projection shared by all
    routed FFN experts (2-D A/B), or one LoRA per expert (3-D A [E, r, in] /
    B [E, out, r], ``lora_expert_mode=per_expert``), added inside the Triton MoE
    between its two GEMMs (the MoE runner's LoRA hooks): gate/up deltas to every
    selected expert's pre-activation, the down delta to every selected expert's
    weighted output. A per-expert adapter applies each slot's own expert adapter.

Enabled with ``--uno-mova-lora``. The draft rows are the ones UNO's LoRA manager
routes to the adapter (``uno_single_adapter_route``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

if TYPE_CHECKING:
    from sglang.srt.lora.lora_manager import LoRAManager

logger = logging.getLogger(__name__)

# Kept in a subdirectory: the LoRA loader reads every *.safetensors next to adapter_model.safetensors.
MOVA_LORA_FILE = "mova/lora.safetensors"
_KEY = re.compile(r"model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight")
_LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.down_proj", "mlp.shared_experts.down_proj")
_GATE_UP = ("mlp", "mlp.shared_experts")  # parents whose gate_proj/up_proj are fused into gate_up_proj
_EXPERTS = ("mlp.experts.gate_proj", "mlp.experts.up_proj", "mlp.experts.down_proj")


def draft_rows(lora_backend, num_tokens: int, device) -> torch.Tensor | None:
    """Row ids of UNO's draft rows, or None outside a UNO draft forward.

    The draft forward is ``batch_size`` blocks of ``[seed, F - 1 draft rows]``; the
    same route tells the regular LoRA layers to skip the seed rows.
    """
    route = getattr(lora_backend.batch_info, "uno_single_adapter_route", None)
    if route is None:
        return None
    assert num_tokens == route.total_rows, (num_tokens, route.total_rows)
    blocks = torch.arange(route.batch_size, device=device)[:, None] * route.forward_width
    return (blocks + torch.arange(1, route.forward_width, device=device)[None, :]).reshape(-1)


def _delta(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    return F.linear(F.linear(x, a), b) * scaling


def shard_lora(proj: str, a: torch.Tensor, b: torch.Tensor, rank: int, size: int):
    """This TP rank's part of a full LoRA ``(A [r, in], B [out, r])`` on a projection of the base model.

    gate/up are column parallel (their output rows are sharded: slice B's rows); down is row parallel (its input
    columns are sharded: slice A's columns, and the per-rank deltas add up to the full one)."""
    if size == 1:
        return a, b
    if proj in ("gate_proj", "up_proj"):
        return a, b.chunk(size, dim=0)[rank].contiguous()
    return a.chunk(size, dim=1)[rank].contiguous(), b


class UnoMovaValueLoRA:
    """Shared value-expert LoRA applied on the draft rows of a UNO block."""

    def __init__(self, a, b, scaling, lora_backend):
        self.a, self.b, self.scaling, self.lora_backend = a, b, scaling, lora_backend

    def __call__(self, hidden_states, routing_weights, selected_experts, expert_weights, values):
        rows = draft_rows(self.lora_backend, hidden_states.shape[0], hidden_states.device)
        if rows is None:
            return values
        x = hidden_states[rows]
        num_experts, width, hidden = expert_weights.shape
        # All experts' projections of these few rows, then the routed ones.
        projected = F.linear(x, expert_weights.reshape(num_experts * width, hidden)).view(-1, num_experts, width)
        index = selected_experts[rows].long()[:, :, None].expand(-1, -1, width)
        projected = projected.gather(1, index) + _delta(x, self.a, self.b, self.scaling)[:, None, :]
        mixed = (F.silu(projected) * routing_weights[rows].to(projected.dtype)[:, :, None]).sum(1)
        values = values.clone()
        values[rows] = mixed.to(values.dtype)
        return values


class UnoMovaPerExpertValueLoRA:
    """Per-expert value LoRA (``lora_expert_mode=per_expert``): every routed value expert has its own adapter.

    ``a`` is [E, r, in] and ``b`` is [E, out, r] (out = the value width of one expert). Same placement as
    :class:`UnoMovaValueLoRA`: the delta of each *selected* expert is added to that expert's projection before the
    value SiLU. All experts' rank activations come from one GEMM, ``b`` is applied per expert with a batched matmul,
    then the selected experts' deltas are gathered, the same all-experts-then-gather scheme as the base projection.
    """

    def __init__(self, a, b, scaling, lora_backend):
        self.rank = a.shape[1]
        self.a_flat = a.reshape(-1, a.shape[-1]).contiguous()  # [E * r, in]
        self.b_t = b.transpose(1, 2).contiguous()  # [E, r, out]
        self.scaling, self.lora_backend = scaling, lora_backend

    def __call__(self, hidden_states, routing_weights, selected_experts, expert_weights, values):
        rows = draft_rows(self.lora_backend, hidden_states.shape[0], hidden_states.device)
        if rows is None:
            return values
        x = hidden_states[rows]
        num_experts, width, hidden = expert_weights.shape
        projected = F.linear(x, expert_weights.reshape(num_experts * width, hidden)).view(-1, num_experts, width)
        index = selected_experts[rows].long()[:, :, None].expand(-1, -1, width)
        activation = F.linear(x, self.a_flat).view(-1, num_experts, self.rank).transpose(0, 1)  # [E, R, r]
        delta = torch.bmm(activation, self.b_t).transpose(0, 1)  # [R, E, out]
        projected = projected.gather(1, index) + delta.gather(1, index) * self.scaling
        mixed = (F.silu(projected) * routing_weights[rows].to(projected.dtype)[:, :, None]).sum(1)
        values = values.clone()
        values[rows] = mixed.to(values.dtype)
        return values


class UnoMovaExpertLoRA:
    """Routed FFN-expert LoRA (one adapter per projection, shared by all experts) on UNO draft rows.

    Called by the MoE runner with the layer input; returns the Triton MoE hooks for
    a UNO draft forward and None otherwise (the runner then keeps its fused path).
    """

    def __init__(self, parts, scaling, lora_backend):
        self.parts = parts  # {"gate_proj" | "up_proj" | "down_proj": (A, B)}
        self.scaling, self.lora_backend = scaling, lora_backend

    def __call__(self, hidden_states):
        rows = draft_rows(self.lora_backend, hidden_states.shape[0], hidden_states.device)
        if rows is None:
            return None
        from sglang.srt.lora.lora_moe_runners import LoRAHooks

        return LoRAHooks(after_gate_up=partial(self.after_gate_up, rows), after_down=partial(self.after_down, rows))

    def after_gate_up(self, rows, hidden_states, gate_up, topk_weights, topk_ids):
        # gate_up: [tokens, topk, 2 * I], gate half first; the delta is the same for every selected expert.
        width = gate_up.shape[-1] // 2
        x = hidden_states[rows]
        for start, proj in ((0, "gate_proj"), (width, "up_proj")):
            if proj in self.parts:
                delta = _delta(x, *self.parts[proj], self.scaling).to(gate_up.dtype)
                gate_up[rows, :, start : start + width] += delta[:, None, :]

    def after_down(self, rows, activated, down, topk_weights, topk_ids):
        # activated: [tokens * topk, I] route-major; down: [tokens, topk, H], already times the routing weight.
        if "down_proj" not in self.parts:
            return
        tokens, topk, _ = down.shape
        h = activated[: tokens * topk].view(tokens, topk, -1)[rows]
        delta = _delta(h, *self.parts["down_proj"], self.scaling) * topk_weights[rows, :, None].to(h.dtype)
        down[rows] += delta.to(down.dtype)


def _expert_lora_loop(x: torch.Tensor, expert_ids: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-slot ``B[e] (A[e] x)`` without scaling: one matmul pair per expert that owns slots (reference / fallback)."""
    out = x.new_zeros(x.shape[0], b.shape[1])
    for e in torch.unique(expert_ids).tolist():
        idx = (expert_ids == e).nonzero(as_tuple=True)[0]
        out[idx] = F.linear(F.linear(x[idx], a[e]), b[e])
    return out


def _expert_lora_grouped(x: torch.Tensor, expert_ids: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Same as :func:`_expert_lora_loop`, with the slots sorted by expert and two ``torch._grouped_mm`` calls.

    Host-sync free (argsort, scatter_add_, cumsum, grouped_mm with device-side offsets, advanced-index copies), hence safe
    inside CUDA graph capture and replay."""
    num_experts = a.shape[0]
    order = torch.argsort(expert_ids, stable=True)
    # Counts per expert without a host sync (torch.bincount on CUDA reads the output size back), so the hooks can run
    # inside a CUDA graph capture.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=expert_ids.device)
    counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
    offsets = torch.cumsum(counts, 0).to(torch.int32)
    rank_act = torch._grouped_mm(x[order], a.transpose(1, 2), offs=offsets)  # [S, r]
    delta = torch._grouped_mm(rank_act, b.transpose(1, 2), offs=offsets)  # [S, out]
    out = torch.empty_like(delta)
    out[order] = delta
    return out


def expert_lora(x: torch.Tensor, expert_ids: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``B[e_s] (A[e_s] x_s)`` for every slot ``s``; ``a`` [E, r, in], ``b`` [E, out, r], ``expert_ids`` [S] int."""
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and hasattr(torch, "_grouped_mm"):
        return _expert_lora_grouped(x, expert_ids.long(), a, b)
    return _expert_lora_loop(x, expert_ids.long(), a, b)


class UnoMovaPerExpertExpertLoRA(UnoMovaExpertLoRA):
    """Routed FFN-expert LoRA with one adapter per expert (``lora_expert_mode=per_expert``) on UNO draft rows.

    ``parts`` maps each projection to ``(A [E, r, in], B [E, out, r])``; expert ``e`` of the adapter is expert ``e`` of
    the model's routed experts (TP=1, so the Triton MoE's ``topk_ids`` are the global expert ids). Placement as in
    :class:`UnoMovaExpertLoRA`: the gate/up deltas of each *selected* expert go to its pre-activation, and the down
    delta (computed on the already LoRA-modified activation) to its output times the routing weight.
    """

    def after_gate_up(self, rows, hidden_states, gate_up, topk_weights, topk_ids):
        width = gate_up.shape[-1] // 2
        topk = topk_ids.shape[1]
        x = hidden_states[rows].repeat_interleave(topk, dim=0)  # [R * topk, H], slot (row, k) = row * topk + k
        ids = topk_ids[rows].reshape(-1)
        for start, proj in ((0, "gate_proj"), (width, "up_proj")):
            if proj in self.parts:
                a, b = self.parts[proj]
                delta = (expert_lora(x, ids, a, b) * self.scaling).to(gate_up.dtype)
                gate_up[rows, :, start : start + width] += delta.view(len(rows), topk, width)

    def after_down(self, rows, activated, down, topk_weights, topk_ids):
        if "down_proj" not in self.parts:
            return
        tokens, topk, _ = down.shape
        h = activated[: tokens * topk].view(tokens, topk, -1)[rows].reshape(len(rows) * topk, -1)
        a, b = self.parts["down_proj"]
        delta = expert_lora(h, topk_ids[rows].reshape(-1), a, b) * self.scaling
        delta = delta.view(len(rows), topk, -1) * topk_weights[rows, :, None].to(delta.dtype)
        down[rows] += delta.to(down.dtype)


class _RowHook:
    """Forward hook adding LoRA deltas to chosen column slices of a linear layer's output."""

    def __init__(self, parts, scaling, lora_backend, all_reduce=None):
        self.parts = parts  # [(column start, column end, A, B)]
        self.scaling, self.lora_backend = scaling, lora_backend
        self.all_reduce = all_reduce  # row-parallel layers that already reduced their output: reduce the delta too

    def __call__(self, module, inputs, output):
        x, (out, *rest) = inputs[0], output
        rows = draft_rows(self.lora_backend, x.shape[0], x.device)
        if rows is None:
            return None
        out = out.clone()
        xr = x[rows]
        for start, end, a, b in self.parts:
            delta = _delta(xr, a, b, self.scaling)
            if self.all_reduce is not None:
                delta = self.all_reduce(delta)
            out[rows, start:end] += delta.to(out.dtype)
        return (out, *rest)


FAST_PARTS = ("row", "gateup", "down", "value")


def fast_parts() -> set:
    """``SGLANG_UNO_MOVA_LORA_FAST``: which draft-row LoRA paths use the fast kernels (default: none).

    ``1``/``all`` = every path, or a comma list of ``row`` (dense / shared-expert forward hooks), ``gateup`` (routed
    experts, before the activation), ``down`` (routed experts, after the down GEMM) and ``value`` (MoVA value experts:
    the LoRA delta is added inside the routed value projection, before its SiLU, instead of recomputing every value
    expert densely for the draft rows)."""
    value = os.environ.get("SGLANG_UNO_MOVA_LORA_FAST", "").strip().lower()
    if value in ("", "0", "false", "off"):
        return set()
    if value in ("1", "all", "true", "on"):
        return set(FAST_PARTS)
    parts = {p.strip() for p in value.split(",") if p.strip()}
    if parts - set(FAST_PARTS):
        raise ValueError(f"SGLANG_UNO_MOVA_LORA_FAST: unknown parts {sorted(parts - set(FAST_PARTS))}, choose from {FAST_PARTS}.")
    return parts


def _draft_route(lora_backend, num_tokens: int):
    route = getattr(lora_backend.batch_info, "uno_single_adapter_route", None)
    if route is None:
        return None
    assert num_tokens == route.total_rows, (num_tokens, route.total_rows)
    return route


def _zero_seed_rows(act: torch.Tensor, route) -> torch.Tensor:
    """Zero the seed row of every ``[seed, F - 1 draft rows]`` block of a rank activation, in place.

    ``B @ 0`` is exactly zero, so adding the resulting delta to the full output leaves the seed rows bit-identical."""
    act.view(route.batch_size, route.forward_width, -1)[:, 0].zero_()
    return act


class _PackedLoRA:
    """LoRAs that read the same input and write adjacent output columns, run as one A GEMM and one B GEMM.

    ``A`` is the row concatenation of the parts' A, ``B`` is block diagonal (the zero blocks add exact zeros), so the
    packed delta covers the parts' columns in order. The LoRA scaling is folded into ``B`` when it is a power of two
    (an exact multiply in bf16); otherwise it is applied to the delta."""

    def __init__(self, parts, scaling: float):
        self.a = torch.cat([a for a, _ in parts], 0).contiguous()
        b_full = parts[0][1].new_zeros(sum(b.shape[0] for _, b in parts), sum(a.shape[0] for a, _ in parts))
        row = col = 0
        for a, b in parts:
            b_full[row : row + b.shape[0], col : col + a.shape[0]] = b
            row, col = row + b.shape[0], col + a.shape[0]
        mantissa, _ = math.frexp(scaling)
        self.post_scale = None if mantissa == 0.5 else scaling
        self.b = (b_full * scaling if self.post_scale is None else b_full).contiguous()
        self.width = row

    def rank_act(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.a)

    def expand(self, act: torch.Tensor) -> torch.Tensor:
        delta = F.linear(act, self.b)
        return delta if self.post_scale is None else delta * self.post_scale

    def delta(self, x: torch.Tensor, route) -> torch.Tensor:
        """Delta for every row of ``x``, exactly zero on the seed rows."""
        return self.expand(_zero_seed_rows(self.rank_act(x), route))


class UnoMovaExpertLoRAFast(UnoMovaExpertLoRA):
    """:class:`UnoMovaExpertLoRA` with the fast paths of :func:`fast_parts` (same math, fewer kernels).

    * ``gateup``: gate and up deltas from one packed LoRA over all rows (seed rows exactly zero), broadcast-added over
      the top-k slots in place, without row gathers or scatters.
    * ``down``: the shared down LoRA is linear, so ``sum_k w_k B A h_k = B (sum_k w_k A h_k)``: one A GEMM over all
      slots, a routing-weighted sum over top-k, one B GEMM per row, added to slot 0 only (the MoE sums the slots next).
    """

    def __init__(self, parts, scaling, lora_backend, fast: set):
        super().__init__(parts, scaling, lora_backend)
        self.fast_gate_up = None
        if "gateup" in fast and "gate_proj" in parts and "up_proj" in parts:
            self.fast_gate_up = _PackedLoRA([parts["gate_proj"], parts["up_proj"]], scaling)
        self.fast_down = None
        if "down" in fast and "down_proj" in parts:
            self.fast_down = _PackedLoRA([parts["down_proj"]], scaling)

    def __call__(self, hidden_states):
        route = _draft_route(self.lora_backend, hidden_states.shape[0])
        if route is None:
            return None
        from sglang.srt.lora.lora_moe_runners import LoRAHooks

        rows = None
        if self.fast_gate_up is None or (self.fast_down is None and "down_proj" in self.parts):
            rows = draft_rows(self.lora_backend, hidden_states.shape[0], hidden_states.device)
        gate_up = partial(self._fast_after_gate_up, route) if self.fast_gate_up is not None else partial(self.after_gate_up, rows)
        down = partial(self._fast_after_down, route) if self.fast_down is not None else partial(self.after_down, rows)
        return LoRAHooks(after_gate_up=gate_up, after_down=down)

    def _fast_after_gate_up(self, route, hidden_states, gate_up, topk_weights, topk_ids):
        delta = self.fast_gate_up.delta(hidden_states, route)
        if delta.shape[-1] != gate_up.shape[-1]:
            raise ValueError(f"UNO MoVA fast gate/up LoRA: delta width {delta.shape[-1]} vs gate_up {gate_up.shape[-1]}.")
        gate_up.add_(delta.to(gate_up.dtype)[:, None, :])

    def _fast_after_down(self, route, activated, down, topk_weights, topk_ids):
        tokens, topk, _ = down.shape
        act = self.fast_down.rank_act(activated[: tokens * topk]).view(tokens, topk, -1)
        act = torch.bmm(topk_weights.to(act.dtype).view(tokens, 1, topk), act).view(tokens, -1)
        delta = self.fast_down.expand(_zero_seed_rows(act, route))
        down[:, 0, :].add_(delta.to(down.dtype))


class UnoMovaValueLoRAFast(UnoMovaValueLoRA):
    """``value`` fast path of :class:`UnoMovaValueLoRA` (same math).

    The base routed value projection runs once over all rows with the LoRA delta added to every selected expert's
    pre-SiLU projection; the delta is one packed LoRA over all rows with the seed rows exactly zero. No dense
    projection through every value expert, no row gathers or scatters. :meth:`project` returns None outside a UNO draft
    forward and for dtypes the fused kernel does not cover (the caller then runs the base path and :meth:`__call__`)."""

    fused = True

    def __init__(self, a, b, scaling, lora_backend):
        super().__init__(a, b, scaling, lora_backend)
        self.packed = _PackedLoRA([(a, b)], scaling)

    def _delta(self, hidden_states, route):
        return self.packed.delta(hidden_states, route)

    def project(self, hidden_states, routing_weights, selected_experts, expert_weights):
        if not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return None
        route = _draft_route(self.lora_backend, hidden_states.shape[0])
        if route is None:
            return None
        from sglang.srt.layers.mova import routed_linear_with_delta

        delta = self._delta(hidden_states, route)
        return routed_linear_with_delta(hidden_states, expert_weights, routing_weights, selected_experts, delta)


class UnoMovaPerExpertValueLoRAFast(UnoMovaPerExpertValueLoRA):
    """``value`` fast path of :class:`UnoMovaPerExpertValueLoRA`: only the selected experts' adapters run (grouped GEMMs
    over the ``tokens * top_k`` slots), and their deltas are added inside the routed value projection before the SiLU."""

    fused = True

    def __init__(self, a, b, scaling, lora_backend):
        super().__init__(a, b, scaling, lora_backend)
        self.a, self.b = a.contiguous(), b.contiguous()

    def project(self, hidden_states, routing_weights, selected_experts, expert_weights):
        if not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return None
        route = _draft_route(self.lora_backend, hidden_states.shape[0])
        if route is None:
            return None
        from sglang.srt.layers.mova import routed_linear_with_delta

        tokens, topk = selected_experts.shape
        x = hidden_states.repeat_interleave(topk, dim=0)  # slot (row, k) = row * topk + k
        delta = (expert_lora(x, selected_experts.reshape(-1), self.a, self.b) * self.scaling).view(tokens, topk, -1)
        delta.view(route.batch_size, route.forward_width, topk, -1)[:, 0].zero_()  # seed rows: exactly the base path
        return routed_linear_with_delta(hidden_states, expert_weights, routing_weights, selected_experts, delta)


class _FastRowHook:
    """Fast :class:`_RowHook` for parts that cover the whole output (no all-reduce): one packed LoRA over all rows,
    seed rows exactly zero, added to the layer output in place (no clone, gather or scatter)."""

    def __init__(self, parts, scaling, lora_backend):
        self.packed = _PackedLoRA([(a, b) for _, _, a, b in parts], scaling)
        self.lora_backend = lora_backend

    def __call__(self, module, inputs, output):
        x, out = inputs[0], output[0]
        route = _draft_route(self.lora_backend, x.shape[0])
        if route is None:
            return None
        delta = self.packed.delta(x, route)
        if delta.shape[-1] != out.shape[-1]:
            raise ValueError(f"UNO MoVA fast LoRA hook: delta width {delta.shape[-1]} vs output {out.shape[-1]}.")
        out.add_(delta.to(out.dtype))
        return None


def _row_parts_cover_output(parts) -> bool:
    """Parts sorted by column start that tile ``[0, end)`` without gaps."""
    pos = 0
    for start, end, _, _ in parts:
        if start != pos:
            return False
        pos = end
    return True


def _attach_expert_lora(moe, lora: UnoMovaExpertLoRA) -> None:
    method = moe.quant_method
    runner = getattr(method, "runner", None)
    if (
        runner is None
        or not runner.runner_backend.is_triton()
        or getattr(method, "_aiter_runner", None) is not None
        or getattr(method, "use_flashinfer_cutlass", False)
        or getattr(method, "use_flashinfer_trtllm_moe", False)
    ):
        raise ValueError("--uno-mova-lora with routed FFN-expert LoRA needs --moe-runner-backend triton.")
    if getattr(method, "w13_swiglu_interleaved", False):
        raise ValueError("--uno-mova-lora with routed FFN-expert LoRA needs SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED off.")
    from sglang.srt.runtime_context import get_exec

    if get_exec().moe.enable_fused_moe_sum_all_reduce:
        raise ValueError("--uno-mova-lora with routed FFN-expert LoRA needs the fused MoE sum all-reduce off.")
    if moe.moe_ep_size != 1:
        raise ValueError("--uno-mova-lora does not support expert parallelism.")
    if moe.moe_tp_size != 1 and isinstance(lora, UnoMovaPerExpertExpertLoRA):
        raise ValueError("--uno-mova-lora supports per-expert FFN LoRA at TP=1 only.")
    runner.uno_lora_hooks = lora


def attach_uno_mova_lora(manager: LoRAManager, model: torch.nn.Module, lora_path: str) -> None:
    """Attach the MoVA LoRA parts of the UNO adapter to the model."""
    from sglang.srt.models.xllm import XllmMoVAAttention

    path = os.path.join(lora_path, MOVA_LORA_FILE)
    if not os.path.exists(path):
        raise ValueError(f"--uno-mova-lora found no {MOVA_LORA_FILE} in {lora_path}.")
    with open(os.path.join(lora_path, "adapter_config.json")) as f:
        config = json.load(f)
    scaling = config["lora_alpha"] / config["r"]
    found = defaultdict(dict)  # (layer, module) -> {"A": tensor, "B": tensor}
    for key, tensor in load_file(path).items():
        match = _KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Unexpected tensor {key} in {path}.")
        found[(int(match.group(1)), match.group(2))][match.group(3)] = tensor
    if any(name == "self_attn.v_experts" for _, name in found) and not any(
        isinstance(m, XllmMoVAAttention) for m in model.modules()
    ):
        raise ValueError("The value-expert LoRA needs a model with MoVA attention layers.")

    from sglang.srt.runtime_context import get_parallel

    tp_group = get_parallel().tp_group
    tp_rank, tp_size = tp_group.rank_in_group, tp_group.world_size
    backend, layers = manager.lora_backend, model.model.layers
    hooks = defaultdict(list)  # (layer, linear module path) -> [(start, end, A, B)]
    experts = defaultdict(dict)  # layer -> {projection: (A, B)}
    attached = 0
    for (layer, name), ab in sorted(found.items()):
        decoder = layers[layer]
        param = next(decoder.parameters())
        # The manager's device, not the layer's: --cpu-offload-gb keeps offloaded layer weights on the CPU.
        a, b = (ab[k].to(device=manager.device, dtype=param.dtype) for k in "AB")
        if name == "self_attn.v_experts" and a.ndim == 3:
            # per-expert value LoRA, native layout: A [E, r, in], B [E, out, r]
            if a.shape[1] != config["r"] or b.shape[2] != config["r"] or a.shape[0] != b.shape[0]:
                raise ValueError(f"Layer {layer} {name}: per-expert LoRA shapes {tuple(a.shape)}/{tuple(b.shape)} vs rank {config['r']}.")
            module = decoder.self_attn
            if module.v_experts.output_size_per_partition != module.v_experts.output_size:
                raise ValueError("--uno-mova-lora supports TP=1 only.")
            if a.shape[0] != module.v_experts.weight.shape[0]:
                raise ValueError(
                    f"Layer {layer} {name}: {a.shape[0]} per-expert adapters for {module.v_experts.weight.shape[0]} value experts."
                )
            cls = UnoMovaPerExpertValueLoRAFast if "value" in fast_parts() else UnoMovaPerExpertValueLoRA
            module.uno_value_lora = cls(a, b, scaling, backend)
            attached += 1
            continue
        if name in _EXPERTS and a.ndim == 3:
            # per-expert routed FFN LoRA, native layout: A [E, r, in], B [E, out, r]
            num_experts = layers[layer].mlp.experts.num_experts
            if a.shape[1] != config["r"] or b.shape[2] != config["r"] or a.shape[0] != b.shape[0]:
                raise ValueError(f"Layer {layer} {name}: per-expert LoRA shapes {tuple(a.shape)}/{tuple(b.shape)} vs rank {config['r']}.")
            if a.shape[0] != num_experts:
                raise ValueError(f"Layer {layer} {name}: {a.shape[0]} per-expert adapters for {num_experts} routed experts.")
            experts[layer][name.rpartition(".")[2]] = (a.contiguous(), b.contiguous())
            attached += 1
            continue
        if a.shape[0] != config["r"] or b.shape[1] != config["r"]:
            raise ValueError(f"Layer {layer} {name}: LoRA shapes {tuple(a.shape)}/{tuple(b.shape)} vs rank {config['r']}.")
        if name == "self_attn.v_experts":
            module = decoder.self_attn
            if module.v_experts.output_size_per_partition != module.v_experts.output_size:
                raise ValueError("--uno-mova-lora supports TP=1 only.")
            cls = UnoMovaValueLoRAFast if "value" in fast_parts() else UnoMovaValueLoRA
            module.uno_value_lora = cls(a, b, scaling, backend)
        elif name in _EXPERTS:
            moe = layers[layer].mlp.experts
            proj = name.rpartition(".")[2]
            experts[layer][proj] = shard_lora(proj, a, b, moe.moe_tp_rank, moe.moe_tp_size)
        elif name in _LINEAR:
            if name.startswith("self_attn.") and tp_size != 1:
                raise ValueError("--uno-mova-lora supports MoVA q/k/v LoRA at TP=1 only.")
            if name.endswith("down_proj"):
                a, b = shard_lora("down_proj", a, b, tp_rank, tp_size)
            hooks[(layer, name)].append((0, b.shape[0], a, b))
        else:
            parent, _, proj = name.rpartition(".")
            if parent not in _GATE_UP or proj not in ("gate_proj", "up_proj"):
                raise ValueError(f"Layer {layer}: unsupported MoVA LoRA module {name}.")
            a, b = shard_lora(proj, a, b, tp_rank, tp_size)
            width = b.shape[0]
            start = 0 if proj == "gate_proj" else width
            hooks[(layer, f"{parent}.gate_up_proj")].append((start, start + width, a, b))
        attached += 1
    fast = fast_parts()
    fast_hooks = 0
    for (layer, name), parts in hooks.items():
        module = layers[layer].get_submodule(name)
        # A row-parallel layer with reduce_results has already summed its ranks: sum the delta's partials the same way.
        reduce = tp_group.all_reduce if tp_size > 1 and name.endswith("down_proj") and module.reduce_results else None
        parts = sorted(parts, key=lambda p: p[0])
        if "row" in fast and reduce is None and _row_parts_cover_output(parts):
            module.register_forward_hook(_FastRowHook(parts, scaling, backend))
            fast_hooks += 1
        else:
            module.register_forward_hook(_RowHook(parts, scaling, backend, reduce))
    for layer, parts in experts.items():
        per_expert = {p[0].ndim == 3 for p in parts.values()}
        if len(per_expert) != 1:
            raise ValueError(f"Layer {layer}: routed experts LoRA mixes per-expert and shared projections.")
        if per_expert.pop():
            lora = UnoMovaPerExpertExpertLoRA(parts, scaling, backend)
        elif fast & {"gateup", "down"}:
            lora = UnoMovaExpertLoRAFast(parts, scaling, backend, fast)
        else:
            lora = UnoMovaExpertLoRA(parts, scaling, backend)
        _attach_expert_lora(layers[layer].mlp.experts, lora)
    logger.info("UNO MoVA LoRA: %d adapters on %d layers (scaling %.1f).", attached, len(layers), scaling)
    if fast:
        logger.info("UNO MoVA LoRA fast paths %s: %d/%d forward hooks fast.", sorted(fast), fast_hooks, len(hooks))
