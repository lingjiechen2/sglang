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

    def __init__(self, parts, scaling, lora_backend):
        self.parts = parts  # [(column start, column end, A, B)]
        self.scaling, self.lora_backend = scaling, lora_backend

    def __call__(self, module, inputs, output):
        x, (out, *rest) = inputs[0], output
        rows = draft_rows(self.lora_backend, x.shape[0], x.device)
        if rows is None:
            return None
        out = out.clone()
        xr = x[rows]
        for start, end, a, b in self.parts:
            out[rows, start:end] += _delta(xr, a, b, self.scaling).to(out.dtype)
        return (out, *rest)


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
    if moe.moe_tp_size != 1 or moe.moe_ep_size != 1:
        raise ValueError("--uno-mova-lora supports TP=1 only.")
    runner.uno_lora_hooks = lora


def attach_uno_mova_lora(manager: LoRAManager, model: torch.nn.Module, lora_path: str) -> None:
    """Attach the MoVA LoRA parts of the UNO adapter to the model."""
    from sglang.srt.models.xllm import XllmMoVAAttention

    path = os.path.join(lora_path, MOVA_LORA_FILE)
    if not any(isinstance(m, XllmMoVAAttention) for m in model.modules()):
        raise ValueError("--uno-mova-lora needs a model with MoVA attention layers.")
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
            module.uno_value_lora = UnoMovaPerExpertValueLoRA(a, b, scaling, backend)
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
            module.uno_value_lora = UnoMovaValueLoRA(a, b, scaling, backend)
        elif name in _EXPERTS:
            experts[layer][name.rpartition(".")[2]] = (a, b)
        elif name in _LINEAR:
            hooks[(layer, name)].append((0, b.shape[0], a, b))
        else:
            parent, _, proj = name.rpartition(".")
            if parent not in _GATE_UP or proj not in ("gate_proj", "up_proj"):
                raise ValueError(f"Layer {layer}: unsupported MoVA LoRA module {name}.")
            width = b.shape[0]
            start = 0 if proj == "gate_proj" else width
            hooks[(layer, f"{parent}.gate_up_proj")].append((start, start + width, a, b))
        attached += 1
    for (layer, name), parts in hooks.items():
        module = layers[layer].get_submodule(name)
        module.register_forward_hook(_RowHook(parts, scaling, backend))
    for layer, parts in experts.items():
        per_expert = {p[0].ndim == 3 for p in parts.values()}
        if len(per_expert) != 1:
            raise ValueError(f"Layer {layer}: routed experts LoRA mixes per-expert and shared projections.")
        cls = UnoMovaPerExpertExpertLoRA if per_expert.pop() else UnoMovaExpertLoRA
        _attach_expert_lora(layers[layer].mlp.experts, cls(parts, scaling, backend))
    logger.info("UNO MoVA LoRA: %d adapters on %d layers (scaling %.1f).", attached, len(layers), scaling)
