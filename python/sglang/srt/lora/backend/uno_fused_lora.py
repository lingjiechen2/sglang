"""UNO draft LoRA folded into the base GEMM.

UNO runs one fixed adapter on draft rows only. Folding the adapter's LoRA-A
rows into the base weight (``[W; A]``) turns every LoRA-wrapped linear layer
into one GEMM plus one Triton "expand" kernel that adds the per-slice LoRA-B
products, skips each request's seed row and writes the contiguous output.
Base-only forwards keep using ``W``, now a row view of the folded weight, so
their inputs and results are unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _uno_lora_expand_kernel(
    full,  # [M, N + U * R]: base output, then one LoRA-A block per unique slice
    full_row_stride,
    lora_b,  # [N, >= R]: LoRA-B rows of every output slice, in output order
    lora_b_row_stride,
    out,  # [M, N] contiguous
    slice_ends,  # [S] int32 exclusive end column of each output slice
    slice_blocks,  # [S] int32 LoRA-A block used by each output slice
    M,
    N,
    forward_width,
    scaling,
    R: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    rows64 = rows.to(tl.int64)

    col_slice = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for s in tl.static_range(NUM_SLICES - 1):
        col_slice += (cols >= tl.load(slice_ends + s)).to(tl.int32)
    col_block = tl.load(slice_blocks + col_slice, mask=col_mask, other=-1)

    # The first row of every request is its seed token, which stays base-only.
    lora_rows = row_mask & (rows % forward_width != 0)
    ranks = tl.arange(0, R)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in tl.static_range(NUM_BLOCKS):
        sel = col_mask & (col_block == block)
        if tl.sum(sel.to(tl.int32), axis=0) > 0:
            h = tl.load(
                full + rows64[:, None] * full_row_stride + (N + block * R) + ranks[None, :],
                mask=lora_rows[:, None],
                other=0.0,
            )
            b = tl.load(
                lora_b + cols.to(tl.int64)[:, None] * lora_b_row_stride + ranks[None, :],
                mask=sel[:, None],
                other=0.0,
            )
            acc += tl.dot(h, tl.trans(b))

    mask = row_mask[:, None] & col_mask[None, :]
    base = tl.load(
        full + rows64[:, None] * full_row_stride + cols[None, :], mask=mask, other=0.0
    )
    result = base.to(tl.float32) + scaling * acc
    tl.store(out + rows64[:, None] * N + cols[None, :], result.to(base.dtype), mask=mask)


@triton.jit
def _uno_lora_expand_silu_mul_kernel(
    full,  # [M, 2I + U * R]: gate | up base outputs, then LoRA-A blocks
    full_row_stride,
    lora_b,  # [2I, >= R]: gate rows then up rows
    lora_b_row_stride,
    out,  # [M, I] contiguous
    M,
    inter,
    forward_width,
    scaling,
    GATE_BLOCK: tl.constexpr,
    UP_BLOCK: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """silu(gate) * up of a folded gate_up layer, LoRA added on draft rows."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < inter
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)
    lora_rows = row_mask & (rows % forward_width != 0)
    ranks = tl.arange(0, R)
    a_base = full + rows64[:, None] * full_row_stride + 2 * inter + ranks[None, :]
    h_gate = tl.load(a_base + GATE_BLOCK * R, mask=lora_rows[:, None], other=0.0)
    h_up = tl.load(a_base + UP_BLOCK * R, mask=lora_rows[:, None], other=0.0)
    b_gate = tl.load(
        lora_b + cols64[:, None] * lora_b_row_stride + ranks[None, :],
        mask=col_mask[:, None],
        other=0.0,
    )
    b_up = tl.load(
        lora_b + (cols64 + inter)[:, None] * lora_b_row_stride + ranks[None, :],
        mask=col_mask[:, None],
        other=0.0,
    )
    mask = row_mask[:, None] & col_mask[None, :]
    base_ptr = full + rows64[:, None] * full_row_stride + cols64[None, :]
    gate = tl.load(base_ptr, mask=mask, other=0.0)
    up = tl.load(base_ptr + inter, mask=mask, other=0.0)
    # Round to the layer's dtype first, as the unfused gate_up output would be.
    gate = (gate.to(tl.float32) + scaling * tl.dot(h_gate, tl.trans(b_gate))).to(gate.dtype)
    up = (up.to(tl.float32) + scaling * tl.dot(h_up, tl.trans(b_up))).to(up.dtype)
    g = gate.to(tl.float32)
    # Same formula as sglang's act_and_mul SiLU.
    act = g / (1.0 + tl.exp(-g)) * up.to(tl.float32)
    tl.store(out + rows64[:, None] * inter + cols64[None, :], act.to(gate.dtype), mask=mask)


class UnoFusedLoRA(msgspec.Struct, frozen=True):
    """One LoRA-wrapped linear layer folded for UNO draft forwards."""

    # [N + U * R, K]; rows [0, N) are the base weight the layer itself reads.
    weight_cat: torch.Tensor
    # [N, >= R] view of this adapter's LoRA-B rows.
    lora_b: torch.Tensor
    slice_ends: torch.Tensor
    slice_blocks: torch.Tensor
    num_slices: int
    num_blocks: int
    out_features: int
    rank: int
    scaling: float
    # Output width of each folded layer, in order.
    out_splits: tuple[int, ...] = ()
    # LoRA-A block of the gate and up slices, for a folded gate_up layer.
    gate_up_blocks: tuple[int, ...] = ()

    def base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The folded layers' base outputs only (no LoRA), as one GEMM."""
        return torch.mm(x, self.weight_cat[: self.out_features].t())

    def forward_silu_mul(self, x: torch.Tensor, *, forward_width: int) -> torch.Tensor:
        """silu(gate) * up of a folded two-slice gate_up layer, in one pass."""
        full = torch.mm(x, self.weight_cat.t())
        rows = full.shape[0]
        inter = self.out_features // 2
        out = torch.empty((rows, inter), dtype=full.dtype, device=full.device)
        block_m, block_n = (16, 32) if rows <= 16 else (64, 64)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(inter, block_n))
        _uno_lora_expand_silu_mul_kernel[grid](
            full,
            full.stride(0),
            self.lora_b,
            self.lora_b.stride(0),
            out,
            rows,
            inter,
            forward_width,
            self.scaling,
            GATE_BLOCK=self.gate_up_blocks[0],
            UP_BLOCK=self.gate_up_blocks[1],
            R=self.rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return out

    def forward(self, x: torch.Tensor, *, forward_width: int) -> torch.Tensor:
        full = torch.mm(x, self.weight_cat.t())
        rows = full.shape[0]
        out = torch.empty((rows, self.out_features), dtype=full.dtype, device=full.device)
        # Measured on H200 with Qwen3.8-27B layer shapes (rank 128).
        if rows <= 16:
            block_m, block_n, num_warps, num_stages = 16, 32, 4, 2
        else:
            block_m, block_n, num_warps, num_stages = 64, 128, 4, 1
        grid = (triton.cdiv(rows, block_m), triton.cdiv(self.out_features, block_n))
        _uno_lora_expand_kernel[grid](
            full,
            full.stride(0),
            self.lora_b,
            self.lora_b.stride(0),
            out,
            self.slice_ends,
            self.slice_blocks,
            rows,
            self.out_features,
            forward_width,
            self.scaling,
            R=self.rank,
            NUM_SLICES=self.num_slices,
            NUM_BLOCKS=self.num_blocks,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def _unique_lora_a_blocks(
    slice_a_blocks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[int]]:
    """Deduplicate per-slice LoRA-A blocks; e.g. q, k and v share one HF A."""
    blocks: list[torch.Tensor] = []
    slice_blocks: list[int] = []
    for block in slice_a_blocks:
        for index, seen in enumerate(blocks):
            if torch.equal(seen, block):
                slice_blocks.append(index)
                break
        else:
            slice_blocks.append(len(blocks))
            blocks.append(block)
    return blocks, slice_blocks


def _lora_slices(module: torch.nn.Module, weight_index: int, rank: int):
    """(base weight, slice offsets, per-slice A blocks, B rows), or None."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.lora.layers import (
        ColumnParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )

    if isinstance(module, QKVParallelLinearWithLoRA):
        a_buffer, b_buffer = module.A_buffer_qkv, module.B_buffer_qkv
    elif isinstance(module, (ColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA)):
        a_buffer, b_buffer = module.A_buffer, module.B_buffer
    else:
        return None
    base = module.base_layer
    weight = base.weight
    if (
        not isinstance(base.quant_method, UnquantizedLinearMethod)
        or base.bias is not None
        or weight.dtype not in (torch.bfloat16, torch.float16)
        or not weight.is_contiguous()
    ):
        return None
    out_features = weight.shape[0]
    offsets = [int(v) for v in module.output_offset_cpu.tolist()]
    num_slices = len(offsets) - 1
    lora_b = b_buffer[weight_index, :, :rank]
    if offsets[-1] != out_features or num_slices < 1 or lora_b.shape[0] != out_features:
        return None
    lora_a = a_buffer[weight_index]
    a_blocks = [lora_a[s * rank : (s + 1) * rank] for s in range(num_slices)]
    return weight, offsets, a_blocks, lora_b


def fold_uno_lora_group(
    modules: list[torch.nn.Module], *, weight_index: int, rank: int, scaling: float
) -> Optional[UnoFusedLoRA]:
    """Fold LoRA-wrapped linear layers that read the same input into one GEMM.

    Each layer's base weight becomes a row view of the shared folded weight, in
    group order, so ``weight_cat[:out_features]`` is the layers' fused base
    GEMM. Returns None, leaving every layer untouched, if any cannot be folded.
    """
    parts = [_lora_slices(module, weight_index, rank) for module in modules]
    if any(part is None for part in parts):
        return None
    in_features = parts[0][0].shape[1]
    if any(w.shape[1] != in_features or w.dtype != parts[0][0].dtype for w, *_ in parts):
        return None

    slice_ends: list[int] = []
    slice_a_blocks: list[torch.Tensor] = []
    out_features = 0
    for weight, offsets, a_blocks, _ in parts:
        slice_ends.extend(out_features + end for end in offsets[1:])
        slice_a_blocks.extend(a_blocks)
        out_features += weight.shape[0]
    blocks, slice_blocks = _unique_lora_a_blocks(slice_a_blocks)

    weight0 = parts[0][0]
    weight_cat = torch.empty(
        (out_features + len(blocks) * rank, in_features),
        dtype=weight0.dtype,
        device=weight0.device,
    )
    row = 0
    for weight, *_ in parts:
        rows = weight.shape[0]
        weight_cat[row : row + rows].copy_(weight.data)
        # The layer keeps reading its base weight, now a row view of weight_cat.
        weight.data = weight_cat[row : row + rows]
        row += rows
    for block in blocks:
        weight_cat[row : row + rank].copy_(block)
        row += rank
    lora_b = (
        parts[0][3]
        if len(parts) == 1
        else torch.cat([part[3] for part in parts]).contiguous()
    )

    device = weight0.device
    return UnoFusedLoRA(
        weight_cat=weight_cat,
        lora_b=lora_b,
        slice_ends=torch.tensor(slice_ends, dtype=torch.int32, device=device),
        slice_blocks=torch.tensor(slice_blocks, dtype=torch.int32, device=device),
        num_slices=len(slice_ends),
        num_blocks=len(blocks),
        out_features=out_features,
        rank=rank,
        scaling=scaling,
        out_splits=tuple(part[0].shape[0] for part in parts),
        gate_up_blocks=tuple(slice_blocks) if len(slice_blocks) == 2 else (),
    )


def fold_uno_lora_layer(
    module: torch.nn.Module, *, weight_index: int, rank: int, scaling: float
) -> Optional[UnoFusedLoRA]:
    """Fold one LoRA-wrapped linear layer, or return None if it cannot be."""
    return fold_uno_lora_group(
        [module], weight_index=weight_index, rank=rank, scaling=scaling
    )
