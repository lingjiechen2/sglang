"""Quantized copies of UNO's folded draft weights (``--uno-draft-weight-quant``).

With ``--uno-fused-lora`` every draft GEMM reads one folded weight ``[W; A]``
(see lora/backend/uno_fused_lora.py). A quantized draft replaces that GEMM with a
weight-only Marlin GEMM (FP8 per-channel or INT4 round-to-nearest, group 128)
over the same folded rows, so the base weight and the LoRA-A rows are read in the
low-precision copy. Verify forwards keep reading the bf16 rows ``W``, and the
worker verifies the seed row and the clean token in that bf16 verify, so
committed tokens still follow the target model exactly.
"""

from __future__ import annotations

import os

import torch

UNO_DRAFT_WEIGHT_QUANT_KINDS = ("none", "bf16", "fp8", "int4")

_FP8_MAX = 448.0
# Marlin dequantizes in registers, so it beats the bf16 GEMM only at small M. On
# H200 with Qwen3.8-27B linears (all layers per forward): INT4 5.8 / 11.6 ms vs bf16
# 12.7 / 13.4 ms at M = 8 / 64; FP8 8.1 ms at M = 8, 13.7 ms at M = 64. Larger draft
# batches fall back to the bf16 folded GEMM. UNO_DRAFT_QUANT_MAX_ROWS overrides.
_DEFAULT_MAX_ROWS = {"int4": 64, "fp8": 48}

# INT4 Marlin launch configs for the draft GEMMs: (N, K) -> {rows bucket:
# (pipeline stages, thread_k, thread_n, threads, blocks per SM)}; thread_k = -1
# keeps Marlin's own tile. The default launch (one block per SM, 4 stages) streams
# these small-M GEMMs at 1.0-2.6 TB/s on H200; more resident blocks or deeper
# pipelines keep more weight loads in flight. Measured on H200 with Qwen3.8-27B
# folded draft shapes (one CUDA graph per class), one forward's draft GEMMs plus
# lm_head: 5.90 -> 4.97 ms at 8 rows, 6.38 -> 5.76 at 16, 8.89 -> 7.42 at 32; draft
# GEMMs 12.04 -> 11.42 ms at 64; with the 64-row bucket retuned (stages 6/8) and
# o_proj-shaped GEMMs left in bf16 above 32 rows, 11.37 -> 10.78 ms. Shapes or row
# counts not listed, or tiles that do not divide the weight, use the default launch.
# UNO_DRAFT_MARLIN_TUNED=0 disables the table.
_INT4_MARLIN_LAUNCH = {
    (5248, 6144): {  # attention o_proj, GDN out_proj
        8: (4, 256, 64, 128, 2),
        16: (4, 256, 64, 128, 2),
        32: (4, 256, 64, 128, 2),
    },
    (14464, 5120): {  # attention qkv
        8: (4, 256, 64, 128, 3),
        16: (8, 128, 64, 128, 3),
        32: (8, 128, 64, 128, 2),
        64: (6, 128, 128, 256, 1),
    },
    (16640, 5120): {  # GDN in_proj
        8: (4, 256, 64, 128, 2),
        16: (4, 256, 64, 128, 2),
        32: (8, 128, 64, 128, 2),
        64: (6, 128, 128, 256, 1),
    },
    (34944, 5120): {  # MLP gate_up
        8: (4, 256, 128, 256, 2),
        16: (8, 64, 128, 128, 3),
        32: (4, 64, 128, 128, 2),
        64: (6, -1, -1, -1, 2),
    },
    (5248, 17408): {  # MLP down
        8: (4, 256, 64, 128, 2),
        16: (4, 256, 64, 128, 2),
        32: (8, 128, 64, 128, 2),
        64: (8, 128, 128, 256, 1),
    },
    (248320, 5120): {  # lm_head
        8: (4, 256, 128, 256, 2),
        16: (8, 64, 128, 128, 3),
        64: (6, -1, -1, -1, 1),
    },
}
# Above these row counts the bf16 GEMM is faster than any measured Marlin launch, so
# the draft keeps the bf16 weight there (H200, 64 rows: o_proj / GDN out_proj 1.06
# vs 0.91 ms and 0.36 vs 0.31 ms per forward; the Marlin 64-row bucket elsewhere).
_INT4_BF16_ABOVE_ROWS = {(5248, 6144): 32}
_LAUNCH_ROW_BUCKETS = (8, 16, 32, 64)
_MAX_BLOCKS_PER_SM = 4


def _int4_marlin_launch(n: int, k: int, rows: int):
    table = _INT4_MARLIN_LAUNCH.get((n, k))
    if table is None:
        return None
    bucket = next((b for b in _LAUNCH_ROW_BUCKETS if rows <= b), None)
    launch = table.get(bucket)
    if launch is not None and launch[1] > 0 and (k % launch[1] or n % launch[2]):
        return None  # the tile must divide the weight
    return launch


def _pad_rows(weight: torch.Tensor, multiple: int) -> torch.Tensor:
    pad = -weight.size(0) % multiple
    if pad == 0:
        return weight
    return torch.cat([weight, weight.new_zeros(pad, weight.size(1))], dim=0)


def _pack_int4_rows(q: torch.Tensor) -> torch.Tensor:
    """[K, N] values in [0, 16) -> GPTQ int32 [K / 8, N]; row 8r + i sits at bits 4i of row r."""
    k, n = q.shape
    q = q.to(torch.int64).view(k // 8, 8, n)
    packed = torch.zeros(k // 8, n, dtype=torch.int64, device=q.device)
    for i in range(8):
        packed |= q[:, i, :] << (4 * i)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32).contiguous()


class UnoHessianCollector:
    """Draft-GEMM stand-in that also accumulates X^T X (fp32) of its inputs and saves
    the mean once UNO_DRAFT_HESSIAN_ROWS rows were seen (eager forwards only)."""

    max_rows = 1 << 30

    def __init__(self, weight: torch.Tensor, key: str, out_dir: str) -> None:
        self.weight, self.key, self.out_dir = weight, key, out_dir
        k = weight.size(1)
        limit = int(os.environ.get("UNO_DRAFT_HESSIAN_MAX_K", "8192"))
        self.hessian = (
            torch.zeros(k, k, dtype=torch.float32, device=weight.device) if k <= limit else None
        )
        self.rows = 0
        self.target = int(os.environ.get("UNO_DRAFT_HESSIAN_ROWS", "50000"))
        self.saved = False

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.saved:
            xf = x.float()
            self.hessian.addmm_(xf.t(), xf)
            self.rows += x.size(0)
            if self.rows >= self.target:
                os.makedirs(self.out_dir, exist_ok=True)
                torch.save((self.hessian / self.rows).cpu(), os.path.join(self.out_dir, f"{self.key}.pt"))
                self.saved = True
                self.hessian = None
        return torch.mm(x, self.weight.t())


# GPTQ column order: UNO_DRAFT_GPTQ_ACT_ORDER=1 compensates the columns in
# decreasing Hessian-diagonal order with static (original-layout) groups, so the
# Marlin layout and the runtime GEMM are unchanged (no g_idx / activation perm).
# =2 also takes each group's scale from the compensated weights when its first
# column comes up in act order (like the natural-order GPTQ below).
_GPTQ_ACT_ORDER = int(os.environ.get("UNO_DRAFT_GPTQ_ACT_ORDER", "0"))


def _gptq_int4_act_order(
    weight: torch.Tensor, hessian: torch.Tensor, group: int, dtype, block: int = 128,
    dynamic_scales: bool = False,
):
    """GPTQ with act order and static groups: the group scales come from the
    original weights; the columns are quantized and their errors compensated in
    decreasing diag(H) order. Same returns as _gptq_int4."""
    w = weight.float().clone()
    n, k = w.shape
    h = hessian.float().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    scales = (
        w.view(n, k // group, group).abs().amax(dim=2).clamp_min(1e-12) / 7.0
    ).to(dtype).float().t().contiguous()  # [K / group, N]
    perm = torch.argsort(torch.diag(h), descending=True)
    w = w[:, perm]
    h = h[perm][:, perm]
    h.diagonal().add_(0.01 * torch.diag(h).mean())
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    col_group = perm // group
    col_scales = scales[col_group].t()  # [N, K] scale of each permuted column
    seen = torch.zeros(k // group, dtype=torch.bool)
    pos = torch.empty(k, dtype=torch.long)
    pos[perm] = torch.arange(k)
    members = pos.view(k // group, group).to(w.device)  # permuted positions of each group
    col_group_cpu = col_group.cpu()
    q_perm = torch.empty(n, k, dtype=torch.int32, device=w.device)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            g = int(col_group_cpu[i1 + i])
            if dynamic_scales and not seen[g]:
                seen[g] = True
                # Current values of the group's columns: this block's from w1 (up to
                # date), later blocks' from w (pending this block's lazy update).
                idx = members[g]
                vals = torch.where(
                    ((idx >= i1) & (idx < i2))[None, :],
                    w1[:, (idx - i1).clamp(0, i2 - i1 - 1)],
                    w[:, idx],
                )
                sc_g = (vals.abs().amax(dim=1).clamp_min(1e-12) / 7.0).to(dtype).float()
                scales[g] = sc_g
                col_scales[:, members[g]] = sc_g[:, None]
            sc = col_scales[:, i1 + i]
            wc = w1[:, i]
            qc = (wc / sc).round().clamp(-8, 7)
            q_perm[:, i1 + i] = (qc + 8).to(torch.int32)
            e = (wc - qc * sc) / h1[i, i]
            w1[:, i:] -= e[:, None] * h1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    q_all = torch.empty_like(q_perm)
    q_all[:, perm] = q_perm
    return q_all.t().contiguous(), scales


def _gptq_int4(weight: torch.Tensor, hessian: torch.Tensor, group: int, dtype, block: int = 128):
    """GPTQ (Frantar et al.) for symmetric INT4, group-wise scales chosen from the
    error-compensated weights at each group start. weight [N, K]; returns
    (q codes [K, N] in [0, 16), scales [K / group, N] rounded to ``dtype``)."""
    w = weight.float().clone()
    n, k = w.shape
    h = hessian.float().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(0.01 * torch.diag(h).mean())
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q_all = torch.empty(n, k, dtype=torch.int32, device=w.device)
    scales = torch.empty(k // group, n, dtype=torch.float32, device=w.device)
    assert block % group == 0 or group % block == 0
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                g_end = col + group
                seg = torch.cat([w1[:, i:], w[:, i2:g_end]], dim=1) if g_end > i2 else w1[:, i : i + group]
                sc = (seg.abs().amax(dim=1).clamp_min(1e-12) / 7.0).to(dtype).float()
                scales[col // group] = sc
            sc = scales[col // group]
            wc = w1[:, i]
            qc = (wc / sc).round().clamp(-8, 7)
            q_all[:, col] = (qc + 8).to(torch.int32)
            e = (wc - qc * sc) / h1[i, i]
            w1[:, i:] -= e[:, None] * h1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q_all.t().contiguous(), scales


class UnoMarlinDraftWeight:
    """A weight-only Marlin copy of a ``[N, K]`` bf16 weight; ``__call__(x) == x @ W.T``."""

    # INT4 scale group along K; UNO_DRAFT_INT4_GROUP overrides (Marlin: 32, 64, 128).
    group = int(os.environ.get("UNO_DRAFT_INT4_GROUP", "128"))

    def __init__(
        self,
        weight: torch.Tensor,
        kind: str,
        hessian: torch.Tensor = None,
        cache_path: str = None,
    ) -> None:
        from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
            marlin_permute_scales,
            scalar_types,
        )

        if kind not in ("fp8", "int4"):
            raise ValueError(f"No quantized UNO draft weight for {kind!r}")
        self.kind = kind
        self.max_rows = int(os.environ.get("UNO_DRAFT_QUANT_MAX_ROWS", _DEFAULT_MAX_ROWS[kind]))
        self.out_features = weight.size(0)
        weight = _pad_rows(weight.detach(), 64)
        self.n, self.k = weight.shape
        if self.k % self.group:
            raise ValueError(f"Marlin draft weights need K % {self.group} == 0, got K={self.k}")
        device = weight.device
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.workspace = marlin_make_workspace(device)
        # Tuned launches use up to _MAX_BLOCKS_PER_SM blocks per SM, one lock each.
        self.tuned = kind == "int4" and os.environ.get("UNO_DRAFT_MARLIN_TUNED", "1") != "0"
        if self.tuned:
            self.workspace = marlin_make_workspace(device, _MAX_BLOCKS_PER_SM)
            self.max_rows = min(
                self.max_rows, _INT4_BF16_ABOVE_ROWS.get((self.n, self.k), self.max_rows)
            )
        if kind == "fp8":
            from sglang.srt.layers.quantization.marlin_utils_fp8 import (
                fp8_fused_exponent_bias_into_scales,
            )

            w = weight.float()
            scale = w.abs().amax(dim=1).clamp_min(1e-12) / _FP8_MAX  # per output channel
            fp8 = (w / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            packed = fp8.contiguous().view(torch.int32).t().contiguous()  # [K / 4, N]
            self.qweight = gptq_marlin_repack(packed, self.empty, self.k, self.n, 8)
            scales = marlin_permute_scales(
                scale.to(weight.dtype).view(1, self.n), self.k, self.n, -1
            )
            self.scales = fp8_fused_exponent_bias_into_scales(scales).contiguous()
            self.quant_type = scalar_types.float8_e4m3fn
        else:
            cached = (
                torch.load(cache_path, map_location=device)
                if cache_path is not None and os.path.exists(cache_path)
                else None
            )
            if cached is not None:
                # Marlin-ready tensors saved by an earlier GPTQ run of this weight.
                self.qweight, self.scales = cached["qweight"], cached["scales"]
                self.quant_type = scalar_types.uint4b8
            elif hessian is not None:
                # weight was padded to N; the Hessian is over K and unaffected.
                if _GPTQ_ACT_ORDER:
                    q, scale = _gptq_int4_act_order(
                        weight, hessian, self.group, weight.dtype,
                        dynamic_scales=_GPTQ_ACT_ORDER == 2,
                    )
                else:
                    q, scale = _gptq_int4(weight, hessian, self.group, weight.dtype)
            else:
                w = weight.t().float().contiguous()  # [K, N]
                groups = w.view(self.k // self.group, self.group, self.n)
                scale = groups.abs().amax(dim=1).clamp_min(1e-12) / 7.0  # [K / group, N]
                q = (groups / scale[:, None, :]).round().clamp(-8, 7) + 8
                q = q.view(self.k, self.n).to(torch.int32)
            if cached is None:
                self.qweight = gptq_marlin_repack(_pack_int4_rows(q), self.empty, self.k, self.n, 4)
                self.scales = marlin_permute_scales(
                    scale.to(weight.dtype), self.k, self.n, self.group
                ).contiguous()
                self.quant_type = scalar_types.uint4b8
                if hessian is not None and cache_path is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save({"qweight": self.qweight, "scales": self.scales}, cache_path)

    def _gemm(self, x: torch.Tensor, out=None) -> torch.Tensor:
        from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm, gptq_marlin_gemm_cfg

        # Row count is fixed per captured graph, so the launch is static per graph.
        launch = _int4_marlin_launch(self.n, self.k, x.size(0)) if self.tuned else None
        if launch is not None:
            stages, thread_k, thread_n, threads, blocks_per_sm = launch
            return gptq_marlin_gemm_cfg(
                x, self.qweight, self.scales, self.workspace, self.quant_type, self.n, self.k,
                thread_k=thread_k, thread_n=thread_n, num_threads=threads,
                blocks_per_sm=blocks_per_sm, pipe_stages=stages, out=out,
            )
        return gptq_marlin_gemm(
            x, out, self.qweight, self.scales, None, None, self.empty, self.empty,
            self.workspace, self.quant_type, x.size(0), self.n, self.k,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self._gemm(x)
        # Consumers read rows through stride(0), so the padded columns stay a view.
        return y if self.n == self.out_features else y[:, : self.out_features]

    def nbytes(self) -> int:
        return (self.qweight.numel() * self.qweight.element_size()
                + self.scales.numel() * self.scales.element_size())
