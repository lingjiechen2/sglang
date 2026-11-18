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

import logging
import os

import torch

logger = logging.getLogger(__name__)

UNO_DRAFT_WEIGHT_QUANT_KINDS = ("none", "bf16", "fp8", "int4")

_FP8_MAX = 448.0
# Marlin dequantizes in registers, so it beats the bf16 GEMM only at small M. On
# H200 with Qwen3.8-27B linears (all layers per forward): INT4 5.8 / 11.6 ms vs bf16
# 12.7 / 13.4 ms at M = 8 / 64; FP8 8.1 ms at M = 8, 13.7 ms at M = 64. Larger draft
# batches fall back to the bf16 folded GEMM. UNO_DRAFT_QUANT_MAX_ROWS overrides.
_DEFAULT_MAX_ROWS = {"int4": 64, "fp8": 48}

# INT4 draft GEMM kernel, from --uno-draft-int4-kernel (UNO_DRAFT_INT4_KERNEL overrides):
#
#   "marlin"        (default) Always Marlin -- the behaviour before Humming existed, so the
#                   default leaves an existing INT4 draft bit for bit as it was. Note the
#                   row limit travels with the kernel: Marlin keeps the quantized draft to
#                   64 rows and falls back to the bf16 GEMM above it, Humming to 256, so
#                   switching kernels DOES change the draft (and the acceptance length)
#                   above 64 rows even though the two kernels agree bit for bit below it.
#   "fit"           Pick ONE kernel at startup from an upper bound on this server's draft
#                   row count, max_running_requests * speculative_num_draft_tokens: Marlin
#                   below UNO_DRAFT_HUMMING_MIN_ROWS rows, Humming from there on. Only the
#                   chosen kernel's packing is built, so the server holds one INT4 copy. A
#                   server launched at concurrency 1 with B8 drafts 8 rows and gets Marlin;
#                   at concurrency 8 it drafts 64 rows and gets Humming.
#   "humming"       Always Humming (Marlin's packing is dropped after it is built).
#   "auto"          Build BOTH packings and dispatch per CUDA graph on the actual row count.
#                   Optimal at every row count one server sees -- an in-flight count that
#                   dips to 1-3 during ramp-up and the tail still gets Marlin -- but it
#                   costs two INT4 copies (measured 25.2 GB vs 12.6 GB of draft weights on
#                   Qwen3.8-27B skip75). The 11-16 c=8 ladder and the 11-18 c=1 A/B were
#                   both measured in this mode; against "humming" at c=8 the difference is
#                   the 3.0% of decode steps with 1-3 requests in flight, ~0.5% of decode
#                   time (estimated from the in-flight histogram, not measured directly).
#
# Humming gets the same GPTQ codes and scales as Marlin, so W4A16 Humming and Marlin agree
# bit for bit and the kernel choice never moves the acceptance length. On H200 with the
# Qwen3.8-27B folded draft shapes, one forward's draft GEMMs: Marlin 4.69 / 7.06 / 11.26 /
# 21.65 ms vs Humming 5.54 / 6.54 / 8.33 / 10.95 ms at 8 / 32 / 64 / 128 rows -- hence the
# 32-row crossover below. With Humming the INT4 draft also beats the bf16 GEMM (13.9 ms)
# above 64 rows, so the row limit rises to 256.
_INT4_KERNEL_CHOICES = ("fit", "marlin", "humming", "auto")
_HUMMING_MIN_ROWS = int(os.environ.get("UNO_DRAFT_HUMMING_MIN_ROWS", "32"))
_RESOLVED_INT4_KERNEL: str | None = None
_WARNED_UNSIZED = False


def _int4_kernel_setting() -> str:
    """The requested setting: UNO_DRAFT_INT4_KERNEL, else --uno-draft-int4-kernel."""
    setting = os.environ.get("UNO_DRAFT_INT4_KERNEL")
    if setting is None:
        try:
            from sglang.srt.runtime_context import get_spec

            setting = getattr(get_spec(), "uno_draft_int4_kernel", "marlin")
        except Exception:  # no runtime context (unit tests, tooling): take the default
            setting = "marlin"
    if setting not in _INT4_KERNEL_CHOICES:
        raise ValueError(
            f"UNO_DRAFT_INT4_KERNEL must be one of {_INT4_KERNEL_CHOICES}, got {setting!r}"
        )
    return setting


def peak_draft_rows() -> int | None:
    """Upper bound on the draft row count this server can issue, or None if unreadable.

    This is an UPPER BOUND, not the exact peak: seed-in-verify drafts
    speculative_num_draft_tokens - 1 rows per request, so the real peak is one row per
    in-flight request lower (concurrency 1 with B8 reports 9 and drafts 8; concurrency 8
    reports 72 and drafts 64). Deriving the exact width would mean re-deriving
    seed_in_verify and tree mode here; the bound only ever errs toward Humming, and since
    it is compared against an aggregate crossover (UNO_DRAFT_HUMMING_MIN_ROWS, measured
    over all the draft GEMMs at once) that slack is within the threshold's own precision.
    Pass --uno-draft-int4-kernel explicitly when the choice has to be exact.

    Reads already-parsed server args, so it does not depend on init order. Returns None
    both when max_running_requests is unset (the scheduler then sizes it from free memory
    and the peak is effectively unbounded) and when the args cannot be read at all.
    """
    try:
        from sglang.srt.runtime_context import get_schedule, get_spec

        concurrency = getattr(get_schedule(), "max_running_requests", None)
        width = getattr(get_spec(), "speculative_num_draft_tokens", None)
    except Exception:
        return None
    if not concurrency or not width:
        return None
    return int(concurrency) * int(width)


def int4_kernel() -> str:
    """Which INT4 kernel this server uses: "marlin", "humming", or "auto" (both built)."""
    global _RESOLVED_INT4_KERNEL
    if _RESOLVED_INT4_KERNEL is not None:
        return _RESOLVED_INT4_KERNEL
    setting = _int4_kernel_setting()
    if setting != "fit":
        _RESOLVED_INT4_KERNEL = setting
        return _RESOLVED_INT4_KERNEL
    rows = peak_draft_rows()
    if rows is None:
        # Either max_running_requests is unset -- the server can then go as wide as memory
        # allows, so take the kernel that wins there -- or the args are not readable yet.
        # The second case must NOT be cached: caching it would pin a server that only ever
        # drafts 8 rows to Humming for its whole life and quietly lose ~18% on every draft
        # GEMM. Warn once (this runs per folded weight) and resolve again on the next call.
        global _WARNED_UNSIZED
        if not _WARNED_UNSIZED:
            _WARNED_UNSIZED = True
            logger.warning(
                "UNO draft INT4 kernel: cannot size the draft from max_running_requests; "
                "using humming for now. Pass --uno-draft-int4-kernel to pin the choice."
            )
        return "humming"
    resolved = "humming" if rows >= _HUMMING_MIN_ROWS else "marlin"
    _RESOLVED_INT4_KERNEL = resolved
    logger.info(
        "UNO draft INT4 kernel: %s (peak draft rows %d, crossover %d); "
        "pass --uno-draft-int4-kernel to choose explicitly.",
        resolved,
        rows,
        _HUMMING_MIN_ROWS,
    )
    return resolved


# Env switches removed with the features they drove, as name -> (why, values that asked for
# what this code still does). A value that asked for something else is rejected rather than
# ignored: a stale script that still passes one would otherwise start cleanly and quietly
# produce numbers for a different configuration than the caller believes they asked for. A
# value that happens to name the surviving behaviour only gets a warning -- sbatch inherits
# the submitting shell, so a leftover export must not be able to fail an unrelated job.
_REMOVED_ENV = {
    "UNO_DRAFT_INT4_ACT": (
        "W4A8 draft activations were removed on 2026-11-18; the INT4 draft GEMMs are "
        "weight-only (W4A16). It cut one forward's draft GEMMs from 7.49 to 6.52 ms at 64 "
        "rows but cost ~1.8-2% acceptance length, and nothing shipped with it on.",
        ("bf16",),
    ),
    "UNO_DRAFT_HUMMING_PDL": (
        "the Humming PDL patch was removed on 2026-11-18. Its +1.2% at c=8 was a single "
        "seed, smaller than the baseline's own 4-seed spread, and it rewrote the installed "
        "humming package at runtime.",
        ("0",),
    ),
    "UNO_DRAFT_INT4_GROUP": (
        "the INT4 scale group is fixed at 128 since 2026-11-18. Per-channel scales only "
        "ever won on the removed W4A8 path.",
        ("128",),
    ),
}


def _reject_removed_env() -> None:
    for name, (why, kept) in _REMOVED_ENV.items():
        value = os.environ.get(name)
        if value in (None, ""):
            continue
        if value in kept:
            logger.warning("%s is no longer read: %s Unset it.", name, why)
            continue
        raise ValueError(f"{name}={value!r} is no longer supported: {why} Unset it.")


_reject_removed_env()


# The Humming draft GEMMs are weight-only: INT4 weights, bf16 activations (W4A16). A W4A8
# variant (UNO_DRAFT_INT4_ACT=int8, per-row int8 activations on the int8 wgmma path) was
# measured and dropped on 2026-11-18: it cut one forward's draft GEMMs from 7.49 to 6.52 ms
# at 64 rows but cost ~1.8-2% acceptance length, and nothing shipped with it on.
_HUMMING_MAX_ROWS = 256


def default_max_rows(kind: str) -> int:
    """Largest draft row count that reads the quantized weight (bf16 above it)."""
    if kind == "int4" and int4_kernel() in ("humming", "auto"):
        return _HUMMING_MAX_ROWS
    return _DEFAULT_MAX_ROWS[kind]

# INT4 Marlin launch configs for the draft GEMMs: (N, K) -> {rows bucket:
# (pipeline stages, thread_k, thread_n, threads, blocks per SM)}; thread_k = -1
# keeps Marlin's own tile. The default launch (one block per SM, 4 stages) streams
# these small-M GEMMs at 1.0-2.6 TB/s on H200; more resident blocks or deeper
# pipelines keep more weight loads in flight. Measured on H200 with Qwen3.8-27B
# folded draft shapes (one CUDA graph per class), one forward's draft GEMMs plus
# lm_head: 5.90 -> 4.97 ms at 8 rows, 6.38 -> 5.76 at 16, 8.89 -> 7.42 at 32; draft
# GEMMs 12.04 -> 11.42 ms at 64. Shapes or row counts not listed, or tiles that do
# not divide the weight, use the default launch. UNO_DRAFT_MARLIN_TUNED=0 disables
# the table.
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
    },
}
_LAUNCH_ROW_BUCKETS = (8, 16, 32, 64)
_MAX_BLOCKS_PER_SM = 4

# Humming tuning configs for the draft GEMMs in the row bucket (56, 64] (linear B8
# at 8 running requests drafts 64 rows): (N, K) -> Humming TuningConfig. Humming's
# own sm90 heuristic picks block (64, 256, 64) / warp (64, 32, 64) with stream-K for
# every shape. A sweep over tile geometry, pipeline stages, CTAs per SM, stream-K,
# raster order and activation multicast on H200 (Qwen3.8-27B folded draft shapes,
# 6 weight copies per shape in one CUDA graph) cut one forward's draft GEMMs from
# 8.30 to 7.49 ms at 64 rows. Other row buckets keep the heuristic; the table is
# used only on 132-SM devices (num_sms is part of each config).
# UNO_DRAFT_HUMMING_TUNED=0 disables it.
_HUMMING_TUNED_ROWS = 64
_HUMMING_TUNED_SMS = 132
_HUMMING_COMMON = {
    "use_f16_accum": False,
    "use_warp_spec": True,
    "use_tma": True,
    "use_mbarrier": True,
    "num_sms": _HUMMING_TUNED_SMS,
    "multi_cast_size_a": 1,
}
_HUMMING_TUNED = {
    (14464, 5120): dict(  # attention qkv: 27.0 -> 23.9 us
        _HUMMING_COMMON, block_shape=[64, 128, 128], warp_shape=[64, 32, 64], use_stream_k=False,
        num_stages=5, num_ctas_per_sm=1, raster_group_m=1, reduce_overlap_last_stage_only=True,
    ),
    (16640, 5120): dict(  # GDN in_proj: 28.7 -> 24.2 us
        _HUMMING_COMMON, block_shape=[64, 128, 128], warp_shape=[64, 32, 64], use_stream_k=False,
        num_stages=5, num_ctas_per_sm=1, raster_group_m=1, reduce_overlap_last_stage_only=False,
    ),
    (5248, 6144): dict(  # attention o_proj, GDN out_proj: 15.9 -> 15.3 us
        _HUMMING_COMMON, block_shape=[64, 128, 128], warp_shape=[64, 32, 64], use_stream_k=True,
        num_stages=4, num_ctas_per_sm=1, raster_group_m=1, reduce_overlap_last_stage_only=True,
    ),
    (34944, 5120): dict(  # MLP gate_up: 54.0 -> 48.0 us
        _HUMMING_COMMON, block_shape=[64, 256, 64], warp_shape=[64, 64, 64], use_stream_k=True,
        num_stages=3, num_ctas_per_sm=2, raster_group_m=77, reduce_overlap_last_stage_only=True,
    ),
    (5248, 17408): dict(  # MLP down: 31.4 -> 29.7 us
        _HUMMING_COMMON, block_shape=[64, 128, 128], warp_shape=[64, 32, 64], use_stream_k=True,
        num_stages=4, num_ctas_per_sm=1, raster_group_m=22, reduce_overlap_last_stage_only=True,
    ),
    (248320, 5120): dict(  # draft lm_head (once per forward): 351 -> 286 us
        _HUMMING_COMMON, block_shape=[64, 512, 64], warp_shape=[64, 64, 64], use_stream_k=True,
        num_stages=6, num_ctas_per_sm=1, raster_group_m=1, reduce_overlap_last_stage_only=True,
    ),
    # Rank-0 folds of skip-adapter layers (UNO_DRAFT_QUANT_SKIPPED), own sweep at 64 rows
    # (multicast winners left out). qkv / o / in_proj rank-0 shapes keep the folded-shape configs
    # (same winners); MLP shapes differ:
    (34816, 5120): dict(  # MLP gate_up, rank 0: heuristic 54.0 -> 46.4 us
        _HUMMING_COMMON, block_shape=[64, 512, 64], warp_shape=[64, 64, 64], use_stream_k=True,
        num_stages=6, num_ctas_per_sm=1, raster_group_m=1, reduce_overlap_last_stage_only=True,
    ),
    (5120, 17408): dict(  # MLP down, rank 0: heuristic 30.7 -> 28.5 us
        _HUMMING_COMMON, block_shape=[64, 256, 64], warp_shape=[64, 32, 64], use_stream_k=True,
        num_stages=6, num_ctas_per_sm=1, raster_group_m=1, reduce_overlap_last_stage_only=True,
    ),
}


def _humming_tuning(layer, n: int, k: int):
    """Humming's heuristic table with the tuned (56, 64] bucket replaced, as JSON, or None."""
    if os.environ.get("UNO_DRAFT_HUMMING_TUNED", "1") == "0":
        return None
    # A base-only (rank 0) fold of a layer the adapter skips is the folded shape minus its
    # LoRA-A rows (128 per block); the tiles of the folded shape divide it as well.
    cfg = None
    for extra_rows in (0, 128, 256):
        cfg = _HUMMING_TUNED.get((n + extra_rows, k))
        if cfg is not None:
            break
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if cfg is None or sms != _HUMMING_TUNED_SMS:
        return None
    import json

    from humming.tune import get_heuristics_config

    buckets = [list(entry) for entry in get_heuristics_config(layer.humming_config)]
    hits = [b for b in buckets if b[0] < _HUMMING_TUNED_ROWS <= b[1]]
    if len(hits) != 1:
        return None
    hits[0][2] = dict(cfg)
    return json.dumps(buckets)


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

    # INT4 scale group along K (Marlin accepts 32, 64, 128). Fixed since 2026-11-18, when
    # UNO_DRAFT_INT4_GROUP and the per-channel (group = 0) path went away with W4A8:
    # per-channel scales only ever won there, because the int8 kernel could then accumulate
    # int32 over all of K and rescale once instead of once per group (one forward 6.31 vs
    # 6.72 ms). On the W4A16 path they just lose accuracy and have no tuned table.
    group = 128

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
        self.max_rows = int(os.environ.get("UNO_DRAFT_QUANT_MAX_ROWS", default_max_rows(kind)))
        self.humming = None
        self.humming_tuning = None
        self.humming_min_rows = 0  # unused while self.humming is None; keep it defined
        self.out_features = weight.size(0)
        weight = _pad_rows(weight.detach(), 64)
        self.n, self.k = weight.shape
        self.group = type(self).group
        if self.k % self.group:
            raise ValueError(f"Marlin draft weights need K % {self.group} == 0, got K={self.k}")
        device = weight.device
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.workspace = marlin_make_workspace(device)
        # Tuned launches use up to _MAX_BLOCKS_PER_SM blocks per SM, one lock each.
        self.tuned = kind == "int4" and os.environ.get("UNO_DRAFT_MARLIN_TUNED", "1") != "0"
        if self.tuned:
            self.workspace = marlin_make_workspace(device, _MAX_BLOCKS_PER_SM)
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
            kernel = int4_kernel()
            if kernel in ("humming", "auto"):
                self.humming_min_rows = 0 if kernel == "humming" else _HUMMING_MIN_ROWS
                self.humming = self._build_humming(weight.dtype)
                if self.humming_min_rows == 0:
                    self.qweight = self.scales = None  # Marlin copy no longer read

    def gptq_codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The INT4 codes [K, N] in [0, 16) and group scales [K / group, N] behind
        the Marlin tensors: scales un-permuted, codes read back from Marlin itself
        (identity rows give (q - 8) * s exactly in bf16)."""
        from sglang.srt.layers.quantization.marlin_utils import get_scale_perms

        perm = torch.tensor(get_scale_perms()[0], device=self.scales.device)  # group scales
        scales = self.scales.reshape(-1, perm.numel())[:, torch.argsort(perm)].reshape(-1, self.n)
        codes = torch.empty(self.k, self.n, dtype=torch.int32, device=scales.device)
        step = 512
        for i in range(0, self.k, step):
            rows = min(step, self.k - i)
            eye = torch.zeros(rows, self.k, dtype=scales.dtype, device=scales.device)
            eye[torch.arange(rows), i + torch.arange(rows)] = 1
            deq = self._gemm(eye).float()  # [rows, N] = (q - 8) * s
            s = scales[(i + torch.arange(rows, device=scales.device)) // self.group].float()
            codes[i : i + rows] = (deq / s).round().clamp(-8, 7).to(torch.int32) + 8
        return codes, scales

    def _build_humming(self, dtype) -> torch.nn.Module:
        from humming.layer import HummingLayer

        codes, scales = self.gptq_codes()
        layer = HummingLayer(
            shape_n=self.n,
            shape_k=self.k,
            weight_config={"quant_method": "gptq", "bits": 4, "group_size": self.group, "sym": True},
            input_config=None,  # weight-only: bf16 activations (W4A16)
            torch_dtype=dtype,
            pad_n_to_multiple=256,  # Humming's N tiles must divide N
        ).to(scales.device)
        layer.load_from_tensors({"qweight": _pack_int4_rows(codes), "scales": scales.to(dtype)})
        layer.transform()
        self.humming_tuning = _humming_tuning(layer, self.n, self.k)
        return layer

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
        # Row count is fixed per captured graph, so the kernel choice is static too.
        if self.humming is not None and x.size(0) >= self.humming_min_rows:
            if self.humming_tuning is not None:
                y = self.humming(x, tuning_config=self.humming_tuning)
            else:
                y = self.humming(x)
        else:
            y = self._gemm(x)
        # Consumers read rows through stride(0), so the padded columns stay a view.
        return y if y.size(1) == self.out_features else y[:, : self.out_features]

    def nbytes(self) -> int:
        total = sum(t.numel() * t.element_size() for t in (self.qweight, self.scales) if t is not None)
        if self.humming is not None:
            total += sum(p.numel() * p.element_size() for p in self.humming.parameters())
        return total
