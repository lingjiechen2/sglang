from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_MAX_THREAD_N = 256


@cache_once
def _jit_gptq_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "gptq_marlin",
        *args,
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{args}>")],
    )


@cache_once
def _jit_gptq_marlin_cfg_module(dtype: torch.dtype, pipe_stages: int) -> Module:
    args = make_cpp_args(dtype, pipe_stages)
    dtype_args = make_cpp_args(dtype)
    return load_jit(
        "gptq_marlin_cfg",
        *args,
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm_cfg", f"gptq_marlin_gemm_cfg<{dtype_args}>")],
        extra_cuda_cflags=[
            f"-DSGL_MARLIN_PIPE_STAGES={pipe_stages}",
            "-DSGL_MARLIN_SMALL_M_TILES",
        ],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c is None:
        c = torch.empty((size_m, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx is not None
        and perm is not None
        and g_idx.numel() > 0
        and perm.numel() > 0
    )

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_m_block = min(((size_m + 15) // 16) * 16, 64)
        c_tmp = torch.empty(
            sms * max_m_block * _MAX_THREAD_N,
            dtype=torch.float32,
            device=device,
        )
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Convert Optional tensors to empty tensors
    global_scale_t = _or_empty(global_scale, device, a.dtype)
    b_zeros_t = _or_empty(b_zeros, device, torch.int32)
    g_idx_t = _or_empty(g_idx, device, torch.int32)
    perm_t = _or_empty(perm, device, torch.int32)

    module = _jit_gptq_marlin_module(a.dtype)
    module.gptq_marlin_gemm(
        a,
        b_q_weight,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        c,
        c_tmp,
        a_tmp,
        workspace,
        b_q_type.id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c


def gptq_marlin_gemm_cfg(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_n: int,
    size_k: int,
    *,
    thread_k: int,
    thread_n: int,
    num_threads: int,
    blocks_per_sm: int,
    pipe_stages: int = 4,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Marlin GEMM of uint4b8 weights with an explicit launch config (no zero
    points, act order or global scale; full K; fp32 reduce). ``thread_k = -1``
    keeps the default thread tile and one block per SM.

    ``workspace`` must hold one zeroed lock per thread block
    (``SMs * blocks_per_sm``). Each ``pipe_stages`` value is its own JIT module.
    """
    device = a.device
    size_m = a.size(0)
    c = out if out is not None else torch.empty((size_m, size_n), dtype=a.dtype, device=device)
    if size_m == 0:
        return c
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    max_m_block = min(((size_m + 15) // 16) * 16, 64)
    # fp32 partial tile per thread block.
    c_tmp = torch.empty(
        sms * blocks_per_sm * max_m_block * max(thread_n, _MAX_THREAD_N),
        dtype=torch.float32,
        device=device,
    )
    empty = torch.empty(0, dtype=a.dtype, device=device)
    empty_i = torch.empty(0, dtype=torch.int32, device=device)
    module = _jit_gptq_marlin_cfg_module(a.dtype, pipe_stages)
    module.gptq_marlin_gemm_cfg(
        a,
        b_q_weight,
        b_scales,
        empty,
        empty_i,
        empty_i,
        empty_i,
        c,
        c_tmp,
        empty,
        workspace,
        b_q_type.id,
        True,
        False,
        True,
        False,
        thread_k,
        thread_n,
        num_threads,
        blocks_per_sm,
    )
    return c
