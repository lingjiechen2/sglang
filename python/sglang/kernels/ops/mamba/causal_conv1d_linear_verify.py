"""Causal conv1d for a linear-chain verify of F tokens per request.

Specialized for the deduplicated intermediate conv window (consecutive draft
windows share storage): per channel the window memory is one sliding sequence
of F + width - 2 values, so each program writes it as one contiguous tile
instead of one strided store per (token, window slot). Results match
``causal_conv1d_update`` with ``intermediate_conv_window`` on that layout.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_linear_verify_kernel(
    x,  # [B, dim, F] (any strides)
    stride_x_seq,
    stride_x_dim,
    stride_x_tok,
    conv_state,  # [slots, dim, width - 1]
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    conv_state_indices,  # [B] int
    weight,  # [dim, width]
    stride_w_dim,
    stride_w_tok,
    bias,
    window,  # base of the window view's physical memory
    stride_win_seq,
    stride_win_dim,
    stride_win_pos,  # stride between consecutive sliding positions
    window_indices,  # [B] int
    out,  # [B, dim, F]
    stride_o_seq,
    stride_o_dim,
    stride_o_tok,
    dim,
    F: tl.constexpr,
    WIDTH: tl.constexpr,
    NP2_SEQ: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    seq = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = channels < dim
    state_slot = tl.load(conv_state_indices + seq).to(tl.int64)
    window_slot = tl.load(window_indices + seq).to(tl.int64)
    # Padding requests carry a negative slot; they produce no output or state.
    valid = (state_slot >= 0) & (window_slot >= 0)

    # Sliding sequence s[0 .. WIDTH-2] = conv_state, s[WIDTH-1 + t] = x[t].
    pos = tl.arange(0, NP2_SEQ)
    from_state = pos < WIDTH - 1
    tok = pos - (WIDTH - 1)
    from_x = (pos >= WIDTH - 1) & (tok < F)
    seq_vals = tl.load(
        conv_state
        + state_slot * stride_cs_seq
        + channels[:, None] * stride_cs_dim
        + pos[None, :] * stride_cs_tok,
        mask=valid & c_mask[:, None] & from_state[None, :],
        other=0.0,
    ).to(tl.float32)
    seq_vals += tl.load(
        x
        + seq * stride_x_seq
        + channels[:, None] * stride_x_dim
        + tok[None, :] * stride_x_tok,
        mask=valid & c_mask[:, None] & from_x[None, :],
        other=0.0,
    ).to(tl.float32)

    # Window for token t = s[t + 1 .. t + WIDTH - 1]; the dedup memory holds
    # s[1 .. F + WIDTH - 2] at consecutive sliding positions.
    win_pos = pos - 1
    win_mask = (pos >= 1) & (pos <= F + WIDTH - 2)
    tl.store(
        window
        + window_slot * stride_win_seq
        + channels[:, None] * stride_win_dim
        + win_pos[None, :] * stride_win_pos,
        seq_vals.to(window.dtype.element_ty),
        mask=valid & c_mask[:, None] & win_mask[None, :],
    )

    if HAS_BIAS:
        b = tl.load(bias + channels, mask=c_mask, other=0.0).to(tl.float32)
    load_mask = valid & c_mask
    for t in tl.static_range(F):
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        if HAS_BIAS:
            acc += b
        for j in tl.static_range(WIDTH):
            w = tl.load(
                weight + channels * stride_w_dim + j * stride_w_tok, mask=c_mask, other=0.0
            ).to(tl.float32)
            # s[t + j]: a conv_state column or an input token, fixed at compile time.
            if t + j < WIDTH - 1:
                col = tl.load(
                    conv_state
                    + state_slot * stride_cs_seq
                    + channels * stride_cs_dim
                    + (t + j) * stride_cs_tok,
                    mask=load_mask,
                    other=0.0,
                )
            else:
                col = tl.load(
                    x + seq * stride_x_seq + channels * stride_x_dim
                    + (t + j - (WIDTH - 1)) * stride_x_tok,
                    mask=load_mask,
                    other=0.0,
                )
            acc += w * col.to(tl.float32)
        if SILU:
            acc = acc / (1.0 + tl.exp(-acc))
        tl.store(
            out + seq * stride_o_seq + channels * stride_o_dim + t * stride_o_tok,
            acc.to(out.dtype.element_ty),
            mask=valid & c_mask,
        )

    # Only after the outputs read the old state: like causal_conv1d_update,
    # leave conv_state at the state after all F tokens (s[F .. F + WIDTH - 2]);
    # the speculative commit then restores the accepted step's window.
    state_pos = pos - F
    state_mask = (pos >= F) & (pos <= F + WIDTH - 2)
    tl.store(
        conv_state
        + state_slot * stride_cs_seq
        + channels[:, None] * stride_cs_dim
        + state_pos[None, :] * stride_cs_tok,
        seq_vals.to(conv_state.dtype.element_ty),
        mask=valid & c_mask[:, None] & state_mask[None, :],
    )



def is_dedup_conv_window(window: torch.Tensor, width: int) -> bool:
    """True for the sliding (deduplicated) window view ``[slots, F, dim, width-1]``."""
    return window.dim() == 4 and window.stride(1) == window.stride(3) and width >= 2


def causal_conv1d_linear_verify(
    x: torch.Tensor,  # [B, dim, F]
    conv_state: torch.Tensor,  # [slots, dim, width - 1]
    weight: torch.Tensor,  # [dim, width]
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    conv_state_indices: torch.Tensor,  # [B]
    intermediate_conv_window: torch.Tensor,  # dedup view [slots, F, dim, width-1]
    intermediate_state_indices: torch.Tensor,  # [B]
    block_c: Optional[int] = None,
) -> torch.Tensor:
    batch, dim, F = x.shape
    if block_c is None:
        # Measured on H200 at dim 10240, F 8: small batches need more programs.
        block_c = 32 if batch < 4 else 128
    width = weight.shape[1]
    out = torch.empty_like(x)
    window = intermediate_conv_window
    grid = (batch, triton.cdiv(dim, block_c))
    _causal_conv1d_linear_verify_kernel[grid](
        x, x.stride(0), x.stride(1), x.stride(2),
        conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        conv_state_indices,
        weight, weight.stride(0), weight.stride(1),
        bias if bias is not None else weight,
        window, window.stride(0), window.stride(2), window.stride(1),
        intermediate_state_indices,
        out, out.stride(0), out.stride(1), out.stride(2),
        dim,
        F=F,
        WIDTH=width,
        NP2_SEQ=triton.next_power_of_2(F + width - 1),
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out
