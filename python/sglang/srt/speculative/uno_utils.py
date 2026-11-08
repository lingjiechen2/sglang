from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl
from flashinfer import top_k as _flashinfer_top_k

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_speculative_verify_target_probs,
)
from sglang.srt.speculative.spec_utils import fast_sample
from sglang.srt.speculative.uno_rectifier import UnoRectifier

_SPARSE_TOP_K_LIMIT = 128
_SOFTMAX_CHUNK = 8192


@triton.jit
def _softmax_chunk_stats_kernel(x, row_max, row_sum, vocab, CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    v = tl.load(x + row.to(tl.int64) * vocab + offs, mask=offs < vocab, other=-float("inf")).to(tl.float32)
    m = tl.max(v, axis=0)
    tl.store(row_max + row * NUM_CHUNKS + chunk, m)
    # A fully masked chunk (all -inf) contributes nothing; exp(-inf - -inf) is NaN.
    s = tl.where(m == -float("inf"), 0.0, tl.sum(tl.exp(v - m), axis=0))
    tl.store(row_sum + row * NUM_CHUNKS + chunk, s)


@triton.jit
def _softmax_normalize_kernel(x, out, row_max, row_sum, vocab, CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    k = tl.arange(0, NUM_CHUNKS)
    used = k < tl.num_programs(1)
    maxes = tl.load(row_max + row * NUM_CHUNKS + k, mask=used, other=-float("inf"))
    m = tl.max(maxes, axis=0)
    sums = tl.load(row_sum + row * NUM_CHUNKS + k, mask=used, other=0.0)
    total = tl.sum(tl.where(used, sums * tl.exp(maxes - m), 0.0), axis=0)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    v = tl.load(x + row.to(tl.int64) * vocab + offs, mask=offs < vocab, other=-float("inf")).to(tl.float32)
    tl.store(out + row.to(tl.int64) * vocab + offs, tl.exp(v - m) / total, mask=offs < vocab)


def _split_row_softmax(logits: torch.Tensor) -> torch.Tensor:
    """fp32 softmax over the last dim with each row split across many blocks.

    The verify and draft rows are few (B * (F + 1)) and the vocabulary large, so
    torch's one-block-per-row softmax leaves most SMs idle (9 x 248320 on H200:
    61 -> 10 us).
    """
    logits = logits.contiguous()
    rows, vocab = logits.shape
    num_chunks = triton.next_power_of_2(triton.cdiv(vocab, _SOFTMAX_CHUNK))
    # Only the first cdiv(vocab, CHUNK) columns are written and read.
    row_max = torch.empty((rows, num_chunks), dtype=torch.float32, device=logits.device)
    row_sum = torch.empty((rows, num_chunks), dtype=torch.float32, device=logits.device)
    out = torch.empty((rows, vocab), dtype=torch.float32, device=logits.device)
    grid = (rows, triton.cdiv(vocab, _SOFTMAX_CHUNK))
    _softmax_chunk_stats_kernel[grid](
        logits, row_max, row_sum, vocab, CHUNK=_SOFTMAX_CHUNK, NUM_CHUNKS=num_chunks, num_warps=8
    )
    _softmax_normalize_kernel[grid](
        logits, out, row_max, row_sum, vocab, CHUNK=_SOFTMAX_CHUNK, NUM_CHUNKS=num_chunks, num_warps=8
    )
    return out


def _normalize_sparse_topk_probs(
    topk_logits: torch.Tensor,
    temperatures: torch.Tensor,
    valid: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Normalize a compact top-k support with top-k-first top-p semantics."""
    scaled = topk_logits.float() / temperatures
    scaled = scaled.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    probs = probs.masked_fill((cdf - probs) > top_ps, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.compile(dynamic=True)
def _sparse_rejection_from_support(
    candidates: torch.Tensor,
    target_ids: torch.Tensor,
    target_probs: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    accept_uniforms: torch.Tensor,
    final_uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact p/q rejection sampling on compact linear-chain supports."""
    batch_size, forward_width = candidates.shape
    num_proposals = forward_width - 1

    if num_proposals == 0:
        final_ids = target_ids[:, 0]
        final_probs = target_probs[:, 0]
        accepted_counts = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=candidates.device,
        )
    else:
        proposal_ids = candidates[:, 1:]
        p_proposal = torch.where(
            target_ids[:, :num_proposals].eq(proposal_ids.unsqueeze(-1)),
            target_probs[:, :num_proposals],
            torch.zeros(
                (),
                dtype=target_probs.dtype,
                device=target_probs.device,
            ),
        ).sum(dim=-1)
        q_proposal = torch.where(
            draft_ids.eq(proposal_ids.unsqueeze(-1)),
            draft_probs,
            torch.zeros(
                (),
                dtype=draft_probs.dtype,
                device=draft_probs.device,
            ),
        ).sum(dim=-1)
        ratios = torch.where(
            q_proposal > 0,
            p_proposal / q_proposal,
            torch.zeros_like(p_proposal),
        ).clamp_(max=1.0)
        accepted_flags = accept_uniforms[:, :num_proposals] < ratios
        accepted_counts = accepted_flags.to(torch.int32).cumprod(dim=1).sum(dim=1)

        batch_indices = torch.arange(batch_size, device=candidates.device)
        final_rows = accepted_counts.to(torch.long)
        final_ids = target_ids[batch_indices, final_rows]
        final_probs = target_probs[batch_indices, final_rows]

        rejected = final_rows < num_proposals
        draft_rows = final_rows.clamp(max=num_proposals - 1)
        final_draft_ids = draft_ids[batch_indices, draft_rows]
        final_draft_probs = draft_probs[batch_indices, draft_rows]
        q_on_target = torch.where(
            final_ids.unsqueeze(2).eq(final_draft_ids.unsqueeze(1)),
            final_draft_probs.unsqueeze(1),
            torch.zeros(
                (),
                dtype=final_draft_probs.dtype,
                device=final_draft_probs.device,
            ),
        ).sum(dim=2)
        correction_probs = (final_probs - q_on_target).clamp_min_(0.0)
        correction_sum = correction_probs.sum(dim=1, keepdim=True)
        correction_probs = torch.where(
            correction_sum > 0,
            correction_probs / correction_sum.clamp_min(1e-12),
            final_probs,
        )
        final_probs = torch.where(
            rejected[:, None],
            correction_probs,
            final_probs,
        )

    cdf = torch.cumsum(final_probs, dim=-1)
    thresholds = final_uniforms * final_probs.sum(dim=-1)
    sampled_offsets = (cdf <= thresholds[:, None]).sum(dim=-1)
    sampled_offsets.clamp_(max=final_ids.shape[-1] - 1)
    bonus = (
        final_ids.gather(1, sampled_offsets[:, None]).squeeze(1).to(candidates.dtype)
    )
    return accepted_counts, bonus


@torch.compile(dynamic=True)
def _build_sparse_target_support_tensors(
    next_token_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    batch_size: int,
    forward_width: int,
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact support using the fastest available top-k primitive."""
    rows = batch_size * forward_width
    topk_logits, topk_ids = _flashinfer_top_k(
        next_token_logits.contiguous(),
        max_top_k,
        sorted=True,
        deterministic=False,
    )

    expanded_temperatures = torch.repeat_interleave(
        temperatures,
        forward_width,
        dim=0,
    ).reshape(rows, -1)
    expanded_top_ks = torch.repeat_interleave(
        top_ks,
        forward_width,
        dim=0,
    ).reshape(rows, 1)
    expanded_top_ps = torch.repeat_interleave(
        top_ps,
        forward_width,
        dim=0,
    ).reshape(rows, 1)
    ranks = torch.arange(
        max_top_k,
        dtype=expanded_top_ks.dtype,
        device=next_token_logits.device,
    )[None, :]
    probs = _normalize_sparse_topk_probs(
        topk_logits,
        expanded_temperatures,
        ranks < expanded_top_ks,
        expanded_top_ps,
    )
    return (
        topk_ids.view(batch_size, forward_width, max_top_k),
        probs.view(batch_size, forward_width, max_top_k),
    )


def _build_sparse_target_support(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    batch_size: int,
    forward_width: int,
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact target token IDs/probabilities without a dense scatter."""
    if bool(getattr(sampling_info, "need_top_p_sampling", False)):
        top_ps = sampling_info.top_ps
    else:
        top_ps = torch.ones(
            (batch_size,),
            dtype=torch.float32,
            device=next_token_logits.device,
        )

    return _build_sparse_target_support_tensors(
        next_token_logits,
        sampling_info.temperatures,
        sampling_info.top_ks,
        top_ps,
        batch_size,
        forward_width,
        max_top_k,
    )


def _sample_from_support(
    support_ids: torch.Tensor,
    support_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample one token from every compact support row."""
    flat_ids = support_ids.flatten(0, 1)
    flat_probs = support_probs.flatten(0, 1)
    _, offsets = fast_sample(flat_probs)
    return flat_ids.gather(1, offsets).view(support_ids.shape[:2])


@dataclass(frozen=True)
class UnoDraftDistribution:
    """The exact q used to sample future UNO proposal rows."""

    probs: torch.Tensor
    token_ids: torch.Tensor | None = None


def _run_sparse_rejection(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    draft_distribution: UnoDraftDistribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    if draft_distribution.token_ids is None:
        raise RuntimeError("Sparse UNO verification requires sparse draft q.")

    batch_size, forward_width = candidates.shape
    support_ids, support_probs = _build_sparse_target_support(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        batch_size=batch_size,
        forward_width=forward_width,
        max_top_k=max_top_k,
    )

    # Preserve the legacy path's two RNG draws and tensor shapes. The final
    # uniforms select the correction/bonus; the first F - 1 acceptance coins
    # are consumed by a linear chain.
    accept_uniforms = torch.rand(
        (batch_size, forward_width),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    final_uniforms = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    return _sparse_rejection_from_support(
        candidates,
        support_ids,
        support_probs,
        draft_distribution.token_ids,
        draft_distribution.probs,
        accept_uniforms,
        final_uniforms,
    )


def _build_dense_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    batch_size: int,
    forward_width: int,
    max_top_k: int,
    uniform_top_k_value: int | None,
) -> torch.Tensor:
    """Build the dense sampling distribution used by SGLang verification."""
    return build_speculative_verify_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=forward_width,
        bs=batch_size,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
        use_sparse_topk=True,
        dense_softmax=_split_row_softmax,
    )


def _sample_from_dense_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token from every dense distribution row."""
    _, token_ids = fast_sample(probs.flatten(0, 1))
    return token_ids.view(probs.shape[:2])


_REJECT_CHUNK = 8192
_SPLIT_REJECTION = os.environ.get("UNO_SPLIT_REJECTION", "0") == "1"


@triton.jit
def _chain_accept_kernel(
    candidates, target_probs, draft_probs, uniforms, accepted, sample_row,
    vocab, F: tl.constexpr, NUM_ROWS_Q: tl.constexpr,
):
    """Linear-chain p/q acceptance (same test as the classic kernel): returns the
    accepted count and the row whose residual (or, all accepted, whose target p)
    the correction token is sampled from (row == F - 1 means plain target p)."""
    b = tl.program_id(0)
    n = 0
    go = 1
    for step in tl.static_range(1, F):
        if go == 1:
            tok = tl.load(candidates + b * F + step).to(tl.int64)
            p = tl.load(target_probs + (b * F + step - 1).to(tl.int64) * vocab + tok)
            q = tl.load(draft_probs + (b * NUM_ROWS_Q + step - 1).to(tl.int64) * vocab + tok)
            coin = tl.load(uniforms + b * F + step - 1)
            if ((q > 0.0) & (q <= 1.0)) & (coin * q < p):
                n += 1
            else:
                go = 0
    tl.store(accepted + b, n)
    tl.store(sample_row + b, n)


@triton.jit
def _residual_race_kernel(
    target_probs, draft_probs, sample_row, seed, best_score, best_idx,
    vocab, F: tl.constexpr, NUM_ROWS_Q: tl.constexpr, CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,
):
    """Exponential race over one vocabulary chunk: argmax r_i / E_i with E_i ~ Exp(1)
    picks i with probability r_i / sum(r), so the chunk maxima reduce to an exact
    sample of the residual max(p - q, 0) (or of p when every proposal passed)."""
    b = tl.program_id(0)
    c = tl.program_id(1)
    row = tl.load(sample_row + b)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    p = tl.load(target_probs + (b * F + row).to(tl.int64) * vocab + offs, mask=mask, other=0.0)
    if row < F - 1:
        q = tl.load(draft_probs + (b * NUM_ROWS_Q + row).to(tl.int64) * vocab + offs, mask=mask, other=0.0)
        q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
        r = tl.maximum(p - q, 0.0)
    else:
        r = p
    u = tl.rand(seed, (b.to(tl.int64) * vocab + offs).to(tl.int32))
    e = -tl.log(tl.maximum(u, 1e-30))
    score = tl.where(mask & (r > 0.0), r / e, -1.0)
    best = tl.max(score, axis=0)
    idx = tl.argmax(score, axis=0)
    tl.store(best_score + b * NUM_CHUNKS + c, best)
    tl.store(best_idx + b * NUM_CHUNKS + c, c * CHUNK + idx)


@triton.jit
def _race_reduce_kernel(best_score, best_idx, out_tokens, vocab, NUM_CHUNKS: tl.constexpr, CHUNKS_PAD: tl.constexpr):
    b = tl.program_id(0)
    k = tl.arange(0, CHUNKS_PAD)
    m = k < NUM_CHUNKS
    sc = tl.load(best_score + b * NUM_CHUNKS + k, mask=m, other=-2.0)
    ix = tl.load(best_idx + b * NUM_CHUNKS + k, mask=m, other=0)
    j = tl.argmax(sc, axis=0)
    tok = tl.sum(tl.where(k == j, ix, 0), axis=0)
    # Degenerate residual (all zero): fall back to the last token like the classic kernel.
    tok = tl.where(tl.max(sc, axis=0) < 0.0, vocab - 1, tok)
    tl.store(out_tokens + b, tok)


def _split_dense_rejection(candidates, target_probs, draft_probs, accept_uniforms):
    """Multi-block replacement for the classic chain kernel's correction sampling
    (UNO_SPLIT_REJECTION=1): same acceptance test and coins; the correction token
    is an exact sample of the same distribution via an exponential race split
    across the vocabulary instead of one block's two-pass CDF scan."""
    batch_size, width = candidates.shape
    vocab = target_probs.shape[-1]
    num_rows_q = draft_probs.shape[1]
    dev = candidates.device
    tp = target_probs.contiguous()
    dp = draft_probs.contiguous()
    accepted = torch.empty((batch_size,), dtype=torch.int32, device=dev)
    sample_row = torch.empty((batch_size,), dtype=torch.int32, device=dev)
    _chain_accept_kernel[(batch_size,)](
        candidates.contiguous(), tp, dp, accept_uniforms.contiguous(), accepted, sample_row,
        vocab, F=width, NUM_ROWS_Q=num_rows_q,
    )
    num_chunks = triton.cdiv(vocab, _REJECT_CHUNK)
    best_score = torch.empty((batch_size, num_chunks), dtype=torch.float32, device=dev)
    best_idx = torch.empty((batch_size, num_chunks), dtype=torch.int32, device=dev)
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    _residual_race_kernel[(batch_size, num_chunks)](
        tp, dp, sample_row, seed, best_score, best_idx,
        vocab, F=width, NUM_ROWS_Q=num_rows_q, CHUNK=_REJECT_CHUNK, NUM_CHUNKS=num_chunks, num_warps=8,
    )
    tokens = torch.empty((batch_size,), dtype=torch.int64, device=dev)
    _race_reduce_kernel[(batch_size,)](
        best_score, best_idx, tokens, vocab, NUM_CHUNKS=num_chunks,
        CHUNKS_PAD=triton.next_power_of_2(num_chunks),
    )
    return accepted, tokens.to(candidates.dtype)


def _run_dense_rejection(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None,
    draft_distribution: UnoDraftDistribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SGLang's fused linear-chain p/q rejection kernel."""
    if draft_distribution.token_ids is not None:
        raise RuntimeError("Dense UNO verification requires dense draft q.")

    batch_size, forward_width = candidates.shape
    target_probs = _build_dense_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        batch_size=batch_size,
        forward_width=forward_width,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
    )
    accept_uniforms = torch.rand(
        (batch_size, forward_width),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    if _SPLIT_REJECTION:
        return _split_dense_rejection(
            candidates, target_probs, draft_distribution.probs, accept_uniforms
        )
    final_uniforms = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accepted_counts,
    ) = _get_or_create_chain_verify_buffers(
        bs=batch_size,
        draft_token_num=forward_width,
        device=next_token_logits.device,
    )
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accepted_counts,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=accept_uniforms,
        uniform_samples_for_final_sampling=final_uniforms,
        target_probs=target_probs,
        draft_probs=draft_distribution.probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    rows = torch.arange(batch_size, device=candidates.device)
    bonus_positions = accept_index[
        rows,
        accepted_counts.to(torch.long),
    ].to(torch.long)
    bonus = predicts[bonus_positions].to(candidates.dtype)
    return accepted_counts, bonus


@dataclass
class UnoSamplingResult:
    output_ids: torch.Tensor
    accept_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    next_seed_tokens: torch.Tensor


@dataclass
class UnoTreeSamplingResult:
    output_ids: torch.Tensor
    accept_lens: torch.Tensor


def build_uno_draft_input(
    *,
    seed_tokens: torch.Tensor,
    forward_width: int,
    vocab_size: int,
    noise_tokens: torch.Tensor | None = None,  # for testing
    hint_tokens: torch.Tensor | None = None,
    hint_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build one ``[seed, uniform noise...]`` row per request.

    Random noise is sampled independently from ``[0, vocab_size)``.

    Supplying ``noise_tokens`` bypasses random generation. This is used
    by deterministic tests and must have shape
    ``(batch_size, forward_width - 1)``.

    Draft recycling: ``hint_tokens`` ``[batch, W]`` and ``hint_lens`` ``[batch]``
    replace the first ``hint_lens[b]`` noise slots of request ``b`` with the
    unchecked proposals of the previous cycle. The noise is drawn first either way,
    so the RNG stream does not depend on the hints. Without hints the result is
    unchanged.

    ``forward_width == 1`` never generates noise or consumes RNG state.
    """
    seed_tokens = seed_tokens.reshape(-1).to(dtype=torch.int64)
    batch_size = seed_tokens.numel()
    noise_shape = (batch_size, forward_width - 1)

    if forward_width == 1:
        return seed_tokens[:, None]

    if noise_tokens is None:
        noise_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=noise_shape,
            dtype=torch.int64,
            device=seed_tokens.device,
        )
    else:
        noise_tokens = noise_tokens.to(
            device=seed_tokens.device,
            dtype=torch.int64,
        )

    if hint_tokens is not None:
        width = hint_tokens.shape[1]
        if hint_lens is None or width > forward_width - 1:
            raise ValueError("UNO recycle hints need hint_lens and fit the noise window.")
        use_hint = torch.arange(width, device=seed_tokens.device)[None, :] < hint_lens[
            :, None
        ].to(seed_tokens.device)
        noise_tokens = noise_tokens.clone()
        noise_tokens[:, :width] = torch.where(
            use_hint,
            hint_tokens.to(device=seed_tokens.device, dtype=torch.int64),
            noise_tokens[:, :width],
        )

    draft_input_ids = seed_tokens.new_empty((batch_size, forward_width))
    draft_input_ids[:, 0].copy_(seed_tokens)
    draft_input_ids[:, 1:].copy_(noise_tokens)

    return draft_input_ids


def extract_uno_recycle_tail(
    candidates: torch.Tensor,  # [B, F]
    accept_lens: torch.Tensor,  # [B]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unchecked proposals after the correction token, for the next cycle's noise window.

    A cycle commits ``accept_lens`` tokens: ``candidates[:, :accept_lens - 1]`` and the
    correction (the target's token at the first rejected proposal, or the bonus token).
    ``candidates[:, accept_lens:]`` were never verified and propose the positions that
    follow the correction, i.e. noise slots 0, 1, ... of the next draft input.
    Returns ``(tail [B, F - 2], tail_lens [B])``; entries past ``tail_lens`` are zero.
    """
    batch_size, forward_width = candidates.shape
    width = max(forward_width - 2, 0)
    accept_lens = accept_lens.to(torch.int64)
    lens = (forward_width - accept_lens).clamp(min=0, max=width)
    cols = torch.arange(width, device=candidates.device)[None, :]
    source = (accept_lens[:, None] + cols).clamp(max=forward_width - 1)
    tail = candidates.gather(1, source) * (cols < lens[:, None])
    return tail, lens


def sample_uno_candidates(
    *,
    draft_logits: torch.Tensor,  # [B, F, V]
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None = None,
    rectifier: UnoRectifier | None = None,
    include_clean: bool = False,
) -> tuple[torch.Tensor, UnoDraftDistribution]:
    """Sample 1 clean token from seed and F-1 draft tokens.
    Depending on max_top_k value, may use sparse representations for efficiency.
    candidates: [B, F] sampled tokens including clean and draft.
    draft_distribution: probabilities of the draft tokens for rejection sampling;
    with ``include_clean`` (--uno-draft-weight-quant) also of the clean token, which
    the bf16 verify then accepts or rejects like the drafts.
    """
    if rectifier is not None and draft_logits.shape[1] > 1:
        return _sample_rectified_candidates(
            draft_logits=draft_logits,
            sampling_info=sampling_info,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
            rectifier=rectifier,
            include_clean=include_clean,
        )
    first = 0 if include_clean else 1
    batch_size, forward_width, vocab_size = draft_logits.shape
    flat_logits = draft_logits.reshape(-1, vocab_size)
    if max_top_k <= _SPARSE_TOP_K_LIMIT:
        support_ids, support_probs = _build_sparse_target_support(
            next_token_logits=flat_logits,
            sampling_info=sampling_info,
            batch_size=batch_size,
            forward_width=forward_width,
            max_top_k=max_top_k,
        )
        candidates = _sample_from_support(support_ids, support_probs)
        draft_distribution = UnoDraftDistribution(
            token_ids=support_ids[:, first:],
            probs=support_probs[:, first:],
        )
    else:
        probs = _build_dense_probs(
            next_token_logits=flat_logits,
            sampling_info=sampling_info,
            batch_size=batch_size,
            forward_width=forward_width,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )
        candidates = _sample_from_dense_probs(probs)
        draft_distribution = UnoDraftDistribution(probs=probs[:, first:])
    return candidates, draft_distribution


def _sample_rectified_candidates(
    *,
    draft_logits: torch.Tensor,  # [B, F, V]
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None,
    rectifier: UnoRectifier,
    include_clean: bool = False,
) -> tuple[torch.Tensor, UnoDraftDistribution]:
    """Sample the clean root as usual and the F-1 drafts from the rectifier."""
    batch_size, forward_width, vocab_size = draft_logits.shape
    sparse = max_top_k <= _SPARSE_TOP_K_LIMIT
    clean_ids = clean_probs = None
    if not include_clean:
        clean_root_tokens = sample_uno_clean_root(
            seed_tokens=None,
            draft_logits=draft_logits,
            sampling_info=sampling_info,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )
    elif sparse:
        # Same per-request top-k/top-p q, laid out on the rectifier's support width
        # so the clean row and the draft rows share one sparse q tensor.
        if max_top_k > rectifier.num_candidates:
            raise ValueError(
                f"max top-k {max_top_k} exceeds the rectifier lattice "
                f"({rectifier.num_candidates} candidates)."
            )
        clean_ids, clean_probs = _build_sparse_target_support(
            next_token_logits=draft_logits[:, 0, :],
            sampling_info=sampling_info,
            batch_size=batch_size,
            forward_width=1,
            max_top_k=rectifier.num_candidates,
        )
        clean_root_tokens = _sample_from_support(clean_ids, clean_probs)[:, 0]
    else:
        clean_probs = _build_dense_probs(
            next_token_logits=draft_logits[:, 0, :].contiguous(),
            sampling_info=sampling_info,
            batch_size=batch_size,
            forward_width=1,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )
        clean_root_tokens = _sample_from_dense_probs(clean_probs)[:, 0]
    if sampling_info.need_top_p_sampling:
        top_ps = sampling_info.top_ps
    else:
        top_ps = torch.ones(
            (batch_size,), dtype=torch.float32, device=draft_logits.device
        )
    rectified = rectifier.sample(
        draft_logits=draft_logits[:, 1:, :],
        first_prev_tokens=clean_root_tokens,
        temperatures=sampling_info.temperatures,
        top_ks=sampling_info.top_ks,
        top_ps=top_ps,
    )
    candidates = torch.cat(
        (clean_root_tokens.to(torch.int64)[:, None], rectified.drafts), dim=1
    )
    if sparse:
        if include_clean:
            return candidates, UnoDraftDistribution(
                token_ids=torch.cat((clean_ids, rectified.candidates), dim=1),
                probs=torch.cat((clean_probs.float(), rectified.probs), dim=1),
            )
        return candidates, UnoDraftDistribution(
            token_ids=rectified.candidates,
            probs=rectified.probs,
        )
    # The dense chain kernel reads q over the whole vocabulary.
    dense_probs = torch.zeros(
        (batch_size, forward_width - 1, vocab_size),
        dtype=torch.float32,
        device=draft_logits.device,
    )
    dense_probs.scatter_(2, rectified.candidates, rectified.probs)
    if include_clean:
        dense_probs = torch.cat((clean_probs.float(), dense_probs), dim=1)
    return candidates, UnoDraftDistribution(probs=dense_probs)


def rectify_uno_greedy_candidates(
    *,
    candidates: torch.Tensor,  # [B, F] argmax of every draft row
    draft_logits: torch.Tensor,  # [B, F, V]
    rectifier: UnoRectifier,
) -> torch.Tensor:
    """Replace the F-1 argmax drafts with the rectified argmax chain."""
    if candidates.shape[1] == 1:
        return candidates
    drafts = rectifier.greedy(
        draft_logits=draft_logits[:, 1:, :],
        first_prev_tokens=candidates[:, 0],
    )
    return torch.cat((candidates[:, :1], drafts.to(candidates.dtype)), dim=1)


def sample_uno_clean_root(
    *,
    seed_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None = None,
) -> torch.Tensor:
    """Sample the clean root using the current UNO sampling path."""
    del seed_tokens
    candidates, _ = sample_uno_candidates(
        draft_logits=draft_logits[:, :1, :].contiguous(),
        sampling_info=sampling_info,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
    )
    return candidates[:, 0]


def sample_uno_tree_target_tokens(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    batch_size: int,
    verify_width: int,
    max_top_k: int,
) -> torch.Tensor:
    """Sample one target token per verify node from compact top-k support."""
    support_ids, support_probs = _build_sparse_target_support(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        batch_size=batch_size,
        forward_width=verify_width,
        max_top_k=max_top_k,
    )
    return _sample_from_support(support_ids, support_probs)


def pack_uno_tree_result(
    *,
    clean_root_tokens: torch.Tensor | None,
    eagle_predict: torch.Tensor,
    eagle_accept_lens: torch.Tensor,
    draft_width: int,
    batch_size: int | None = None,
) -> UnoTreeSamplingResult:
    """Convert an internal EAGLE tree result into UNO's public row.

    ``clean_root_tokens`` is None for seed-root trees: the EAGLE path already
    starts after the seed, so it is the whole row (up to F accepted + bonus)."""
    if clean_root_tokens is None:
        verify_width = eagle_predict.numel() // batch_size
        predict_rows = eagle_predict.reshape(batch_size, verify_width)
        output_ids = predict_rows.new_zeros((batch_size, draft_width + 1), dtype=torch.int64)
        width = min(draft_width + 1, verify_width)
        output_ids[:, :width].copy_(predict_rows[:, :width].to(dtype=output_ids.dtype))
        return UnoTreeSamplingResult(output_ids=output_ids, accept_lens=eagle_accept_lens)
    clean_root_tokens = clean_root_tokens.reshape(-1).to(dtype=torch.int64)
    batch_size = clean_root_tokens.numel()
    verify_width = eagle_predict.numel() // batch_size
    predict_rows = eagle_predict.reshape(batch_size, verify_width)

    output_ids = clean_root_tokens.new_zeros((batch_size, draft_width + 1))
    output_ids[:, 0].copy_(clean_root_tokens)
    output_ids[:, 1:].copy_(predict_rows[:, :draft_width].to(dtype=output_ids.dtype))
    return UnoTreeSamplingResult(
        output_ids=output_ids,
        accept_lens=eagle_accept_lens + 1,
    )


def pack_uno_result(
    *,
    candidates: torch.Tensor,  # [B, F]
    accepted_proposal_counts: torch.Tensor,  # [B]
    bonus_tokens: torch.Tensor,  # [B]
    committed_frontiers: torch.Tensor,  # [B]
) -> UnoSamplingResult:
    """Pack acceptance into fixed-width UNO output rows."""
    batch_size, forward_width = candidates.shape
    output_ids = candidates.new_zeros((batch_size, forward_width + 1))
    output_ids[:, :forward_width].copy_(candidates)
    output_ids.scatter_(
        1,
        (accepted_proposal_counts.to(torch.long) + 1)[:, None],
        bonus_tokens[:, None],
    )

    accept_lens = accepted_proposal_counts + 2
    new_seq_lens = committed_frontiers + accept_lens.to(committed_frontiers.dtype)
    return UnoSamplingResult(
        output_ids=output_ids,
        accept_lens=accept_lens,
        new_seq_lens=new_seq_lens,
        next_seed_tokens=bonus_tokens,
    )


def run_uno_sampling(
    *,
    candidates: torch.Tensor,  # [B, F]
    next_token_logits: torch.Tensor,  # [B x F, V]
    sampling_info: Any,
    committed_frontiers: torch.Tensor,  # [B]
    draft_distribution: UnoDraftDistribution,
    max_top_k: int,
    uniform_top_k_value: int | None = None,
) -> UnoSamplingResult:
    """Verify sampled UNO proposals against target p and pack the result."""
    if draft_distribution.token_ids is not None:
        accepted, bonus = _run_sparse_rejection(
            candidates=candidates,
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            max_top_k=max_top_k,
            draft_distribution=draft_distribution,
        )
    else:
        accepted, bonus = _run_dense_rejection(
            candidates=candidates,
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            draft_distribution=draft_distribution,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )
    return pack_uno_result(
        candidates=candidates,
        accepted_proposal_counts=accepted,
        bonus_tokens=bonus,
        committed_frontiers=committed_frontiers,
    )
