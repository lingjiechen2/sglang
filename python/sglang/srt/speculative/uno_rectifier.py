"""Vanilla Markov rectifier for UNO linear draft rows.

The rectifier adds a previous-token bias ``W2 @ W1[prev]`` to every UNO draft
row (the DSpark vanilla Markov head), where ``prev`` is the token sampled for
the row before it and the clean root conditions the first draft row.  Drafts
are sampled from a lattice of each row's top-K base candidates, so the
transition biases of one step are a single batched GEMM and the full
vocabulary is never biased; the exact q of every row is returned for p/q
verification, which keeps the verifier lossless.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl
from flashinfer import top_k as _flashinfer_top_k
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

_RECTIFIER_CONFIG_FILE = "rectifier_config.json"
_RECTIFIER_WEIGHTS_FILE = "rectifier_model.safetensors"

# Measured on BBQ-7B drafts: base top-256 holds all but ~0.7% of q's mass.
_DEFAULT_NUM_CANDIDATES = 256


@triton.jit
def _rectified_chain_kernel(
    candidate_logits,  # [B, S, K] base draft logits of each row's candidates
    candidates,  # [B, S, K] int64 token of each candidate
    biases,  # [B, S, K, K] bias of candidate k given previous candidate p
    uniforms,  # [B, S, K] fp32, unused when GREEDY
    temperatures,  # [B] fp32
    top_ks,  # [B] int
    top_ps,  # [B] fp32
    drafts,  # [B, S] int64 output
    draft_probs,  # [B, S, K] fp32 output q, unused when GREEDY
    S,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GREEDY: tl.constexpr,
):
    """One program walks one request's draft chain left to right."""
    batch = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_K)
    in_bounds = offsets < K
    if not GREEDY:
        inverse_temperature = 1.0 / tl.load(temperatures + batch).to(tl.float32)
        top_k = tl.minimum(tl.load(top_ks + batch).to(tl.int32), K)
        top_p = tl.load(top_ps + batch).to(tl.float32)
    # The first draft row is conditioned on the clean root, stored in slot 0.
    previous = 0
    for step in range(S):
        row = (batch * S + step) * K
        bias = tl.load(
            biases + (row + previous) * K + offsets, mask=in_bounds, other=0.0
        ).to(tl.float32)
        base = tl.load(candidate_logits + row + offsets, mask=in_bounds, other=0.0)
        scores = tl.where(in_bounds, base.to(tl.float32) + bias, float("-inf"))
        if GREEDY:
            # Break ties toward the lowest token, as torch.argmax does, so the
            # chain does not depend on the top-k kernel's candidate order.
            best_score = tl.max(scores, axis=0)
            tokens = tl.load(candidates + row + offsets, mask=in_bounds, other=0)
            best_token = tl.min(
                tl.where(scores == best_score, tokens, 0x7FFFFFFFFFFFFFFF), axis=0
            )
            previous = tl.argmax(
                ((tokens == best_token) & in_bounds).to(tl.int32), axis=0
            )
        else:
            # Top-k first, then top-p, matching UNO's sparse target support.
            ranked_scores = tl.sort(scores, descending=True)
            kth_score = tl.min(
                tl.where(offsets < top_k, ranked_scores, float("inf")), axis=0
            )
            keep = in_bounds & (scores >= kth_score)
            scaled = tl.where(keep, scores * inverse_temperature, float("-inf"))
            probs = tl.exp(scaled - tl.max(scaled, axis=0))
            probs = probs / tl.sum(probs, axis=0)
            ranked_probs = tl.sort(probs, descending=True)
            preceding = tl.cumsum(ranked_probs, axis=0) - ranked_probs
            cutoff = tl.min(
                tl.where(preceding <= top_p, ranked_probs, float("inf")), axis=0
            )
            probs = tl.where(keep & (probs >= cutoff), probs, 0.0)
            probs = probs / tl.sum(probs, axis=0)
            tl.store(draft_probs + row + offsets, probs, mask=in_bounds)
            # Gumbel-max on log q samples exactly the published q.
            uniform = tl.load(uniforms + row + offsets, mask=in_bounds, other=0.5)
            uniform = tl.minimum(tl.maximum(uniform, 1e-10), 1.0 - 1e-7)
            gumbel = -tl.log(-tl.log(uniform))
            previous = tl.argmax(
                tl.where(probs > 0.0, tl.log(probs) + gumbel, float("-inf")), axis=0
            )
        tl.store(drafts + batch * S + step, tl.load(candidates + row + previous))


class UnoRectifiedDrafts(msgspec.Struct, frozen=True):
    """Drafts sampled from the rectified lattice and the exact q behind them."""

    # [B, S] int64.
    drafts: torch.Tensor
    # [B, S, K] int64 lattice token of each q entry.
    candidates: torch.Tensor
    # [B, S, K] fp32 q over ``candidates``; zero outside the kept support.
    probs: torch.Tensor


class UnoRectifier:
    """DSpark vanilla Markov head applied to UNO draft rows."""

    def __init__(
        self,
        *,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        num_candidates: int = _DEFAULT_NUM_CANDIDATES,
    ):
        if markov_w1.ndim != 2 or markov_w1.shape != markov_w2.shape:
            raise ValueError(
                "UNO rectifier weights must both be [vocab, rank]; got "
                f"{tuple(markov_w1.shape)} and {tuple(markov_w2.shape)}."
            )
        # [V, r]: W1 embeds the previous token, W2 rows score each candidate.
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2
        self.vocab_size = int(markov_w1.shape[0])
        self.num_candidates = min(int(num_candidates), self.vocab_size)

    def _build_lattice(
        self,
        draft_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K base candidates per row and every candidate transition bias.

        Returns values [B, S, K], candidates [B, S, K] and biases [B, S, K, K],
        where biases[b, j, p, k] is the bias of candidate k of row j given
        candidate p of row j - 1 (the clean root for j = 0).
        """
        batch_size, num_rows, vocab_size = draft_logits.shape
        if vocab_size != self.vocab_size:
            raise RuntimeError(
                f"UNO rectifier vocab ({self.vocab_size}) does not match the "
                f"draft logits ({vocab_size})."
            )
        num_candidates = self.num_candidates
        flat_logits = draft_logits.reshape(batch_size * num_rows, vocab_size)
        if num_candidates < vocab_size:
            values, candidates = _flashinfer_top_k(
                flat_logits.contiguous(),
                num_candidates,
                sorted=True,
                deterministic=False,
            )
        else:
            values, candidates = torch.topk(flat_logits, num_candidates, dim=-1)
        shape = (batch_size, num_rows, num_candidates)
        values = values.view(shape).contiguous()
        candidates = candidates.to(torch.int64).view(shape).contiguous()
        prev_tokens = torch.cat(
            (
                first_prev_tokens.reshape(batch_size, 1, 1)
                .to(torch.int64)
                .expand(batch_size, 1, num_candidates),
                candidates[:, :-1],
            ),
            dim=1,
        )
        biases = torch.matmul(
            torch.nn.functional.embedding(prev_tokens, self.markov_w1),
            torch.nn.functional.embedding(candidates, self.markov_w2).transpose(-1, -2),
        )
        return values, candidates, biases

    @torch.inference_mode()
    def sample(
        self,
        *,
        draft_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> UnoRectifiedDrafts:
        """Sample one rectified draft chain per request.

        ``draft_logits`` is [B, S, V] for the S draft rows; ``first_prev_tokens``
        [B] is the clean root.  Per-request temperature, top-k and top-p apply
        to the rectified candidates; top_k = 1 rows are greedy.
        """
        batch_size, num_rows, _ = draft_logits.shape
        values, candidates, biases = self._build_lattice(
            draft_logits, first_prev_tokens
        )
        num_candidates = candidates.shape[-1]
        drafts = torch.empty(
            (batch_size, num_rows), dtype=torch.int64, device=draft_logits.device
        )
        probs = torch.empty(
            candidates.shape, dtype=torch.float32, device=draft_logits.device
        )
        uniforms = torch.rand(
            candidates.shape, dtype=torch.float32, device=draft_logits.device
        )
        _rectified_chain_kernel[(batch_size,)](
            values,
            candidates,
            biases,
            uniforms,
            temperatures.reshape(batch_size).to(torch.float32).contiguous(),
            top_ks.reshape(batch_size).contiguous(),
            top_ps.reshape(batch_size).to(torch.float32).contiguous(),
            drafts,
            probs,
            num_rows,
            K=num_candidates,
            BLOCK_K=triton.next_power_of_2(num_candidates),
            GREEDY=False,
        )
        return UnoRectifiedDrafts(drafts=drafts, candidates=candidates, probs=probs)

    @torch.inference_mode()
    def greedy(
        self,
        *,
        draft_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return the [B, S] argmax chain of the rectified candidates."""
        batch_size, num_rows, _ = draft_logits.shape
        values, candidates, biases = self._build_lattice(
            draft_logits, first_prev_tokens
        )
        num_candidates = candidates.shape[-1]
        drafts = torch.empty(
            (batch_size, num_rows), dtype=torch.int64, device=draft_logits.device
        )
        _rectified_chain_kernel[(batch_size,)](
            values,
            candidates,
            biases,
            values,
            values,
            drafts,
            values,
            drafts,
            values,
            num_rows,
            K=num_candidates,
            BLOCK_K=triton.next_power_of_2(num_candidates),
            GREEDY=True,
        )
        return drafts


def _strip_rectifier_prefix(name: str) -> str:
    # Some exports keep the trainer's module prefix on the tensor names.
    return name.removeprefix("rectifier.")


def load_uno_rectifier(
    *,
    adapter_path: str,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[UnoRectifier]:
    """Load the vanilla rectifier exported next to the UNO LoRA, if any."""
    if not os.path.isdir(adapter_path):
        adapter_path = snapshot_download(
            adapter_path,
            allow_patterns=[_RECTIFIER_CONFIG_FILE, _RECTIFIER_WEIGHTS_FILE],
        )
    config_path = os.path.join(adapter_path, _RECTIFIER_CONFIG_FILE)
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        config = json.load(f)
    if config.get("type") != "vanilla":
        raise ValueError(
            f"Only vanilla UNO rectifiers are supported; got {config.get('type')!r} "
            f"in {config_path}."
        )
    if int(config["vocab_size"]) != vocab_size:
        raise ValueError(
            f"UNO rectifier vocab ({config['vocab_size']}) does not match the "
            f"model vocab ({vocab_size})."
        )
    state = {
        _strip_rectifier_prefix(name): tensor
        for name, tensor in load_file(
            os.path.join(adapter_path, _RECTIFIER_WEIGHTS_FILE)
        ).items()
    }
    rectifier = UnoRectifier(
        markov_w1=state["markov_w1.weight"].to(device=device, dtype=dtype),
        markov_w2=state["markov_w2.weight"].to(device=device, dtype=dtype),
    )
    logger.info(
        "Loaded UNO rectifier from %s (rank %d, %d candidates).",
        adapter_path,
        rectifier.markov_w1.shape[1],
        rectifier.num_candidates,
    )
    return rectifier
