"""Contracts for the UNO linear draft rectifier."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.uno_rectifier import UnoRectifier, load_uno_rectifier
from sglang.srt.speculative.uno_utils import (
    _SPARSE_TOP_K_LIMIT,
    UnoDraftDistribution,
    _build_sparse_target_support,
    _normalize_sparse_topk_probs,
    rectify_uno_greedy_candidates,
    run_uno_sampling,
    sample_uno_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _rectifier(vocab_size, rank, scale, num_candidates):
    generator = torch.Generator(device=_DEVICE).manual_seed(0)
    weights = [
        torch.randn(vocab_size, rank, generator=generator, device=_DEVICE) * scale
        for _ in range(2)
    ]
    return UnoRectifier(
        markov_w1=weights[0], markov_w2=weights[1], num_candidates=num_candidates
    )


def _sampling_info(temperatures, top_ks, top_ps):
    return SimpleNamespace(
        temperatures=torch.tensor(temperatures, device=_DEVICE).view(-1, 1),
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device=_DEVICE),
        top_ps=torch.tensor(top_ps, device=_DEVICE),
        need_top_p_sampling=any(p < 1.0 for p in top_ps),
        need_top_k_sampling=any(k != TOP_K_ALL for k in top_ks),
        is_all_greedy=all(k <= 1 for k in top_ks),
    )


def _full_vocab_rectified_q(logits, bias, temperature, top_k, top_p):
    """DSpark's full-vocabulary rectified logits under UNO's target truncation."""
    scores, order = torch.sort(logits + bias, descending=True)
    ranks = torch.arange(scores.numel(), device=scores.device)
    probs = _normalize_sparse_topk_probs(
        scores[None],
        torch.tensor([[temperature]], device=scores.device),
        (ranks < top_k)[None],
        torch.tensor([[top_p]], device=scores.device),
    )[0]
    return torch.zeros_like(probs).scatter_(0, order, probs)


def _target_p(target_row, top_k, top_p):
    """The target distribution exactly as UNO verification builds it."""
    sampling_info = _sampling_info([1.0], [top_k], [top_p])
    if top_k <= _SPARSE_TOP_K_LIMIT:
        ids, probs = _build_sparse_target_support(
            next_token_logits=target_row,
            sampling_info=sampling_info,
            batch_size=1,
            forward_width=1,
            max_top_k=top_k,
        )
        return torch.zeros(target_row.shape[-1], device=_DEVICE).scatter_(
            0, ids[0, 0], probs[0, 0]
        )
    return build_speculative_verify_target_probs(
        next_token_logits=target_row,
        sampling_info=sampling_info,
        draft_token_num=1,
        bs=1,
        max_top_k=top_k,
    )[0, 0]


def _total_variation(tokens, probs):
    counts = torch.bincount(tokens, minlength=probs.numel()).float()
    return 0.5 * (counts / counts.sum() - probs).abs().sum().item()


class TestUnoRectifier(CustomTestCase):
    def test_full_lattice_q_equals_full_vocab_rectified_distribution(self):
        """With K = V the lattice q is the full-vocabulary rectified q of
        every row, conditioned on the draft actually sampled before it."""
        vocab_size, num_rows = 64, 4
        rectifier = _rectifier(vocab_size, 8, 0.5, num_candidates=vocab_size)
        params = [
            (1.0, TOP_K_ALL, 1.0),
            (0.7, 5, 1.0),
            (1.3, TOP_K_ALL, 0.8),
            (1.0, 1, 1.0),
            (0.6, 10, 0.5),
            (1.0, vocab_size, 0.95),
        ]
        batch_size = len(params)
        torch.manual_seed(1)
        draft_logits = torch.randn(batch_size, num_rows, vocab_size, device=_DEVICE)
        first_prev_tokens = torch.randint(vocab_size, (batch_size,), device=_DEVICE)
        temperatures, top_ks, top_ps = zip(*params)
        sampling_info = _sampling_info(temperatures, top_ks, top_ps)

        rectified = rectifier.sample(
            draft_logits=draft_logits,
            first_prev_tokens=first_prev_tokens,
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            top_ps=sampling_info.top_ps,
        )

        for b, (temperature, top_k, top_p) in enumerate(params):
            prev = first_prev_tokens[b]
            for j in range(num_rows):
                bias = rectifier.markov_w2 @ rectifier.markov_w1[prev]
                expected = _full_vocab_rectified_q(
                    draft_logits[b, j], bias, temperature, top_k, top_p
                )
                published = torch.zeros(vocab_size, device=_DEVICE).scatter_(
                    0, rectified.candidates[b, j], rectified.probs[b, j]
                )
                torch.testing.assert_close(published, expected, atol=1e-5, rtol=1e-4)
                draft = rectified.drafts[b, j]
                self.assertGreater(published[draft].item(), 0.0)
                prev = draft

    def _emitted_token_tv(self, *, top_k, top_p, corrupt_q):
        """TV distance between the first emitted token and the target p."""
        vocab_size, batch_size = 64, 400_000
        rectifier = _rectifier(vocab_size, 8, 2.0, num_candidates=16)
        torch.manual_seed(2)
        draft_rows = torch.randn(2, vocab_size, device=_DEVICE) * 2
        target_rows = torch.randn(2, vocab_size, device=_DEVICE) * 2
        draft_logits = draft_rows.expand(batch_size, 2, vocab_size).contiguous()
        target_logits = target_rows.repeat(batch_size, 1)
        sampling_info = _sampling_info(
            [1.0] * batch_size, [top_k] * batch_size, [top_p] * batch_size
        )
        max_top_k = top_k

        candidates, draft_distribution = sample_uno_candidates(
            draft_logits=draft_logits,
            sampling_info=sampling_info,
            max_top_k=max_top_k,
            rectifier=rectifier,
        )
        if corrupt_q:
            # The plain draft distribution instead of the rectified q.
            _, base_q = sample_uno_candidates(
                draft_logits=draft_logits,
                sampling_info=sampling_info,
                max_top_k=max_top_k,
            )
            draft_distribution = UnoDraftDistribution(
                probs=base_q.probs, token_ids=base_q.token_ids
            )
        result = run_uno_sampling(
            candidates=candidates,
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            committed_frontiers=torch.zeros(
                batch_size, dtype=torch.int64, device=_DEVICE
            ),
            draft_distribution=draft_distribution,
            max_top_k=max_top_k,
        )
        return _total_variation(
            result.output_ids[:, 1], _target_p(target_rows[:1], top_k, top_p)
        )

    def test_rectified_drafts_keep_verification_lossless(self):
        """Every emitted token follows the target p on the sparse (top-k) and
        dense (full-vocabulary) verification paths."""
        for top_k, top_p in ((20, 0.9), (TOP_K_ALL, 1.0), (TOP_K_ALL, 0.8)):
            with self.subTest(top_k=top_k, top_p=top_p):
                tv = self._emitted_token_tv(top_k=top_k, top_p=top_p, corrupt_q=False)
                # Sampling noise of TV at 400k draws over 64 tokens is about 0.005.
                self.assertLess(tv, 0.015, f"TV={tv:.4f}")

    def test_wrong_q_is_detected(self):
        """Control for the lossless test: verifying rectified drafts against
        the unrectified q measurably biases the emitted tokens."""
        tv = self._emitted_token_tv(top_k=20, top_p=0.9, corrupt_q=True)
        self.assertGreater(tv, 0.03, f"TV={tv:.4f}")

    def test_zero_rectifier_greedy_chain_is_plain_argmax(self):
        vocab_size = 1000
        rectifier = _rectifier(vocab_size, 8, 0.0, num_candidates=256)
        torch.manual_seed(3)
        draft_logits = torch.randn(5, 6, vocab_size, device=_DEVICE)
        argmax = draft_logits.argmax(dim=-1)
        rectified = rectify_uno_greedy_candidates(
            candidates=argmax, draft_logits=draft_logits, rectifier=rectifier
        )
        torch.testing.assert_close(rectified, argmax, rtol=0, atol=0)

    def test_zero_rectifier_greedy_ties_match_torch_argmax(self):
        """bf16 logits tie often; the rectified argmax must break ties toward
        the lowest token like torch.argmax, not by top-k candidate order."""
        vocab_size = 1000
        rectifier = _rectifier(vocab_size, 8, 0.0, num_candidates=256)
        torch.manual_seed(4)
        draft_logits = (torch.randn(64, 4, vocab_size, device=_DEVICE) * 0.5).round()
        argmax = draft_logits.argmax(dim=-1)
        rectified = rectify_uno_greedy_candidates(
            candidates=argmax, draft_logits=draft_logits, rectifier=rectifier
        )
        torch.testing.assert_close(rectified, argmax, rtol=0, atol=0)

    def test_loader_contract(self):
        vocab_size, rank = 32, 4
        with tempfile.TemporaryDirectory() as adapter_path:
            self.assertIsNone(
                load_uno_rectifier(
                    adapter_path=adapter_path,
                    vocab_size=vocab_size,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
            )
            w1 = torch.randn(vocab_size, rank)
            w2 = torch.randn(vocab_size, rank)
            save_file(
                {"rectifier.markov_w1.weight": w1, "rectifier.markov_w2.weight": w2},
                os.path.join(adapter_path, "rectifier_model.safetensors"),
            )
            with open(os.path.join(adapter_path, "rectifier_config.json"), "w") as f:
                json.dump({"type": "vanilla", "vocab_size": vocab_size}, f)

            rectifier = load_uno_rectifier(
                adapter_path=adapter_path,
                vocab_size=vocab_size,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            torch.testing.assert_close(rectifier.markov_w1, w1)
            torch.testing.assert_close(rectifier.markov_w2, w2)
            with self.assertRaises(ValueError):
                load_uno_rectifier(
                    adapter_path=adapter_path,
                    vocab_size=vocab_size + 1,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
