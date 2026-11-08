"""The rectified draft distribution q must never exceed 1.

Triton's fp32 division is approximate, so a one-token nucleus used to
renormalize to 1 + 1 ulp.  The chain verifier treats q > 1 as invalid and always
rejects that draft token (still lossless, but about 3% shorter acceptance).
The clamp lives in the shared rectifier kernel, so this one test covers every
model that loads a rectifier.
"""

import unittest

import torch

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.uno_rectifier import UnoRectifier
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _rectifier(vocab_size, rank, scale, num_candidates, seed):
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    weights = [
        torch.randn(vocab_size, rank, generator=generator, device=_DEVICE) * scale
        for _ in range(2)
    ]
    return UnoRectifier(
        markov_w1=weights[0], markov_w2=weights[1], num_candidates=num_candidates
    )


class TestUnoRectifierQBound(CustomTestCase):
    def _sample(self, seed, temperature, top_k, top_p, logit_scale):
        vocab_size, batch_size, num_rows = 512, 64, 8
        rectifier = _rectifier(vocab_size, 8, 0.5, num_candidates=32, seed=seed)
        torch.manual_seed(seed)
        # Peaked logits make a one-token nucleus the common case.
        draft_logits = (
            torch.randn(batch_size, num_rows, vocab_size, device=_DEVICE) * logit_scale
        )
        first_prev_tokens = torch.randint(vocab_size, (batch_size,), device=_DEVICE)
        return rectifier.sample(
            draft_logits=draft_logits,
            first_prev_tokens=first_prev_tokens,
            temperatures=torch.full((batch_size, 1), temperature, device=_DEVICE),
            top_ks=torch.full((batch_size,), top_k, dtype=torch.int32, device=_DEVICE),
            top_ps=torch.full((batch_size,), top_p, device=_DEVICE),
        )

    def test_q_never_exceeds_one(self):
        # top-p well below the largest probability leaves a one-token nucleus.
        settings = [
            (1.0, TOP_K_ALL, 0.05, 8.0),
            (1.0, TOP_K_ALL, 0.30, 8.0),
            (0.7, 20, 0.50, 6.0),
            (1.0, 20, 0.95, 10.0),
            (1.3, TOP_K_ALL, 0.10, 4.0),
        ]
        worst = 0.0
        for seed in range(20):
            for temperature, top_k, top_p, logit_scale in settings:
                out = self._sample(seed, temperature, top_k, top_p, logit_scale)
                worst = max(worst, out.probs.max().item())
                self.assertLessEqual(
                    out.probs.max().item(),
                    1.0,
                    f"q > 1 at seed={seed} T={temperature} top_k={top_k} top_p={top_p}",
                )
        self.assertGreater(worst, 0.99, "settings never produced a one-token nucleus")

    def test_published_q_still_sums_to_one(self):
        out = self._sample(seed=0, temperature=1.0, top_k=TOP_K_ALL, top_p=0.3,
                           logit_scale=8.0)
        torch.testing.assert_close(
            out.probs.sum(dim=-1),
            torch.ones_like(out.probs.sum(dim=-1)),
            atol=1e-5,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
