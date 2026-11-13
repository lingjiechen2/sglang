"""Contracts for SGLANG_SIMULATE_ACC_LEN on linear UNO (simulate_uno_acceptance)."""

import unittest

import torch

from sglang.srt.speculative.uno_utils import simulate_uno_acceptance
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

B, F, V = 3, 8, 50


def run(acc_len, mode="real-draft-token"):
    torch.manual_seed(0)
    candidates = torch.randint(0, V, (B, F))
    logits = torch.randn(B * F, V)
    seq = torch.tensor([10, 20, 30], dtype=torch.int32)
    out = simulate_uno_acceptance(
        candidates=candidates, next_token_logits=logits, committed_seq_lens=seq, simulate_acc_len=acc_len,
        simulate_acc_method="match-expected", simulate_acc_token_mode=mode)
    return candidates, logits.view(B, F, V), seq, out


class TestUnoSimulateAcc(CustomTestCase):
    def test_real_draft_tokens_layout(self):
        candidates, logits, seq, (output_ids, accept_lens, new_seq_lens, correction) = run(4.0)
        self.assertEqual(accept_lens.tolist(), [4] * B)  # candidates[0] + 2 drafts + correction
        self.assertEqual(new_seq_lens.tolist(), [14, 24, 34])
        self.assertTrue(torch.equal(output_ids[:, :3], candidates[:, :3]))
        self.assertTrue(torch.equal(correction, logits[:, 2].argmax(-1)))
        self.assertTrue(torch.equal(output_ids[:, 3], correction))
        self.assertEqual(tuple(output_ids.shape), (B, F + 1))

    def test_bounds(self):
        self.assertEqual(run(1.0)[3][1].tolist(), [2] * B)  # a round always commits candidates[0] + correction
        self.assertEqual(run(99.0)[3][1].tolist(), [F + 1] * B)

    def test_fractional_length_matches_in_expectation(self):
        torch.manual_seed(1)
        lens = []
        for i in range(4000):
            lens.append(int(simulate_uno_acceptance(
                candidates=torch.zeros((1, F), dtype=torch.int64), next_token_logits=torch.zeros(F, V),
                committed_seq_lens=torch.zeros(1, dtype=torch.int32), simulate_acc_len=4.36,
                simulate_acc_method="match-expected", simulate_acc_token_mode="fixed")[1][0]))
        self.assertAlmostEqual(sum(lens) / len(lens), 4.36, delta=0.05)

    def test_fixed_token_mode(self):
        _, _, _, (output_ids, _, _, correction) = run(3.0, mode="fixed")
        self.assertTrue(bool((output_ids == 100).all()) and bool((correction == 100).all()))


if __name__ == "__main__":
    unittest.main()
