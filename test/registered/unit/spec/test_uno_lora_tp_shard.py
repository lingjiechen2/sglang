"""A LoRA sharded over TP ranks must add up to the full LoRA.

gate/up are column parallel: every rank adds B_r (A x) to its own output slice, so the slices concatenate to B (A x).
down is row parallel: every rank applies B (A_r h_r) to its input slice h_r, and the partial deltas sum to B (A h).
"""

import unittest

import torch

from sglang.srt.speculative.uno_mova_lora import shard_lora
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RANK, HIDDEN, INTER, TOKENS = 8, 32, 64, 5


class TestUnoLoraTpShard(CustomTestCase):
    def test_gate_up_slices_concatenate_to_full_delta(self):
        torch.manual_seed(0)
        x = torch.randn(TOKENS, HIDDEN, dtype=torch.float64)
        a, b = torch.randn(RANK, HIDDEN, dtype=torch.float64), torch.randn(INTER, RANK, dtype=torch.float64)
        for size in (1, 2, 4, 8):
            parts = [shard_lora("gate_proj", a, b, rank, size) for rank in range(size)]
            delta = torch.cat([x @ pa.T @ pb.T for pa, pb in parts], dim=1)
            torch.testing.assert_close(delta, x @ a.T @ b.T)

    def test_down_partial_deltas_sum_to_full_delta(self):
        torch.manual_seed(1)
        h = torch.randn(TOKENS, INTER, dtype=torch.float64)
        a, b = torch.randn(RANK, INTER, dtype=torch.float64), torch.randn(HIDDEN, RANK, dtype=torch.float64)
        for size in (1, 2, 4, 8):
            shards = h.chunk(size, dim=1)
            parts = [shard_lora("down_proj", a, b, rank, size) for rank in range(size)]
            delta = sum(hr @ pa.T @ pb.T for hr, (pa, pb) in zip(shards, parts))
            torch.testing.assert_close(delta, h @ a.T @ b.T)


if __name__ == "__main__":
    unittest.main()
