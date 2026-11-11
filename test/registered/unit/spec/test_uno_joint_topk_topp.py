"""CPU contracts: UNO's target support uses the AR sampler's joint top-k / top-p semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.uno_utils import (
    _build_dense_probs,
    _normalize_sparse_topk_probs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def joint_reference(logits, temperature, top_k, top_p):
    """Top-p nucleus of the full distribution intersected with the top-k set, renormalized."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order]
    preceding = torch.cumsum(sorted_probs, dim=0) - sorted_probs
    keep_sorted = (preceding <= top_p) & (torch.arange(len(order)) < top_k)
    out = torch.zeros_like(probs)
    out[order[keep_sorted]] = probs[order[keep_sorted]]
    return out / out.sum()


def top_k_first_reference(logits, temperature, top_k, top_p):
    """The previous UNO semantics: renormalize the top-k set, then cut the nucleus on that."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    order = torch.argsort(probs, descending=True)[:top_k]
    sub = probs[order] / probs[order].sum()
    preceding = torch.cumsum(sub, dim=0) - sub
    keep = preceding <= top_p
    out = torch.zeros_like(probs)
    out[order[keep]] = sub[keep]
    return out / out.sum()


class TestUnoJointTopkTopp(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = 300
        # Flat rows (much mass outside the top-k set), peaked rows, and mixed rows.
        self.logits = torch.cat(
            [
                torch.randn(4, self.vocab) * 0.5,
                torch.randn(4, self.vocab) * 4.0,
                torch.randn(4, self.vocab) * 1.5,
            ]
        )

    def _sparse(self, top_ks, top_p, temperature=1.0):
        rows = self.logits.shape[0]
        max_k = int(max(top_ks))
        topk_logits, topk_ids = torch.topk(self.logits, max_k, dim=-1)
        temps = torch.full((rows, 1), temperature)
        lse = torch.logsumexp(self.logits.float() / temps, dim=-1, keepdim=True)
        ks = torch.tensor(top_ks).reshape(rows, 1)
        valid = torch.arange(max_k)[None, :] < ks
        probs = _normalize_sparse_topk_probs(
            topk_logits, temps, valid, torch.full((rows, 1), top_p), lse
        )
        dense = torch.zeros(rows, self.vocab, dtype=probs.dtype).scatter_(1, topk_ids, probs)
        return dense

    def test_sparse_matches_joint_reference(self):
        rows = self.logits.shape[0]
        for top_p, temperature in [(0.95, 1.0), (0.8, 1.0), (1.0, 1.0), (0.95, 0.7)]:
            top_ks = [20] * rows
            got = self._sparse(top_ks, top_p, temperature)
            for i in range(rows):
                ref = joint_reference(self.logits[i], temperature, 20, top_p)
                torch.testing.assert_close(got[i].double(), ref, atol=2e-6, rtol=0)

    def test_per_row_top_k_below_the_compact_width(self):
        rows = self.logits.shape[0]
        top_ks = [3, 7, 20, 11] * (rows // 4)
        got = self._sparse(top_ks, 0.95)
        for i in range(rows):
            ref = joint_reference(self.logits[i], 1.0, top_ks[i], 0.95)
            torch.testing.assert_close(got[i].double(), ref, atol=2e-6, rtol=0)

    def test_old_top_k_first_semantics_differ_on_flat_rows(self):
        # Guards that this test can tell the two semantics apart.
        diffs = []
        for i in range(4):
            joint = joint_reference(self.logits[i], 1.0, 20, 0.95)
            first = top_k_first_reference(self.logits[i], 1.0, 20, 0.95)
            diffs.append(0.5 * (joint - first).abs().sum().item())
        self.assertGreater(max(diffs), 0.01)

    def test_dense_path_matches_joint_reference(self):
        rows, width = self.logits.shape[0], 2
        batch = rows // width
        top_ks = torch.full((batch,), 200, dtype=torch.int32)
        top_ks[::3] = 40
        sampling_info = SimpleNamespace(
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            temperatures=torch.ones(batch, 1),
            top_ks=top_ks,
            top_ps=torch.full((batch,), 0.9),
        )
        out = _build_dense_probs(
            next_token_logits=self.logits,
            sampling_info=sampling_info,
            batch_size=batch,
            forward_width=width,
            max_top_k=200,
            uniform_top_k_value=None,
        ).reshape(rows, -1)
        for i in range(rows):
            ref = joint_reference(self.logits[i], 1.0, int(top_ks[i // width]), 0.9)
            torch.testing.assert_close(out[i].double(), ref, atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
