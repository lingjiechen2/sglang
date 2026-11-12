"""Contracts: UNO's target support uses the AR sampler's joint top-k / top-p semantics.

Runs on CUDA when available, so the dense path exercises the flashinfer renorm kernels it uses in
production (they only accept CUDA tensors); on CPU-only hosts it checks the torch fallback."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.uno_utils import (
    _build_dense_probs,
    _build_sparse_target_support_tensors,
    _normalize_sparse_topk_probs,
    joint_top_k_top_p_probs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def joint_reference(logits, temperature, top_k, top_p):
    """Top-p nucleus of the full distribution intersected with the top-k set, renormalized.

    flashinfer's rule: keep a token when fewer than top_k tokens are strictly more likely and the
    mass of the tokens strictly more likely is below top_p, so tied tokens share one decision."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order]
    above = sorted_probs[None, :] > sorted_probs[:, None]
    mass_above = (sorted_probs[None, :] * above).sum(-1)
    keep_sorted = (mass_above < top_p) & (above.sum(-1) < top_k)
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
        ).to(_DEVICE)

    def _sparse(self, top_ks, top_p, temperature=1.0):
        rows = self.logits.shape[0]
        max_k = int(max(top_ks))
        topk_logits, topk_ids = torch.topk(self.logits, max_k, dim=-1)
        temps = torch.full((rows, 1), temperature, device=_DEVICE)
        lse = torch.logsumexp(self.logits.float() / temps, dim=-1, keepdim=True)
        ks = torch.tensor(top_ks, device=_DEVICE).reshape(rows, 1)
        valid = torch.arange(max_k, device=_DEVICE)[None, :] < ks
        probs = _normalize_sparse_topk_probs(
            topk_logits, temps, valid, torch.full((rows, 1), top_p, device=_DEVICE), lse
        )
        dense = torch.zeros(rows, self.vocab, dtype=probs.dtype, device=_DEVICE).scatter_(1, topk_ids, probs)
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
        top_ks = torch.full((batch,), 200, dtype=torch.int32, device=_DEVICE)
        top_ks[::3] = 40
        sampling_info = SimpleNamespace(
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            temperatures=torch.ones(batch, 1, device=_DEVICE),
            top_ks=top_ks,
            top_ps=torch.full((batch,), 0.9, device=_DEVICE),
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

    def test_tree_dense_joint_matches_reference_and_not_top_k_first(self):
        # The tree verify's dense branch (eagle_sample) calls joint_top_k_top_p_probs for UNO. Rows
        # mix top-k values the way a batch of top-k 20 and top-k off requests does.
        rows = self.logits.shape[0]
        probs = torch.softmax(self.logits.float(), dim=-1)
        top_ks = torch.tensor([20, 200, self.vocab, 40] * (rows // 4), dtype=torch.int32, device=_DEVICE)
        top_ps = torch.full((rows,), 0.95, device=_DEVICE)
        got = joint_top_k_top_p_probs(probs, top_ks, top_ps)
        sequential_tv = []
        for i in range(rows):
            ref = joint_reference(self.logits[i], 1.0, int(top_ks[i]), 0.95)
            torch.testing.assert_close(got[i].double(), ref, atol=2e-6, rtol=0)
            self.assertTrue(torch.equal(got[i] > 0, (ref > 0).to(got.device)))
            first = top_k_first_reference(self.logits[i], 1.0, int(top_ks[i]), 0.95)
            sequential_tv.append(0.5 * (ref - first).abs().sum().item())
        # The sequential top-k then top-p order (the previous tree dense branch) is a different
        # distribution on the flat rows.
        self.assertGreater(max(sequential_tv), 0.01)

    def test_tie_at_the_nucleus_edge_keeps_both_tokens(self):
        # Tokens 6 and 7 have the same logit. 0.93 of the mass is strictly above them, so the
        # nucleus keeps both (0.93 < 0.95); a cumulative sum over the sorted order would see 0.955
        # before the second one and drop it.
        head = torch.tensor([0.5, 0.2, 0.1, 0.06, 0.04, 0.03, 0.025, 0.025], dtype=torch.float64)
        tail = torch.full((self.vocab - head.numel(),), 0.02 / (self.vocab - head.numel()), dtype=torch.float64)
        logits = torch.log(torch.cat([head, tail])).float().to(_DEVICE)
        logits[7] = logits[6]
        rows = 1
        topk_logits, topk_ids = torch.topk(logits[None], 20, dim=-1)
        temps = torch.ones(rows, 1, device=_DEVICE)
        lse = torch.logsumexp(logits[None], dim=-1, keepdim=True)
        probs = _normalize_sparse_topk_probs(
            topk_logits, temps, torch.ones_like(topk_logits, dtype=torch.bool),
            torch.full((rows, 1), 0.95, device=_DEVICE), lse,
        )
        dense = torch.zeros(rows, self.vocab, device=_DEVICE).scatter_(1, topk_ids, probs)[0]
        self.assertGreater(dense[6].item(), 0.0)
        self.assertEqual(dense[6].item(), dense[7].item())
        self.assertEqual(int((dense > 0).sum()), 8)
        torch.testing.assert_close(dense.double(), joint_reference(logits, 1.0, 20, 0.95), atol=2e-6, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "flashinfer sampling kernels need CUDA")
    def test_sparse_support_matches_flashinfer_on_bf16_ties(self):
        # bf16-rounded logits tie often, like real model logits. The compact sparse support must keep
        # exactly the tokens flashinfer's joint top-k / top-p (the dense path and the AR sampler) keeps.
        torch.manual_seed(1)
        rows, vocab = 64, 512
        logits = (torch.randn(rows, vocab, device=_DEVICE) * 2.0).to(torch.bfloat16).float()
        checked = edge_tie = k_tie = 0
        for top_k in (20, 40):
            for top_p in (0.8, 0.9, 0.95, 1.0):
                ids, probs = _build_sparse_target_support_tensors(
                    logits, torch.ones(rows, 1, device=_DEVICE),
                    torch.full((rows,), top_k, dtype=torch.int32, device=_DEVICE),
                    torch.full((rows,), top_p, device=_DEVICE), rows, 1, top_k,
                )
                sparse = torch.zeros(rows, vocab, device=_DEVICE).scatter_(1, ids.reshape(rows, -1), probs.reshape(rows, -1))
                info = SimpleNamespace(
                    need_top_k_sampling=True, need_top_p_sampling=True,
                    temperatures=torch.ones(rows, 1, device=_DEVICE),
                    top_ks=torch.full((rows,), top_k, dtype=torch.int32, device=_DEVICE),
                    top_ps=torch.full((rows,), top_p, device=_DEVICE),
                )
                dense = _build_dense_probs(
                    next_token_logits=logits, sampling_info=info, batch_size=rows, forward_width=1,
                    max_top_k=top_k, uniform_top_k_value=None,
                ).reshape(rows, vocab)
                for r in range(rows):
                    kept = sparse[r] > 0
                    self.assertTrue(torch.equal(kept, dense[r] > 0), f"row {r} top_k {top_k} top_p {top_p}")
                    torch.testing.assert_close(sparse[r], dense[r], atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(sparse[r].double(), joint_reference(logits[r], 1.0, top_k, top_p), atol=2e-6, rtol=0)
                    last = logits[r][kept].min()
                    edge_tie += int((logits[r][kept] == last).sum() > 1)
                    k_tie += int(kept.sum() > top_k)
                    checked += 1
        self.assertGreater(edge_tie, 0)  # ties at the kept edge occurred
        self.assertGreater(k_tie, 0)  # ... and some straddled the k-th rank (support wider than k)
        self.assertEqual(checked, rows * 8)


if __name__ == "__main__":
    unittest.main()
