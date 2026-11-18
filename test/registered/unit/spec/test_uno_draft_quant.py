"""Contracts for --uno-draft-weight-quant: Marlin draft weights, the bf16 fallback above the row
limit, a draft conv pass that leaves conv_state alone, and the clean row in the draft q."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.mamba.causal_conv1d_linear_verify import causal_conv1d_linear_verify
from sglang.srt.lora.backend.uno_fused_lora import UnoFusedLoRA
from sglang.srt.speculative.uno_draft_quant import UnoMarlinDraftWeight
from sglang.srt.speculative.uno_rectifier import UnoRectifier
from sglang.srt.speculative.uno_utils import (
    _build_sparse_target_support,
    sample_uno_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _int4_reference(weight, group):
    """RTN int4 (group along K, symmetric, zero point 8) dequantized back to fp32."""
    w = weight.t().float()  # [K, N]
    k, n = w.shape
    g = w.view(k // group, group, n)
    scale = g.abs().amax(dim=1).clamp_min(1e-12) / 7.0  # q uses the fp32 scale ...
    q = (g / scale[:, None, :]).round().clamp(-8, 7)
    stored = scale.to(weight.dtype).float()  # ... Marlin stores it in the weight dtype
    return (q * stored[:, None, :]).view(k, n).t()


class TestUnoDraftQuant(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weight = (torch.randn(700, 1024, device=_DEVICE) * 0.02).to(torch.bfloat16)
        self.x = torch.randn(8, 1024, device=_DEVICE, dtype=torch.bfloat16)

    def test_int4_matches_dequantized_reference(self):
        w = UnoMarlinDraftWeight(self.weight, "int4")
        got = w(self.x).float()
        self.assertEqual(got.shape, (8, 700))  # padded output rows are cut
        ref = self.x.float() @ _int4_reference(self.weight, w.group).t()
        full = self.x.float() @ self.weight.float().t()
        err_ref = ((got - ref).norm() / ref.norm()).item()
        err_full = ((got - full).norm() / full.norm()).item()
        print(f"int4 rel err vs dequantized reference {err_ref:.4f}, vs bf16 weight {err_full:.4f}")
        self.assertLess(err_ref, 0.01)  # bf16 output rounding only
        self.assertLess(err_full, 0.15)  # int4 noise, not a wrong layout

    def test_humming_reads_the_marlin_gptq_codes(self):
        try:
            import humming  # noqa: F401
        except ImportError:
            self.skipTest("humming-kernels not installed")
        w = UnoMarlinDraftWeight(self.weight, "int4")
        codes, scales = w.gptq_codes()
        deq = ((codes.float() - 8) * scales.float().repeat_interleave(w.group, dim=0)).t()[:700]
        ref_w = _int4_reference(self.weight, w.group)
        torch.testing.assert_close(deq, ref_w, rtol=0, atol=1e-6)  # exact codes and scales
        layer = w._build_humming(torch.bfloat16)
        for m in (8, 64):
            x = torch.randn(m, 1024, device=_DEVICE, dtype=torch.bfloat16)
            got = layer(x)[:, :700].float()
            ref = w._gemm(x)[:, :700].float()
            err = ((got - ref).norm() / ref.norm()).item()
            print(f"humming vs marlin M={m} rel diff {err:.5f}")
            self.assertLess(err, 0.01)  # same codes, only accumulation order differs

    def test_humming_tuned_bucket_matches_heuristic(self):
        try:
            import humming  # noqa: F401
        except ImportError:
            self.skipTest("humming-kernels not installed")
        import json
        from unittest import mock

        import sglang.srt.speculative.uno_draft_quant as q

        if torch.cuda.get_device_properties(0).multi_processor_count != q._HUMMING_TUNED_SMS:
            self.skipTest("tuned Humming table is for 132-SM devices")
        w = UnoMarlinDraftWeight(self.weight, "int4")
        cfg = dict(q._HUMMING_TUNED[(16640, 5120)])  # block (64, 128, 128): legal for N=704, K=1024
        with mock.patch.dict(q._HUMMING_TUNED, {(w.n, w.k): cfg}):
            layer = w._build_humming(torch.bfloat16)
        table = json.loads(w.humming_tuning)
        tuned = [e for e in table if e[2].get("block_shape") == cfg["block_shape"] and e[2].get("num_stages") == 5]
        self.assertEqual(len(tuned), 1)  # only the (56, 64] bucket is replaced
        self.assertTrue(tuned[0][0] < 64 <= tuned[0][1])
        for m in (48, 64):
            x = torch.randn(m, 1024, device=_DEVICE, dtype=torch.bfloat16)
            got = layer(x, tuning_config=w.humming_tuning)[:, :700].float()
            ref = layer(x)[:, :700].float()
            err = ((got - ref).norm() / ref.norm()).item()
            print(f"humming tuned vs heuristic M={m} rel diff {err:.5f}")
            self.assertLess(err, 0.01)  # same codes, only tile order / split-K differ

    def test_rank0_fold_is_the_quantized_base_gemm(self):
        # UNO_DRAFT_QUANT_SKIPPED: a layer the adapter skips folds with rank 0 (no LoRA expand).
        import torch.nn.functional as F

        n = 1024
        w = (torch.randn(2 * n, 1024, device=_DEVICE) * 0.02).to(torch.bfloat16)
        empty = torch.zeros(0, dtype=torch.int32, device=_DEVICE)
        fused = UnoFusedLoRA(
            weight_cat=w, lora_b=torch.zeros(2 * n, 0, device=_DEVICE, dtype=torch.bfloat16),
            slice_ends=empty, slice_blocks=empty, num_slices=2, num_blocks=1, out_features=2 * n,
            rank=0, scaling=1.0, out_splits=(2 * n,), gate_up_blocks=(0, 0),
            draft_weight=UnoMarlinDraftWeight(w, "int4"),
        )
        x = torch.randn(64, 1024, device=_DEVICE, dtype=torch.bfloat16)
        base = fused.draft_weight(x).float()
        got = fused.forward(x, forward_width=8).float()
        torch.testing.assert_close(got, base, rtol=0, atol=0)
        act = fused.forward_silu_mul(x, forward_width=8).float()
        ref = F.silu(base[:, :n]) * base[:, n:]
        err = ((act - ref).norm() / ref.norm()).item()
        print(f"rank-0 fold silu_mul rel diff {err:.5f}")
        self.assertLess(err, 0.01)

    def test_fp8_close_to_bf16(self):
        got = UnoMarlinDraftWeight(self.weight, "fp8")(self.x).float()
        full = self.x.float() @ self.weight.float().t()
        rel = (got - full).norm() / full.norm()
        self.assertLess(rel.item(), 0.05)

    def test_fused_gemm_uses_bf16_above_the_row_limit(self):
        draft = UnoMarlinDraftWeight(self.weight, "int4")
        draft.max_rows = 8
        fused = SimpleNamespace(draft_weight=draft, weight_cat=self.weight)
        small = UnoFusedLoRA._draft_gemm(fused, self.x)
        torch.testing.assert_close(small, draft(self.x), rtol=0, atol=0)
        big = torch.cat([self.x, self.x])
        torch.testing.assert_close(UnoFusedLoRA._draft_gemm(fused, big), torch.mm(big, self.weight.t()), rtol=0, atol=0)
        none = SimpleNamespace(draft_weight=None, weight_cat=self.weight)
        torch.testing.assert_close(UnoFusedLoRA._draft_gemm(none, self.x), torch.mm(self.x, self.weight.t()), rtol=0, atol=0)

    def test_draft_conv_pass_keeps_conv_state(self):
        batch, dim, num_tokens, width = 4, 256, 8, 4
        slots = batch + 2
        x = torch.randn(batch, dim, num_tokens, device=_DEVICE, dtype=torch.bfloat16)
        conv_state = torch.randn(slots, dim, width - 1, device=_DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(dim, width, device=_DEVICE, dtype=torch.bfloat16)
        idx = torch.arange(batch, device=_DEVICE, dtype=torch.int32)

        def run(update):
            phys = torch.zeros(slots, dim, num_tokens + width - 2, device=_DEVICE, dtype=torch.bfloat16)
            window = phys.as_strided((slots, num_tokens, dim, width - 1),
                                     (phys.stride(0), phys.stride(2), phys.stride(1), phys.stride(2)))
            state = conv_state.clone()
            out = causal_conv1d_linear_verify(
                x, state, weight, None, "silu", conv_state_indices=idx,
                intermediate_conv_window=window, intermediate_state_indices=idx,
                update_conv_state=update,
            )
            return out, state, phys

        out_u, state_u, phys_u = run(True)
        out_k, state_k, phys_k = run(False)
        torch.testing.assert_close(out_k, out_u, rtol=0, atol=0)
        torch.testing.assert_close(phys_k, phys_u, rtol=0, atol=0)
        torch.testing.assert_close(state_k, conv_state, rtol=0, atol=0)
        self.assertFalse(torch.equal(state_u, conv_state))

    def test_clean_row_joins_the_draft_distribution(self):
        batch, width, vocab = 3, 5, 64
        logits = torch.randn(batch, width, vocab, device=_DEVICE)
        info = SimpleNamespace(
            temperatures=torch.ones(batch, 1, device=_DEVICE),
            top_ks=torch.full((batch,), 1 << 30, dtype=torch.int32, device=_DEVICE),
            top_ps=torch.ones(batch, device=_DEVICE),
            need_top_k_sampling=False, need_top_p_sampling=False, need_min_p_sampling=False,
        )
        torch.manual_seed(1)
        c0, d0 = sample_uno_candidates(draft_logits=logits, sampling_info=info, max_top_k=1 << 30)
        torch.manual_seed(1)
        c1, d1 = sample_uno_candidates(draft_logits=logits, sampling_info=info, max_top_k=1 << 30,
                                       include_clean=True)
        torch.testing.assert_close(c1, c0, rtol=0, atol=0)
        probs0, probs1 = d0.probs.reshape(batch, width - 1, vocab), d1.probs.reshape(batch, width, vocab)
        torch.testing.assert_close(probs1[:, 1:], probs0, rtol=0, atol=0)
        torch.testing.assert_close(probs1[:, 0], torch.softmax(logits[:, 0].float(), -1), rtol=1e-5, atol=1e-6)


    def test_clean_row_joins_the_rectified_sparse_distribution(self):
        # top-k 20 takes the sparse path: the clean support (num_candidates + tie pad) is wider
        # than the rectifier lattice, so the drafts are padded with q = 0 columns.
        batch, width, vocab, num_candidates = 3, 6, 512, 32
        gen = torch.Generator(device=_DEVICE).manual_seed(0)
        w1, w2 = (torch.randn(vocab, 16, generator=gen, device=_DEVICE) * 0.3 for _ in range(2))
        rectifier = UnoRectifier(markov_w1=w1, markov_w2=w2, num_candidates=num_candidates)
        logits = torch.randn(batch, width, vocab, device=_DEVICE) * 3
        info = SimpleNamespace(
            temperatures=torch.ones(batch, 1, device=_DEVICE),
            top_ks=torch.full((batch,), 20, dtype=torch.int32, device=_DEVICE),
            top_ps=torch.full((batch,), 0.95, device=_DEVICE),
            need_top_k_sampling=True, need_top_p_sampling=True, need_min_p_sampling=False,
        )
        cands, dist = sample_uno_candidates(draft_logits=logits, sampling_info=info, max_top_k=20,
                                            rectifier=rectifier, include_clean=True)
        self.assertEqual(tuple(cands.shape), (batch, width))
        self.assertIsNotNone(dist.token_ids)
        self.assertEqual(dist.token_ids.shape[:2], (batch, width))
        dense = torch.zeros(batch, width, vocab, device=_DEVICE).scatter_add_(
            2, dist.token_ids.long(), dist.probs.float())
        torch.testing.assert_close(dense.sum(-1), torch.ones(batch, width, device=_DEVICE),
                                   rtol=0, atol=1e-5)
        self.assertTrue(bool((dense.gather(2, cands[:, :, None].long()) > 0).all()))
        clean_ids, clean_probs = _build_sparse_target_support(
            next_token_logits=logits[:, 0, :], sampling_info=info, batch_size=batch,
            forward_width=1, max_top_k=num_candidates)
        clean_dense = torch.zeros(batch, vocab, device=_DEVICE).scatter_add_(
            1, clean_ids[:, 0].long(), clean_probs[:, 0].float())
        torch.testing.assert_close(dense[:, 0], clean_dense, rtol=0, atol=0)
        pad = dist.token_ids.shape[-1] - num_candidates
        self.assertGreater(pad, 0)
        self.assertTrue(bool((dist.probs[:, 1:, num_candidates:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
