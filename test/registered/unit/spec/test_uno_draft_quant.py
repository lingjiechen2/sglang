"""Contracts for --uno-draft-weight-quant: Marlin draft weights, the bf16 fallback above the row
limit, a draft conv pass that leaves conv_state alone, and the clean row in the draft q."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.mamba.causal_conv1d_linear_verify import causal_conv1d_linear_verify
from sglang.srt.lora.backend.uno_fused_lora import UnoFusedLoRA
from sglang.srt.speculative.uno_draft_quant import UnoMarlinDraftWeight
from sglang.srt.speculative.uno_utils import sample_uno_candidates
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


if __name__ == "__main__":
    unittest.main()
