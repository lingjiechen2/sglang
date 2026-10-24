"""UNO's MoVA routed FFN-expert LoRA through the Triton MoE hooks.

The reference is the HF adapter's semantics (duo3 ``k2h_adapter.py``): one LoRA per
projection shared by every routed expert, each selected expert computing
``down(silu(gate(x)) * up(x))`` with its LoRA'd projections, weighted by the
routing weight; UNO draft rows only.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.uno_mova_lora import UnoMovaExpertLoRA
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, TOPK, HIDDEN, INTER, RANK, SCALE = 16, 4, 256, 128, 8, 16.0


def _backend(route):
    return SimpleNamespace(batch_info=SimpleNamespace(uno_single_adapter_route=route))


def _moe(hidden, w1, w2, topk_weights, topk_ids, hooks):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _fused_moe_kernel_sequence,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    config = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4, num_stages=3)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 16, E)
    return _fused_moe_kernel_sequence(
        hidden, w1, w2, topk_weights, topk_ids, sorted_ids, expert_ids, num_post, config, None, False, False,
        b1=None, b2=None, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
        per_channel_quant=False, w1_scale=None, w2_scale=None, w1_zp=None, w2_zp=None, a1_scale=None,
        a2_scale=None, block_shape=None, activation="silu", is_gated=True, no_combine=False, inplace=False,
        apply_router_weight_on_input=False, routed_scaling_factor=None, gemm1_alpha=None, gemm1_limit=None,
        filter_expert=False, hooks=hooks,
    )


def _reference(x, w1, w2, topk_weights, topk_ids, parts, lora_rows):
    """Per (token, expert) loop in fp32, LoRA on ``lora_rows`` only."""
    x, w1, w2 = x.float(), w1.float(), w2.float()
    parts = {k: (a.float(), b.float()) for k, (a, b) in parts.items()}

    def lora(name, t, on):
        if not on or name not in parts:
            return 0
        a, b = parts[name]
        return (t @ a.T) @ b.T * SCALE

    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        on = t in lora_rows
        for k in range(TOPK):
            e = topk_ids[t, k].item()
            gate = w1[e, :INTER] @ x[t] + lora("gate_proj", x[t], on)
            up = w1[e, INTER:] @ x[t] + lora("up_proj", x[t], on)
            h = F.silu(gate) * up
            out[t] += topk_weights[t, k] * (w2[e] @ h + lora("down_proj", h, on))
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestUnoMovaExpertLoRA(CustomTestCase):
    def setUp(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        dev, bf = "cuda", torch.bfloat16
        self.route = SimpleNamespace(batch_size=3, forward_width=4, total_rows=12)
        self.m = 12
        self.x = torch.randn(self.m, HIDDEN, device=dev, generator=g).to(bf)
        self.w1 = (torch.randn(E, 2 * INTER, HIDDEN, device=dev, generator=g) / HIDDEN**0.5).to(bf)
        self.w2 = (torch.randn(E, HIDDEN, INTER, device=dev, generator=g) / INTER**0.5).to(bf)
        self.topk_ids = torch.rand(self.m, E, device=dev, generator=g).topk(TOPK, -1).indices.to(torch.int32)
        self.topk_weights = torch.rand(self.m, TOPK, device=dev, generator=g) * 2.5

        def ab(out_dim, in_dim):
            a = torch.randn(RANK, in_dim, device=dev, generator=g) / in_dim**0.5
            b = torch.randn(out_dim, RANK, device=dev, generator=g) / RANK**0.5 / SCALE
            return a.to(bf), b.to(bf)

        self.parts = {
            "gate_proj": ab(INTER, HIDDEN),
            "up_proj": ab(INTER, HIDDEN),
            "down_proj": ab(HIDDEN, INTER),
        }
        self.draft = {r for r in range(self.m) if r % 4 != 0}

    def _run(self, parts, route):
        lora = UnoMovaExpertLoRA(parts, SCALE, _backend(route))
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            get_parallel().override(tp_group=None),
        ):
            return _moe(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, lora(self.x))

    def _check(self, parts):
        out = self._run(parts, self.route).float()
        ref = _reference(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, parts, self.draft)
        base = _reference(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, parts, set())
        # bf16 kernel vs fp32 loop: the LoRA-free kernel is already ~5e-3 off per row.
        rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
        self.assertLess(rel.max().item(), 1.5e-2)
        drafts = sorted(self.draft)
        # The LoRA is a visible change, so the tolerance above cannot pass by accident.
        self.assertGreater((ref[drafts] - base[drafts]).abs().max().item(), 0.3)

    def test_gate_up_down_match_the_per_expert_reference(self):
        self._check(self.parts)

    def test_each_projection_alone(self):
        for name in self.parts:
            with self.subTest(name=name):
                self._check({name: self.parts[name]})

    def test_seed_rows_bitwise_equal_to_no_lora(self):
        out = self._run(self.parts, self.route)
        plain = self._run(self.parts, None)  # no route: the hooks are None
        seeds = [r for r in range(self.m) if r % 4 == 0]
        self.assertTrue(torch.equal(out[seeds], plain[seeds]))
        self.assertIsNone(UnoMovaExpertLoRA(self.parts, SCALE, _backend(None))(self.x))


if __name__ == "__main__":
    unittest.main()
