"""SGLANG_UNO_MOVA_LORA_FAST: the fast draft-row LoRA paths match the default ones.

Row hooks (dense / shared-expert projections) run on CPU in fp32; the routed-expert hooks run through the Triton MoE
on CUDA in bf16. Seed rows must stay bit-identical to a LoRA-free forward in every mode.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.uno_mova_lora import (
    UnoMovaExpertLoRA,
    UnoMovaExpertLoRAFast,
    _FastRowHook,
    _row_parts_cover_output,
    _RowHook,
    fast_parts,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _backend(route):
    return SimpleNamespace(batch_info=SimpleNamespace(uno_single_adapter_route=route))


class TestFastParts(CustomTestCase):
    def test_parse(self):
        for value, parts in (("", set()), ("0", set()), ("1", {"row", "gateup", "down"}), ("all", {"row", "gateup", "down"}),
                             ("row,down", {"row", "down"})):
            with mock.patch.dict(os.environ, {"SGLANG_UNO_MOVA_LORA_FAST": value}):
                self.assertEqual(fast_parts(), parts)
        with mock.patch.dict(os.environ, {"SGLANG_UNO_MOVA_LORA_FAST": "row,bogus"}):
            with self.assertRaisesRegex(ValueError, "bogus"):
                fast_parts()


class TestFastRowHook(CustomTestCase):
    def _case(self, scale):
        g = torch.Generator().manual_seed(1)
        width, hidden, rank = 5, 7, 3
        x = torch.randn(12, hidden, generator=g)
        out = torch.randn(12, 2 * width, generator=g)
        a = [torch.randn(rank, hidden, generator=g) for _ in range(2)]
        b = [torch.randn(width, rank, generator=g) for _ in range(2)]
        parts = [(0, width, a[0], b[0]), (width, 2 * width, a[1], b[1])]
        route = SimpleNamespace(batch_size=3, forward_width=4, total_rows=12)
        ref, _ = _RowHook(parts, scale, _backend(route))(None, (x,), (out, None))
        fast_out = out.clone()
        self.assertIsNone(_FastRowHook(parts, scale, _backend(route))(None, (x,), (fast_out, None)))
        torch.testing.assert_close(fast_out, ref)
        seeds = [r for r in range(12) if r % 4 == 0]
        self.assertTrue(torch.equal(fast_out[seeds], out[seeds]))
        self.assertFalse(torch.allclose(fast_out[1], out[1]))
        untouched = out.clone()
        self.assertIsNone(_FastRowHook(parts, scale, _backend(None))(None, (x,), (untouched, None)))
        self.assertTrue(torch.equal(untouched, out))

    def test_power_of_two_scale_folded(self):
        self._case(16.0)

    def test_other_scale(self):
        self._case(3.0)

    def test_cover(self):
        z = torch.zeros(1)
        self.assertTrue(_row_parts_cover_output([(0, 4, z, z), (4, 8, z, z)]))
        self.assertFalse(_row_parts_cover_output([(0, 4, z, z), (5, 8, z, z)]))
        self.assertFalse(_row_parts_cover_output([(4, 8, z, z)]))


E, TOPK, HIDDEN, INTER, RANK, SCALE = 16, 4, 256, 128, 8, 16.0


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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFastExpertLoRA(CustomTestCase):
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

        self.parts = {"gate_proj": ab(INTER, HIDDEN), "up_proj": ab(INTER, HIDDEN), "down_proj": ab(HIDDEN, INTER)}

    def _run(self, lora):
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            get_parallel().override(tp_group=None),
        ):
            return _moe(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, lora(self.x) if lora else None)

    def _check(self, fast):
        base = self._run(UnoMovaExpertLoRA(self.parts, SCALE, _backend(self.route))).float()
        out = self._run(UnoMovaExpertLoRAFast(self.parts, SCALE, _backend(self.route), fast)).float()
        plain = self._run(None).float()
        # Same math in another order: the bf16 difference must be far below the LoRA's own effect.
        effect = (base - plain).norm(dim=-1)
        diff = (out - base).norm(dim=-1)
        drafts = [r for r in range(self.m) if r % 4 != 0]
        self.assertLess((diff[drafts] / effect[drafts]).max().item(), 2e-2)
        seeds = [r for r in range(self.m) if r % 4 == 0]
        self.assertTrue(torch.equal(out[seeds], plain[seeds]))

    def test_all_fast(self):
        self._check({"gateup", "down"})

    def test_each_fast_part_alone(self):
        for fast in ({"gateup"}, {"down"}):
            with self.subTest(fast=fast):
                self._check(fast)

    def test_base_forward_returns_no_hooks(self):
        self.assertIsNone(UnoMovaExpertLoRAFast(self.parts, SCALE, _backend(None), {"gateup", "down"})(self.x))


if __name__ == "__main__":
    unittest.main()
