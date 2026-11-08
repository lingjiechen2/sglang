"""UNO's per-expert routed FFN-expert LoRA (UnoMovaPerExpertExpertLoRA) through the Triton MoE hooks.

Reference: a per (token, expert) fp32 loop in which every selected expert applies *its own* adapter
(``lora_expert_mode=per_expert``): gate/up deltas before the activation, the down delta on the LoRA'd activation,
all times the routing weight; UNO draft rows only. Also checks that equal per-expert adapters reproduce the shared
hook, that an expert's adapter only touches the rows that select it, and that the grouped GEMM equals the loop.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.uno_mova_lora import (
    UnoMovaExpertLoRA,
    UnoMovaPerExpertExpertLoRA,
    _expert_lora_grouped,
    _expert_lora_loop,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, TOPK, HIDDEN, INTER, RANK, SCALE = 16, 4, 256, 128, 8, 16.0


def _backend(route):
    return SimpleNamespace(batch_info=SimpleNamespace(uno_single_adapter_route=route))


def _moe(hidden, w1, w2, topk_weights, topk_ids, hooks):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import _fused_moe_kernel_sequence
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

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
    """Per (token, expert) loop in fp32; the LoRA of expert ``e`` is ``parts[name][0][e]`` / ``[1][e]``."""
    x, w1, w2 = x.float(), w1.float(), w2.float()
    parts = {k: (a.float(), b.float()) for k, (a, b) in parts.items()}

    def lora(name, e, t, on):
        if not on or name not in parts:
            return 0
        a, b = parts[name]
        return (t @ a[e].T) @ b[e].T * SCALE

    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        on = t in lora_rows
        for k in range(TOPK):
            e = topk_ids[t, k].item()
            gate = w1[e, :INTER] @ x[t] + lora("gate_proj", e, x[t], on)
            up = w1[e, INTER:] @ x[t] + lora("up_proj", e, x[t], on)
            h = F.silu(gate) * up
            out[t] += topk_weights[t, k] * (w2[e] @ h + lora("down_proj", e, h, on))
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestUnoMovaPerExpertExpertLoRA(CustomTestCase):
    def setUp(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        dev, bf = "cuda", torch.bfloat16
        self.g, self.bf = g, bf
        self.route = SimpleNamespace(batch_size=3, forward_width=4, total_rows=12)
        self.m = 12
        self.x = torch.randn(self.m, HIDDEN, device=dev, generator=g).to(bf)
        self.w1 = (torch.randn(E, 2 * INTER, HIDDEN, device=dev, generator=g) / HIDDEN**0.5).to(bf)
        self.w2 = (torch.randn(E, HIDDEN, INTER, device=dev, generator=g) / INTER**0.5).to(bf)
        self.topk_ids = torch.rand(self.m, E, device=dev, generator=g).topk(TOPK, -1).indices.to(torch.int32)
        self.topk_weights = torch.rand(self.m, TOPK, device=dev, generator=g) * 2.5
        self.parts = {
            "gate_proj": self.ab(INTER, HIDDEN),
            "up_proj": self.ab(INTER, HIDDEN),
            "down_proj": self.ab(HIDDEN, INTER),
        }
        self.draft = {r for r in range(self.m) if r % 4 != 0}

    def ab(self, out_dim, in_dim):
        a = torch.randn(E, RANK, in_dim, device="cuda", generator=self.g) / in_dim**0.5
        b = torch.randn(E, out_dim, RANK, device="cuda", generator=self.g) / RANK**0.5 / SCALE
        return a.to(self.bf), b.to(self.bf)

    def _run(self, parts, route, cls=UnoMovaPerExpertExpertLoRA):
        lora = cls(parts, SCALE, _backend(route))
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            get_parallel().override(tp_group=None),
        ):
            return _moe(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, lora(self.x))

    def _check(self, parts):
        out = self._run(parts, self.route).float()
        ref = _reference(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, parts, self.draft)
        base = _reference(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, parts, set())
        rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
        self.assertLess(rel.max().item(), 1.5e-2)
        drafts = sorted(self.draft)
        self.assertGreater((ref[drafts] - base[drafts]).abs().max().item(), 0.3)  # the LoRA is a visible change

    def test_gate_up_down_match_the_per_expert_reference(self):
        self._check(self.parts)

    def test_each_projection_alone(self):
        for name in self.parts:
            with self.subTest(name=name):
                self._check({name: self.parts[name]})

    def test_wrong_expert_adapters_are_detected(self):
        # Rolling the adapters by one expert must move the output well beyond the check tolerance.
        out = self._run(self.parts, self.route).float()
        rolled = {k: (a.roll(1, 0), b.roll(1, 0)) for k, (a, b) in self.parts.items()}
        ref = _reference(self.x, self.w1, self.w2, self.topk_weights, self.topk_ids, rolled, self.draft)
        rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
        self.assertGreater(rel[sorted(self.draft)].min().item(), 5e-2)

    def test_equal_adapters_match_the_shared_hook(self):
        shared = {k: (a[0], b[0]) for k, (a, b) in self.parts.items()}
        tiled = {k: (a.unsqueeze(0).expand(E, -1, -1).contiguous(), b.unsqueeze(0).expand(E, -1, -1).contiguous())
                 for k, (a, b) in shared.items()}
        per_expert = self._run(tiled, self.route).float()
        reference = self._run(shared, self.route, cls=UnoMovaExpertLoRA).float()
        rel = (per_expert - reference).norm(dim=-1) / reference.norm(dim=-1)
        self.assertLess(rel.max().item(), 1e-2)

    def test_an_adapter_only_touches_the_rows_that_select_it(self):
        base = self._run(self.parts, self.route).float()
        drafts = torch.tensor([r in self.draft for r in range(self.m)], device="cuda")
        selected = (self.topk_ids[:, :, None] == torch.arange(E, device="cuda")).any(1)  # [m, E]
        for e in range(E):
            changed = {k: (a.clone(), b.clone()) for k, (a, b) in self.parts.items()}
            for k in changed:
                changed[k][1][e] += 0.5
            out = self._run(changed, self.route).float()
            moved = (out - base).abs().amax(-1) > 1e-6
            self.assertTrue(torch.equal(moved, selected[:, e] & drafts), f"expert {e}")

    def test_seed_rows_bitwise_equal_to_no_lora_and_no_route_gives_no_hooks(self):
        out = self._run(self.parts, self.route)
        plain = self._run(self.parts, None)
        seeds = [r for r in range(self.m) if r % 4 == 0]
        self.assertTrue(torch.equal(out[seeds], plain[seeds]))
        self.assertIsNone(UnoMovaPerExpertExpertLoRA(self.parts, SCALE, _backend(None))(self.x))

    def test_grouped_mm_equals_loop_including_empty_experts(self):
        g = self.g
        a = torch.randn(E, RANK, HIDDEN, device="cuda", generator=g).to(self.bf)
        b = torch.randn(E, INTER, RANK, device="cuda", generator=g).to(self.bf)
        x = torch.randn(37, HIDDEN, device="cuda", generator=g).to(self.bf)
        ids = torch.randint(0, E, (37,), device="cuda", generator=g)
        ids[ids == 3] = 4  # expert 3 has no slot
        ids[ids == 9] = 8
        grouped, loop = _expert_lora_grouped(x, ids, a, b).float(), _expert_lora_loop(x, ids, a, b).float()
        rel = (grouped - loop).norm(dim=-1) / loop.norm(dim=-1)
        self.assertLess(rel.max().item(), 1e-2)

    def test_grouped_path_is_cuda_graph_safe(self):
        # No host sync inside the grouped path: capture, replay with new data copied into the static input, compare with eager.
        a = torch.randn(E, RANK, HIDDEN, device="cuda", generator=self.g).to(self.bf)
        b = torch.randn(E, INTER, RANK, device="cuda", generator=self.g).to(self.bf)
        x = torch.randn(40, HIDDEN, device="cuda", generator=self.g).to(self.bf)
        ids = torch.randint(0, E, (40,), device="cuda", generator=self.g)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _expert_lora_grouped(x, ids, a, b)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _expert_lora_grouped(x, ids, a, b)
        for seed in (1, 2):  # replay with different routing than at capture time
            g = torch.Generator(device="cuda").manual_seed(seed)
            x.copy_(torch.randn(40, HIDDEN, device="cuda", generator=g).to(self.bf))
            ids.copy_(torch.randint(0, E, (40,), device="cuda", generator=g))
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out, _expert_lora_grouped(x, ids, a, b)))


if __name__ == "__main__":
    unittest.main()
