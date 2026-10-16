"""Contracts for UNO's folded draft LoRA (--uno-fused-lora)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.lora.backend.uno_fused_lora import (
    fold_uno_lora_group,
    fold_uno_lora_layer,
)
from sglang.srt.lora.layers import MergedColumnParallelLinearWithLoRA
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"
_RANK = 16
_SCALING = 2.0


def _module(slice_sizes, shared_a_slices, in_features=96):
    """A LoRA-wrapped layer stub; slices listed in shared_a_slices reuse A."""
    out = sum(slice_sizes)
    weight = torch.randn(out, in_features, device=_DEVICE, dtype=torch.bfloat16)
    a_blocks, shared = [], None
    for s in range(len(slice_sizes)):
        if s in shared_a_slices:
            shared = shared if shared is not None else torch.randn(_RANK, in_features)
            a_blocks.append(shared)
        else:
            a_blocks.append(torch.randn(_RANK, in_features))
    # Buffers carry a second (unused) slot so weight_index matters.
    lora_a = torch.zeros(2, len(slice_sizes) * _RANK, in_features)
    lora_a[1] = torch.cat(a_blocks) * 0.1
    lora_b = torch.zeros(2, out, _RANK)
    lora_b[1] = torch.randn(out, _RANK) * 0.1
    offsets = [0]
    for size in slice_sizes:
        offsets.append(offsets[-1] + size)
    base = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        bias=None,
        quant_method=UnquantizedLinearMethod(),
    )
    # A real wrapper type (folding dispatches on it) without a real base layer.
    module = MergedColumnParallelLinearWithLoRA.__new__(MergedColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(module)
    module.base_layer = base
    module.A_buffer = lora_a.to(_DEVICE, torch.bfloat16)
    module.B_buffer = lora_b.to(_DEVICE, torch.bfloat16)
    module.output_offset_cpu = torch.tensor(offsets, dtype=torch.int32)
    return module


def _reference(module, weight, x, forward_width):
    """Base GEMM plus per-slice LoRA on every non-seed row, in fp32."""
    a = module.A_buffer[1].float()
    b = module.B_buffer[1].float()
    offsets = module.output_offset_cpu.tolist()
    out = x.float() @ weight.float().t()
    hidden = x.float() @ a.t()
    for s in range(len(offsets) - 1):
        delta = hidden[:, s * _RANK : (s + 1) * _RANK] @ b[offsets[s] : offsets[s + 1]].t()
        delta[::forward_width] = 0.0
        out[:, offsets[s] : offsets[s + 1]] += _SCALING * delta
    return out


class TestUnoFusedLoRA(CustomTestCase):
    def _check(self, slice_sizes, shared_a_slices, num_requests, forward_width):
        torch.manual_seed(0)
        module = _module(slice_sizes, shared_a_slices)
        weight = module.base_layer.weight.data.clone()
        fused = fold_uno_lora_layer(
            module, weight_index=1, rank=_RANK, scaling=_SCALING
        )
        # Base-only forwards read an unchanged weight.
        torch.testing.assert_close(module.base_layer.weight.data, weight, rtol=0, atol=0)
        x = torch.randn(num_requests * forward_width, weight.shape[1], device=_DEVICE)
        x = x.to(torch.bfloat16)
        out = fused.forward(x, forward_width=forward_width)
        torch.testing.assert_close(
            out.float(), _reference(module, weight, x, forward_width), rtol=2e-2, atol=2e-1
        )
        return fused

    def test_shared_a_slices_fold_once(self):
        """GDN q|k|v share one HF LoRA-A; folding keeps one copy of it."""
        fused = self._check([32, 32, 96, 96], {0, 1, 2}, num_requests=3, forward_width=8)
        self.assertEqual(fused.num_blocks, 2)
        self.assertEqual(fused.weight_cat.shape[0], 256 + 2 * _RANK)

    def test_narrow_slices_inside_one_tile(self):
        """Two 48-wide slices (GDN b|a) share every 128-column tile."""
        self._check([48, 48], set(), num_requests=1, forward_width=8)

    def test_group_matches_concatenated_layers(self):
        """GDN folds qkvz and ba (same input) into one GEMM and one expand."""
        torch.manual_seed(1)
        qkvz = _module([32, 32, 96, 96], {0, 1, 2})
        ba = _module([48, 48], set())
        weights = [qkvz.base_layer.weight.data.clone(), ba.base_layer.weight.data.clone()]
        fused = fold_uno_lora_group(
            [qkvz, ba], weight_index=1, rank=_RANK, scaling=_SCALING
        )
        self.assertEqual(fused.out_splits, (256, 96))
        self.assertEqual(fused.num_blocks, 4)
        x = torch.randn(2 * 8, 96, device=_DEVICE).to(torch.bfloat16)
        expected = torch.cat(
            [
                _reference(qkvz, weights[0], x, 8),
                _reference(ba, weights[1], x, 8),
            ],
            dim=1,
        )
        torch.testing.assert_close(
            fused.forward(x, forward_width=8).float(), expected, rtol=2e-2, atol=2e-1
        )
        torch.testing.assert_close(
            fused.base_forward(x).float(),
            torch.cat([x.float() @ w.float().t() for w in weights], dim=1),
            rtol=2e-2,
            atol=2e-1,
        )
        for module, weight in zip((qkvz, ba), weights):
            torch.testing.assert_close(module.base_layer.weight.data, weight, rtol=0, atol=0)

    def test_single_slice_many_requests(self):
        self._check([200], set(), num_requests=20, forward_width=4)


if __name__ == "__main__":
    unittest.main()
