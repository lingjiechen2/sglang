"""Contracts for UNO's MoVA value-expert LoRA."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.mova import routed_linear_reference
from sglang.srt.speculative.uno_mova_lora import (
    UnoMovaValueLoRA,
    _RowHook,
    attach_uno_mova_lora,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _backend(route):
    return SimpleNamespace(batch_info=SimpleNamespace(uno_single_adapter_route=route))


class TestUnoMovaValueLoRA(CustomTestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.experts, self.width, self.hidden, self.rank, self.k = 6, 8, 16, 4, 3
        self.weights = torch.randn(self.experts, self.width, self.hidden, generator=g)
        self.a = torch.randn(self.rank, self.hidden, generator=g)
        self.b = torch.randn(self.width, self.rank, generator=g)
        self.scale = 2.0
        self.route = SimpleNamespace(batch_size=2, forward_width=4, total_rows=8)
        self.x = torch.randn(8, self.hidden, generator=g)
        self.sel = torch.stack([torch.randperm(self.experts, generator=g)[: self.k] for _ in range(8)])
        self.w = torch.rand(8, self.k, generator=g)
        self.base = routed_linear_reference(self.x, self.weights, self.w, self.sel)

    def _lora(self, route):
        return UnoMovaValueLoRA(self.a, self.b, self.scale, _backend(route))

    def test_draft_rows_get_the_lora_before_silu_and_seed_rows_do_not(self):
        out = self._lora(self.route)(self.x, self.w, self.sel, self.weights, self.base)
        draft = [row for row in range(8) if row % 4 != 0]
        for row in range(8):
            if row in draft:
                delta = F.linear(F.linear(self.x[row], self.a), self.b) * self.scale
                z = torch.einsum("knh,h->kn", self.weights[self.sel[row]], self.x[row]) + delta
                expected = (F.silu(z) * self.w[row, :, None]).sum(0)
                torch.testing.assert_close(out[row], expected)
            else:
                torch.testing.assert_close(out[row], self.base[row])
        self.assertFalse(torch.allclose(out[1], self.base[1]))

    def test_base_forwards_are_untouched(self):
        out = self._lora(None)(self.x, self.w, self.sel, self.weights, self.base)
        self.assertIs(out, self.base)

    def test_zero_lora_b_is_the_base_projection(self):
        lora = UnoMovaValueLoRA(self.a, torch.zeros_like(self.b), self.scale, _backend(self.route))
        out = lora(self.x, self.w, self.sel, self.weights, self.base)
        torch.testing.assert_close(out, self.base)

    def test_attach_fails_without_mova_layers(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, "MoVA attention"):
                attach_uno_mova_lora(SimpleNamespace(lora_backend=None), torch.nn.Linear(2, 2), path)


class TestUnoMovaRowHook(CustomTestCase):
    def test_hook_adds_each_slice_on_draft_rows_only(self):
        g = torch.Generator().manual_seed(1)
        width, hidden, rank, scale = 5, 7, 3, 4.0
        x = torch.randn(8, hidden, generator=g)
        out = torch.randn(8, 2 * width, generator=g)
        a = [torch.randn(rank, hidden, generator=g) for _ in range(2)]
        b = [torch.randn(width, rank, generator=g) for _ in range(2)]
        parts = [(0, width, a[0], b[0]), (width, 2 * width, a[1], b[1])]
        route = SimpleNamespace(batch_size=2, forward_width=4, total_rows=8)
        hook = _RowHook(parts, scale, _backend(route))
        new, bias = hook(None, (x,), (out, None))
        self.assertIsNone(bias)
        for row in range(8):
            if row % 4 == 0:
                torch.testing.assert_close(new[row], out[row])
            else:
                expected = out[row].clone()
                for start, end, pa, pb in parts:
                    expected[start:end] += F.linear(F.linear(x[row], pa), pb) * scale
                torch.testing.assert_close(new[row], expected)
        self.assertIsNone(_RowHook(parts, scale, _backend(None))(None, (x,), (out, None)))


if __name__ == "__main__":
    unittest.main()
