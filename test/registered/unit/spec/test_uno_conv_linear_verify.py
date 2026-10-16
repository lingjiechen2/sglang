"""The linear-verify conv matches causal_conv1d_update on the deduplicated
sliding conv window used by GDN speculative verify."""

import unittest

import torch

from sglang.kernels.ops.mamba.causal_conv1d_linear_verify import (
    causal_conv1d_linear_verify,
    is_dedup_conv_window,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _dedup_window(slots, num_tokens, dim, width):
    """The memory pool's sliding view: [slots, F, dim, width-1] over [slots, dim, F+width-2]."""
    phys = torch.zeros(slots, dim, num_tokens + width - 2, device=_DEVICE, dtype=torch.bfloat16)
    view = phys.as_strided(
        (slots, num_tokens, dim, width - 1),
        (phys.stride(0), phys.stride(2), phys.stride(1), phys.stride(2)),
    )
    return phys, view


class TestUnoConvLinearVerify(CustomTestCase):
    def _check(self, batch, dim, num_tokens, width, with_bias, padded_request):
        torch.manual_seed(batch + dim)
        slots = batch + 3
        x = (
            torch.randn(batch * num_tokens, dim, device=_DEVICE, dtype=torch.bfloat16)
            .view(batch, num_tokens, dim)
            .transpose(1, 2)
        )
        conv_state = torch.randn(slots, dim, width - 1, device=_DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(dim, width, device=_DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(dim, device=_DEVICE, dtype=torch.bfloat16) if with_bias else None
        state_idx = torch.randperm(slots, device=_DEVICE)[:batch].to(torch.int32)
        window_idx = torch.randperm(slots, device=_DEVICE)[:batch].to(torch.int32)
        if padded_request:
            state_idx[-1] = -1
            window_idx[-1] = -1

        ref_phys, ref_window = _dedup_window(slots, num_tokens, dim, width)
        new_phys, new_window = _dedup_window(slots, num_tokens, dim, width)
        self.assertTrue(is_dedup_conv_window(new_window, width))
        ref_state, new_state = conv_state.clone(), conv_state.clone()

        ref = causal_conv1d_update(
            x, ref_state, weight, bias, "silu",
            conv_state_indices=state_idx,
            intermediate_conv_window=ref_window,
            intermediate_state_indices=window_idx,
            pad_slot_id=-1,
        )
        new = causal_conv1d_linear_verify(
            x, new_state, weight, bias, "silu",
            conv_state_indices=state_idx,
            intermediate_conv_window=new_window,
            intermediate_state_indices=window_idx,
        )
        live = slice(0, batch - 1) if padded_request else slice(0, batch)
        torch.testing.assert_close(new[live].float(), ref[live].float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(new_phys, ref_phys, rtol=0, atol=0)
        torch.testing.assert_close(new_state, ref_state, rtol=0, atol=0)

    def test_qwen38_shape(self):
        self._check(batch=16, dim=10240, num_tokens=8, width=4, with_bias=False, padded_request=False)

    def test_single_request_with_bias(self):
        self._check(batch=1, dim=320, num_tokens=8, width=4, with_bias=True, padded_request=False)

    def test_padded_request_is_skipped(self):
        self._check(batch=4, dim=192, num_tokens=5, width=4, with_bias=False, padded_request=True)


if __name__ == "__main__":
    unittest.main()
