import unittest

import torch
from flashinfer.quantization import segment_packbits

from sglang.kernels.ops.speculative.pack_verify_mask import pack_verify_mask_into
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPackVerifyMask(CustomTestCase):
    """Packed verify masks match FlashInfer's bit packing and keep padding bytes."""

    def test_packed_bits_and_padding(self):
        for width, lengths in [(4, [8196]), (3, [1, 7, 8191]), (5, [13, 17, 16379])]:
            with self.subTest(width=width, lengths=lengths):
                indptr = (
                    torch.tensor([0] + lengths, dtype=torch.int32)
                    .cumsum(0)
                    .int()
                    .cuda()
                )
                mask = torch.randint(
                    0, 2, (sum(lengths) * width,), device="cuda"
                ).bool()
                expected, packed_indptr = segment_packbits(
                    mask, indptr * width, bitorder="little"
                )
                result = torch.full(
                    (expected.numel() + 256,), 173, dtype=torch.uint8, device="cuda"
                )
                pack_verify_mask_into(
                    mask,
                    indptr,
                    packed_indptr,
                    result,
                    batch_size=len(lengths),
                    query_width=width,
                    max_kv_len=max(lengths),
                )
                torch.testing.assert_close(
                    result[: expected.numel()], expected, rtol=0, atol=0
                )
                self.assertTrue(torch.all(result[expected.numel() :] == 173).item())


if __name__ == "__main__":
    unittest.main()
