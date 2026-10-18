import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPackVerifyMask(CustomTestCase):
    """Packed verify masks match FlashInfer's bit packing and keep padding bytes."""


if __name__ == "__main__":
    unittest.main()
