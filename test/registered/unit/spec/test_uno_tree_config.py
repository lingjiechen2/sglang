"""Startup validation for UNO configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _handle_uno
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestUnoTreeConfig(CustomTestCase):
    def test_parent_list_overflow_is_rejected_at_startup(self):
        """An invalid tree must not survive startup and crash on first decode."""

        server_args = SimpleNamespace(
            device="cuda",
            enable_deterministic_inference=False,
            enable_strict_thinking=False,
            speculative_draft_model_path=None,
            uno_lora_path="/tmp/uno-lora",
            speculative_num_draft_tokens=8,
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            uno_draft_weight_quant="none",
            uno_draft_kv_window=0,
        )

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook.resolving_view",
                side_effect=lambda args: args,
            ),
            patch("sglang.srt.arg_groups.speculative_hook.declare_resolution"),
            self.assertRaisesRegex(ValueError, "parent-list ABI"),
        ):
            _handle_uno(server_args)

    def test_quantized_tree_needs_a_node_for_the_seed_root(self):
        """Quantized tree drafts verify the seed as the root: Q >= F + 1."""

        server_args = SimpleNamespace(
            device="cuda",
            enable_deterministic_inference=False,
            enable_strict_thinking=False,
            speculative_draft_model_path=None,
            uno_lora_path="/tmp/uno-lora",
            speculative_num_draft_tokens=8,
            speculative_num_steps=7,
            speculative_eagle_topk=4,
            uno_draft_weight_quant="int4",
            uno_draft_kv_window=0,
            uno_fused_lora=True,
            uno_enable_recycle=False,
            uno_mova_lora=False,
        )

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook.resolving_view",
                side_effect=lambda args: args,
            ),
            patch("sglang.srt.arg_groups.speculative_hook.declare_resolution"),
            self.assertRaisesRegex(ValueError, "Q >= F \\+ 1"),
        ):
            _handle_uno(server_args)

    def test_windowed_tree_needs_a_node_for_the_seed_root(self):
        """Windowed tree drafts (--uno-draft-kv-window) verify the seed as the root too: Q >= F + 1."""

        server_args = SimpleNamespace(
            device="cuda",
            enable_deterministic_inference=False,
            enable_strict_thinking=False,
            speculative_draft_model_path=None,
            uno_lora_path="/tmp/uno-lora",
            speculative_num_draft_tokens=8,
            speculative_num_steps=7,
            speculative_eagle_topk=4,
            uno_draft_weight_quant="none",
            uno_draft_kv_window=2048,
        )

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook.resolving_view",
                side_effect=lambda args: args,
            ),
            patch("sglang.srt.arg_groups.speculative_hook.declare_resolution"),
            self.assertRaisesRegex(ValueError, "Q >= F \\+ 1"),
        ):
            _handle_uno(server_args)


if __name__ == "__main__":
    unittest.main()
