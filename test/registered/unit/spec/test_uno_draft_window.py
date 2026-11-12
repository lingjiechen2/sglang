"""Contracts for the opt-in UNO draft KV window settings (UNO_DRAFT_KV_WINDOW / _LAYERS)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.uno_draft_window import (
    DRAFT_WINDOW_ENV,
    DRAFT_WINDOW_LAYERS_ENV,
    check_layers,
    draft_window_from_env,
    full_attention_backend,
    parse_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnoDraftWindow(CustomTestCase):
    def test_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(DRAFT_WINDOW_ENV, None)
            os.environ.pop(DRAFT_WINDOW_LAYERS_ENV, None)
            self.assertEqual(draft_window_from_env(), (0, None))

    def test_env_values(self):
        with patch.dict(os.environ, {DRAFT_WINDOW_ENV: "2048", DRAFT_WINDOW_LAYERS_ENV: "0, 2,4-6"}):
            self.assertEqual(draft_window_from_env(), (2048, frozenset({0, 2, 4, 5, 6})))
        with patch.dict(os.environ, {DRAFT_WINDOW_ENV: "-1"}):
            with self.assertRaises(ValueError):
                draft_window_from_env()

    def test_layer_spec_errors(self):
        self.assertIsNone(parse_layer_ids("  "))
        for bad in ("1,,2", "5-3"):
            with self.assertRaises(ValueError):
                parse_layer_ids(bad)
        check_layers(None, 4)
        check_layers(frozenset({0, 3}), 4)
        with self.assertRaises(ValueError):
            check_layers(frozenset({4}), 4)

    def test_backend_lookup(self):
        full = SimpleNamespace(uno_draft_window=-1)
        self.assertIs(full_attention_backend(full), full)
        self.assertIs(full_attention_backend(SimpleNamespace(full_attn_backend=full)), full)
        with self.assertRaises(RuntimeError):
            full_attention_backend(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
