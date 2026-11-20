"""Contracts for the UNO draft KV window settings (--uno-draft-kv-window / --uno-draft-kv-window-layers)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.uno_draft_window import (
    REMOVED_ENV,
    check_layers,
    draft_window,
    full_attention_backend,
    parse_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _window(window=0, layers=None, env=None):
    spec = SimpleNamespace(uno_draft_kv_window=window, uno_draft_kv_window_layers=layers)
    clean = {name: "" for name in REMOVED_ENV}
    with patch("sglang.srt.speculative.uno_draft_window.get_spec", return_value=spec), patch.dict(
        os.environ, {**clean, **(env or {})}
    ):
        return draft_window()


class TestUnoDraftWindow(CustomTestCase):
    def test_off_by_default(self):
        self.assertEqual(_window(), (0, None))

    def test_flag_values(self):
        self.assertEqual(_window(2048, "0, 2,4-6"), (2048, frozenset({0, 2, 4, 5, 6})))
        with self.assertRaises(ValueError):
            _window(-1)

    def test_removed_env_is_rejected(self):
        # A stale export of the old switches must fail loudly, not run without the window.
        for name in REMOVED_ENV:
            with self.assertRaises(ValueError):
                _window(env={name: "1"})
        self.assertEqual(_window(env={"UNO_DRAFT_KV_WINDOW": "0"}), (0, None))

    def test_layer_spec_errors(self):
        self.assertIsNone(parse_layer_ids("  "))
        self.assertIsNone(parse_layer_ids(None))
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

    def test_seed_shift_keeps_c0_row_dense_and_sparse(self):
        import torch

        from sglang.srt.speculative.uno_utils import UnoDraftDistribution, shift_uno_seed_into_verify

        batch, width, k = 2, 4, 3
        seed = torch.tensor([7, 9])
        cands = torch.arange(batch * width).view(batch, width)
        probs = torch.rand(batch, width, k)
        ids = torch.randint(0, 50, (batch, width, k))
        for dist in (UnoDraftDistribution(probs=probs), UnoDraftDistribution(probs=probs, token_ids=ids)):
            out, q = shift_uno_seed_into_verify(seed_tokens=seed, candidates=cands, draft_distribution=dist)
            self.assertTrue(torch.equal(out, torch.cat((seed[:, None], cands[:, :-1]), dim=1)))
            self.assertTrue(torch.equal(q.probs, probs[:, :-1]))
            self.assertTrue(q.token_ids is None if dist.token_ids is None else torch.equal(q.token_ids, ids[:, :-1]))
        out, q = shift_uno_seed_into_verify(seed_tokens=seed, candidates=cands, draft_distribution=None)
        self.assertIsNone(q)


if __name__ == "__main__":
    unittest.main()
