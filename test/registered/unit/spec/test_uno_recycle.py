"""Contracts for UNO draft recycling: tail extraction and hint placement in the draft input."""

import unittest

import torch

from sglang.srt.speculative.uno_utils import build_uno_draft_input, extract_uno_recycle_tail
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_VOCAB = 1000


class TestUnoRecycleTail(CustomTestCase):
    def test_tail_matches_reference_for_every_accept_length(self):
        # HF reference (uno_generate): committed = props[:1] + accepted + correction,
        # tail = props[committed.shape[1]:] when committed.shape[1] < block_size.
        for width in (3, 4, 8, 16):
            candidates = torch.arange(100, 100 + width)[None, :].repeat(width, 1)
            accept_lens = torch.arange(2, width + 2)  # accepted proposals 0..F-1 -> 2..F+1
            tail, lens = extract_uno_recycle_tail(candidates, accept_lens)
            self.assertEqual(tail.shape, (width, width - 2))
            for row, committed in enumerate(accept_lens.tolist()):
                want = candidates[row, committed:].tolist() if committed < width else []
                self.assertEqual(lens[row].item(), len(want))
                self.assertEqual(tail[row, : len(want)].tolist(), want)
                self.assertTrue(bool((tail[row, len(want) :] == 0).all()))

    def test_full_acceptance_and_first_rejection_edges(self):
        candidates = torch.arange(10, 18)[None, :]  # F = 8
        _, lens = extract_uno_recycle_tail(candidates, torch.tensor([9]))  # all 7 accepted + bonus
        self.assertEqual(lens.item(), 0)
        tail, lens = extract_uno_recycle_tail(candidates, torch.tensor([2]))  # first proposal rejected
        self.assertEqual((lens.item(), tail[0].tolist()), (6, [12, 13, 14, 15, 16, 17]))


class TestUnoRecycleDraftInput(CustomTestCase):
    def test_without_hints_the_input_is_unchanged(self):
        seeds = torch.tensor([5, 6, 7])
        torch.manual_seed(0)
        plain = build_uno_draft_input(seed_tokens=seeds, forward_width=8, vocab_size=_VOCAB)
        torch.manual_seed(0)
        zero = build_uno_draft_input(
            seed_tokens=seeds,
            forward_width=8,
            vocab_size=_VOCAB,
            hint_tokens=torch.full((3, 6), 999),
            hint_lens=torch.zeros(3, dtype=torch.int64),
        )
        self.assertTrue(torch.equal(plain, zero))

    def test_hints_replace_only_their_slots_and_keep_the_noise_stream(self):
        seeds = torch.tensor([5, 6])
        hints = torch.tensor([[900, 901, 902, 903], [910, 911, 912, 913]])
        torch.manual_seed(1)
        plain = build_uno_draft_input(seed_tokens=seeds, forward_width=6, vocab_size=_VOCAB)
        torch.manual_seed(1)
        out = build_uno_draft_input(
            seed_tokens=seeds,
            forward_width=6,
            vocab_size=_VOCAB,
            hint_tokens=hints,
            hint_lens=torch.tensor([3, 1]),
        )
        self.assertEqual(out[0, :4].tolist(), [5, 900, 901, 902])
        self.assertEqual(out[1, :2].tolist(), [6, 910])
        self.assertTrue(torch.equal(out[0, 4:], plain[0, 4:]))  # noise beyond the hints is identical
        self.assertTrue(torch.equal(out[1, 2:], plain[1, 2:]))
        self.assertTrue(torch.equal(out[:, 0], seeds))

    def test_hints_land_on_the_absolute_positions_they_propose(self):
        # Label every proposal with the absolute position it proposes: candidate j of a cycle
        # whose seed sits at C proposes position C + 1 + j. After a cycle commits `accept_len`
        # tokens the new seed sits at C' = C + accept_len and draft slot j sits at C' + j.
        width, seed_pos = 8, 40
        candidates = (seed_pos + 1 + torch.arange(width))[None, :]
        for accept_len in range(2, width + 2):
            new_seed_pos = seed_pos + accept_len
            tail, lens = extract_uno_recycle_tail(candidates, torch.tensor([accept_len]))
            ids = build_uno_draft_input(
                seed_tokens=torch.tensor([new_seed_pos]),
                forward_width=width,
                vocab_size=10**6,
                hint_tokens=tail,
                hint_lens=lens,
            )
            for slot in range(1, 1 + lens.item()):
                self.assertEqual(ids[0, slot].item(), new_seed_pos + slot)

    def test_hint_width_must_fit_the_noise_window(self):
        with self.assertRaises(ValueError):
            build_uno_draft_input(
                seed_tokens=torch.tensor([1]),
                forward_width=4,
                vocab_size=_VOCAB,
                hint_tokens=torch.zeros(1, 4, dtype=torch.int64),
                hint_lens=torch.tensor([1]),
            )


if __name__ == "__main__":
    unittest.main()
