"""Contracts: UNO's prefix-cache track step for linear-attention states.

After a verify UNO commits ``rows`` states following ``track_from`` tokens; the state after
row j covers ``track_from + j + 1`` tokens. The snapshot row must be the one landing on the
track boundary, the same crossing the scheduler records (``_mamba_check_track_boundary``)."""

from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.uno_utils import uno_mamba_track_steps
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cases(interval, max_rows):
    starts, rows = zip(*[(s, r) for s in range(3 * interval) for r in range(max_rows + 1)])
    return torch.tensor(starts), torch.tensor(rows)


class TestUnoMambaTrack(CustomTestCase):
    def test_step_lands_on_the_boundary(self):
        for interval, max_rows in [(8, 8), (16, 7), (64, 16)]:
            starts, rows = _cases(interval, max_rows)
            got = uno_mamba_track_steps(starts, rows, interval)
            for s, r, step in zip(starts.tolist(), rows.tolist(), got.tolist()):
                hits = [j for j in range(r) if (s + j + 1) % interval == 0]
                self.assertEqual(step, hits[-1] if hits else -1, (interval, s, r))

    def test_matches_the_scheduler_record(self):
        # Scheduler: prev = cur - (accept_lens - 1), crossing iff cur // I != prev // I,
        # snapshot seqlen cur // I * I. UNO's verify commit: track_from = prev, rows = cur - prev.
        interval = 16
        starts, rows = _cases(interval, 15)
        got = uno_mamba_track_steps(starts, rows, interval)
        for s, r, step in zip(starts.tolist(), rows.tolist(), got.tolist()):
            cur = s + r
            crossed = cur // interval != s // interval
            self.assertEqual(step >= 0, crossed)
            if crossed:
                self.assertEqual(s + step + 1, cur // interval * interval)

    def test_matches_the_linear_chain_verify_commit(self):
        # Same step as the generic spec commit (EAGLE / NEXTN) gives a linear chain.
        from sglang.srt.speculative import spec_utils

        interval, width = 16, 8
        starts, rows = _cases(interval, width)
        rows = rows.clamp(min=1)  # the generic path always commits the bonus row
        batch = MagicMock()
        batch.seq_lens = starts
        batch.mamba_track_indices = torch.zeros_like(starts)
        accept_index = (
            torch.arange(len(starts))[:, None] * width + torch.arange(width)[None, :]
        )
        with patch.object(spec_utils, "mamba_track_grid", return_value=interval):
            last, steps = spec_utils._verify_commit_step_indices(
                batch=batch, accept_index=accept_index, accept_lens=rows, draft_token_num=width
            )
        self.assertTrue(torch.equal(last, rows - 1))
        self.assertTrue(torch.equal(steps, uno_mamba_track_steps(starts, rows, interval)))


if __name__ == "__main__":
    import unittest

    unittest.main()
