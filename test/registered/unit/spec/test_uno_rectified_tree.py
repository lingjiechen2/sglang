"""UNO tree proposals built from rectified per-parent children."""

import heapq
import unittest

import torch

from sglang.srt.speculative.uno_rectifier import UnoRectifier
from sglang.srt.speculative.uno_tree import build_uno_tree_proposal
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _rectifier(vocab_size, rank, scale, num_candidates):
    w1 = torch.randn(vocab_size, rank, device=_DEVICE) * scale
    w2 = torch.randn(vocab_size, rank, device=_DEVICE) * scale
    return UnoRectifier(markov_w1=w1, markov_w2=w2, num_candidates=num_candidates)


def _reference_tree(root, logits, rectifier, temperature, max_nodes, top_k):
    """Best-first tree where each node's children are its rectified top-k.

    ``logits`` is [S, V]; the lattice is the full vocabulary, so a node's
    children are the top-k of logits[depth] + W1[node token] @ W2^T.
    """
    w1 = rectifier.markov_w1.double()
    w2 = rectifier.markov_w2.double()

    def children(token, depth):
        scores = (logits[depth].double() + w1[token] @ w2.T) / temperature
        log_probs = torch.log_softmax(scores, dim=-1)
        values, tokens = torch.topk(log_probs, top_k)
        return list(zip(values.tolist(), tokens.tolist()))

    nodes = [(int(root), 0, 0.0, -1)]  # token, depth, mass, parent
    frontier = []

    def expose(index):
        token, depth, mass, _ = nodes[index]
        if depth >= logits.shape[0]:
            return
        for rank, (log_prob, child) in enumerate(children(token, depth)):
            heapq.heappush(frontier, (-(mass + log_prob), depth + 1, rank, child, index))

    expose(0)
    while len(nodes) < max_nodes:
        neg_mass, depth, _, token, parent = heapq.heappop(frontier)
        nodes.append((token, depth, -neg_mass, parent))
        expose(len(nodes) - 1)
    return [n[0] for n in nodes[1:]], [n[3] for n in nodes[1:]]


class TestUnoRectifiedTree(CustomTestCase):
    def _proposals(self, rectifier, logits, roots, temperatures, max_nodes, top_k):
        plain = build_uno_tree_proposal(
            roots, logits, max_nodes=max_nodes, candidate_top_k=top_k,
            temperature=temperatures,
        )
        plain = (plain.draft_tokens.clone(), plain.top_scores_index.clone(),
                 plain.parent_list.clone())
        children = rectifier.tree_children(
            draft_logits=logits, root_tokens=roots, temperatures=temperatures,
            top_k=top_k,
        )
        rectified = build_uno_tree_proposal(
            roots, logits, max_nodes=max_nodes, candidate_top_k=top_k,
            temperature=temperatures, rectified_children=children,
        )
        return plain, rectified

    def test_zero_rectifier_full_lattice_matches_plain_tree(self):
        torch.manual_seed(0)
        batch, depths, vocab, max_nodes, top_k = 3, 5, 128, 16, 4
        rectifier = _rectifier(vocab, 8, 0.0, num_candidates=vocab)
        logits = torch.randn(batch, depths, vocab, device=_DEVICE) * 3
        roots = torch.randint(0, vocab, (batch,), device=_DEVICE)
        temperatures = torch.tensor([1.0, 0.7, 0.0], device=_DEVICE)
        plain, rectified = self._proposals(
            rectifier, logits, roots, temperatures, max_nodes, top_k
        )
        torch.testing.assert_close(rectified.draft_tokens, plain[0], rtol=0, atol=0)
        torch.testing.assert_close(rectified.top_scores_index, plain[1], rtol=0, atol=0)
        torch.testing.assert_close(rectified.parent_list, plain[2], rtol=0, atol=0)

    def test_rectified_tree_matches_reference(self):
        torch.manual_seed(1)
        batch, depths, vocab, max_nodes, top_k = 4, 6, 96, 20, 4
        rectifier = _rectifier(vocab, 16, 0.6, num_candidates=vocab)
        logits = torch.randn(batch, depths, vocab, device=_DEVICE) * 2
        roots = torch.randint(0, vocab, (batch,), device=_DEVICE)
        temperatures = torch.tensor([1.0, 0.8, 1.3, 1.0], device=_DEVICE)
        _, proposal = self._proposals(
            rectifier, logits, roots, temperatures, max_nodes, top_k
        )
        for b in range(batch):
            tokens, parents = _reference_tree(
                roots[b], logits[b], rectifier, float(temperatures[b]), max_nodes, top_k
            )
            self.assertEqual(proposal.draft_tokens[b].tolist(), tokens)
            # EAGLE's edge encoding is parent_node * K + rank.
            self.assertEqual(
                (proposal.top_scores_index[b] // top_k).tolist(), parents
            )

    def test_rectifier_changes_children_per_parent(self):
        torch.manual_seed(2)
        batch, depths, vocab, max_nodes, top_k = 2, 4, 64, 14, 4
        rectifier = _rectifier(vocab, 16, 2.0, num_candidates=32)
        logits = torch.randn(batch, depths, vocab, device=_DEVICE)
        roots = torch.randint(0, vocab, (batch,), device=_DEVICE)
        temperatures = torch.ones(batch, device=_DEVICE)
        plain, rectified = self._proposals(
            rectifier, logits, roots, temperatures, max_nodes, top_k
        )
        self.assertFalse(torch.equal(rectified.draft_tokens, plain[0]))
        # Every node sits under a parent that precedes it, and siblings differ.
        for b in range(batch):
            parents = (rectified.top_scores_index[b] // top_k).tolist()
            tokens = rectified.draft_tokens[b].tolist()
            self.assertTrue(all(p <= i for i, p in enumerate(parents)))
            seen = set()
            for p, t in zip(parents, tokens):
                self.assertNotIn((p, t), seen)
                seen.add((p, t))


if __name__ == "__main__":
    unittest.main()
