"""UNO tree GDN kernels match a per-node reference of the generic tree verify
(the same formulas as causal_conv1d_update / fused_sigmoid_gating_delta_rule
in tree mode): the fused tree conv, the ancestor-masked chunkwise verify and
the path commit."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.gdn_uno_tree import (
    uno_tree_conv,
    uno_tree_gdn_commit,
    uno_tree_gdn_verify,
    uno_tree_structure,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"


def _random_tree(batch, num_nodes, chain):
    """Parents (parent before child), EAGLE first-child / next-sibling links."""
    parents = torch.full((batch, num_nodes), -1, dtype=torch.int64)
    for b in range(batch):
        for i in range(1, num_nodes):
            parents[b, i] = i - 1 if chain else int(torch.randint(0, i, (1,)))
    next_token = torch.full((batch, num_nodes), -1, dtype=torch.int64)
    next_sibling = torch.full((batch, num_nodes), -1, dtype=torch.int64)
    for b in range(batch):
        last_child = {}
        for i in range(1, num_nodes):
            p = int(parents[b, i])
            if p in last_child:
                next_sibling[b, last_child[p]] = i
            else:
                next_token[b, p] = i
            last_child[p] = i
    return parents, next_token.to(_DEVICE), next_sibling.to(_DEVICE)


def _reference_conv(x, conv_state, weight, cache_indices, parents, batch, num_nodes):
    """Node i convolves [.., grand-parent, parent, i]; the conv state precedes the root."""
    dim, width = weight.shape
    out = torch.empty(batch * num_nodes, dim, device=_DEVICE)
    windows = torch.empty(batch, num_nodes, dim, width - 1, device=_DEVICE, dtype=x.dtype)
    for b in range(batch):
        state = conv_state[int(cache_indices[b])]  # [dim, W - 1], newest last
        for i in range(num_nodes):
            taps, cur, missing = [], i, 0
            for _ in range(width):
                if cur >= 0:
                    taps.append(x[b * num_nodes + cur])
                    cur = int(parents[b, cur])
                else:
                    missing += 1
                    taps.append(state[:, width - 1 - missing])
            taps = taps[::-1]  # oldest first
            acc = sum(taps[j].float() * weight[:, j].float() for j in range(width))
            out[b * num_nodes + i] = acc / (1 + torch.exp(-acc))
            windows[b, i] = torch.stack(taps[1:], dim=-1)
    return out, windows


def _reference_rule(q, k, v, a, b, A_log, dt_bias, state, cache_indices, parents,
                    batch, num_nodes, H, HV, K, V):
    """Every node continues its parent's state, as the generic tree verify does."""
    ratio = HV // H
    out = torch.empty(batch * num_nodes, HV, V, device=_DEVICE)
    node_states = torch.empty(batch, num_nodes, HV, V, K, device=_DEVICE)
    for bi in range(batch):
        for i in range(num_nodes):
            t = bi * num_nodes + i
            p = int(parents[bi, i])
            h = (state[int(cache_indices[bi])] if p < 0 else node_states[bi, p]).clone()
            qt = q[t].float().view(H, K).repeat_interleave(ratio, 0)
            kt = k[t].float().view(H, K).repeat_interleave(ratio, 0)
            qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * K**-0.5
            kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            vt = v[t].float().view(HV, V)
            x = a[t].float() + dt_bias
            g = -torch.exp(A_log) * torch.where(x <= 20, torch.log1p(torch.exp(x)), x)
            beta = torch.sigmoid(b[t].float())
            h = h * torch.exp(g)[:, None, None]  # h: [HV, V, K]
            delta = (vt - torch.einsum("hvk,hk->hv", h, kt)) * beta[:, None]
            h = h + delta[:, :, None] * kt[:, None, :]
            node_states[bi, i] = h
            out[t] = torch.einsum("hvk,hk->hv", h, qt)
    return out, node_states


def _paths(parents, batch, num_nodes):
    """Root-to-node path of a random node per request."""
    paths, lens = [], []
    for b in range(batch):
        node = int(torch.randint(0, num_nodes, (1,)))
        path = [node]
        while parents[b, path[-1]] >= 0:
            path.append(int(parents[b, path[-1]]))
        paths.append(path[::-1])
        lens.append(len(path))
    width = max(lens)
    out = torch.full((batch, width), -1, dtype=torch.int32)
    for b, p in enumerate(paths):
        out[b, : len(p)] = torch.tensor(p, dtype=torch.int32)
    return out.to(_DEVICE), torch.tensor(lens, dtype=torch.int32, device=_DEVICE)


class TestUnoTreeGdn(CustomTestCase):
    H, HV, K, V, W = 4, 12, 128, 128, 4

    def _case(self, batch, num_nodes, chain, seed):
        torch.manual_seed(seed)
        H, HV, K, V, W = self.H, self.HV, self.K, self.V, self.W
        dim = 2 * H * K + HV * V
        slots, node_stride = batch + 3, num_nodes + 2
        tokens = batch * num_nodes
        parents, next_token, next_sibling = _random_tree(batch, num_nodes, chain)
        parent_buf = ancestor_buf = None
        if not chain:
            parent_buf = torch.empty(batch + 1, num_nodes, dtype=torch.int32, device=_DEVICE)
            ancestor_buf = torch.empty(batch + 1, num_nodes, num_nodes, dtype=torch.int8, device=_DEVICE)
            uno_tree_structure(next_token, next_sibling, parent_buf, ancestor_buf,
                               num_nodes, max_depth=num_nodes - 1)
            self.assertEqual(parent_buf[:batch, :num_nodes].tolist(), parents.tolist())
            parent_buf, ancestor_buf = parent_buf[:batch], ancestor_buf[:batch]

        # Projection output is a strided view, as UNO's fused LoRA hands it over.
        x = torch.randn(tokens, dim + 64, device=_DEVICE, dtype=torch.bfloat16)[:, :dim]
        conv_state = (
            torch.randn(slots, W - 1, dim, device=_DEVICE, dtype=torch.bfloat16)
            .transpose(1, 2)
        )
        weight = torch.randn(dim, W, device=_DEVICE, dtype=torch.bfloat16) * 0.5
        cache_indices = torch.randperm(slots, device=_DEVICE)[:batch].to(torch.int32)
        rows = torch.arange(batch, device=_DEVICE, dtype=torch.int32)

        # --- conv: outputs and per-node windows.
        new_window = torch.zeros(batch + 1, node_stride, dim, W - 1, device=_DEVICE,
                                 dtype=torch.bfloat16)
        ref_conv, ref_window = _reference_conv(
            x, conv_state, weight, cache_indices, parents, batch, num_nodes
        )
        new_conv = uno_tree_conv(
            x, conv_state, weight, None, "silu", cache_indices, new_window, rows,
            num_nodes, parent_buf,
        )
        torch.testing.assert_close(new_conv.float(), ref_conv, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(new_window[:batch, :num_nodes], ref_window, rtol=0, atol=0)

        # --- delta rule: outputs vs the per-node recurrence.
        conv = new_conv
        q, k, v = conv.split([H * K, H * K, HV * V], dim=-1)
        a = torch.randn(tokens, HV, device=_DEVICE, dtype=torch.bfloat16)
        b = torch.randn(tokens, HV, device=_DEVICE, dtype=torch.bfloat16)
        A_log = torch.rand(HV, device=_DEVICE) * 0.5
        dt_bias = torch.rand(HV, device=_DEVICE)
        state = torch.randn(slots, HV, V, K, device=_DEVICE) * 0.05
        ref_out, node_states = _reference_rule(
            q, k, v, a, b, A_log, dt_bias, state, cache_indices, parents,
            batch, num_nodes, H, HV, K, V,
        )
        saved_k = torch.empty(1, (batch + 1) * node_stride, H * K, device=_DEVICE, dtype=torch.bfloat16)
        saved_v = torch.empty(1, (batch + 1) * node_stride, HV * V, device=_DEVICE, dtype=torch.bfloat16)
        saved_g = torch.empty(1, (batch + 1) * node_stride, HV, device=_DEVICE)
        saved_beta = torch.empty_like(saved_g)
        new_out = uno_tree_gdn_verify(
            q, k, v, a, b, A_log, dt_bias, state, cache_indices,
            saved_k[0], saved_v[0], saved_g[0], saved_beta[0], rows,
            num_nodes=num_nodes, node_stride=node_stride, num_k_heads=H,
            head_k_dim=K, head_v_dim=V, ancestors=ancestor_buf,
        )
        torch.testing.assert_close(new_out.float(), ref_out, rtol=3e-2, atol=3e-2)

        # --- commit along a random accepted path vs the stored node state.
        path, lens = _paths(parents, batch, num_nodes)
        last = path.gather(1, (lens - 1).long()[:, None])[:, 0].long()
        expected = node_states[torch.arange(batch, device=_DEVICE), last]
        committed = state.clone()
        uno_tree_gdn_commit(
            committed.unsqueeze(0), saved_k, saved_v, saved_g, saved_beta,
            cache_indices, rows, path, lens, node_stride=node_stride, num_k_heads=H,
        )
        torch.testing.assert_close(committed[cache_indices.long()], expected, rtol=1e-4, atol=1e-4)
        untouched = torch.ones(slots, dtype=torch.bool, device=_DEVICE)
        untouched[cache_indices.long()] = False
        torch.testing.assert_close(committed[untouched], state[untouched], rtol=0, atol=0)

    def test_tree_after_uncommitted_seed(self):
        """Draft chain saves its seed; the tree verify and the commit fold the
        seed in without it ever being committed on its own."""
        torch.manual_seed(3)
        H, HV, K, V, W = self.H, self.HV, self.K, self.V, self.W
        batch, F, Q = 3, 8, 16
        dim = 2 * H * K + HV * V
        slots, offset = batch + 2, Q
        node_stride = Q + F
        cache_indices = torch.randperm(slots, device=_DEVICE)[:batch].to(torch.int32)
        rows = torch.arange(batch, device=_DEVICE, dtype=torch.int32)
        conv_state = torch.randn(slots, W - 1, dim, device=_DEVICE, dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.randn(dim, W, device=_DEVICE, dtype=torch.bfloat16) * 0.5
        A_log = torch.rand(HV, device=_DEVICE) * 0.5
        dt_bias = torch.rand(HV, device=_DEVICE)
        state = torch.randn(slots, HV, V, K, device=_DEVICE) * 0.05
        saved = dict(
            saved_k=torch.empty(1, (batch + 1) * node_stride, H * K, device=_DEVICE, dtype=torch.bfloat16),
            saved_v=torch.empty(1, (batch + 1) * node_stride, HV * V, device=_DEVICE, dtype=torch.bfloat16),
            saved_g=torch.empty(1, (batch + 1) * node_stride, HV, device=_DEVICE),
        )
        saved["saved_beta"] = torch.empty_like(saved["saved_g"])
        seed_x = torch.empty(batch + 1, dim, device=_DEVICE, dtype=torch.bfloat16)
        window = torch.zeros(batch + 1, Q, dim, W - 1, device=_DEVICE, dtype=torch.bfloat16)

        def run(x, a, b, num_nodes, parents_buf, anc_buf, **kw):
            conv = uno_tree_conv(x, conv_state, weight, None, "silu", cache_indices, window,
                                 rows, num_nodes, parents_buf, seed_x=seed_x,
                                 save_seed=kw["save_seed"], has_seed=kw["has_seed"])
            q, k, v = conv.split([H * K, H * K, HV * V], dim=-1)
            out = uno_tree_gdn_verify(
                q, k, v, a, b, A_log, dt_bias, state, cache_indices,
                saved["saved_k"][0], saved["saved_v"][0], saved["saved_g"][0],
                saved["saved_beta"][0], rows, num_nodes=num_nodes, node_stride=node_stride,
                num_k_heads=H, head_k_dim=K, head_v_dim=V, ancestors=anc_buf,
                max_depth=F - 1, save_offset=kw["offset"], seed_node=kw["seed"])
            return conv, q, k, v, out

        # Draft chain (F rows) from the committed state.
        xd = torch.randn(batch * F, dim, device=_DEVICE, dtype=torch.bfloat16)
        ad = torch.randn(batch * F, HV, device=_DEVICE, dtype=torch.bfloat16)
        bd = torch.randn(batch * F, HV, device=_DEVICE, dtype=torch.bfloat16)
        chain_parents = torch.arange(-1, F - 1).repeat(batch, 1)
        _, qd, kd, vd, _ = run(xd, ad, bd, F, None, None, save_seed=True, has_seed=False,
                               offset=offset, seed=None)
        _, chain_windows = _reference_conv(xd, conv_state, weight, cache_indices,
                                           chain_parents, batch, F)
        _, chain_states = _reference_rule(qd, kd, vd, ad, bd, A_log, dt_bias, state,
                                          cache_indices, chain_parents, batch, F, H, HV, K, V)
        # Reference committed state after the seed alone.
        state1, conv1 = state.clone(), conv_state.clone()
        for bi in range(batch):
            state1[int(cache_indices[bi])] = chain_states[bi, 0]
            conv1[int(cache_indices[bi])] = chain_windows[bi, 0]

        # Tree verify from the uncommitted state with the seed folded in.
        parents, next_token, next_sibling = _random_tree(batch, Q, chain=False)
        parent_buf = torch.empty(batch + 1, Q, dtype=torch.int32, device=_DEVICE)
        anc_buf = torch.empty(batch + 1, Q, Q, dtype=torch.int8, device=_DEVICE)
        uno_tree_structure(next_token, next_sibling, parent_buf, anc_buf, Q, max_depth=Q - 1)
        xt = torch.randn(batch * Q, dim, device=_DEVICE, dtype=torch.bfloat16)
        at = torch.randn(batch * Q, HV, device=_DEVICE, dtype=torch.bfloat16)
        bt = torch.randn(batch * Q, HV, device=_DEVICE, dtype=torch.bfloat16)
        conv_t, qt, kt, vt, out_t = run(xt, at, bt, Q, parent_buf[:batch], anc_buf[:batch],
                                        save_seed=False, has_seed=True, offset=0, seed=offset)
        ref_conv, ref_windows = _reference_conv(xt, conv1, weight, cache_indices, parents, batch, Q)
        torch.testing.assert_close(conv_t.float(), ref_conv, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(window[:batch], ref_windows, rtol=0, atol=0)
        ref_out, tree_states = _reference_rule(qt, kt, vt, at, bt, A_log, dt_bias, state1,
                                               cache_indices, parents, batch, Q, H, HV, K, V)
        torch.testing.assert_close(out_t.float(), ref_out, rtol=3e-2, atol=3e-2)

        # One commit: seed + accepted path, from the never-advanced state.
        path, lens = _paths(parents, batch, Q)
        last = path.gather(1, (lens - 1).long()[:, None])[:, 0].long()
        committed = state.clone()
        uno_tree_gdn_commit(committed.unsqueeze(0), saved["saved_k"], saved["saved_v"],
                            saved["saved_g"], saved["saved_beta"], cache_indices, rows,
                            path, lens, node_stride=node_stride, num_k_heads=H, seed_node=offset)
        expected = tree_states[torch.arange(batch, device=_DEVICE), last]
        torch.testing.assert_close(committed[cache_indices.long()], expected, rtol=1e-4, atol=1e-4)

    def test_tree_q16(self):
        self._case(batch=3, num_nodes=16, chain=False, seed=0)

    def test_tree_q32(self):
        self._case(batch=2, num_nodes=32, chain=False, seed=1)

    def test_tree_q64(self):
        self._case(batch=2, num_nodes=64, chain=False, seed=4)

    def test_chain_draft_f8(self):
        self._case(batch=4, num_nodes=8, chain=True, seed=2)


if __name__ == "__main__":
    unittest.main()
