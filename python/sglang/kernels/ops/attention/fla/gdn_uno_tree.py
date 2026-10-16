"""GDN (gated delta rule) kernels for UNO tree verification.

The generic tree verify of a GDN layer stores every node's full recurrent
state (HV x V x K fp32 per node and layer) so the accepted node's state can
be copied back afterwards.  UNO tree mode instead

  * convolves each node over its own ancestors in one fused kernel
    (``uno_tree_conv``), writing only the small per-node conv windows;
  * computes all node outputs from the committed state in chunkwise (WY)
    form with the causal mask replaced by the ancestor mask
    (``uno_tree_gdn_verify``), saving each node's raw k, v, g and beta;
  * commits by replaying the accepted path from those saved inputs
    (``uno_tree_gdn_commit``) with the recurrent kernel's update order.

A tree is given either by EAGLE's ``retrieve_next_token`` /
``retrieve_next_sibling`` (verify pass) or is a chain (UNO draft pass).
Node 0 is the root; a parent always precedes its children.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _uno_tree_structure_kernel(
    next_token,
    next_sibling,
    stride_nt_seq,
    stride_nt_tok,
    stride_ns_seq,
    stride_ns_tok,
    parents_out,  # [B, NODE_STRIDE] int32
    ancestors_out,  # [B, NODE_STRIDE] int64, bit j: node j is an ancestor-or-self
    NUM_NODES: tl.constexpr,
    NP2: tl.constexpr,
    NODE_STRIDE: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
):
    """Parents and ancestor bit masks of one request's tree, once per forward."""
    seq = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, NP2)
    live = idx < NUM_NODES
    first_child = tl.load(
        next_token + seq * stride_nt_seq + idx * stride_nt_tok, mask=live, other=-1
    )
    sibling = tl.load(
        next_sibling + seq * stride_ns_seq + idx * stride_ns_tok, mask=live, other=-1
    )
    parents = tl.full((NP2,), -1, tl.int32)
    # Same derivation as the generic tree conv: a node's first child gets the
    # node as parent, its next sibling gets the node's parent.
    for t in range(NUM_NODES):
        child = tl.sum(tl.where(idx == t, first_child, 0))
        parents = tl.where((idx == child) & (child >= 0), t, parents)
        nxt = tl.sum(tl.where(idx == t, sibling, 0))
        parent_t = tl.sum(tl.where(idx == t, parents, 0))
        parents = tl.where((idx == nxt) & (nxt >= 0), parent_t, parents)
    one = tl.full((NP2,), 1, tl.int64)
    bits = tl.where(live, one << idx.to(tl.int64), 0)
    cur = idx
    for _ in tl.static_range(MAX_DEPTH):
        hit = idx[None, :] == cur[:, None]
        cur = tl.where(cur >= 0, tl.sum(tl.where(hit, parents[None, :], 0), axis=1), -1)
        bits = bits | tl.where(cur >= 0, one << tl.maximum(cur, 0).to(tl.int64), 0)
    tl.store(parents_out + seq * NODE_STRIDE + idx, parents, mask=live)
    tl.store(ancestors_out + seq * NODE_STRIDE + idx, bits, mask=live)


def uno_tree_structure(
    next_token: torch.Tensor,  # [B, Q]
    next_sibling: torch.Tensor,
    parents: torch.Tensor,  # [>= B, node_stride] int32 (out)
    ancestors: torch.Tensor,  # [>= B, node_stride] int64 (out)
    num_nodes: int,
    max_depth: int,
) -> None:
    assert num_nodes <= 63 and parents.stride(0) == ancestors.stride(0)
    _uno_tree_structure_kernel[(next_token.shape[0],)](
        next_token,
        next_sibling,
        next_token.stride(0),
        next_token.stride(1),
        next_sibling.stride(0),
        next_sibling.stride(1),
        parents,
        ancestors,
        NUM_NODES=num_nodes,
        NP2=triton.next_power_of_2(num_nodes),
        NODE_STRIDE=parents.stride(0),
        MAX_DEPTH=max(min(max_depth, num_nodes - 1), 1),
        num_warps=1,
    )


@triton.jit
def _uno_tree_conv_kernel(
    x,  # [B * Q, dim] node inputs (row stride stride_x)
    stride_x,
    conv_state,  # [slots, dim, W - 1]
    stride_cs_slot,
    stride_cs_dim,
    stride_cs_win,
    cache_indices,  # [B]
    weight,  # [dim, W]
    stride_w_dim,
    stride_w_tap,
    bias,
    out,  # [B * Q, dim] contiguous
    window,  # [rows, nodes, dim, W - 1]
    stride_win_row,
    stride_win_node,
    stride_win_dim,
    stride_win_pos,
    window_rows,  # [B]
    parents,  # [B, NODE_STRIDE] int32 (IS_TREE)
    seed_x,  # [rows, dim] raw input of the seed token (SAVE_SEED / HAS_SEED)
    stride_seed,
    dim,
    Q: tl.constexpr,
    NODE_STRIDE: tl.constexpr,
    W: tl.constexpr,
    BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    IS_TREE: tl.constexpr,
    SAVE_SEED: tl.constexpr,
    HAS_SEED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    seq = row // Q
    node = (row % Q).to(tl.int32)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    c_mask = channels < dim
    slot = tl.load(cache_indices + seq).to(tl.int64)
    win_row = tl.load(window_rows + seq).to(tl.int64)
    valid = slot >= 0

    acc = tl.zeros((BC,), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + channels, mask=c_mask, other=0.0).to(tl.float32)
    # Level 0 is the node, level l its l-th ancestor; past the root the taps
    # continue with the uncommitted seed token (HAS_SEED) and then the conv
    # state, newest column first (cur = -1, -2, ...).
    cur = node
    for level in tl.static_range(W):
        from_x = cur >= 0
        safe = tl.maximum(cur, 0)
        missing = tl.maximum(-cur, 0)
        if HAS_SEED:
            from_seed = cur == -1
            state_col = W - missing
        else:
            from_seed = cur < -W
            state_col = W - 1 - missing
        tap = tl.load(
            x + (seq * Q + safe) * stride_x + channels,
            mask=c_mask & from_x,
            other=0.0,
        )
        tap += tl.load(
            seed_x + win_row * stride_seed + channels,
            mask=c_mask & valid & from_seed,
            other=0.0,
        )
        tap += tl.load(
            conv_state
            + slot * stride_cs_slot
            + channels * stride_cs_dim
            + tl.maximum(state_col, 0) * stride_cs_win,
            mask=c_mask & valid & (cur < 0) & (from_seed == 0),
            other=0.0,
        )
        if SAVE_SEED:
            if level == 0:
                # The draft pass's row 0 is the seed; tree verify consumes it.
                tl.store(seed_x + win_row * stride_seed + channels, tap,
                         mask=c_mask & valid & (node == 0))
        w = tl.load(weight + channels * stride_w_dim + (W - 1 - level) * stride_w_tap,
                    mask=c_mask, other=0.0)
        acc += tap.to(tl.float32) * w.to(tl.float32)
        if level < W - 1:
            # Window after this node, oldest first: [.., parent, node].
            tl.store(
                window
                + win_row * stride_win_row
                + node * stride_win_node
                + channels * stride_win_dim
                + (W - 2 - level) * stride_win_pos,
                tap.to(window.dtype.element_ty),
                mask=c_mask & valid,
            )
        if IS_TREE:
            parent = tl.load(parents + seq * NODE_STRIDE + safe)
        else:
            parent = safe - 1
        cur = tl.where(from_x, parent, cur - 1)
    if SILU:
        acc = acc / (1.0 + tl.exp(-acc))
    tl.store(out + row * dim + channels, acc.to(out.dtype.element_ty), mask=c_mask)


def uno_tree_conv(
    x: torch.Tensor,  # [B * Q, dim], feature dim contiguous
    conv_state: torch.Tensor,  # [slots, dim, W - 1]
    weight: torch.Tensor,  # [dim, W]
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    cache_indices: torch.Tensor,  # [B]
    window: torch.Tensor,  # [rows, nodes >= Q, dim, W - 1]
    window_rows: torch.Tensor,  # [B]
    num_nodes: int,
    parents: Optional[torch.Tensor] = None,  # [B, node_stride]; None = chain
    seed_x: Optional[torch.Tensor] = None,  # [rows, dim]
    save_seed: bool = False,
    has_seed: bool = False,
) -> torch.Tensor:
    """``save_seed``: store row 0's input (the draft seed) into ``seed_x``;
    ``has_seed``: the seed precedes the root (its state is not committed)."""
    tokens, dim = x.shape
    assert x.stride(1) == 1 and tokens % num_nodes == 0
    batch = tokens // num_nodes
    width = weight.shape[1]
    out = torch.empty((tokens, dim), dtype=x.dtype, device=x.device)
    is_tree = parents is not None
    block_c = 256
    _uno_tree_conv_kernel[(tokens, triton.cdiv(dim, block_c))](
        x, x.stride(0),
        conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        cache_indices,
        weight, weight.stride(0), weight.stride(1),
        bias if bias is not None else weight,
        out,
        window, window.stride(0), window.stride(1), window.stride(2), window.stride(3),
        window_rows,
        parents if is_tree else cache_indices,
        seed_x if seed_x is not None else x,
        seed_x.stride(0) if seed_x is not None else 0,
        dim,
        Q=num_nodes,
        NODE_STRIDE=parents.stride(0) if is_tree else 0,
        W=width,
        BC=block_c,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        IS_TREE=is_tree,
        SAVE_SEED=save_seed,
        HAS_SEED=has_seed,
        num_warps=4,
    )
    return out


@triton.jit
def _uno_tree_gdn_verify_kernel(
    q, k, v, a, b,
    stride_q, stride_k, stride_v, stride_a, stride_b,
    A_log, dt_bias,
    state,  # [slots, HV, V, K] fp32 (slot stride stride_state)
    stride_state,
    cache_indices,  # [B]
    out,  # [B * Q, HV, V]
    saved_k, saved_v, saved_g, saved_beta,  # [rows * NODE_STRIDE, ...]
    save_rows,  # [B]
    ancestors,  # [B, ANC_STRIDE] int64 ancestor bits (IS_TREE)
    scale,
    SAVE_OFFSET: tl.constexpr,
    SEED_NODE: tl.constexpr,
    HAS_SEED: tl.constexpr,
    Q: tl.constexpr,
    QP: tl.constexpr,
    NODE_STRIDE: tl.constexpr,
    ANC_STRIDE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_TREE: tl.constexpr,
    PRECISION: tl.constexpr,
    LOG_DEPTH: tl.constexpr,
):
    """Chunkwise (WY) delta rule over a tree, from the committed state S0.

    With G_i the summed log decay along node i's path P(i):
      u_j = beta_j (v_j - e^{G_j} k_j S0 - sum_{l in P(j), l != j} e^{G_j - G_l} (k_j . k_l) u_l)
      o_i = e^{G_i} q_i S0 + sum_{j in P(i)} e^{G_i - G_j} (q_i . k_j) u_j
    (q, k l2-normalized, q scaled).  The system (I + L) u = rhs has L strictly
    "ancestor-lower" and nilpotent of index <= MAX_DEPTH + 1, so
    (I + L)^-1 = (I - L)(I + L^2)(I + L^4)...: LOG_DEPTH small matmuls, and no
    node order beyond parents existing is assumed.
    """
    i_b = tl.program_id(0).to(tl.int64)
    i_hv = tl.program_id(1)
    i_v = tl.program_id(2)
    i_h = i_hv // (HV // H)
    slot = tl.load(cache_indices + i_b).to(tl.int64)
    save_row = tl.load(save_rows + i_b).to(tl.int64)

    o_q = tl.arange(0, QP)
    q_ok = o_q < Q
    o_v = i_v * BV + tl.arange(0, BV)
    rows = i_b * Q + tl.where(q_ok, o_q, 0)

    # Gates, exactly as the recurrent kernel forms them.
    a_n = tl.load(a + rows * stride_a + i_hv, mask=q_ok, other=0.0).to(tl.float32)
    b_n = tl.load(b + rows * stride_b + i_hv, mask=q_ok, other=0.0).to(tl.float32)
    x_n = a_n + tl.load(dt_bias + i_hv).to(tl.float32)
    softplus = tl.where(x_n <= 20.0, tl.log(1.0 + tl.exp(x_n)), x_n)
    g_n = tl.where(q_ok, -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus, 0.0)
    beta_n = 1.0 / (1.0 + tl.exp(-b_n))

    # Ancestor-or-self mask.
    if IS_TREE:
        bits = tl.load(ancestors + i_b * ANC_STRIDE + o_q, mask=q_ok, other=0)
        anc = ((bits[:, None] >> o_q[None, :].to(tl.int64)) & 1) != 0
    else:
        anc = o_q[None, :] <= o_q[:, None]
    anc = anc & q_ok[:, None] & q_ok[None, :]

    G = tl.sum(tl.where(anc, g_n[None, :], 0.0), axis=1)
    decay = tl.where(anc, tl.exp(G[:, None] - G[None, :]), 0.0)
    eG = tl.exp(G)

    kk = tl.zeros([QP, QP], dtype=tl.float32)
    qk = tl.zeros([QP, QP], dtype=tl.float32)
    ks = tl.zeros([QP, BV], dtype=tl.float32)
    qs = tl.zeros([QP, BV], dtype=tl.float32)
    k_sq = tl.zeros([QP], dtype=tl.float32)
    q_sq = tl.zeros([QP], dtype=tl.float32)
    store_k = (i_v == 0) & (i_hv % (HV // H) == 0) & (save_row >= 0)
    if HAS_SEED:
        # The seed token sits between the committed state S0 and the root:
        # S1 = e^{g_s} S0 + k_s u_s^T, folded in below.
        seed_row = save_row * NODE_STRIDE + SEED_NODE
        kks = tl.zeros([QP], dtype=tl.float32)
        qks = tl.zeros([QP], dtype=tl.float32)
        kss = tl.zeros([BV], dtype=tl.float32)
        ss_sq = 0.0
    for kc in tl.static_range(0, K, BK):
        o_k = kc + tl.arange(0, BK)
        k_raw = tl.load(k + rows[:, None] * stride_k + i_h * K + o_k[None, :],
                        mask=q_ok[:, None], other=0.0)
        b_q = tl.load(q + rows[:, None] * stride_q + i_h * K + o_k[None, :],
                      mask=q_ok[:, None], other=0.0).to(tl.float32)
        b_k = k_raw.to(tl.float32)
        if store_k:
            tl.store(saved_k + (save_row * NODE_STRIDE + SAVE_OFFSET + o_q[:, None]) * (H * K)
                     + i_h * K + o_k[None, :], k_raw, mask=q_ok[:, None])
        k_sq += tl.sum(b_k * b_k, axis=1)
        q_sq += tl.sum(b_q * b_q, axis=1)
        kk += tl.dot(b_k, tl.trans(b_k), input_precision=PRECISION)
        qk += tl.dot(b_q, tl.trans(b_k), input_precision=PRECISION)
        h0 = tl.load(state + slot * stride_state + i_hv * K * V
                     + o_v[None, :] * K + o_k[:, None],
                     mask=(slot >= 0) & (o_v[None, :] < V), other=0.0).to(tl.float32)
        ks += tl.dot(b_k, h0, input_precision=PRECISION)
        qs += tl.dot(b_q, h0, input_precision=PRECISION)
        if HAS_SEED:
            k_s = tl.load(saved_k + seed_row * (H * K) + i_h * K + o_k).to(tl.float32)
            ss_sq += tl.sum(k_s * k_s)
            kks += tl.sum(b_k * k_s[None, :], axis=1)
            qks += tl.sum(b_q * k_s[None, :], axis=1)
            kss += tl.sum(k_s[:, None] * h0, axis=0)
    r_k = 1.0 / tl.sqrt(k_sq + 1e-6)
    r_q = scale / tl.sqrt(q_sq + 1e-6)
    kk = kk * r_k[:, None] * r_k[None, :]
    qk = qk * r_q[:, None] * r_k[None, :]
    ks = ks * r_k[:, None]
    qs = qs * r_q[:, None]
    if HAS_SEED:
        r_s = 1.0 / tl.sqrt(ss_sq + 1e-6)
        g_s = tl.load(saved_g + seed_row * HV + i_hv)
        beta_s = tl.load(saved_beta + seed_row * HV + i_hv)
        v_s = tl.load(saved_v + seed_row * (HV * V) + i_hv * V + o_v).to(tl.float32)
        decay_s = tl.exp(g_s)
        u_s = beta_s * (v_s - decay_s * kss * r_s)
        ks = decay_s * ks + (kks * r_k * r_s)[:, None] * u_s[None, :]
        qs = decay_s * qs + (qks * r_q * r_s)[:, None] * u_s[None, :]

    v_raw = tl.load(v + rows[:, None] * stride_v + i_hv * V + o_v[None, :],
                    mask=q_ok[:, None], other=0.0)
    if save_row >= 0:
        node_rows = save_row * NODE_STRIDE + SAVE_OFFSET + o_q
        tl.store(saved_v + node_rows[:, None] * (HV * V) + i_hv * V + o_v[None, :],
                 v_raw, mask=q_ok[:, None])
        if i_v == 0:
            tl.store(saved_g + node_rows * HV + i_hv, g_n, mask=q_ok)
            tl.store(saved_beta + node_rows * HV + i_hv, beta_n, mask=q_ok)

    strict = anc & (o_q[:, None] != o_q[None, :])
    eye = (o_q[:, None] == o_q[None, :]).to(tl.float32)
    neg_l = tl.where(strict, -beta_n[:, None] * kk * decay, 0.0)
    inv = eye + neg_l
    power = neg_l
    for _ in tl.static_range(LOG_DEPTH - 1):
        power = tl.dot(power, power, input_precision=PRECISION)
        inv = tl.dot(inv, eye + power, input_precision=PRECISION)
    rhs = beta_n[:, None] * (v_raw.to(tl.float32) - eG[:, None] * ks)
    u = tl.dot(inv, rhs, input_precision=PRECISION)
    o = eG[:, None] * qs + tl.dot(qk * decay, u, input_precision=PRECISION)
    tl.store(out + (rows[:, None] * HV + i_hv) * V + o_v[None, :],
             o.to(out.dtype.element_ty), mask=q_ok[:, None])


def uno_tree_gdn_verify(
    q: torch.Tensor,  # [B * Q, H * K] (row stride may exceed H * K)
    k: torch.Tensor,
    v: torch.Tensor,  # [B * Q, HV * V]
    a: torch.Tensor,  # [B * Q, HV]
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,  # [slots, HV, V, K] fp32
    cache_indices: torch.Tensor,  # [B]
    saved_k: torch.Tensor,  # [rows * node_stride, H * K]
    saved_v: torch.Tensor,  # [rows * node_stride, HV * V]
    saved_g: torch.Tensor,  # [rows * node_stride, HV] fp32
    saved_beta: torch.Tensor,
    save_rows: torch.Tensor,  # [B]
    num_nodes: int,
    node_stride: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    scale: Optional[float] = None,
    ancestors: Optional[torch.Tensor] = None,  # [B, *] int64 bits; None = chain
    max_depth: Optional[int] = None,
    block_v: Optional[int] = None,
    num_warps: Optional[int] = None,
    block_k: int = 64,
    precision: str = "tf32",
    save_offset: int = 0,
    seed_node: Optional[int] = None,
) -> torch.Tensor:
    """``max_depth`` bounds the deepest node (root = 0); defaults to Q - 1."""
    tokens = q.shape[0]
    batch = tokens // num_nodes
    num_v_heads = state.shape[1]
    assert state.dtype == torch.float32 and state.stride(-1) == 1
    assert q.stride(1) == k.stride(1) == v.stride(1) == 1
    if scale is None:
        scale = head_k_dim**-0.5
    out = torch.empty((tokens, num_v_heads, head_v_dim), dtype=v.dtype, device=v.device)
    is_tree = ancestors is not None
    qp = max(16, triton.next_power_of_2(num_nodes))
    if max_depth is None or max_depth > num_nodes - 1:
        max_depth = num_nodes - 1
    max_depth = max(max_depth, 1)
    if block_v is None:
        block_v = 64
    if num_warps is None:
        # Measured on H200 (HV 48, K=V 128, 8-16 nodes, TF32 like ReplaySSM).
        num_warps = 4 if batch <= 8 else 2
    _uno_tree_gdn_verify_kernel[(batch, num_v_heads, triton.cdiv(head_v_dim, block_v))](
        q, k, v, a, b,
        q.stride(0), k.stride(0), v.stride(0), a.stride(0), b.stride(0),
        A_log, dt_bias,
        state, state.stride(0),
        cache_indices,
        out,
        saved_k, saved_v, saved_g, saved_beta,
        save_rows,
        ancestors if is_tree else cache_indices,
        scale,
        SAVE_OFFSET=save_offset,
        SEED_NODE=seed_node if seed_node is not None else 0,
        HAS_SEED=seed_node is not None,
        Q=num_nodes,
        QP=qp,
        NODE_STRIDE=node_stride,
        ANC_STRIDE=ancestors.stride(0) if is_tree else 0,
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=block_k,
        BV=block_v,
        IS_TREE=is_tree,
        PRECISION=precision,
        LOG_DEPTH=max(1, max_depth.bit_length()),
        num_warps=num_warps,
        num_stages=2,
    )
    return out


@triton.jit
def _uno_tree_gdn_commit_kernel(
    state,  # [layers, slots, HV, V, K] fp32
    stride_state_layer,
    stride_state_slot,
    saved_k, saved_v, saved_g, saved_beta,  # [layers, rows * NODE_STRIDE, ...]
    stride_sk_layer, stride_sv_layer, stride_sg_layer,
    state_indices,  # [B]
    save_rows,  # [B]
    path,  # [B, P] accepted nodes in order, root first
    stride_path,
    path_lens,  # [B]
    SEED_NODE: tl.constexpr,
    HAS_SEED: tl.constexpr,
    NODE_STRIDE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """Advance each slot along its accepted path (recurrent update order)."""
    i_b = tl.program_id(0).to(tl.int64)
    i_lh = tl.program_id(1)
    i_v = tl.program_id(2)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    slot = tl.load(state_indices + i_b).to(tl.int64)
    save_row = tl.load(save_rows + i_b).to(tl.int64)
    n = tl.load(path_lens + i_b)
    if (slot >= 0) & (save_row >= 0) & ((n > 0) | HAS_SEED):
        o_k = tl.arange(0, K)
        o_v = i_v * BV + tl.arange(0, BV)
        p_h = (state + layer * stride_state_layer + slot * stride_state_slot
               + i_hv * K * V + o_v[None, :] * K + o_k[:, None])
        h = tl.load(p_h)
        if HAS_SEED:
            n_steps = n + 1
        else:
            n_steps = n
        for s in range(0, n_steps):
            if HAS_SEED:
                node = tl.load(path + i_b * stride_path + tl.maximum(s - 1, 0)).to(tl.int64)
                node = tl.where(s == 0, SEED_NODE, node)
            else:
                node = tl.load(path + i_b * stride_path + s).to(tl.int64)
            r = save_row * NODE_STRIDE + node
            b_k = tl.load(saved_k + layer * stride_sk_layer + r * (H * K) + i_h * K + o_k).to(tl.float32)
            b_v = tl.load(saved_v + layer * stride_sv_layer + r * (HV * V) + i_hv * V + o_v).to(tl.float32)
            b_g = tl.load(saved_g + layer * stride_sg_layer + r * HV + i_hv)
            b_beta = tl.load(saved_beta + layer * stride_sg_layer + r * HV + i_hv)
            b_k = b_k * (1.0 / tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
            h *= tl.exp(b_g)
            b_v -= tl.sum(h * b_k[:, None], 0)
            b_v *= b_beta
            h += b_k[:, None] * b_v[None, :]
        tl.store(p_h, h)


def uno_tree_gdn_commit(
    state: torch.Tensor,  # [layers, slots, HV, V, K] fp32
    saved_k: torch.Tensor,  # [layers, rows * node_stride, H * K]
    saved_v: torch.Tensor,
    saved_g: torch.Tensor,
    saved_beta: torch.Tensor,
    state_indices: torch.Tensor,  # [B]
    save_rows: torch.Tensor,  # [B]
    path: torch.Tensor,  # [B, P] local node ids (root first)
    path_lens: torch.Tensor,  # [B]
    node_stride: int,
    num_k_heads: int,
    seed_node: Optional[int] = None,
) -> None:
    """``seed_node``: replay that saved node before the path (merged seed)."""
    layers, _, num_v_heads, head_v_dim, head_k_dim = state.shape
    batch = state_indices.shape[0]
    if batch == 0:
        return
    assert state.stride(-1) == 1 and state.stride(2) == head_v_dim * head_k_dim
    assert saved_g.stride() == saved_beta.stride()
    block_v = 32
    _uno_tree_gdn_commit_kernel[(batch, layers * num_v_heads, head_v_dim // block_v)](
        state, state.stride(0), state.stride(1),
        saved_k, saved_v, saved_g, saved_beta,
        saved_k.stride(0), saved_v.stride(0), saved_g.stride(0),
        state_indices, save_rows,
        path, path.stride(0), path_lens,
        SEED_NODE=seed_node if seed_node is not None else 0,
        HAS_SEED=seed_node is not None,
        NODE_STRIDE=node_stride,
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BV=block_v,
        num_warps=4,
    )
