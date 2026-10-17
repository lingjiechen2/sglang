"""UNO tree mode on GDN hybrids: per-layer verify and the path commit.

Enabled only by UNO tree mode (``enable_uno_tree_gdn`` attaches the state to
the mamba pool); every other speculative path keeps the generic GDN verify
that stores one recurrent state per node.  See
``sglang.kernels.ops.attention.fla.gdn_uno_tree`` for the kernels.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.fla.gdn_uno_tree import (
    uno_tree_conv,
    uno_tree_gdn_commit,
    uno_tree_gdn_verify,
    uno_tree_structure,
)
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
)
from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled


class UnoTreeGdnState:
    """Raw per-node k, v, g and beta of the last GDN verify, per layer.

    Rows follow the speculative scratch (``intermediate_conv_window``): row r
    node j lives at ``r * node_stride + j``.  Tree verify nodes use
    ``[0, tree_nodes)``; the chain draft pass saves at ``draft_offset`` so its
    seed (row 0) survives the verify: the seed's state is never committed on
    its own but replayed together with the accepted path.
    """

    def __init__(self, spec_state, num_nodes: int, max_depth: int):
        conv = spec_state.conv[0]  # [layers, slots, dim, W - 1]
        window = spec_state.intermediate_conv_window[0]  # [layers, rows, nodes, dim, W - 1]
        temporal = spec_state.temporal  # [layers, slots, HV, V, K]
        layers, rows, tree_nodes = window.shape[0], window.shape[1], window.shape[2]
        if tree_nodes < num_nodes:
            raise ValueError(
                f"UNO tree GDN needs {num_nodes} scratch nodes; the pool has {tree_nodes}."
            )
        self.draft_offset = tree_nodes
        node_stride = tree_nodes + max_depth + 1
        num_v_heads, head_v_dim, head_k_dim = temporal.shape[2:]
        key_dim = (conv.shape[2] - num_v_heads * head_v_dim) // 2
        self.node_stride = node_stride
        self.max_depth = max_depth
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_k_heads = key_dim // head_k_dim
        tokens = rows * node_stride
        device, dtype = conv.device, conv.dtype
        self.saved_k = torch.empty((layers, tokens, key_dim), dtype=dtype, device=device)
        self.saved_v = torch.empty(
            (layers, tokens, num_v_heads * head_v_dim), dtype=dtype, device=device
        )
        self.saved_g = torch.empty((layers, tokens, num_v_heads), dtype=torch.float32, device=device)
        self.saved_beta = torch.empty_like(self.saved_g)
        # Raw conv input of each request's draft seed (draft row 0).
        self.seed_x = torch.empty((layers, rows, conv.shape[2]), dtype=dtype, device=device)
        # Tree structure of the current verify, built by the first GDN layer.
        self.parents = torch.empty((rows, tree_nodes), dtype=torch.int32, device=device)
        self.ancestors = torch.empty(
            (rows, tree_nodes, tree_nodes), dtype=torch.int8, device=device
        )


def enable_uno_tree_gdn(
    req_to_token_pool, num_nodes: int, max_depth: int
) -> UnoTreeGdnState:
    """``max_depth``: deepest tree node below the root (UNO: draft width - 1)."""
    mamba_pool = req_to_token_pool.mamba_pool
    state = UnoTreeGdnState(
        req_to_token_pool.get_speculative_mamba2_params_all_layers(),
        num_nodes,
        max_depth,
    )
    mamba_pool.uno_tree_state = state
    return state


def get_uno_tree_gdn(req_to_token_pool) -> Optional[UnoTreeGdnState]:
    mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
    return getattr(mamba_pool, "uno_tree_state", None)


def uno_tree_gdn_target_verify(
    *,
    state: UnoTreeGdnState,
    layer,
    layer_index: int,
    mixed_qkv: torch.Tensor,  # [B * Q, qkv_dim]
    a: torch.Tensor,
    b: torch.Tensor,
    layer_cache,
    cache_indices: torch.Tensor,  # [B]
    scratch_rows: torch.Tensor,  # [B]
    num_nodes: int,
    retrieve_next_token: Optional[torch.Tensor],
    retrieve_next_sibling: Optional[torch.Tensor],
) -> torch.Tensor:
    """One GDN layer of a UNO tree (or chain) verify; returns [1, B*Q, HV, V]."""
    batch = cache_indices.shape[0]
    parents = ancestors = None
    if retrieve_next_token is not None:
        parents, ancestors = state.parents[:batch], state.ancestors[:batch]
        if layer_index == 0:
            # Same tree in every layer of this forward.
            uno_tree_structure(
                retrieve_next_token,
                retrieve_next_sibling,
                parents,
                ancestors,
                num_nodes,
                min(state.max_depth, num_nodes - 1),
            )
    conv_out = uno_tree_conv(
        mixed_qkv,
        layer_cache.conv[0],
        layer.conv_weights,
        layer.bias,
        layer.activation,
        cache_indices,
        layer_cache.intermediate_conv_window[0],
        scratch_rows,
        num_nodes,
        parents,
        seed_x=state.seed_x[layer_index],
        save_seed=parents is None,
        has_seed=parents is not None,
    )
    q = conv_out[:, : layer.q_dim]
    k = conv_out[:, layer.q_dim : layer.q_dim + layer.k_dim]
    v = conv_out[:, layer.q_dim + layer.k_dim :]
    out = uno_tree_gdn_verify(
        q, k, v,
        a.reshape(a.shape[0], -1),
        b.reshape(b.shape[0], -1),
        layer.A_log,
        layer.dt_bias,
        layer_cache.temporal,
        cache_indices,
        state.saved_k[layer_index],
        state.saved_v[layer_index],
        state.saved_g[layer_index],
        state.saved_beta[layer_index],
        scratch_rows,
        num_nodes=num_nodes,
        node_stride=state.node_stride,
        num_k_heads=state.num_k_heads,
        head_k_dim=state.head_k_dim,
        head_v_dim=state.head_v_dim,
        ancestors=ancestors,
        max_depth=min(state.max_depth, num_nodes - 1),
        save_offset=0 if parents is not None else state.draft_offset,
        seed_node=state.draft_offset if parents is not None else None,
    )
    return out.unsqueeze(0)


def _scratch_rows(req_pool_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    # Matches select_verify_intermediate_state_indices for live requests.
    if pp_spec_stable_rows_enabled():
        return req_pool_indices[:batch_size].to(torch.int32)
    return torch.arange(batch_size, dtype=torch.int32, device=req_pool_indices.device)


def commit_uno_tree_path(
    req_to_token_pool,
    req_pool_indices: torch.Tensor,
    path: torch.Tensor,  # [B, P] local node ids, root first, -1 padded
    path_lens: torch.Tensor,  # [B]
    with_seed: bool = False,
) -> None:
    """Advance every request's GDN state and conv window along its path
    (after the uncommitted draft seed when ``with_seed``)."""
    state = get_uno_tree_gdn(req_to_token_pool)
    batch_size = path_lens.shape[0]
    if batch_size == 0:
        return
    spec_state = req_to_token_pool.get_speculative_mamba2_params_all_layers()
    state_indices = req_to_token_pool.get_mamba_indices(req_pool_indices[:batch_size])
    rows = _scratch_rows(req_pool_indices, batch_size)
    path = path.to(torch.int32)
    path_lens = path_lens.to(torch.int32)
    uno_tree_gdn_commit(
        spec_state.temporal,
        state.saved_k,
        state.saved_v,
        state.saved_g,
        state.saved_beta,
        state_indices,
        rows,
        path,
        path_lens,
        node_stride=state.node_stride,
        num_k_heads=state.num_k_heads,
        seed_node=state.draft_offset if with_seed else None,
    )
    last_nodes = path.gather(1, (path_lens.clamp_min(1) - 1).to(torch.int64)[:, None])[:, 0]
    last_nodes = torch.where(path_lens > 0, last_nodes, -1)
    src_rows = None if not pp_spec_stable_rows_enabled() else rows
    for conv, window in zip(spec_state.conv, spec_state.intermediate_conv_window):
        fused_conv_window_scatter_with_mask(
            conv, window, state_indices, last_nodes, src_rows
        )


def commit_uno_tree_after_verify(
    req_to_token_pool,
    req_pool_indices: torch.Tensor,
    accept_index: torch.Tensor,  # [B, P] flat node indices, -1 padded
    accept_lens: torch.Tensor,  # [B]
    num_nodes: int,
) -> None:
    batch_size = accept_lens.shape[0]
    offsets = torch.arange(batch_size, device=accept_index.device)[:, None] * num_nodes
    path = torch.where(accept_index >= 0, accept_index - offsets, -1)
    commit_uno_tree_path(
        req_to_token_pool, req_pool_indices, path, accept_lens, with_seed=True
    )
