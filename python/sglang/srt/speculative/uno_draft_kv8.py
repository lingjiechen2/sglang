"""FP8 (e4m3) shadow KV for the UNO draft pass (opt-in, ``UNO_DRAFT_KV8=1``).

The draft's full-attention layers read an fp8 copy of the KV cache (half the bytes);
the verify keeps reading the bf16 cache, so outputs stay exact and only the proposals
change. The shadow is indexed by KV slot like the main pool and kept current without
touching the pool's write paths: prompts are converted eagerly the first time a
request reaches the draft (``prepare``); inside the draft forward every request's last
32 committed positions are re-converted (a cycle commits at most F + 1 < 32 tokens) and
the pass's own rows are written next to the bf16 ones. Scales are 1 (descale 1); the
query is cast to fp8 as in sglang's fp8-KV FA3 path. Needs page_size 1.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

FP8 = torch.float8_e4m3fn
REFRESH = 32


class UnoDraftKV8:
    def __init__(self, *, layer_ids, num_slots, kv_heads, head_dim, num_req_slots, device):
        self.layer_index = {lid: i for i, lid in enumerate(layer_ids)}
        shape = (len(layer_ids), num_slots, 1, kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=FP8, device=device)
        self.v = torch.zeros(shape, dtype=FP8, device=device)
        self.sig = [None] * num_req_slots
        logger.info("UNO draft fp8 shadow KV: %.2f GB", 2 * self.k.numel() / 2**30)

    @torch.no_grad()
    def prepare(self, token_to_kv_pool, req_to_token, reqs):
        """Eager, host-only bookkeeping: convert the prompt KV of newly admitted requests.
        ``reqs`` = [(request slot, request id, prompt length)]; no GPU sync."""
        for slot, rid, plen in reqs:
            if self.sig[slot] == rid or plen <= 0:
                continue
            self.sig[slot] = rid
            slots = req_to_token[slot, :plen].long()
            for lid, li in self.layer_index.items():
                k = token_to_kv_pool.get_key_buffer(lid)
                v = token_to_kv_pool.get_value_buffer(lid)
                self.k[li].view(-1, *self.k.shape[-2:])[slots] = k.view(-1, *k.shape[-2:])[slots].to(FP8)
                self.v[li].view(-1, *self.v.shape[-2:])[slots] = v.view(-1, *v.shape[-2:])[slots].to(FP8)

    def attend(self, fa_call, q, k_new, v_new, out_cache_loc, key_cache, value_cache, page_table,
               cache_seqlens, cu_seqlens_q, cu_seqlens_k, num_rows, layer, scale):
        li = self.layer_index[layer.layer_id]
        kvh = layer.tp_k_head_num
        sk = self.k[li].view(-1, kvh, self.k.shape[-1])
        sv = self.v[li].view(-1, kvh, self.v.shape[-1])
        kc = key_cache.view(-1, kvh, key_cache.shape[-1])
        vc = value_cache.view(-1, kvh, value_cache.shape[-1])
        dev = q.device
        # 1. committed tokens since the last pass: the last REFRESH positions < prefix
        prefix = cache_seqlens.to(torch.long) - num_rows
        pos = (prefix[:, None] - REFRESH + torch.arange(REFRESH, device=dev)[None, :]).clamp(
            0, page_table.shape[1] - 1
        )
        ref = page_table.gather(1, pos).reshape(-1).long()
        sk[ref] = kc[ref].to(FP8)
        sv[ref] = vc[ref].to(FP8)
        # 2. this pass's own rows
        loc = out_cache_loc.long()
        sk[loc] = k_new.view(-1, kvh, sk.shape[-1]).to(FP8)
        sv[loc] = v_new.view(-1, kvh, sv.shape[-1]).to(FP8)
        # 3. fp8 attention over the shadow (descale 1)
        ones = torch.ones(page_table.shape[0], kvh, dtype=torch.float32, device=dev)
        res = fa_call(
            q=q.to(FP8),
            k_cache=self.k[li],
            v_cache=self.v[li],
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=num_rows,
            softmax_scale=scale,
            causal=True,
            k_descale=ones,
            v_descale=ones,
        )
        return res[0] if isinstance(res, tuple) else res


def make_kv8(model_runner):
    from sglang.srt.runtime_context import get_parallel

    if os.environ.get("UNO_DRAFT_KV8", "0") != "1":
        return None
    cfg = model_runner.model_config
    hf = getattr(cfg.hf_config, "text_config", cfg.hf_config)
    pool = model_runner.token_to_kv_pool
    mapping = getattr(pool, "full_attention_layer_id_mapping", None)
    if mapping is not None:  # hybrid (GDN + full attention) pool
        layer_ids = sorted(mapping)
    else:
        types = getattr(hf, "layer_types", None)
        layer_ids = [i for i, t in enumerate(types) if t == "full_attention"] if types else list(range(cfg.num_hidden_layers))
    return UnoDraftKV8(
        layer_ids=layer_ids,
        # Exact slot count of the full-attention KV buffers (a hybrid pool's size
        # does not count them).
        num_slots=pool.get_key_buffer(layer_ids[0]).shape[0],
        kv_heads=cfg.get_num_kv_heads(get_parallel().tp_size),
        head_dim=cfg.head_dim,
        num_req_slots=model_runner.req_to_token_pool.size + 1,
        device=model_runner.device,
    )
