"""Quest-style query-aware sparse attention for the UNO draft pass (opt-in).

The draft's full-attention layers read, per KV head, only the Quest-selected pages of
the cached prefix plus the most recent keys; the verify keeps the full cache, so outputs
stay exact and only the proposals change.

Per (full-attention layer, request slot, page of 16 positions, KV head) the key-wise
minimum and maximum are kept in bf16 (``kmin`` / ``kmax``). Committed keys never change,
so a page is built once: eagerly for the prefill (``prepare``) and, inside the draft
forward, by recomputing the last two pages of every request (a cycle accepts at most
F + 1 < 16 tokens). Selection per KV head: upper bound ``q+ . kmax + q- . kmin`` maxed
over the draft rows of the head's query group, top ``(budget - recent) / 16`` pages
among the pages entirely before the recent window. Attention: one FA3 call per KV head
on a compact page table [selected pages | last ``recent`` keys | this pass's rows]
(bottom-right causal). Prefixes not longer than ``budget`` read the whole cache. Every
shape is static, so the draft CUDA graphs capture it. Needs page_size 1.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

PAGE = 16


@triton.jit
def _quest_score_kernel(
    q_ptr,  # [B, KVH, R, D] bf16 (contiguous)
    kmin_ptr,  # layer view [slots, pages, KVH, D] bf16
    kmax_ptr,
    slot_ptr,  # [B] int64
    valid_ptr,  # [B] int64: pages entirely before the recent window
    out_ptr,  # [B, KVH, pages] fp32
    stride_ms, stride_mp, stride_mh,
    pages, KVH: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    BR: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr,
    MINMAX: tl.constexpr = False,
):
    bh = tl.program_id(0)
    pb = tl.program_id(1)
    b = bh // KVH
    h = bh % KVH
    p = pb * BP + tl.arange(0, BP)
    valid = tl.load(valid_ptr + b)
    out = out_ptr + bh * pages + p
    if pb * BP >= valid:
        tl.store(out, tl.full([BP], float("-inf"), tl.float32), mask=p < pages)
        return
    slot = tl.load(slot_ptr + b)
    r = tl.arange(0, BR)
    acc = tl.zeros([BR, BP], dtype=tl.float32)
    acc2 = tl.zeros([BR, BP], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        qt = tl.load(q_ptr + ((b * KVH + h) * R + r[:, None]) * D + d[None, :], mask=r[:, None] < R, other=0.0)
        koff = slot * stride_ms + p[None, :] * stride_mp + h * stride_mh + d[:, None]
        pm = p[None, :] < valid
        kmx = tl.load(kmax_ptr + koff, mask=pm, other=0.0)
        kmn = tl.load(kmin_ptr + koff, mask=pm, other=0.0)
        if MINMAX:
            acc += tl.dot(qt, kmx)
            acc2 += tl.dot(qt, kmn)
        else:
            qp = tl.where(qt > 0, qt, 0.0).to(kmx.dtype)
            qn = tl.where(qt < 0, qt, 0.0).to(kmn.dtype)
            acc += tl.dot(qp, kmx) + tl.dot(qn, kmn)
    if MINMAX:
        acc = tl.maximum(acc, acc2)
    acc = tl.where(r[:, None] < R, acc, float("-inf"))
    ub = tl.max(acc, axis=0)
    ub = tl.where(p < valid, ub, float("-inf"))
    tl.store(out, ub, mask=p < pages)


class UnoDraftQuest:
    def __init__(self, *, layer_ids, num_slots, max_context, kv_heads, head_dim, device, budget, recent=64):
        if (budget - recent) < PAGE or (budget - recent) % PAGE:
            raise ValueError("UNO Quest budget - recent must be a positive multiple of 16")
        self.budget, self.recent = budget, recent
        self.num_pages_sel = (budget - recent) // PAGE
        self.layer_index = {lid: i for i, lid in enumerate(layer_ids)}
        self.max_pages = (max_context + PAGE - 1) // PAGE
        shape = (len(layer_ids), num_slots, self.max_pages, kv_heads, head_dim)
        self.kmin = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.kmax = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.sig = [None] * num_slots  # request id owning each request slot (host side)
        # Exact fp32 scoring in PyTorch (kernel tests); the Triton kernel otherwise.
        self.reference_scores = os.environ.get("UNO_DRAFT_QUEST_REFERENCE", "0") == "1"
        # Debug: compare kept page min/max with the cache every eager pass.
        self.check_meta = os.environ.get("UNO_DRAFT_QUEST_CHECK", "0") == "1"
        # Page score: "bound" = Quest's upper bound sum_d max(q_d kmin_d, q_d kmax_d);
        # "minmax" = max(q . kmin, q . kmax), the page's min and max key vectors as keys.
        self.minmax = os.environ.get("UNO_DRAFT_QUEST_SCORE", "bound") == "minmax"
        self.check_worst = 0.0
        logger.info(
            "UNO draft Quest: budget %d (%d pages + %d recent), metadata %.2f GB",
            budget, self.num_pages_sel, recent, 2 * self.kmin.numel() * 2 / 2**30,
        )

    # ------------------------------------------------------------------ metadata
    @staticmethod
    def _page_minmax(k: torch.Tensor, valid: torch.Tensor):
        """k [..., pages, PAGE, kvh, d], valid [..., pages, PAGE] -> (min, max) [..., pages, kvh, d]."""
        v = valid[..., None, None]
        kf = k.float()
        kmin = kf.masked_fill(~v, float("inf")).amin(-3)
        kmax = kf.masked_fill(~v, float("-inf")).amax(-3)
        empty = ~valid.any(-1)[..., None, None]
        return kmin.masked_fill(empty, 0).to(torch.bfloat16), kmax.masked_fill(empty, 0).to(torch.bfloat16)

    @torch.no_grad()
    def prepare(self, token_to_kv_pool, req_to_token, reqs):
        """Eager, host-only bookkeeping: build the prompt pages of newly admitted
        requests. ``reqs`` = [(request slot, request id, prompt length)]; no GPU sync."""
        for slot, rid, plen in reqs:
            if self.sig[slot] == rid:
                continue  # later pages are completed inside the draft graph
            self.sig[slot] = rid
            # Complete prompt pages; the graph refreshes the last two pages of every
            # pass (a pass commits at most F + 1 < 16 tokens), so decode pages stay current.
            upto = plen // PAGE
            if upto <= 0:
                continue
            pos = torch.arange(upto * PAGE, device=self.kmin.device)
            slots = req_to_token[slot, pos].long()
            for lid, li in self.layer_index.items():
                k = token_to_kv_pool.get_key_buffer(lid)
                k = k.view(k.shape[0], -1, k.shape[-1])[slots].view(upto, PAGE, -1, k.shape[-1])
                valid = torch.ones(upto, PAGE, dtype=torch.bool, device=k.device)
                mn, mx = self._page_minmax(k, valid)
                self.kmin[li, slot, :upto] = mn
                self.kmax[li, slot, :upto] = mx

    def _check_meta(self, li, kc, page_table, prefix, slot):
        """Debug: max |kept - exact| over every complete page of every request."""
        worst = 0.0
        for b in range(page_table.shape[0]):
            p = int(prefix[b])
            pages = p // PAGE
            if pages <= 0:
                continue
            k = kc[page_table[b, : pages * PAGE].long()].float().view(pages, PAGE, kc.shape[1], -1)
            s = int(slot[b])
            err = max(
                (self.kmin[li, s, :pages].float() - k.amin(1).bfloat16().float()).abs().max().item(),
                (self.kmax[li, s, :pages].float() - k.amax(1).bfloat16().float()).abs().max().item(),
            )
            worst = max(worst, err)
        if worst > self.check_worst:
            self.check_worst = worst
            logger.warning("UNO Quest metadata check: new worst |kept - exact| = %.4g", worst)

    # ------------------------------------------------------------------ attention
    def attend(self, fa_call, q, key_cache, value_cache, page_table, cache_seqlens, cu_seqlens_q,
               num_rows, layer, req_pool_indices, scale):
        li = self.layer_index[layer.layer_id]
        kvh = layer.tp_k_head_num
        heads, d = q.shape[1], q.shape[2]
        g = heads // kvh
        batch = page_table.shape[0]
        dev = q.device
        kc = key_cache.view(-1, kvh, key_cache.shape[-1])
        n = cache_seqlens.to(torch.long)
        prefix = n - num_rows
        slot = req_pool_indices[:batch].to(torch.long)

        # 1. refresh the last two pages of every request (positions < prefix)
        p0 = ((prefix - 1).clamp_min(0) // PAGE - 1).clamp_min(0)  # first of the two pages
        pos = p0[:, None] * PAGE + torch.arange(2 * PAGE, device=dev)[None, :]
        valid = pos < prefix[:, None]
        kslots = page_table.gather(1, pos.clamp(max=page_table.shape[1] - 1)).long()
        k2 = kc[kslots].view(batch, 2, PAGE, kvh, -1)
        mn, mx = self._page_minmax(k2, valid.view(batch, 2, PAGE))
        pages2 = (p0[:, None] + torch.arange(2, device=dev)[None, :]).clamp(max=self.max_pages - 1)
        # Padded graph rows (prefix <= 0) write to the spare last slot.
        dummy = torch.full_like(slot, self.kmin.shape[1] - 1)
        rows = torch.where(prefix > 0, slot, dummy)[:, None].expand(-1, 2)
        self.kmin[li].index_put_((rows.reshape(-1), pages2.reshape(-1)), mn.reshape(batch * 2, kvh, -1))
        self.kmax[li].index_put_((rows.reshape(-1), pages2.reshape(-1)), mx.reshape(batch * 2, kvh, -1))

        if self.check_meta and not torch.cuda.is_current_stream_capturing():
            self._check_meta(li, kc, page_table, prefix, slot)
        # 2. page upper bounds per KV head (pages entirely before the recent window)
        q_rows = torch.zeros(batch * num_rows, heads, d, dtype=q.dtype, device=dev)
        q_rows[: q.shape[0]] = q[: batch * num_rows]
        qg = q_rows.view(batch, num_rows, kvh, g, d).permute(0, 2, 1, 3, 4).reshape(batch, kvh, num_rows * g, d)
        valid = ((prefix - self.recent).clamp_min(0) // PAGE)
        if self.reference_scores:
            kmin = self.kmin[li].index_select(0, slot).float()
            kmax = self.kmax[li].index_select(0, slot).float()
            qf = qg.float()
            if self.minmax:
                ub = torch.maximum(torch.matmul(qf, kmax.permute(0, 2, 3, 1)),
                                   torch.matmul(qf, kmin.permute(0, 2, 3, 1))).amax(2)
            else:
                ub = (torch.matmul(qf.clamp_min(0), kmax.permute(0, 2, 3, 1))
                      + torch.matmul(qf.clamp_max(0), kmin.permute(0, 2, 3, 1))).amax(2)
            page_ids = torch.arange(self.max_pages, device=dev)
            ub = ub.masked_fill(~(page_ids[None, :] < valid[:, None])[:, None, :], float("-inf"))
        else:
            qg = qg.contiguous()
            ub = torch.empty(batch, kvh, self.max_pages, dtype=torch.float32, device=dev)
            kmin_l, kmax_l = self.kmin[li], self.kmax[li]
            rows_q = num_rows * g
            _quest_score_kernel[(batch * kvh, triton.cdiv(self.max_pages, 64))](
                qg, kmin_l, kmax_l, slot, valid, ub,
                kmin_l.stride(0), kmin_l.stride(1), kmin_l.stride(2),
                self.max_pages, KVH=kvh, R=rows_q, D=d,
                BR=triton.next_power_of_2(rows_q), BP=64, BD=64, MINMAX=self.minmax, num_warps=4,
            )
        top = ub.topk(self.num_pages_sel, dim=-1).indices  # [B, kvh, P]

        # 3. compact per-head token indices: [selected pages | recent | rows]
        width = self.num_pages_sel * PAGE + self.recent + num_rows
        sel = (top[..., None] * PAGE + torch.arange(PAGE, device=dev)).reshape(batch, kvh, -1)
        tail = (prefix - self.recent)[:, None] + torch.arange(self.recent + num_rows, device=dev)[None, :]
        idx = torch.cat((sel, tail[:, None, :].expand(-1, kvh, -1)), dim=-1)  # [B, kvh, width]
        short = (prefix <= self.budget)[:, None, None]
        col = torch.arange(width, device=dev)[None, None, :]
        idx = torch.where(short, torch.minimum(col, (n - 1).clamp_min(0)[:, None, None]), idx).clamp_min(0)
        lens = torch.where(prefix <= self.budget, n, torch.full_like(n, width)).to(cache_seqlens.dtype)
        lens = torch.minimum(lens, n.to(lens.dtype))
        cu_k = torch.nn.functional.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))

        # 4. one FA3 call per KV head
        out = torch.empty(q.shape[0], heads, value_cache.shape[-1], dtype=q.dtype, device=dev)
        for h in range(kvh):
            table = page_table.gather(1, idx[:, h]).to(page_table.dtype)
            res = fa_call(
                q=q[:, h * g : (h + 1) * g].contiguous(),
                k_cache=key_cache[..., h : h + 1, :],
                v_cache=value_cache[..., h : h + 1, :],
                page_table=table,
                cache_seqlens=lens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_k,
                max_seqlen_q=num_rows,
                softmax_scale=scale,
                causal=True,
            )
            out[:, h * g : (h + 1) * g] = res[0] if isinstance(res, tuple) else res
        return out


def make_quest(model_runner):
    from sglang.srt.runtime_context import get_parallel

    budget = int(os.environ.get("UNO_DRAFT_QUEST_BUDGET", "0"))
    if budget <= 0:
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
    return UnoDraftQuest(
        layer_ids=layer_ids,
        num_slots=model_runner.req_to_token_pool.size + 1,
        max_context=model_runner.model_config.context_len,
        kv_heads=cfg.get_num_kv_heads(get_parallel().tp_size),
        head_dim=cfg.head_dim,
        device=model_runner.device,
        budget=budget,
        recent=int(os.environ.get("UNO_DRAFT_QUEST_RECENT", "64")),
    )
