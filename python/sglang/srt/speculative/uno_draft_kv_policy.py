"""Research path: KV-cache compression policies applied to the UNO draft pass only.

The draft's full-attention layers attend over a policy-selected (or fake-quantized)
view of the cached prefix; the verify keeps the full cache, so outputs stay exact and
only the proposals (acceptance length) change. Attention is computed in PyTorch, per
request, so this is for measuring acceptance, not speed: run with CUDA graphs off.

Policies (prefix budget B tokens, ``UNO_DRAFT_KV_BUDGET``):
  full       every prefix key (checks the path itself)
  window     the last B prefix keys
  streaming  4 sink keys + the last B - 4 (StreamingLLM)
  quest      per KV head, the top pages (16 tokens) by the query-aware upper bound
             sum_d max(q_d * kmin_d, q_d * kmax_d) of the current draft queries, plus
             the last 64 keys (Quest)
  snapkv     per KV head, the top keys by attention of the first draft pass of a
             request (avg-pooled, kernel 7), frozen; later keys all kept (SnapKV,
             with the first draft queries standing in for the prompt's observation
             window)
  h2o        per KV head, B / 2 heavy hitters by accumulated attention plus the last
             B / 2 keys, evicted step by step (H2O)
  kivi       every key, 2-bit fake quantization (keys per channel over 32-token
             groups, values per token over 32-channel groups) except the last 128
             (KIVI)
"""

from __future__ import annotations

import math
import os

import torch

_SINKS = 4
_QUEST_PAGE = 16
_QUEST_RECENT = 64
_SNAP_KERNEL = 7
_SNAP_RECENT = 64
_KIVI_GROUP = 32
_KIVI_RESIDUAL = 128
_KIVI_BITS = 2


def make_policy():
    name = os.environ.get("UNO_DRAFT_KV_POLICY", "")
    if not name:
        return None
    budget = int(os.environ.get("UNO_DRAFT_KV_BUDGET", "2048"))
    return UnoDraftKVPolicy(name, budget)


def _fake_quant(x: torch.Tensor, dim: int, bits: int) -> torch.Tensor:
    """Asymmetric min/max fake quantization reducing over ``dim``."""
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    levels = 2**bits - 1
    scale = (hi - lo).clamp_min(1e-6) / levels
    return ((x - lo) / scale).round().clamp(0, levels) * scale + lo


class UnoDraftKVPolicy:
    def __init__(self, name: str, budget: int):
        if name not in ("full", "window", "streaming", "quest", "snapkv", "h2o", "kivi"):
            raise ValueError(f"Unknown UNO_DRAFT_KV_POLICY {name!r}")
        self.name = name
        self.budget = budget
        # (request slot, layer) -> per-request state; keyed by the request's first KV
        # slot too so a reused request slot starts fresh.
        self.state: dict = {}

    # ------------------------------------------------------------------ selection
    def _visible(self, key, sid, layer_id, q, k, prefix):
        """Boolean [kvh, prefix] mask of the prefix keys the draft may read."""
        kvh = k.shape[1]
        dev = k.device
        mask = torch.zeros(kvh, prefix, dtype=torch.bool, device=dev)
        if prefix <= self.budget or self.name in ("full", "kivi"):
            mask[:] = True
            if self.name in ("snapkv", "h2o"):
                self.state.pop(key, None)
            return mask
        b = self.budget
        if self.name == "window":
            mask[:, prefix - b :] = True
        elif self.name == "streaming":
            mask[:, :_SINKS] = True
            mask[:, prefix - (b - _SINKS) :] = True
        elif self.name == "quest":
            pages = math.ceil(prefix / _QUEST_PAGE)
            pad = pages * _QUEST_PAGE - prefix
            kp = torch.nn.functional.pad(k[:prefix].float(), (0, 0, 0, 0, 0, pad))
            kp = kp.view(pages, _QUEST_PAGE, kvh, -1)
            fill = torch.zeros(pages * _QUEST_PAGE, dtype=torch.bool, device=dev)
            fill[prefix:] = True
            fill = fill.view(pages, _QUEST_PAGE)[:, :, None, None]
            kmin = kp.masked_fill(fill, float("inf")).amin(1)  # [pages, kvh, d]
            kmax = kp.masked_fill(fill, float("-inf")).amax(1)
            g = q.shape[1] // kvh
            qq = q.float().view(q.shape[0], kvh, g, -1)  # [F, kvh, g, d]
            ub = torch.maximum(
                torch.einsum("fhgd,phd->hfgp", qq, kmin), torch.einsum("fhgd,phd->hfgp", qq, kmax)
            ).amax(dim=(1, 2))  # [kvh, pages]
            keep_pages = max(1, (b - _QUEST_RECENT) // _QUEST_PAGE)
            top = ub.topk(min(keep_pages, pages), dim=1).indices
            page_mask = torch.zeros(kvh, pages, dtype=torch.bool, device=dev)
            page_mask.scatter_(1, top, True)
            mask = page_mask.repeat_interleave(_QUEST_PAGE, dim=1)[:, :prefix]
            mask[:, prefix - _QUEST_RECENT :] = True
        elif self.name == "snapkv":
            st = self.state.get(key)
            if st is None or st["sid"] != sid or st["p0"] > prefix:
                probs = self._probs(q, k[:prefix], kvh)  # [kvh, prefix]
                pooled = torch.nn.functional.avg_pool1d(
                    probs[:, None, :], _SNAP_KERNEL, stride=1, padding=_SNAP_KERNEL // 2
                )[:, 0, :prefix]
                sel = torch.zeros(kvh, prefix, dtype=torch.bool, device=dev)
                keep = max(1, b - _SNAP_RECENT)
                sel.scatter_(1, pooled[:, : prefix - _SNAP_RECENT].topk(min(keep, prefix - _SNAP_RECENT), dim=1).indices, True)
                sel[:, prefix - _SNAP_RECENT :] = True
                st = {"sid": sid, "p0": prefix, "sel": sel}
                self.state[key] = st
            mask[:, : st["p0"]] = st["sel"]
            mask[:, st["p0"] :] = True
        elif self.name == "h2o":
            st = self.state.get(key)
            recent = b // 2
            heavy = b - recent
            if st is None or st["sid"] != sid:
                acc = self._probs(q, k[:prefix], kvh)  # bootstrap: full attention once
                st = {"sid": sid, "acc": acc, "evicted": torch.zeros_like(acc, dtype=torch.bool)}
                self.state[key] = st
            acc, evicted = st["acc"], st["evicted"]
            if acc.shape[1] < prefix:  # newly committed keys
                grow = prefix - acc.shape[1]
                acc = torch.cat((acc, acc.new_zeros(kvh, grow)), 1)
                evicted = torch.cat((evicted, evicted.new_zeros(kvh, grow)), 1)
            acc, evicted = acc[:, :prefix], evicted[:, :prefix]
            older = prefix - recent
            score = acc[:, :older].masked_fill(evicted[:, :older], float("-inf"))
            keep = score.topk(min(heavy, older), dim=1).indices
            mask[:, prefix - recent :] = True
            mask[:, :older].scatter_(1, keep, True)
            evicted = evicted | ~mask  # eviction is permanent
            st["acc"], st["evicted"] = acc, evicted
        return mask

    @staticmethod
    def _probs(q, k, kvh):
        """Mean attention probability per KV head over the draft queries: [kvh, n]."""
        g = q.shape[1] // kvh
        qq = q.float().view(q.shape[0], kvh, g, -1)
        s = torch.einsum("fhgd,nhd->hfgn", qq, k.float()) / math.sqrt(q.shape[-1])
        return s.softmax(-1).mean(dim=(1, 2))

    # ------------------------------------------------------------------ attention
    def attend(self, q, key_cache, value_cache, page_table, cache_seqlens, cu_seqlens_q, layer, req_pool_indices, scale):
        """q [tokens, H, d] -> output [tokens, H, dv] over the policy's prefix view."""
        kvh = layer.tp_k_head_num
        if key_cache.dim() == 4 and key_cache.shape[1] != 1:
            raise ValueError("UNO draft KV policies index the page table as token slots (page_size 1)")
        kc = key_cache.view(-1, kvh, key_cache.shape[-1])
        vc = value_cache.view(-1, kvh, value_cache.shape[-1])
        out = torch.empty(q.shape[0], q.shape[1], vc.shape[-1], dtype=q.dtype, device=q.device)
        lens = cache_seqlens.tolist()
        cu = cu_seqlens_q.tolist()
        rows = req_pool_indices.tolist()
        for b in range(len(lens)):
            n, qs, qe = lens[b], cu[b], cu[b + 1]
            if qe <= qs:
                continue
            f = qe - qs
            prefix = n - f
            slots = page_table[b, :n].long()
            k, v = kc[slots], vc[slots]
            qb = q[qs:qe]
            key = (rows[b], layer.layer_id)
            vis = self._visible(key, int(slots[0]), layer.layer_id, qb, k, prefix)
            if self.name == "kivi" and prefix > _KIVI_RESIDUAL:
                cut = prefix - _KIVI_RESIDUAL
                cut -= cut % _KIVI_GROUP
                if cut > 0:
                    kq = k[:cut].float().view(cut // _KIVI_GROUP, _KIVI_GROUP, kvh, -1)
                    kq = _fake_quant(kq, dim=1, bits=_KIVI_BITS).view(cut, kvh, -1)
                    d = v.shape[-1]
                    vq = v[:cut].float().view(cut, kvh, d // _KIVI_GROUP, _KIVI_GROUP)
                    vq = _fake_quant(vq, dim=3, bits=_KIVI_BITS).view(cut, kvh, d)
                    k = torch.cat((kq.to(k.dtype), k[cut:]))
                    v = torch.cat((vq.to(v.dtype), v[cut:]))
            g = qb.shape[1] // kvh
            qq = qb.float().view(f, kvh, g, -1)
            s = torch.einsum("fhgd,nhd->hfgn", qq, k.float()) * scale  # [kvh, f, g, n]
            allow = torch.ones(kvh, f, n, dtype=torch.bool, device=q.device)
            allow[:, :, :prefix] = vis[:, None, :]
            causal = torch.arange(n, device=q.device)[None, :] <= (prefix + torch.arange(f, device=q.device))[:, None]
            allow &= causal[None]
            s = s.masked_fill(~allow[:, :, None, :], float("-inf"))
            p = s.softmax(-1)
            if self.name == "h2o":
                st = self.state.get(key)
                if st is not None:
                    st["acc"][:, :prefix] += p[..., :prefix].mean(dim=(1, 2))
            o = torch.einsum("hfgn,nhd->fhgd", p, v.float())
            out[qs:qe] = o.reshape(f, kvh * g, -1).to(out.dtype)
        return out
