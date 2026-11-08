"""Env-gated acceptance diagnostics (UNO_ACCEPT_DEBUG=<json path>): per draft position, accumulate
sum_x min(p, q) (expected acceptance of that row), |supp q|, |supp p|, q mass, and the top_p values seen."""
import json
import os

import torch

PATH = os.environ.get("UNO_ACCEPT_DEBUG")
_acc = {}
_calls = 0


def enabled():
    return bool(PATH)


def _add(key, vals):
    vals = vals.double().cpu()
    cur = _acc.setdefault(key, [[0.0] * vals.shape[-1], 0])
    s = vals.sum(0).tolist()
    cur[0] = [a + b for a, b in zip(cur[0], s)]
    cur[1] += vals.shape[0]


def record(p, q, top_ps=None, tag="verify"):
    """p, q: [B, S, V] target / draft distributions of the S draft rows."""
    global _calls
    if not PATH:
        return
    with torch.no_grad():
        p = p.float(); q = q.float()
        _add(f"{tag}.summin", torch.minimum(p, q).sum(-1))
        _add(f"{tag}.supp_q", (q > 0).sum(-1))
        _add(f"{tag}.supp_p", (p > 0).sum(-1))
        _add(f"{tag}.mass_q", q.sum(-1))
        _add(f"{tag}.mass_p", p.sum(-1))
        _add(f"{tag}.p_on_suppq", (p * (q > 0)).sum(-1))
        if top_ps is not None:
            _add(f"{tag}.top_p", top_ps.reshape(-1, 1).float())
    _calls += 1
    if _calls % 50 == 0:
        dump()


def dump():
    out = {k: {"mean": [x / n for x in s], "rows": n} for k, (s, n) in _acc.items()}
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"calls": _calls, "stats": out}, f)
    os.replace(tmp, PATH)


def record_realized(r, reached, accepted, tag):
    """r: [B, S] min(1, p(x)/q(x)) of the drafted token; reached/accepted: [B, S] 0/1 (row reached / accepted)."""
    if not PATH:
        return
    with torch.no_grad():
        reached = reached.float(); accepted = accepted.float(); r = r.float()
        _add(f"{tag}.reached", reached)
        _add(f"{tag}.r_on_reached", r * reached)
        _add(f"{tag}.acc_on_reached", accepted * reached)
        _add(f"{tag}.q_x_zero", (r != r).float())  # NaN marker (q(x)=0 & p(x)=0)
