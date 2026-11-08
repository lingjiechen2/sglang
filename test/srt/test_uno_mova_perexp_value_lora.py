"""Per-expert value LoRA (UnoMovaPerExpertValueLoRA) against a per-row reference and against the shared version, on CPU."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[2] / "python/sglang/srt/speculative/uno_mova_lora.py"
spec = importlib.util.spec_from_file_location("uno_mova_lora_under_test", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

E, WIDTH, HIDDEN, R, K = 6, 8, 16, 4, 2
BATCH, FWIDTH = 3, 5
SCALING = 2.0


def backend(active=True):
    route = SimpleNamespace(total_rows=BATCH * FWIDTH, batch_size=BATCH, forward_width=FWIDTH) if active else None
    return SimpleNamespace(batch_info=SimpleNamespace(uno_single_adapter_route=route))


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    n = BATCH * FWIDTH
    hidden = torch.randn(n, HIDDEN, generator=g, dtype=torch.double)
    weights = torch.randn(E, WIDTH, HIDDEN, generator=g, dtype=torch.double)
    selected = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
    routing = torch.softmax(torch.randn(n, K, generator=g, dtype=torch.double), dim=-1)
    values = torch.randn(n, WIDTH, generator=g, dtype=torch.double)
    a = torch.randn(E, R, HIDDEN, generator=g, dtype=torch.double)
    b = torch.randn(E, WIDTH, R, generator=g, dtype=torch.double)
    return hidden, weights, selected, routing, values, a, b


def reference(hidden, weights, selected, routing, values, a, b):
    out = values.clone()
    for block in range(BATCH):
        for j in range(1, FWIDTH):  # draft rows: every row of a block except the seed row
            i = block * FWIDTH + j
            acc = torch.zeros(WIDTH, dtype=hidden.dtype)
            for k in range(K):
                e = int(selected[i, k])
                proj = weights[e] @ hidden[i] + SCALING * (b[e] @ (a[e] @ hidden[i]))
                acc += F.silu(proj) * routing[i, k]
            out[i] = acc
    return out


def test_matches_reference_and_leaves_seed_rows_alone():
    hidden, weights, selected, routing, values, a, b = inputs()
    lora = mod.UnoMovaPerExpertValueLoRA(a, b, SCALING, backend())
    out = lora(hidden, routing, selected, weights, values)
    ref = reference(hidden, weights, selected, routing, values, a, b)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    for block in range(BATCH):  # seed rows keep the base values
        torch.testing.assert_close(out[block * FWIDTH], values[block * FWIDTH], rtol=0, atol=0)
    assert (out - values).abs().max() > 1e-3


def test_equal_experts_match_the_shared_version():
    hidden, weights, selected, routing, values, a, b = inputs(1)
    a1, b1 = a[:1].expand(E, -1, -1).contiguous(), b[:1].expand(E, -1, -1).contiguous()
    per_expert = mod.UnoMovaPerExpertValueLoRA(a1, b1, SCALING, backend())
    shared = mod.UnoMovaValueLoRA(a[0], b[0], SCALING, backend())
    torch.testing.assert_close(
        per_expert(hidden, routing, selected, weights, values),
        shared(hidden, routing, selected, weights, values), rtol=1e-12, atol=1e-12)


def test_other_experts_do_not_leak():
    hidden, weights, selected, routing, values, a, b = inputs(2)
    base = mod.UnoMovaPerExpertValueLoRA(a, b, SCALING, backend())(hidden, routing, selected, weights, values)
    b2 = b.clone()
    unused = [e for e in range(E) if not (selected == e).any()]
    target = unused[0] if unused else 0
    b2[target] += 1.0
    changed = mod.UnoMovaPerExpertValueLoRA(a, b2, SCALING, backend())(hidden, routing, selected, weights, values)
    uses = (selected == target).any(dim=1)
    uses = uses & torch.tensor([i % FWIDTH != 0 for i in range(BATCH * FWIDTH)])
    diff = (changed - base).abs().amax(dim=1) > 1e-9
    assert torch.equal(diff, uses), (diff, uses)


def test_outside_a_uno_draft_forward_values_are_untouched():
    hidden, weights, selected, routing, values, a, b = inputs(3)
    out = mod.UnoMovaPerExpertValueLoRA(a, b, SCALING, backend(active=False))(hidden, routing, selected, weights, values)
    assert out is values


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
