"""Loading for UNO's draft LoRA."""

from __future__ import annotations

import logging
import os

from typing import TYPE_CHECKING

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)

# This name is internal to the model-execution process. It is never exposed as
# a request-selectable serving adapter.
_UNO_INTERNAL_LORA_NAME = "__uno_draft__"

# LoRA pool capacity includes the base-model slot.
_UNO_LORA_POOL_CAPACITY = 2


def init_uno_lora_manager(
    model_runner: ModelRunner,
) -> tuple[LoRAManager, str]:
    """Load and pin the single UNO draft adapter."""

    lora_path = get_spec().uno_lora_path

    uno_ref = LoRARef(
        lora_id=LoRARef.deterministic_id(
            _UNO_INTERNAL_LORA_NAME,
            lora_path,
        ),
        lora_name=_UNO_INTERNAL_LORA_NAME,
        lora_path=lora_path,
        pinned=True,
    )

    manager = LoRAManager(
        base_model=model_runner.model,
        base_hf_config=model_runner.model_config.hf_config,
        max_loras_per_batch=_UNO_LORA_POOL_CAPACITY,
        load_config=model_runner.load_config,
        dtype=model_runner.dtype,
        server_args=model_runner.server_args,
        lora_backend="uno_cublas",  # fast path
        # Infer these from the one trained adapter.
        max_lora_rank=None,
        target_modules=None,
        lora_paths=[uno_ref],
    )

    # LoRAManager construction initially makes only the base slot resident.
    # UNO always needs both fixed choices resident:
    #
    #   None             -> base model
    #   uno_ref.lora_id  -> base model + UNO draft LoRA
    manager.fetch_new_loras({None, uno_ref.lora_id})
    if get_spec().uno_fused_lora:
        _fold_uno_lora(manager, uno_ref.lora_id)
        if get_spec().uno_draft_weight_quant in ("fp8", "int4"):
            _quantize_uno_fused(manager, get_spec().uno_draft_weight_quant)
        elif os.environ.get("UNO_DRAFT_HESSIAN_OUT"):
            _attach_hessian_collectors(manager, os.environ["UNO_DRAFT_HESSIAN_OUT"])
    if get_spec().uno_mova_lora:
        from sglang.srt.speculative.uno_mova_lora import attach_uno_mova_lora

        attach_uno_mova_lora(manager, model_runner.model, lora_path)

    return manager, uno_ref.lora_id


def _fold_uno_lora(manager: LoRAManager, lora_id: str) -> None:
    """Fold the resident UNO adapter's LoRA-A rows into each base GEMM."""
    from sglang.srt.lora.backend.uno_fused_lora import (
        fold_uno_lora_group,
        fold_uno_lora_layer,
    )
    from sglang.srt.lora.layers import BaseLayerWithLoRA
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    adapter = manager.loras[lora_id]
    fold_kwargs = dict(
        weight_index=manager.memory_pool.uid_to_buffer_id[lora_id],
        rank=adapter.config.r,
        scaling=adapter.scaling,
    )
    grouped: set[int] = set()
    num_groups = 0
    # GDN input projections share one input; fold qkvz|ba like the base fused GEMM.
    for module in manager.base_model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        members = [module.in_proj_qkvz, module.in_proj_ba]
        fused = fold_uno_lora_group(members, **fold_kwargs)
        if fused is None:
            continue
        module.uno_fused_in_proj = fused
        grouped.update(id(member) for member in members)
        num_groups += 1

    num_folded = num_kept = 0
    for layer_modules in manager.lora_modules:
        for module in layer_modules.values():
            if id(module) in grouped:
                continue
            fused = fold_uno_lora_layer(module, **fold_kwargs)
            if fused is None:
                num_kept += 1
                continue
            module.uno_fused = fused
            num_folded += 1
    # Dense MLPs fuse the folded gate_up expand with SiLU*mul on draft rows.
    num_mlps = 0
    for module in manager.base_model.modules():
        if not isinstance(module, Qwen2MoeMLP) or not isinstance(
            module.gate_up_proj, BaseLayerWithLoRA
        ):
            continue
        fused = module.gate_up_proj.uno_fused
        if fused is None or len(fused.gate_up_blocks) != 2 or fused.out_features % 2:
            continue
        module.uno_fused_gate_up = fused
        num_mlps += 1
    logger.info(
        "UNO fused LoRA: %d GDN input groups, %d single layers folded, "
        "%d keep the unfused path, %d MLPs fuse SiLU*mul.",
        num_groups,
        num_folded,
        num_kept,
        num_mlps,
    )


def _uno_fused_slots(manager: LoRAManager):
    """(key, fused, setter) for every folded draft GEMM, in a fixed order; keys are
    stable across runs (module names), e.g. for saved calibration statistics."""
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    grouped: set[int] = set()
    slots = []
    for name, module in manager.base_model.named_modules():
        if isinstance(module, Qwen3_5GatedDeltaNet) and module.uno_fused_in_proj is not None:
            grouped.update((id(module.in_proj_qkvz), id(module.in_proj_ba)))
            slots.append((f"{name}.uno_fused_in_proj", module.uno_fused_in_proj,
                          lambda f, m=module: setattr(m, "uno_fused_in_proj", f)))
    num_unfolded = 0
    for layer_id, layer_modules in enumerate(manager.lora_modules):
        for name, module in layer_modules.items():
            if id(module) in grouped:
                continue
            fused = getattr(module, "uno_fused", None)
            if fused is None:
                num_unfolded += 1
                continue
            slots.append((f"layers.{layer_id}.{name}.uno_fused", fused,
                          lambda f, m=module: setattr(m, "uno_fused", f)))
    return slots, num_unfolded


def _refresh_fused_gate_up(manager: LoRAManager) -> None:
    """The MLP holds the same folded gate_up object; point it at the replaced one."""
    from sglang.srt.lora.layers import BaseLayerWithLoRA
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    for module in manager.base_model.modules():
        if isinstance(module, Qwen2MoeMLP) and module.uno_fused_gate_up is not None:
            if isinstance(module.gate_up_proj, BaseLayerWithLoRA):
                module.uno_fused_gate_up = module.gate_up_proj.uno_fused


def _attach_hessian_collectors(manager: LoRAManager, out_dir: str) -> None:
    """Calibration for --uno-draft-weight-quant int4 + UNO_DRAFT_GPTQ_DIR: with the
    bf16 seed layout (quant 'bf16') and no CUDA graph, accumulate X^T X of every
    folded draft GEMM input and save it once UNO_DRAFT_HESSIAN_ROWS rows were seen."""
    import msgspec

    from sglang.srt.speculative.uno_draft_quant import UnoHessianCollector

    import re

    pattern = re.compile(os.environ.get("UNO_DRAFT_HESSIAN_FILTER", ".*"))
    slots, _ = _uno_fused_slots(manager)
    num = 0
    for key, fused, setter in slots:
        if not pattern.search(key):
            continue
        collector = UnoHessianCollector(fused.weight_cat, key, out_dir)
        if collector.hessian is None:
            continue
        setter(msgspec.structs.replace(fused, draft_weight=collector))
        num += 1
    _refresh_fused_gate_up(manager)
    logger.info("UNO draft Hessian collection: %d GEMMs -> %s", num, out_dir)


def _quantize_uno_fused(manager: LoRAManager, kind: str) -> None:
    """Give every folded draft GEMM a quantized copy of its ``[W; A]`` weight.

    Only draft forwards read it (UnoFusedLoRA.forward / forward_silu_mul); verify
    forwards keep the bf16 rows W. Modules left unfolded keep bf16 drafts.
    """
    import msgspec
    import torch

    from sglang.srt.speculative.uno_draft_quant import UnoMarlinDraftWeight

    import re

    gptq_dir = os.environ.get("UNO_DRAFT_GPTQ_DIR") if kind == "int4" else None
    # Mixed precision: INT4 drafts may keep the GEMMs whose key matches
    # UNO_DRAFT_FP8_KEYS (regex) in FP8.
    fp8_keys = os.environ.get("UNO_DRAFT_FP8_KEYS") if kind == "int4" else None
    fp8_pattern = re.compile(fp8_keys) if fp8_keys else None
    num_gemms = num_bytes = num_gptq = num_fp8 = 0
    slots, num_unfolded = _uno_fused_slots(manager)
    for key, fused, setter in slots:
        hessian = cache = None
        if fp8_pattern is not None and fp8_pattern.search(key):
            weight = UnoMarlinDraftWeight(fused.weight_cat, "fp8")
            setter(msgspec.structs.replace(fused, draft_weight=weight))
            num_gemms += 1
            num_fp8 += 1
            num_bytes += weight.nbytes()
            continue
        if gptq_dir:
            # Marlin-ready GPTQ results are cached next to the Hessians; a cached
            # weight needs no Hessian (and no GPTQ pass) on later starts.
            from sglang.srt.speculative.uno_draft_quant import _GPTQ_ACT_ORDER

            suffix = {0: "", 1: ".ao", 2: ".ao2"}[_GPTQ_ACT_ORDER]
            cache = os.path.join(
                gptq_dir, "marlin_cache", f"{key}.int4g{UnoMarlinDraftWeight.group}{suffix}.pt"
            )
            path = os.path.join(gptq_dir, f"{key}.pt")
            if os.path.exists(cache):
                num_gptq += 1
            elif os.path.exists(path):
                hessian = torch.load(path, map_location=fused.weight_cat.device)
                num_gptq += 1
            else:
                cache = None
        weight = UnoMarlinDraftWeight(fused.weight_cat, kind, hessian=hessian, cache_path=cache)
        del hessian
        setter(msgspec.structs.replace(fused, draft_weight=weight))
        num_gemms += 1
        num_bytes += weight.nbytes()
    _refresh_fused_gate_up(manager)
    if gptq_dir or fp8_pattern is not None:
        logger.info(
            "UNO draft INT4: %d of %d GEMMs quantized with GPTQ (%s); %d kept in FP8.",
            num_gptq, num_gemms, gptq_dir, num_fp8,
        )
    # The draft's lm_head (no LoRA) gets a quantized copy too; LogitsProcessor
    # reads it only for UNO draft forwards. UNO_DRAFT_QUANT_LM_HEAD=0 keeps bf16.
    lm_head = getattr(manager.base_model, "lm_head", None)
    lm_head_bytes = 0
    if (
        os.environ.get("UNO_DRAFT_QUANT_LM_HEAD", "1") != "0"
        and lm_head is not None
        and isinstance(getattr(lm_head, "weight", None), torch.Tensor)
        and lm_head.weight.dtype in (torch.bfloat16, torch.float16)
    ):
        lm_head.uno_draft_weight = UnoMarlinDraftWeight(lm_head.weight, kind)
        # Up to 32 rows (c <= 4 with F = 8): steady per cycle c=2 26.2 vs 26.4 ms,
        # c=4 29.3 vs 29.5 ms against the bf16 lm_head; 36-row (old 9-row draft)
        # c=4 blocks were slower with Marlin (32.6 vs 31.7 ms).
        lm_head.uno_draft_weight.max_rows = min(
            lm_head.uno_draft_weight.max_rows,
            int(os.environ.get("UNO_DRAFT_LM_HEAD_MAX_ROWS", "32")),
        )
        lm_head_bytes = lm_head.uno_draft_weight.nbytes()
    torch.cuda.empty_cache()
    logger.info(
        "UNO draft weights (%s): %d folded GEMMs quantized, %.2f GB; "
        "%d LoRA modules outside the folded groups keep bf16 drafts; lm_head %s (%.2f GB).",
        kind,
        num_gemms,
        num_bytes / 1e9,
        num_unfolded,
        "quantized" if lm_head_bytes else "bf16",
        lm_head_bytes / 1e9,
    )


class UnoCudaGraphLoRAState:
    """Retained token-row LoRA routing for UNO draft graph buckets."""

    def __init__(
        self,
        manager: LoRAManager,
        uno_lora_id: str,
        forward_width: int,
    ):
        self.manager = manager
        self.uno_lora_id = uno_lora_id
        self.forward_width = forward_width
        self._draft_batch_infos = {}

    def capture_draft(self, batch_size: int) -> None:
        if batch_size not in self._draft_batch_infos:
            self.manager.prepare_lora_token_segments(
                lora_ids=[None, self.uno_lora_id] * batch_size,
                segment_lens=[1, self.forward_width - 1] * batch_size,
            )
            batch_info = self.manager.lora_backend.batch_info
            batch_info.use_cuda_graph = True
            self._draft_batch_infos[batch_size] = batch_info

        self.activate_draft(batch_size)

    def activate_draft(self, batch_size: int) -> None:
        self.manager.reset_lora_batch()
        self.manager.lora_backend.batch_info = self._draft_batch_infos[batch_size]

    def reset(self) -> None:
        self.manager.reset_lora_batch()
