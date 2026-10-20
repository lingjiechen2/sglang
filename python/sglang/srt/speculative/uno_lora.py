"""Loading for UNO's draft LoRA."""

from __future__ import annotations

import logging

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
