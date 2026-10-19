from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestEngineAdmission(CustomTestCase):
    def _base_layer(self, **overrides):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        quant_method = object.__new__(UnquantizedFusedMoEMethod)
        dispatcher = object.__new__(StandardDispatcher)
        dispatcher.skip_local_expert_mapping = overrides.get("skip_local", False)
        return SimpleNamespace(
            quant_method=quant_method,
            dispatcher=dispatcher,
            w13_weight=torch.zeros(2, 8, 4, dtype=torch.bfloat16),
            w2_weight=torch.zeros(2, 4, 4, dtype=torch.bfloat16),
            num_local_experts=2,
            should_fuse_routed_scaling_factor_in_topk=overrides.get(
                "fused_scaling", False
            ),
            moe_runner_config=SimpleNamespace(
                activation=overrides.get("activation", "silu"),
                is_gated=True,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
                apply_router_weight_on_input=False,
                no_combine=False,
                num_fused_shared_experts=0,
                top_k=2,
                routed_scaling_factor=1.0,
            ),
        )

    def _admit(self, **overrides):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        MoeLoraRunner._admit(self._base_layer(**overrides))

    def test_admits_a_supported_layer(self):
        self._admit()

    def test_weight_family_of_a_bf16_layer(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        assert MoeLoraRunner._weight_family(self._base_layer()) == "bf16"

    def _fp8_layer(self, *, block_quant=True, weight_dtype=torch.float8_e4m3fn):
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        layer = self._base_layer()
        quant_method = object.__new__(Fp8MoEMethod)
        quant_method.block_quant = block_quant
        quant_method.use_mxfp8 = False
        quant_method.weight_block_size = [128, 128] if block_quant else None
        layer.quant_method = quant_method
        layer.w13_weight = layer.w13_weight.to(weight_dtype)
        layer.w2_weight = layer.w2_weight.to(weight_dtype)
        return layer

    def test_weight_family_of_a_block_fp8_layer(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        assert MoeLoraRunner._weight_family(self._fp8_layer()) == "fp8"

    def test_rejects_per_tensor_fp8(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        with self.assertRaisesRegex(NotImplementedError, "128-block"):
            MoeLoraRunner._weight_family(self._fp8_layer(block_quant=False))

    def test_rejects_marlin_repacked_fp8(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        with self.assertRaisesRegex(NotImplementedError, "float8_e4m3fn"):
            MoeLoraRunner._weight_family(self._fp8_layer(weight_dtype=torch.bfloat16))

    def test_fp8_family_selects_its_vendors(self):

        for vendor, rows in (
            ("cutedsl", "expert_major"),
            ("cutedsl", "route_major"),
            ("triton", "route_major"),
            # No masked slab domain: decode plans run the route-major
            # provider, same as the Marlin nvfp4 vendor.
            ("triton", "expert_major"),
        ):
            assert select_provider_cls(rows, "fp8", vendor)
        # Unsupported FP8 vendor choices resolve to the FP8 family default.
        for absent in ("marlin", "deepgemm"):
            assert select_provider_cls(
                "expert_major", "fp8", absent
            ) is select_provider_cls("expert_major", "fp8")

    def test_every_listed_vendor_resolves_and_others_fall_back_to_the_first(self):
        """Mixed-quant layers resolve independently, falling back to their family's vendor."""
        from sglang.srt.lora.moe.base_gemm_provider import VENDORS

        every_vendor = {vendor for vendors in VENDORS.values() for vendor in vendors}
        for family, vendors in VENDORS.items():
            default = select_provider_cls("route_major", family)
            assert default is select_provider_cls("route_major", family, vendors[0])
            for vendor in vendors:
                assert select_provider_cls("expert_major", family, vendor)
            for other in every_vendor - set(vendors):
                assert select_provider_cls("route_major", family, other) is default

    def test_vendors_without_a_masked_domain_serve_expert_major_rows(self):
        """Vendors without a masked domain serve expert-major requests through their route-major class."""

        for vendor, family in (
            ("triton", "bf16"),
            ("triton", "fp8"),
            ("marlin", "nvfp4"),
        ):
            assert select_provider_cls(
                "expert_major", family, vendor
            ) is select_provider_cls("route_major", family, vendor)

    def test_rejects_architectures_with_neither_mma_family(self):
        """SM90 and SM100 are a closed set, not a floor: SM120 reports major 12
        yet has neither WGMMA nor tcgen05, so a ">= SM90" check mis-admits it."""
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        for capability in ((8, 0), (12, 0), (11, 0)):
            with mock.patch(
                "torch.cuda.get_device_capability", lambda device=None: capability
            ):
                with self.assertRaisesRegex(NotImplementedError, "SM90 and SM100"):
                    MoeLoraRunner._admit(self._base_layer())

    def test_rejects_global_expert_id_dispatch(self):
        with self.assertRaisesRegex(NotImplementedError, "EP-local"):
            self._admit(skip_local=True)

    def test_rejects_prefolded_routed_scaling(self):
        """Scaling folded into top-k would be applied twice."""
        with self.assertRaisesRegex(NotImplementedError, "routed scaling"):
            self._admit(fused_scaling=True)

    def test_rejects_unsupported_activation(self):
        with self.assertRaisesRegex(NotImplementedError, "SiLU or ReLU2"):
            self._admit(activation="gelu")

    def test_activation_and_gating_bind_against_resident_slices(self):
        from sglang.srt.lora.moe.plan import ActivationFn, resolve_plans
        from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.lora.utils import Phase

        experts, hidden, inter = 4, 256, 128
        for activation in ActivationFn:
            for is_gated in (True, False):
                with self.subTest(activation=activation, is_gated=is_gated):
                    slices = 2 if is_gated else 1
                    provider = select_provider_cls("route_major", "bf16", "triton")(
                        MoeLoraBf16QuantInfo(
                            w13_weight=torch.zeros(
                                (experts, slices * inter, hidden),
                                device="cuda",
                                dtype=torch.bfloat16,
                            ),
                            w2_weight=torch.zeros(
                                (experts, hidden, inter),
                                device="cuda",
                                dtype=torch.bfloat16,
                            ),
                            num_local_experts=experts,
                            intermediate_size=inter,
                            hidden_size=hidden,
                        )
                    )
                    runner = MoeLoraRunner(
                        providers={"route_major": provider},
                        top_k=2,
                        routed_scaling_factor=1.0,
                        activation=activation,
                        is_gated=is_gated,
                    )
                    selected = resolve_plans(
                        quant_family="bf16",
                        architecture="sm100",
                        is_shared_outer=False,
                        physical_rank=16,
                        activation=activation,
                        hidden_size=hidden,
                        num_local_experts=experts,
                    )[Phase.PREFILL]
                    runner.validate_plan(
                        selected.plan, base_gemm_rows=selected.base_gemm_rows
                    )
                    runner.is_gated = not is_gated
                    with self.assertRaisesRegex(ValueError, "is_gated"):
                        runner.validate_plan(
                            selected.plan, base_gemm_rows=selected.base_gemm_rows
                        )


class TestLayerRunnerBackendResolution(CustomTestCase):
    """Bind the selected LoRA runner, otherwise the quant method's runner."""

    def _construct(self, *, global_backend, base_runner_backend, lora_backend="triton"):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        quant_method = mock.MagicMock()
        base_layer = SimpleNamespace(
            quant_method=quant_method,
            moe_runner_config=MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                layer_id=0,
                top_k=2,
                num_fused_shared_experts=0,
                params_dtype=torch.bfloat16,
                activation="silu",
                inplace=True,
            ),
            dispatcher=None,
            num_local_experts=8,
            should_fuse_routed_scaling_factor_in_topk=False,
            moe_tp_size=1,
            moe_tp_rank=0,
            intermediate_size_per_partition=32,
            runner=SimpleNamespace(runner_backend=base_runner_backend),
            run_moe_core=mock.Mock(),
        )
        backend = SimpleNamespace(name=lora_backend, lora_workspace=object())
        original_run_moe_core = base_layer.run_moe_core
        with (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=global_backend,
            ),
            mock.patch.object(MoeLoraRunner, "from_layer") as bind,
        ):
            layer = FusedMoEWithLoRA(
                base_layer,
                backend,
                experts_shared_outer_loras=True,
                max_lora_rank=16,
            )
        if global_backend.is_lora():
            bind.assert_called_once_with(
                base_layer,
                workspace=backend.lora_workspace,
                is_shared_outer=True,
                physical_rank=16,
            )
            self.assertIs(layer._moe_lora_runner, bind.return_value)
            self.assertIs(layer._base_run_moe_core, original_run_moe_core)
            self.assertEqual(base_layer.run_moe_core, layer._run_moe_core_with_lora)
        return layer

    def test_v2_requires_new_moe_runner(self):
        for backend in (
            MoeRunnerBackend.TRITON,
            MoeRunnerBackend.DEEP_GEMM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
        ):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "mixing new and legacy"),
            ):
                self._construct(
                    global_backend=backend,
                    base_runner_backend=backend,
                    lora_backend="triton_v2",
                )

    def test_v2_accepts_each_new_moe_runner(self):
        for backend in (
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
            MoeRunnerBackend.LORA_MARLIN,
        ):
            with self.subTest(backend=backend):
                layer = self._construct(
                    global_backend=backend,
                    base_runner_backend=MoeRunnerBackend.DEEP_GEMM,
                    lora_backend="triton_v2",
                )
                self.assertIs(layer._lora_runner_backend, backend)

    def test_new_moe_runner_rejects_legacy_dense_backend(self):
        for backend in (
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
            MoeRunnerBackend.LORA_MARLIN,
        ):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "mixing new and legacy"),
            ):
                self._construct(
                    global_backend=backend,
                    base_runner_backend=MoeRunnerBackend.DEEP_GEMM,
                )

    def test_experimental_backends_win_over_their_aliased_base_runner(self):
        """is_marlin() / is_flashinfer_trtllm() alias the experimental backends,
        so the base layer's runner says MARLIN / FLASHINFER_TRTLLM; the layer
        must still bind the experimental engine the server selected."""
        cases = (
            (
                MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
                MoeRunnerBackend.MARLIN,
                "sglang.srt.lora.marlin_lora_temp.lora_layer.init_experimental_sgl_marlin_lora",
            ),
            (
                MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                "sglang.srt.lora.trtllm_lora_temp.lora_layer.init_experimental_sgl_trtllm_lora",
            ),
        )
        for global_backend, base_runner_backend, init_path in cases:
            with mock.patch(init_path):
                layer = self._construct(
                    global_backend=global_backend,
                    base_runner_backend=base_runner_backend,
                )
            self.assertIs(layer._lora_runner_backend, global_backend)

    def test_legacy_path_follows_the_quant_methods_runner(self):
        layer = self._construct(
            global_backend=MoeRunnerBackend.AUTO,
            base_runner_backend=MoeRunnerBackend.TRITON,
        )
        self.assertIs(layer._lora_runner_backend, MoeRunnerBackend.TRITON)


if __name__ == "__main__":
    unittest.main()
