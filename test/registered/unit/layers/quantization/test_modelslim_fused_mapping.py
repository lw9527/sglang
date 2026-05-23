import unittest

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimFusedMapping(CustomTestCase):
    def test_flat_packed_modules_mapping_resolves_fused_mla_prefix(self):
        quant_description = {
            "language_model.model.layers.0.self_attn.q_a_proj.weight": "W8A8",
            "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "W8A8",
        }
        config = ModelSlimConfig(
            {
                **quant_description,
                "packed_modules_mapping": {
                    "fused_qkv_a_proj_with_mqa": [
                        "q_a_proj",
                        "kv_a_proj_with_mqa",
                    ],
                },
            }
        )

        prefix = "language_model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"
        proj_name = prefix.split(".")[-1]
        fused_mapping = config._resolve_fused_modules_mapping(proj_name, "model")

        self.assertIn("fused_qkv_a_proj_with_mqa", fused_mapping)
        prefix_in_quant_config = prefix.replace(
            proj_name, fused_mapping["fused_qkv_a_proj_with_mqa"][0]
        )
        scheme = config.get_linear_scheme(layer=None, prefix=prefix_in_quant_config)

        self.assertIsNotNone(scheme)
        self.assertEqual(
            prefix_in_quant_config,
            "language_model.model.layers.0.self_attn.q_a_proj",
        )


if __name__ == "__main__":
    unittest.main()
