from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_mxfp4_vllm.py"
SPEC = importlib.util.spec_from_file_location("validate_mxfp4_vllm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class ValidateMxfp4VllmTests(unittest.TestCase):
    def test_accepts_weight_only_mxfp4_pack_quantized_config(self) -> None:
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 32,
                        },
                        "input_activations": None,
                    }
                },
            }
        }
        self.assertEqual(
            validator.validate_quantization_config(config),
            config["quantization_config"],
        )

    def test_rejects_quantized_activations_or_wrong_format(self) -> None:
        base = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 32},
                        "input_activations": None,
                    }
                },
            },
        }
        quantized_activations = copy.deepcopy(base)
        quantized_activations["quantization_config"]["config_groups"]["group_0"][
            "input_activations"
        ] = {"num_bits": 8}
        with self.assertRaisesRegex(validator.VllmValidationError, "input activations"):
            validator.validate_quantization_config(quantized_activations)

        wrong_format = copy.deepcopy(base)
        wrong_format["quantization_config"]["format"] = "dense"
        with self.assertRaisesRegex(validator.VllmValidationError, "wrong compressed-tensors"):
            validator.validate_quantization_config(wrong_format)


if __name__ == "__main__":
    unittest.main()
