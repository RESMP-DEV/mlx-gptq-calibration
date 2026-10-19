from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_mxfp4_vllm.py"
CAMPAIGN = SCRIPT.parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json"
SPEC = importlib.util.spec_from_file_location("validate_mxfp4_vllm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class ValidateMxfp4VllmTests(unittest.TestCase):
    def test_missing_required_package_fails_closed(self) -> None:
        error = validator.importlib.metadata.PackageNotFoundError("vllm")
        with patch.object(validator.importlib.metadata, "version", side_effect=error):
            with self.assertRaisesRegex(validator.VllmValidationError, "missing"):
                validator.package_versions()

    def test_campaign_pins_vllm_runtime(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text())
        self.assertEqual(campaign["validation"]["engine"], "vllm")
        self.assertEqual(campaign["validation"]["vllm"], "0.32.0")

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
