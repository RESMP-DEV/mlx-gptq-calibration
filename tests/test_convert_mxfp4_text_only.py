from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/convert_mxfp4_text_only.py"
SPEC = importlib.util.spec_from_file_location("convert_mxfp4_text_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ConvertMxfp4TextOnlyTests(unittest.TestCase):
    def test_flattens_conditional_text_config(self) -> None:
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
            },
            "quantization_config": {
                "format": "mxfp4-pack-quantized",
                "ignore": ["lm_head", "re:^mtp.*"],
            },
        }
        flattened = converter.flatten_config(config)
        self.assertEqual(flattened["architectures"], ["Qwen3_5ForCausalLM"])
        self.assertEqual(flattened["model_type"], "qwen3_5_text")
        self.assertEqual(flattened["hidden_size"], 5120)
        self.assertEqual(flattened["quantization_config"]["ignore"], ["lm_head"])

    def test_rewrites_language_keys_and_drops_auxiliary_weights(self) -> None:
        self.assertEqual(
            converter.tensor_key("model.language_model.layers.0.mlp.down_proj.weight_packed"),
            "model.layers.0.mlp.down_proj.weight_packed",
        )
        self.assertEqual(converter.tensor_key("lm_head.weight"), "lm_head.weight")
        self.assertIsNone(converter.tensor_key("mtp.fc.weight"))
        self.assertIsNone(converter.tensor_key("model.visual.blocks.0.mlp.linear_fc1.weight"))

    def test_source_index_must_have_main_and_mtp_shards(self) -> None:
        with self.assertRaisesRegex(converter.TextOnlyPackagingError, "main and MTP"):
            converter.verify_source_index({"weight_map": {"a": "model.safetensors"}})

    def test_restores_pinned_tokenizer_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_source = root / "source-checkpoint"
            output = root / "artifact"
            tokenizer_source.mkdir()
            output.mkdir()
            for name in converter.TOKENIZER_FILE_NAMES:
                (tokenizer_source / name).write_bytes(f"pinned-{name}".encode())

            hashes = converter.copy_pinned_tokenizer(tokenizer_source, output)

            self.assertEqual(set(hashes), set(converter.TOKENIZER_FILE_NAMES))
            for name in converter.TOKENIZER_FILE_NAMES:
                self.assertEqual((output / name).read_bytes(), f"pinned-{name}".encode())

    def test_missing_pinned_tokenizer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_source = root / "source-checkpoint"
            output = root / "artifact"
            tokenizer_source.mkdir()
            output.mkdir()

            expected = "pinned tokenizer source is missing"
            with self.assertRaisesRegex(converter.TextOnlyPackagingError, expected):
                converter.copy_pinned_tokenizer(tokenizer_source, output)


if __name__ == "__main__":
    unittest.main()
