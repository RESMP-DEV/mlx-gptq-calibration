from __future__ import annotations

import array
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlx_gptq_calibration.campaign import load_campaign

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/quantize_mxfp4_vllm.py"
SPEC = importlib.util.spec_from_file_location("quantize_mxfp4_vllm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)

CAMPAIGN = SCRIPT.parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json"


class QuantizeMxfp4VllmTests(unittest.TestCase):
    def test_campaign_is_pinned_for_compressed_tensors(self) -> None:
        campaign = load_campaign(CAMPAIGN)
        self.assertEqual(campaign.data["output"]["format"], "compressed-tensors")
        self.assertEqual(campaign.data["output"]["scheme"], "MXFP4A16")
        self.assertEqual(
            campaign.data["engine"]["commit"], "6e4b96e467c243cb988e87c0fe97b4d51359f983"
        )
        self.assertEqual(campaign.data["runtime"]["llmcompressor"], "0.14.0")
        self.assertEqual(campaign.data["validation"]["vllm"], "0.32.0")
        self.assertEqual(campaign.data["validation"]["ampere_behavior"], "Marlin W4A16")

    def test_runtime_mismatch_fails_closed(self) -> None:
        data = json.loads(CAMPAIGN.read_text())
        versions = dict(data["runtime"])
        with patch.object(quantize, "package_versions", return_value=versions):
            verified = quantize.verify_runtime(data)
        self.assertEqual(verified["llmcompressor"], "0.14.0")

        versions["transformers"] = "0.0.0"
        with patch.object(quantize, "package_versions", return_value=versions):
            with self.assertRaisesRegex(quantize.QuantizationContractError, "runtime package"):
                quantize.verify_runtime(data)

    def test_calibration_hash_and_shape_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.npy"
            base = b"{'descr':'<i8','fortran_order':False,'shape':(2,4),}"
            header = base + b" " * (64 - 10 - len(base) - 1) + b"\n"
            path.write_bytes(
                b"\x93NUMPY\x01\x00"
                + len(header).to_bytes(2, "little")
                + header
                + array.array("q", range(8)).tobytes()
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fake_numpy = SimpleNamespace(
                load=lambda source, mmap_mode: SimpleNamespace(
                    shape=(2, 4), min=lambda: 0, tolist=lambda: [[0, 1, 2, 3], [4, 5, 6, 7]]
                )
            )
            with patch.dict(sys.modules, {"numpy": fake_numpy}):
                loaded = quantize.load_calibration(path, digest, (2, 4))
            self.assertEqual(loaded.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7]])
            with self.assertRaisesRegex(quantize.QuantizationContractError, "shape mismatch"):
                wrong_shape = SimpleNamespace(
                    load=lambda source, mmap_mode: SimpleNamespace(shape=(4, 2))
                )
                with patch.dict(sys.modules, {"numpy": wrong_shape}):
                    quantize.load_calibration(path, digest, (2, 4))

    def test_source_receipt_repository_and_both_revisions_are_required(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text())
        revision = campaign["model"]["revision"]
        receipt = {
            "status": "verified",
            "repo": campaign["model"]["id"],
            "requested_revision": revision,
            "resolved_revision": revision,
            "failures": [],
        }
        self.assertEqual(quantize.verify_source_receipt(receipt, campaign), receipt)

        receipt["resolved_revision"] = "0" * 40
        expected = "source model verification"
        with self.assertRaisesRegex(quantize.QuantizationContractError, expected):
            quantize.verify_source_receipt(receipt, campaign)


if __name__ == "__main__":
    unittest.main()
