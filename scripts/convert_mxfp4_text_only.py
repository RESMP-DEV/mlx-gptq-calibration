#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TextOnlyPackagingError(ValueError):
    pass


TOKENIZER_FILE_NAMES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        msg = "source config has no text_config"
        raise TextOnlyPackagingError(msg)
    flattened = dict(text)
    flattened["architectures"] = ["Qwen3_5ForCausalLM"]
    flattened["model_type"] = text.get("model_type", "qwen3_5_text")
    flattened["quantization_config"] = dict(config["quantization_config"])
    flattened["quantization_config"]["ignore"] = ["lm_head"]
    return flattened


def tensor_key(source_key: str) -> str | None:
    if source_key == "lm_head.weight":
        return source_key
    prefix = "model.language_model."
    if source_key.startswith(prefix):
        return f"model.{source_key[len(prefix) :]}"
    return None


def verify_source_index(index: Mapping[str, Any], loaded_keys: set[str]) -> None:
    weight_map = index.get("weight_map", {})
    required = {"model.safetensors", "model_mtp.safetensors"}
    if not required.issubset(set(weight_map.values())):
        msg = "source index does not contain the expected main and MTP shards"
        raise TextOnlyPackagingError(msg)
    if any("model.visual." in key for key in weight_map):
        msg = "text-only source unexpectedly contains vision weights"
        raise TextOnlyPackagingError(msg)
    expected_text_keys = {key for key in weight_map if tensor_key(key) is not None}
    loaded_text_keys = {key for key in loaded_keys if tensor_key(key) is not None}
    wrong_shard = sorted(
        key for key in expected_text_keys if weight_map.get(key) != "model.safetensors"
    )
    if expected_text_keys != loaded_text_keys or wrong_shard:
        missing = sorted(expected_text_keys - loaded_text_keys)
        unexpected = sorted(loaded_text_keys - expected_text_keys)
        msg = (
            "indexed text tensors do not match loaded model.safetensors keys: "
            f"missing={missing}, unexpected={unexpected}, wrong_shard={wrong_shard}"
        )
        raise TextOnlyPackagingError(msg)


def copy_pinned_tokenizer(
    tokenizer_source: Path, output: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Restore tokenizer files that Transformers may rewrite while saving a model."""
    hashes = verify_pinned_tokenizer(tokenizer_source, expected_hashes)
    for name in TOKENIZER_FILE_NAMES:
        shutil.copyfile(tokenizer_source / name, output / name)
    return hashes


def verify_pinned_tokenizer(
    tokenizer_source: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Fail before artifact serialization when tokenizer provenance does not match."""
    hashes = {}
    for name in TOKENIZER_FILE_NAMES:
        source_file = tokenizer_source / name
        if not source_file.is_file():
            msg = f"pinned tokenizer source is missing {name}"
            raise TextOnlyPackagingError(msg)
        if set(expected_hashes) != set(TOKENIZER_FILE_NAMES):
            msg = "campaign does not pin exactly the required tokenizer file set"
            raise TextOnlyPackagingError(msg)
        digest = sha256_file(source_file)
        if digest != expected_hashes[name]:
            msg = f"tokenizer source hash mismatch for {name}: {digest} != {expected_hashes[name]}"
            raise TextOnlyPackagingError(msg)
        hashes[name] = digest
    return hashes


def run_conversion(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file, save_file

    campaign = json.loads(args.campaign.read_text())
    if args.output.exists() and any(args.output.iterdir()):
        msg = f"output directory is not empty: {args.output}"
        raise TextOnlyPackagingError(msg)
    source_config = json.loads((args.source / "config.json").read_text())
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())
    flattened = flatten_config(source_config)
    tensors = load_file(str(args.source / "model.safetensors"), device="cpu")
    verify_source_index(source_index, set(tensors))
    text_tensors = {}
    for source_key, tensor in tensors.items():
        target_key = tensor_key(source_key)
        if target_key is not None:
            text_tensors[target_key] = tensor
    packed = sum(key.endswith("weight_packed") for key in text_tensors)
    scaled = sum(key.endswith("weight_scale") for key in text_tensors)
    if (packed, scaled) != (496, 496):
        msg = f"expected 496 packed/scaled tensor pairs, found {(packed, scaled)}"
        raise TextOnlyPackagingError(msg)
    verify_pinned_tokenizer(args.tokenizer_source, campaign["output"]["tokenizer_files"])
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(text_tensors, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in text_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: "model.safetensors" for key in sorted(text_tensors)},
    }
    (args.output / "config.json").write_text(json.dumps(flattened, indent=2, sort_keys=True) + "\n")
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    for name in ("generation_config.json", "recipe.yaml"):
        source_file = args.source / name
        if source_file.is_file():
            shutil.copyfile(source_file, args.output / name)
    tokenizer_files = copy_pinned_tokenizer(
        args.tokenizer_source, args.output, campaign["output"]["tokenizer_files"]
    )
    files = {
        item.name: sha256_file(item) for item in sorted(args.output.iterdir()) if item.is_file()
    }
    receipt = {
        "schema": "mlx-gptq-calibration/mxfp4-text-only-packaging/1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "campaign": campaign["name"],
        "source": str(args.source),
        "output": str(args.output),
        "architectures": flattened["architectures"],
        "tensor_count": len(text_tensors),
        "packed_linear_tensors": packed,
        "scale_tensors": scaled,
        "tokenizer_source": str(args.tokenizer_source),
        "tokenizer_files": tokenizer_files,
        "output_files": files,
        "non_claims": [
            "This packaging pass changes config/key layout only; it does not alter packed values.",
            "Live generation and parity remain separate validation gates.",
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json",
    )
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        required=True,
        help="verified source checkpoint supplying the pinned tokenizer byte-for-byte",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_conversion(args)
    except (OSError, ValueError) as exc:
        print(f"text-only packaging failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
