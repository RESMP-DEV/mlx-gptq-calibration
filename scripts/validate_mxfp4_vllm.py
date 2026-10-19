#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class VllmValidationError(ValueError):
    pass


def package_versions() -> dict[str, str]:
    try:
        return {
            "vllm": importlib.metadata.version("vllm"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        msg = f"required vLLM package is missing: {exc}"
        raise VllmValidationError(msg) from exc


def validate_quantization_config(config: Mapping[str, Any]) -> dict[str, Any]:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        msg = "model config has no quantization_config"
        raise VllmValidationError(msg)
    if quantization.get("quant_method") != "compressed-tensors":
        msg = f"wrong quant method: {quantization.get('quant_method')!r}"
        raise VllmValidationError(msg)
    if quantization.get("format") != "mxfp4-pack-quantized":
        msg = f"wrong compressed-tensors format: {quantization.get('format')!r}"
        raise VllmValidationError(msg)
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        msg = "quantization_config has no config groups"
        raise VllmValidationError(msg)
    weights = [group.get("weights") for group in groups.values()]
    if any(
        not isinstance(weight, Mapping)
        or weight.get("num_bits") != 4
        or weight.get("group_size") != 32
        or weight.get("type") != "float"
        for weight in weights
    ):
        msg = "quantization groups are not MXFP4 group-32 float weights"
        raise VllmValidationError(msg)
    if any(group.get("input_activations") is not None for group in groups.values()):
        msg = "MXFP4A16 must not quantize input activations"
        raise VllmValidationError(msg)
    return dict(quantization)


def gpu_state() -> list[dict[str, Any]] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


def run_validation(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    config = json.loads((args.model / "config.json").read_text())
    quantization = validate_quantization_config(config)
    versions = package_versions()
    if args.expected_vllm and versions["vllm"] != args.expected_vllm:
        msg = f"vLLM version mismatch: expected {args.expected_vllm}, got {versions['vllm']}"
        raise VllmValidationError(msg)
    llm = LLM(
        model=str(args.model),
        quantization="compressed-tensors",
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    started = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    elapsed = time.perf_counter() - started
    if len(outputs) != 1 or not outputs[0].outputs:
        msg = "vLLM returned no generation output"
        raise VllmValidationError(msg)
    completion = outputs[0].outputs[0]
    receipt = {
        "schema": "mlx-gptq-calibration/vllm-mxfp4-validation/1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "runtime": versions,
        "gpus": gpu_state(),
        "model": str(args.model),
        "architectures": config.get("architectures"),
        "quantization_config": quantization,
        "prompt": args.prompt,
        "output": completion.text,
        "token_ids": list(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "generation_elapsed_seconds": round(elapsed, 3),
        "non_claims": [
            "One deterministic generation proves CUDA execution, not accuracy or throughput.",
            "Parity and performance require a separate matched BF16 control protocol.",
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--expected-vllm", default="0.32.0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_validation(args)
    except (KeyError, OSError, ValueError) as exc:
        print(f"vLLM validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
