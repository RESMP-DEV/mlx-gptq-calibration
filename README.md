# MLX GPTQ Calibration

Reproducible calibration campaigns that keep the Mac authoritative while a
bounded SSH-accessed CUDA host performs the expensive GPTQ solve. The returned
artifacts are packed and verified locally as standard `mlx-lm` models.

The first campaign is pinned to:

- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Corpus: `RESMP-DEV/ptq-calibration-corpus@f73806747b6d90b7a4ba1c1f20b027345b3d4354`
- Engine: `RESMP-DEV/mlx-gptq-mxfp@7fe79676bb717b4a282bf3aa11096f617f8f154f`
- Output: native MLX MXFP4, group size 32, activation-Hessian GPTQ with safe
  E8M0 scale search

`calibration.txt` is retokenized with the pinned Qwen tokenizer. The corpus's
included `calibration_256x2048.npy` is intentionally not used because those
token IDs were produced for LFM2.5 and are not portable across tokenizers.

## Architecture

```text
Mac (Git, credentials, pins, acceptance)
  | SSH: prepare/start/status
  | rsync: calibration.txt ->
  v
CUDA host (model cache + layer-streamed GPTQ)
  | rsync: calibrated q/scales artifacts ->
  v
Mac (mlx-lm conversion, injection, generation verification)
```

The checkpoint is a native vision-language wrapper. Transformers exposes a
text-only causal runtime at `model.layers`, while the raw checkpoint uses
`model.language_model.layers` and MLX uses `language_model.model.layers`. The
pinned engine maps those three namespaces explicitly. The vision tower is not
used to generate calibration activations and `mlx-lm` drops it during
conversion, so the final artifact is the Qwen3.8 text-generation model.

The second campaign targets the actual Gemma 4 MoE checkpoint:

- Model: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized@f1e06dc520982d9b9edd76859fdb7ab209449949`
- Architecture: 25.2B total parameters, 3.8B active, 128 experts with top-8 routing
- Source: BF16 weights extracted from Google's Q4_0 quantization-aware training
- Output: native MLX MXFP4 with routed-token Hessians for every active expert

The similarly named `...-unquantized-assistant` repository is only an 839 MB,
four-layer dense speculative-decoding drafter. It is pinned in the campaign as
an optional companion, but it is not substituted for the MoE target.

## Quick start

The project itself has no third-party runtime dependency:

```bash
uv sync
uv run mlx-gptq-calibration verify
uv run mlx-gptq-calibration probe --host YOUR_CUDA_SSH_ALIAS
```

Select campaign two with:

```bash
GEMMA=campaigns/gemma4-26b-a4b-qat-mxfp4.json
uv run mlx-gptq-calibration --campaign "$GEMMA" verify
uv run mlx-gptq-calibration --campaign "$GEMMA" probe --host YOUR_CUDA_SSH_ALIAS
```

The default acceptance target is four CUDA GPUs with at least 20,000 MiB each,
matching a 4x24 GB workstation. A smaller host can be inspected, but
`prepare`/`start` reject it unless `--allow-underprovisioned` is supplied
explicitly.

Prepare the isolated remote environment and pinned 55.6 GB checkpoint:

```bash
uv run mlx-gptq-calibration prepare --host YOUR_CUDA_SSH_ALIAS
```

`prepare` discovers complete pinned snapshots in `~/models` and the Hugging
Face cache, transferring them with checksummed `rsync`; it falls back to a
pinned Hub download on the CUDA host. An explicit source can be selected with
`--local-model PATH`.

Start the resumable CUDA stage in the background, then monitor it:

```bash
uv run mlx-gptq-calibration start --host YOUR_CUDA_SSH_ALIAS
uv run mlx-gptq-calibration status --host YOUR_CUDA_SSH_ALIAS
```

Fetch completed calibration artifacts:

```bash
uv run mlx-gptq-calibration fetch --host YOUR_CUDA_SSH_ALIAS
```

Stage B needs the exact campaign source model on the Mac and an MLX-ready
environment in the pinned engine checkout. For Qwen3.8:

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir ~/models/Qwen-Qwen3.8-27B-1d4bf0f

uv sync --directory ~/mlx-gptq --frozen --extra mlx
~/mlx-gptq/.venv/bin/python ~/mlx-gptq/tools/patch_mlx_lm_tf512.py

uv run mlx-gptq-calibration pack \
  --model ~/models/Qwen-Qwen3.8-27B-1d4bf0f
```

Every phase writes a machine-readable receipt below the ignored
`runs/qwen3.8-27b-mxfp4/` directory. Calibration is crash-safe and resumes from
the layer artifacts already present on the CUDA host.

## Local artifacts

The Mac currently has a verified copy of the exact corpus text, so it is sent
directly instead of downloaded again. `~/Qwen3.6-27B` has the same
Qwen3.5-family 64-layer shape and is useful for compatibility experiments, but
its weights, config, tokenizer metadata, and 15-shard index do not match the
pinned Qwen3.8 18-shard checkpoint. It is never accepted as the model input.

## Safety properties

- Model, dataset, engine, and corpus content are independently pinned.
- Remote work lives under a campaign-specific directory and never owns Git or
  credentials for this project.
- The remote start is detached, logged, resumable, and refuses duplicate runs.
- Hardware requirements are proven from live `nvidia-smi`, not host labels.
- Packing refuses an engine checkout at the wrong commit.
- Published model repositories are private by default.
