# MLX GPTQ Calibration

Reproducible calibration campaigns that keep the Mac authoritative while a
bounded SSH-accessed CUDA host performs the expensive GPTQ solve. The returned
artifacts are packed and verified locally as standard `mlx-lm` models.

The first campaign is pinned to:

- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Corpus: `RESMP-DEV/ptq-calibration-corpus@f73806747b6d90b7a4ba1c1f20b027345b3d4354`
- Engine: `RESMP-DEV/mlx-gptq-mxfp@d103c83bef49b966f6e694da0c36adfc5712846a`
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

The checkpoint is a native vision-language wrapper. This campaign calibrates
the 64-layer language model at `model.language_model.layers`; the vision tower
is not used to generate text calibration activations.

## Quick start

The project itself has no third-party runtime dependency:

```bash
uv sync
uv run mlx-gptq-calibration verify
uv run mlx-gptq-calibration probe --host YOUR_CUDA_SSH_ALIAS
```

The default acceptance target is four CUDA GPUs with at least 20,000 MiB each,
matching a 4x24 GB workstation. A smaller host can be inspected, but
`prepare`/`start` reject it unless `--allow-underprovisioned` is supplied
explicitly.

Prepare the isolated remote environment and pinned 55.6 GB checkpoint:

```bash
uv run mlx-gptq-calibration prepare --host YOUR_CUDA_SSH_ALIAS
```

Start the resumable CUDA stage in the background, then monitor it:

```bash
uv run mlx-gptq-calibration start --host YOUR_CUDA_SSH_ALIAS
uv run mlx-gptq-calibration status --host YOUR_CUDA_SSH_ALIAS
```

Fetch completed calibration artifacts:

```bash
uv run mlx-gptq-calibration fetch --host YOUR_CUDA_SSH_ALIAS
```

Stage B needs the exact pinned Qwen3.8 source model on the Mac and an MLX-ready
environment in the pinned engine checkout:

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir /Users/kearm/models/Qwen3.8-27B-1d4bf0f

uv sync --directory /Users/kearm/mlx-gptq --frozen --extra mlx
/Users/kearm/mlx-gptq/.venv/bin/python \
  /Users/kearm/mlx-gptq/tools/patch_mlx_lm_tf512.py

uv run mlx-gptq-calibration pack \
  --model /Users/kearm/models/Qwen3.8-27B-1d4bf0f
```

Every phase writes a machine-readable receipt below the ignored
`runs/qwen3.8-27b-mxfp4/` directory. Calibration is crash-safe and resumes from
the layer artifacts already present on the CUDA host.

## Local artifacts

The Mac currently has a verified copy of the exact corpus text, so it is sent
directly instead of downloaded again. `/Users/kearm/Qwen3.6-27B` has the same
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
