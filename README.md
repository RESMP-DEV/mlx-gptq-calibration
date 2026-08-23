# MLX GPTQ Calibration

Reproducible calibration campaigns that keep the Mac authoritative while a
bounded SSH-accessed CUDA host performs the expensive GPTQ solve. The returned
artifacts are packed and verified locally as standard `mlx-lm` models.

The first campaign is pinned to:

- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Corpus: `RESMP-DEV/ptq-calibration-corpus@ffa2e74b9361d573167373140f10dfd30c69a327`
- Engine: `RESMP-DEV/mlx-gptq-mxfp@7131c407af8b8991fb5a8148a49cf2f323c7afd3`
- Output: native MLX MXFP4, group size 32, activation-Hessian GPTQ with safe
  E8M0 scale search

The first campaign consumes the dataset's default Qwen3.8-tokenized
`data/calibration_256x2048.npy`. The model-neutral `calibration.txt` remains
available for targets without a published matching matrix; token IDs are never
reused across tokenizers.

## Architecture

```text
Mac (Git, credentials, pins, acceptance)
  | SSH: prepare/start/status
  | rsync: tokenizer-matched calibration.npy ->
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

## Hybrid BF16 router experiment

The Gemma campaign also pins its ordinary BF16 parent and can fetch only the
21.8 MB router state as a safetensors overlay. This supports two distinct
mixed-precision variants without downloading another 51.6 GB checkpoint:

- QAT-derived GPTQ MXFP4 with the QAT router retained in BF16.
- The same packed model with the ordinary parent BF16 router transplanted.

The campaign's packing regex explicitly keeps `router.*`, `gate`, shared-expert
gates, and score-correction paths out of low-bit conversion. This also prevents
the custom pack predicate from accidentally overriding Gemma 4's native router
policy and quantizing `router.proj` to ordinary 4-bit RTN.

See [docs/hybrid-bf16-surgery.md](docs/hybrid-bf16-surgery.md) for the evidence,
ablation matrix, routing metrics, the pinned Qwen3.6-35B-A3B non-QAT control, and
the artifact-level audit of JoyAI-LLM Flash as a second genuine QAT MoE.

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

The practical floor is one CUDA GPU reporting at least 8,000 MiB VRAM. This is
not a full-model residency requirement: the pipeline streams one decoder layer
at a time and derives the per-GPU expert-solve budget from live free memory,
while reserving headroom for the driver, layer, and activation working set.
Sub-8 GB operation can be forced with `--allow-underprovisioned`, but is likely
to lose the practical advantage over local MLX calibration. There is no
multi-GPU requirement or recommendation; every eligible GPU present is used.
Budgets are computed independently, so mixed 24 GB, 32 GB, and 96 GB devices
retain their own expert-chunk capacity instead of being throttled to the
smallest GPU.

Prepare the isolated remote environment and pinned 55.6 GB checkpoint:

```bash
uv run mlx-gptq-calibration prepare --host YOUR_CUDA_SSH_ALIAS
```

`prepare` discovers complete pinned snapshots in `~/models` and the Hugging
Face cache, transferring them with checksummed `rsync`; it falls back to the
independent `hfd` CLI on the CUDA host. The fallback disables Xet, pins the Hub
revision, resumes through aria2, and verifies every LFS SHA-256 or Git blob
before calibration can start. Install `hfd` in the remote `PATH` (or at
`~/.local/bin/hfd`). An explicit local source can be selected with
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

Fetch the pinned parent-BF16 router overlay for the Gemma hybrid:

```bash
uv run mlx-gptq-calibration --campaign "$GEMMA" fetch-overlay
```

Stage B needs the exact campaign source model on the Mac and an MLX-ready
environment in the pinned engine checkout. For Qwen3.8:

```bash
hfd download Qwen/Qwen3.8-27B \
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

The Mac currently has the verified Qwen3.8 token matrix used by campaign one
and the Gemma 4 matrix used by campaign two, so each is sent directly instead
of being downloaded again. `~/Qwen3.6-27B` has the same
Qwen3.5-family 64-layer shape and is useful for compatibility experiments, but
its weights, config, tokenizer metadata, and 15-shard index do not match the
pinned Qwen3.8 18-shard checkpoint. It is never accepted as the model input.

## Safety properties

- Model, dataset, engine, and tokenizer-specific calibration content are
  independently pinned.
- Remote work lives under a campaign-specific directory and never owns Git or
  credentials for this project.
- The remote start is detached, logged, resumable, and refuses duplicate runs.
- Hardware requirements are proven from live `nvidia-smi`, not host labels.
- Packing refuses an engine checkout at the wrong commit.
- Published model repositories are private by default.
