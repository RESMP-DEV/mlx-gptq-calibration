from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from .campaign import Campaign, CampaignError, load_campaign, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "campaigns" / "qwen3.8-27b-mxfp4.json"
LOCAL_CORPUS_CANDIDATES = (
    Path.home() / "datasets/publish/ptq-calibration-corpus/calibration.txt",
    Path.home() / "RESMP-DEV/hf-dataset-card-staging/ptq-calibration-corpus/calibration.txt",
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, check=True, **kwargs)


def _ssh(host: str, command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            host,
            "bash",
            "-lc",
            shlex.quote(command),
        ],
        capture_output=capture,
    )


def _api_json(kind: str, repo_id: str, revision: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
    if kind == "dataset":
        url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def verify_hub(campaign: Campaign) -> dict[str, Any]:
    expected_model = campaign.data["model"]["revision"]
    expected_dataset = campaign.data["dataset"]["revision"]
    model = _api_json("model", campaign.data["model"]["id"], expected_model)
    dataset = _api_json("dataset", campaign.data["dataset"]["id"], expected_dataset)
    if model.get("sha") != expected_model:
        raise CampaignError(
            f"model revision moved: pinned {expected_model}, live {model.get('sha')}"
        )
    if dataset.get("sha") != expected_dataset:
        raise CampaignError(
            f"dataset revision moved: pinned {expected_dataset}, live {dataset.get('sha')}"
        )
    return {
        "model": {
            "id": model.get("id"),
            "sha": model.get("sha"),
            "private": model.get("private"),
            "gated": model.get("gated"),
        },
        "dataset": {
            "id": dataset.get("id"),
            "sha": dataset.get("sha"),
            "private": dataset.get("private"),
            "gated": dataset.get("gated"),
        },
    }


def find_corpus(campaign: Campaign) -> Path:
    candidates: list[Path] = []
    if override := os.environ.get("PTQ_CALIBRATION_CORPUS"):
        candidates.append(Path(override).expanduser())
    candidates.extend(LOCAL_CORPUS_CANDIDATES)
    expected = campaign.data["dataset"]["sha256"]
    mismatches: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        actual = sha256_file(candidate)
        if actual == expected:
            return candidate.resolve()
        mismatches.append(f"{candidate}={actual}")
    detail = f"; mismatches: {', '.join(mismatches)}" if mismatches else ""
    raise CampaignError(
        "pinned calibration.txt is not available locally; set PTQ_CALIBRATION_CORPUS" + detail
    )


def probe_remote(campaign: Campaign, host: str) -> dict[str, Any]:
    remote_root = campaign.remote_root
    script = r"""
set -eu
root="$1"
gpu_rows="$(nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader,nounits)"
python3 - "$root" "$gpu_rows" <<'PY'
import json, os, shutil, socket, subprocess, sys
root, rows = sys.argv[1], sys.argv[2]
expanded = os.path.expanduser(root)
probe_path = expanded
while not os.path.exists(probe_path):
    parent = os.path.dirname(probe_path)
    if parent == probe_path:
        break
    probe_path = parent
usage = shutil.disk_usage(probe_path)
gpus = []
for row in rows.splitlines():
    index, name, memory, driver = [part.strip() for part in row.split(',', 3)]
    gpus.append({"index": int(index), "name": name, "memory_mib": int(memory), "driver": driver})
print(json.dumps({
    "hostname": socket.gethostname(),
    "gpus": gpus,
    "disk_free_bytes": usage.free,
    "uv": shutil.which("uv"),
    "python": sys.version.split()[0],
    "remote_root": expanded,
}))
PY
"""
    command = f"bash -s -- {shlex.quote(remote_root)}"
    result = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            host,
            command,
        ],
        input=script,
        capture_output=True,
    )
    probe = json.loads(result.stdout)
    minimum_count = campaign.data["compute"]["minimum_gpu_count"]
    minimum_vram = campaign.data["compute"]["minimum_vram_mib"]
    probe["requirements"] = {
        "minimum_gpu_count": minimum_count,
        "minimum_vram_mib": minimum_vram,
        "satisfied": len(probe["gpus"]) >= minimum_count
        and all(gpu["memory_mib"] >= minimum_vram for gpu in probe["gpus"][:minimum_count]),
    }
    return probe


def _receipt_path(campaign: Campaign, name: str) -> Path:
    path = ROOT / "runs" / campaign.name / "receipts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_receipt(campaign: Campaign, name: str, payload: dict[str, Any]) -> Path:
    path = _receipt_path(campaign, name)
    body = {"recorded_at": _now(), "campaign": campaign.name, **payload}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def _require_satisfied(probe: dict[str, Any], allow: bool) -> None:
    if probe["requirements"]["satisfied"] or allow:
        return
    got = ", ".join(f"{gpu['name']} ({gpu['memory_mib']} MiB)" for gpu in probe["gpus"])
    required = probe["requirements"]
    raise CampaignError(
        f"CUDA host is underprovisioned: got {len(probe['gpus'])} GPU(s): {got}; "
        f"campaign requires {required['minimum_gpu_count']} GPU(s) with at least "
        f"{required['minimum_vram_mib']} MiB each"
    )


def stage_a_argv(campaign: Campaign, gpu_count: int) -> list[str]:
    root = campaign.remote_root.rstrip("/")
    quant = campaign.data["calibration"]
    devices = ",".join(f"cuda:{index}" for index in range(gpu_count))
    return [
        f"{root}/engine/.venv/bin/python",
        "-m",
        "mlx_gptq.calibrate",
        "--model",
        f"{root}/inputs/model",
        "--output",
        f"{root}/artifacts",
        "--dataset",
        f"{root}/inputs/calibration.txt",
        "--nsamples",
        str(quant["nsamples"]),
        "--seqlen",
        str(quant["seqlen"]),
        "--batch-size",
        str(quant["batch_size"]),
        "--mode",
        quant["mode"],
        "--mxfp-algorithm",
        quant["mxfp_algorithm"],
        "--bits",
        str(quant["bits"]),
        "--group-size",
        str(quant["group_size"]),
        "--clip",
        quant["clip"],
        "--damp",
        str(quant["damp"]),
        "--dtype",
        quant["dtype"],
        "--devices",
        devices,
        "--vram-gb",
        str(quant["vram_gb"]),
        "--layers-attr",
        campaign.data["model"]["layers_attr"],
        "--seed",
        str(quant["seed"]),
    ]


def cmd_verify(campaign: Campaign, _args: argparse.Namespace) -> int:
    corpus = find_corpus(campaign)
    hub = verify_hub(campaign)
    receipt = _write_receipt(
        campaign,
        "verify",
        {
            "hub": hub,
            "corpus": {
                "path": str(corpus),
                "bytes": corpus.stat().st_size,
                "sha256": sha256_file(corpus),
            },
            "engine": campaign.data["engine"],
        },
    )
    print(receipt)
    return 0


def cmd_probe(campaign: Campaign, args: argparse.Namespace) -> int:
    probe = probe_remote(campaign, args.host)
    receipt = _write_receipt(campaign, f"probe-{args.host}", {"host": args.host, **probe})
    print(json.dumps(probe, indent=2))
    print(f"receipt: {receipt}")
    _require_satisfied(probe, args.allow_underprovisioned)
    return 0


def cmd_prepare(campaign: Campaign, args: argparse.Namespace) -> int:
    corpus = find_corpus(campaign)
    verify_hub(campaign)
    probe = probe_remote(campaign, args.host)
    _require_satisfied(probe, args.allow_underprovisioned)

    root = campaign.remote_root.rstrip("/")
    engine = campaign.data["engine"]
    model = campaign.data["model"]
    setup = f"""
set -eu
mkdir -p {shlex.quote(root)}/inputs {shlex.quote(root)}/cache/huggingface
if [ ! -d {shlex.quote(root)}/engine/.git ]; then
  git clone {shlex.quote(engine["repository"])} {shlex.quote(root)}/engine
fi
cd {shlex.quote(root)}/engine
test -z "$(git status --porcelain)"
git fetch origin {shlex.quote(engine["commit"])}
git checkout --detach {shlex.quote(engine["commit"])}
uv sync --frozen --python {shlex.quote(engine["python"])}
uv pip install --python .venv/bin/python accelerate=={shlex.quote(engine["accelerate"])}
"""
    _ssh(args.host, setup)
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            str(corpus),
            f"{args.host}:{root}/inputs/calibration.txt",
        ]
    )
    download = f"""
set -eu
export HF_HUB_CACHE={shlex.quote(root)}/cache/huggingface
export HF_XET_CACHE={shlex.quote(root)}/cache/xet
export HF_XET_HIGH_PERFORMANCE=1
{shlex.quote(root)}/engine/.venv/bin/hf download {shlex.quote(model["id"])} \
  --revision {shlex.quote(model["revision"])} \
  --local-dir {shlex.quote(root)}/inputs/model
sha256sum {shlex.quote(root)}/inputs/calibration.txt
test -f {shlex.quote(root)}/inputs/model/model.safetensors.index.json
"""
    _ssh(args.host, download)
    receipt = _write_receipt(
        campaign,
        f"prepare-{args.host}",
        {"host": args.host, "probe": probe, "remote_root": root},
    )
    print(receipt)
    return 0


def cmd_start(campaign: Campaign, args: argparse.Namespace) -> int:
    probe = probe_remote(campaign, args.host)
    _require_satisfied(probe, args.allow_underprovisioned)
    argv = stage_a_argv(campaign, len(probe["gpus"]))
    root = campaign.remote_root.rstrip("/")
    command = shlex.join(argv)
    start = f"""
set -eu
test -f {shlex.quote(root)}/inputs/model/model.safetensors.index.json
test -f {shlex.quote(root)}/inputs/calibration.txt
if [ -f {shlex.quote(root)}/run.pid ] && \
  kill -0 "$(cat {shlex.quote(root)}/run.pid)" 2>/dev/null; then
  echo 'calibration is already running' >&2
  exit 3
fi
nohup {command} > {shlex.quote(root)}/stage-a.log 2>&1 < /dev/null &
echo $! > {shlex.quote(root)}/run.pid
cat {shlex.quote(root)}/run.pid
"""
    result = _ssh(args.host, start, capture=True)
    receipt = _write_receipt(
        campaign,
        f"start-{args.host}",
        {"host": args.host, "pid": int(result.stdout.strip()), "argv": argv, "probe": probe},
    )
    print(receipt)
    return 0


def cmd_status(campaign: Campaign, args: argparse.Namespace) -> int:
    root = campaign.remote_root.rstrip("/")
    status = f"""
set -u
pid=''
state='not-started'
if [ -f {shlex.quote(root)}/run.pid ]; then
  pid="$(cat {shlex.quote(root)}/run.pid)"
  if kill -0 "$pid" 2>/dev/null; then state='running'; else state='stopped'; fi
fi
printf 'state=%s pid=%s\n' "$state" "$pid"
if [ -f {shlex.quote(root)}/artifacts/manifest.json ]; then
  python3 - {shlex.quote(root)}/artifacts/manifest.json <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print('layers_done=' + str(len(d.get('layers_done', []))))
print('tensor_count=' + str(len(d.get('tensors', {{}}))))
PY
fi
if [ -f {shlex.quote(root)}/stage-a.log ]; then
  tail -n {int(args.lines)} {shlex.quote(root)}/stage-a.log
fi
"""
    result = _ssh(args.host, status, capture=True)
    print(result.stdout, end="")
    return 0


def cmd_fetch(campaign: Campaign, args: argparse.Namespace) -> int:
    root = campaign.remote_root.rstrip("/")
    destination = ROOT / "runs" / campaign.name / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-a",
            "--partial",
            "--checksum",
            f"{args.host}:{root}/artifacts/",
            f"{destination}/",
        ]
    )
    manifest = destination / "manifest.json"
    if not manifest.is_file():
        raise CampaignError(f"fetched artifacts have no manifest: {manifest}")
    payload = json.loads(manifest.read_text())
    receipt = _write_receipt(
        campaign,
        f"fetch-{args.host}",
        {
            "host": args.host,
            "destination": str(destination),
            "manifest_sha256": sha256_file(manifest),
            "layers_done": len(payload.get("layers_done", [])),
            "tensor_count": len(payload.get("tensors", {})),
        },
    )
    print(receipt)
    return 0


def cmd_pack(campaign: Campaign, args: argparse.Namespace) -> int:
    engine = Path(args.engine).expanduser().resolve()
    expected_commit = campaign.data["engine"]["commit"]
    actual_commit = _run(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], capture_output=True
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise CampaignError(f"engine is {actual_commit}; campaign pins {expected_commit}")
    python = engine / ".venv" / "bin" / "python"
    if not python.is_file():
        raise CampaignError(
            f"missing {python}; run `uv sync --directory {engine} --frozen --extra mlx` first"
        )
    source_model = Path(args.model).expanduser().resolve()
    if not (source_model / "model.safetensors.index.json").is_file():
        raise CampaignError("pack requires a complete local checkout of the pinned Qwen3.8 model")
    calib = ROOT / "runs" / campaign.name / "calibration"
    output = Path(args.output).expanduser().resolve()
    argv = [
        str(python),
        "-m",
        "mlx_gptq.pack",
        "--hf-path",
        str(source_model),
        "--calib",
        str(calib),
        "--mlx-path",
        str(output),
        "--verify",
    ]
    _run(argv, cwd=engine)
    receipt = _write_receipt(
        campaign,
        "pack",
        {"source_model": str(source_model), "output": str(output), "engine_commit": actual_commit},
    )
    print(receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="verify pinned Hub revisions and the local corpus")

    for name in ("probe", "prepare", "start"):
        command = sub.add_parser(name)
        command.add_argument("--host", required=True)
        command.add_argument("--allow-underprovisioned", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--host", required=True)
    status.add_argument("--lines", type=int, default=40)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--host", required=True)

    pack = sub.add_parser("pack")
    pack.add_argument("--engine", default="~/mlx-gptq")
    pack.add_argument("--model", required=True)
    pack.add_argument("--output", default="~/models/Qwen3.8-27B-MLX-GPTQ-MXFP4")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        handlers = {
            "verify": cmd_verify,
            "probe": cmd_probe,
            "prepare": cmd_prepare,
            "start": cmd_start,
            "status": cmd_status,
            "fetch": cmd_fetch,
            "pack": cmd_pack,
        }
        return handlers[args.command](campaign, args)
    except (CampaignError, subprocess.CalledProcessError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
