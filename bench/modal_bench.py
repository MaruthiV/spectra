"""Modal-side benchmark: run the Spectra benchmark on an Nvidia A100.

Prereq:
    pip install modal           # local
    modal token new             # one-time: link your Modal account
    cd spectra/demo && pnpm build       # produce demo/dist
    cd spectra/bench && pnpm build      # not strictly needed; runner.ts is run via tsx in the container

Run:
    cd spectra
    modal run bench/modal_bench.py

The script:
    1. Builds the Docker image from bench/docker/Dockerfile (uses local demo/dist + bench/).
    2. Spins up a Modal container with one A100 attached.
    3. The container's CMD serves demo/dist on :8000, then runs runner.ts against it.
    4. We capture stdout + the result JSON from the container's `bench/results/`
       directory (mounted as a volume) and write it to host disk.

NOTE: WebGPU on Linux Chrome requires a working Vulkan stack. Modal A100
containers ship with Nvidia drivers but the Vulkan ICD wiring is non-trivial.
This script is currently a SCAFFOLD; expect to iterate on the Dockerfile until
`vkcube` runs cleanly inside the container before the benchmark itself
will produce meaningful numbers.

Status: scaffold (2026-05-14). Validated locally; Modal end-to-end pending.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Image: built from bench/docker/Dockerfile + the demo's static build.
# ---------------------------------------------------------------------------
app = modal.App("spectra-bench")

REPO_ROOT = pathlib.Path(__file__).parent.parent  # spectra/
DEMO_DIST = REPO_ROOT / "demo" / "dist"
BENCH_DIR = pathlib.Path(__file__).parent

if not DEMO_DIST.exists():
    raise RuntimeError(
        f"{DEMO_DIST} does not exist. Run `pnpm build` in spectra/demo/ first."
    )

image = (
    modal.Image.from_dockerfile(
        str(BENCH_DIR / "docker" / "Dockerfile"),
        context_mount=modal.Mount.from_local_dir(str(REPO_ROOT), remote_path="/spectra-src"),
        # Note: the Dockerfile's COPY paths reference demo/dist + bench/, which is
        # what's at /spectra-src after the context mount. Adjust the Dockerfile or
        # add a build arg if Modal's image build context differs.
    )
)

results_volume = modal.Volume.from_name("spectra-bench-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Function: run the benchmark on an A100.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=15 * 60,           # 15 min total (model download + benchmark)
    volumes={"/results": results_volume},
)
def run_bench() -> dict[str, Any]:
    import subprocess
    proc = subprocess.run(
        ["bash", "-c", (
            "cd /spectra/bench && "
            "npx http-server /spectra/demo -p 8000 --silent & "
            "sleep 2 && "
            "SPECTRA_DEMO_URL=http://localhost:8000/ "
            "SPECTRA_HEADLESS=true "
            "npx tsx runner.ts 2>&1"
        )],
        capture_output=True,
        text=True,
        check=False,
    )
    print("---- stdout ----")
    print(proc.stdout[-4000:])
    print("---- stderr ----")
    print(proc.stderr[-2000:])
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark failed with exit {proc.returncode}")

    # Surface results file produced by runner.ts.
    results_dir = pathlib.Path("/spectra/bench/results")
    files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    if not files:
        raise RuntimeError("benchmark produced no results JSON")
    latest = files[-1]
    payload = json.loads(latest.read_text())
    # Persist to the Modal Volume so we can fetch from the host side.
    out_volpath = pathlib.Path("/results") / latest.name
    out_volpath.write_text(json.dumps(payload, indent=2))
    return payload


@app.local_entrypoint()
def main() -> None:
    print("Submitting Spectra benchmark to Modal A100…")
    result = run_bench.remote()
    print("\n--- benchmark complete ---")
    runs = result.get("runs", [])
    print(f"hostname: {result.get('hostname')} on {result.get('platform')}/{result.get('arch')}")
    print(f"baseline: {result.get('baselineTokPerSec')} tok/s")
    print(f"\n| γ | mean accept | mean tok/s | speedup |")
    print(f"|---|---|---|---|")
    for r in runs:
        print(
            f"| {r['gamma']} | {r['meanAcceptance']:.3f} | "
            f"{r['meanTokensPerSecond']:.1f} | {r['speedupVsBaseline']:.2f}× |"
        )
    print("\nFull JSON saved to Modal Volume `spectra-bench-results`.")
