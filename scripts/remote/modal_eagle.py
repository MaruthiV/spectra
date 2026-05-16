"""Modal app — Phase C (EAGLE custom draft training).

Foundation file: defines the shared App, GPU image, and Modal Volumes used by all
Phase C jobs (corpus build, hidden-state dump, training, eval). Individual jobs
live in sibling files (build_corpus.py, dump_hidden.py, train_eagle.py, etc.)
and import `app`, `image`, and the volume handles from here.

One-time setup (run on host):
    eval "$(/Users/maruthi/miniforge3/bin/conda shell.zsh hook)" && conda activate spectra
    modal token new       # opens browser; auth once

Smoke test (~60s wall, ~$0.07 of H100 time):
    modal run scripts/remote/modal_eagle.py::hello

Expected stdout:
    [modal] running on NVIDIA H100 80GB HBM3 (CUDA 12.x)
    torch=2.5.x  transformers=4.x  flash_attn=2.x
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# App + GPU image. Used by all Phase C jobs.
# ---------------------------------------------------------------------------
app = modal.App("spectra-eagle")

image = (
    # Use the official PyTorch dev image — bundles CUDA 12.4 + cuDNN 9 + torch 2.5.1
    # so flash-attn has everything it needs at compile time, no manual CUDA install.
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "huggingface_hub==0.26.2",
        "safetensors==0.4.5",
        "numpy==1.26.4",
        "tqdm==4.66.5",
        "ninja==1.11.1.1",     # speeds up flash-attn build by ~10x
    )
    # flash-attn must build against the right torch/CUDA; the pytorch base already has both.
    .pip_install("flash-attn==2.7.0.post2", extra_options="--no-build-isolation")
)

# ---------------------------------------------------------------------------
# Volumes. Each phase writes to its own volume so we can blow one away without
# losing the others. Created on first use (`create_if_missing=True`).
# ---------------------------------------------------------------------------
corpus_vol = modal.Volume.from_name("spectra-eagle-corpus", create_if_missing=True)   # C2 output
dump_vol = modal.Volume.from_name("spectra-eagle-dump", create_if_missing=True)        # C3 output (~100 GB)
ckpt_vol = modal.Volume.from_name("spectra-eagle-ckpt", create_if_missing=True)        # C5 output


# ---------------------------------------------------------------------------
# Smoke test: confirm the image, GPU, and our pinned package versions all work.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="H100", timeout=120)
def hello() -> dict[str, str]:
    import torch
    import transformers
    import flash_attn

    info = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU",
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "unknown",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn.__version__,
        "mem_total_gb": (
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}"
            if torch.cuda.is_available()
            else "n/a"
        ),
    }
    print(f"[modal] running on {info['gpu_name']} (CUDA {info['cuda_version']})")
    print(f"torch={info['torch']}  transformers={info['transformers']}  flash_attn={info['flash_attn']}")
    print(f"GPU memory: {info['mem_total_gb']} GB")
    return info


@app.local_entrypoint()
def main() -> None:
    """Default entrypoint: invoke the smoke test from `modal run scripts/remote/modal_eagle.py`."""
    print("Submitting Spectra-EAGLE smoke test to Modal (H100, ~60s)…")
    info = hello.remote()
    print("\n--- Modal returned ---")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("\n✓ C1 setup verified.")
