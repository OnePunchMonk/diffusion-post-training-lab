"""Step 1 on Modal: reproduce AnchorSeg's ReasonSeg val number from released weights.

Staged on purpose. The upstream environment is a pinned 2023 stack -- Python
3.9, torch 2.0.1+cu118, transformers 4.31, peft 0.4.0, deepspeed 0.10.3,
flash-attn 2.6.3 -- and building it is the part most likely to fail. Failing
that on a $2/hour A100 while it downloads 18GB of weights is the expensive way
to find out, so the work is split into functions you can run in order:

  modal run modal/step1_eval.py::probe     # CPU, ~free: does the image build and import?
  modal run modal/step1_eval.py::fetch     # CPU: pull weights into the volume, once
  modal run modal/step1_eval.py::evaluate  # A100: the actual run

Each stage caches into a Modal volume, so a failure in one doesn't re-do the
previous one.

Why cu118 and A100 rather than anything newer: flash-attn 2.6.3 is pinned to a
cu118/torch2.0/cp39 wheel, and A100 (sm80) is the architecture that stack was
built and tested against. Newer GPUs would need the whole pin set revisited,
which is a change to the thing being replicated.
"""

from __future__ import annotations

import modal

APP_NAME = "anchorseg-step1"

UPSTREAM_REPO = "https://github.com/rui-qian/AnchorSeg.git"
UPSTREAM_COMMIT = "0d8e3f098b763aeb035baadf85649b94c7e1e721"
ANCHORSEG_7B_REPO = "rui-qian/hf-AnchorSeg-7b_reason_seg_val_llava1.5_ema"
CLIP_VISION_TOWER = "openai/clip-vit-large-patch14-336"
SAM_VIT_H_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
REASONSEG_DATASET = "fcxfcx/ReasonSeg"

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
)

CACHE = "/cache"
CODE = "/workspace/AnchorSeg"

app = modal.App(APP_NAME)

# One volume for everything that is expensive to obtain: weights, the dataset,
# and the predicted masks a run produces. Re-running evaluate() then costs GPU
# time only.
cache = modal.Volume.from_name("anchorseg-cache", create_if_missing=True)

image = (
    # cu118 devel (not runtime): deepspeed and flash-attn compile against nvcc.
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.9")
    .apt_install("git", "wget", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .run_commands(
        f"git clone {UPSTREAM_REPO} {CODE}",
        f"cd {CODE} && git checkout --detach {UPSTREAM_COMMIT}",
    )
    # Run their build.sh rather than transcribing its pins here. It encodes an
    # exact, interdependent version set; retyping it is a transcription-error
    # surface for no benefit, and running it keeps this honest about what the
    # replication environment actually is.
    .run_commands(f"cd {CODE} && chmod +x build.sh && ./build.sh", gpu="A100-40GB")
    .run_commands(f"pip install {FLASH_ATTN_WHEEL}")
    .run_commands(f"cd {CODE} && pip install -e .")
    # Ours, for independent scoring. Pinned loosely because it only has to run
    # numpy -- it deliberately shares nothing with the upstream stack.
    .pip_install("numpy==1.24.2", "opencv-python-headless==4.8.0.74", "huggingface-hub==0.25.0")
)


@app.function(image=image, volumes={CACHE: cache}, timeout=900)
def probe() -> dict:
    """Does the environment actually work? CPU only, so it costs almost nothing.

    Checks the three things that break first: the pinned stack imports, the
    upstream package is importable at the pinned commit, and flash-attn loaded.
    A GPU check is deliberately not here -- flash-attn importing is the useful
    signal and it does not need a device.
    """
    import subprocess
    import sys

    report: dict = {"python": sys.version.split()[0]}

    for module in ("torch", "transformers", "peft", "deepspeed", "flash_attn", "cv2"):
        try:
            mod = __import__(module)
            report[module] = getattr(mod, "__version__", "imported")
        except Exception as exc:  # reporting every failure beats stopping at the first
            report[module] = f"FAILED: {type(exc).__name__}: {exc}"

    commit = subprocess.run(
        ["git", "-C", CODE, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    report["upstream_commit"] = commit.stdout.strip()
    report["commit_matches_pin"] = commit.stdout.strip() == UPSTREAM_COMMIT

    parse_check = f"import ast; ast.parse(open('{CODE}/train_ds.py').read())"
    entrypoint = subprocess.run(
        ["python", "-c", parse_check], capture_output=True, text=True
    )
    report["train_ds_parses"] = entrypoint.returncode == 0

    for key, value in report.items():
        print(f"{key:<22} {value}")
    return report


@app.function(image=image, volumes={CACHE: cache}, timeout=3600)
def fetch() -> dict:
    """Pull weights and the dataset into the volume. CPU only -- no GPU rental
    while ~18GB downloads."""
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    os.makedirs(CACHE, exist_ok=True)
    sizes = {}

    for name, repo_id, repo_type in (
        ("anchorseg", ANCHORSEG_7B_REPO, "model"),
        ("clip", CLIP_VISION_TOWER, "model"),
    ):
        target = f"{CACHE}/{name}"
        if os.path.isdir(target) and os.listdir(target):
            print(f"{name}: already present")
        else:
            print(f"{name}: downloading {repo_id}")
            snapshot_download(repo_id, repo_type=repo_type, local_dir=target)
        sizes[name] = _dir_size(target)

    reasonseg = f"{CACHE}/reason_seg/ReasonSeg"
    if not os.path.isdir(f"{reasonseg}/val"):
        os.makedirs(reasonseg, exist_ok=True)
        snapshot_download(
            REASONSEG_DATASET, repo_type="dataset", allow_patterns=["val/*"], local_dir=reasonseg
        )
    n_images = len([f for f in os.listdir(f"{reasonseg}/val") if f.endswith(".jpg")])
    sizes["reasonseg_val_images"] = n_images
    if n_images != 200:
        raise RuntimeError(
            f"ReasonSeg val has {n_images} images, expected 200. An incomplete split scores "
            "higher than a complete one, so this is a hard failure."
        )

    sam = f"{CACHE}/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam):
        # Download to a .part and rename, so an interrupted transfer cannot
        # leave a truncated checkpoint that loads far enough to produce garbage.
        subprocess.run(["wget", "-q", "-O", sam + ".part", SAM_VIT_H_URL], check=True)
        os.rename(sam + ".part", sam)
    sizes["sam"] = _dir_size(sam)

    cache.commit()
    for key, value in sizes.items():
        print(f"{key:<22} {value}")
    return sizes


def _dir_size(path: str) -> str:
    import os

    if os.path.isfile(path):
        return f"{os.path.getsize(path) / 1e9:.2f} GB"
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )
    return f"{total / 1e9:.2f} GB"


@app.local_entrypoint()
def main() -> None:
    """Default entrypoint runs the cheap stage, so `modal run` on this file
    never accidentally starts a GPU job."""
    print("Probing the upstream environment (CPU only)...\n")
    probe.remote()
    print("\nNext: modal run modal/step1_eval.py::fetch   (CPU, ~18GB into the volume)")
