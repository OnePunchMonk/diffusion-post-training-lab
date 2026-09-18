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

**Two Pythons, on purpose.** Modal's runtime requires Python >= 3.10 (3.9 is
rejected outright), but upstream needs 3.9 -- the flash-attn wheel is a `cp39`
build and the rest of the pin set is contemporaneous with it. So the container
runs 3.11 for Modal's own machinery, and upstream lives in a separate 3.9
environment at UGROUND_ENV that these functions shell out to. Downgrading
upstream to 3.10 to collapse the two would mean finding a different flash-attn
build and revisiting the pins around it, which changes the environment being
replicated.
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

# Upstream's Python 3.9 environment, separate from the container's own 3.11.
UGROUND_ENV = "/opt/uground"
UGROUND_PY = f"{UGROUND_ENV}/bin/python"
# Scoped to the build steps: putting this on the image-wide PATH would shadow
# the interpreter Modal's runtime needs.
WITH_ENV = f"export PATH={UGROUND_ENV}/bin:$PATH &&"

app = modal.App(APP_NAME)

# One volume for everything that is expensive to obtain: weights, the dataset,
# and the predicted masks a run produces. Re-running evaluate() then costs GPU
# time only.
cache = modal.Volume.from_name("anchorseg-cache", create_if_missing=True)

image = (
    # cu118 devel (not runtime): deepspeed compiles against nvcc.
    # add_python is the container's own interpreter, for Modal's runtime only.
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "wget", "curl", "bzip2", "build-essential", "ninja-build", "libgl1", "libglib2.0-0"
    )
    # micromamba, purely to get a real Python 3.9 that upstream's cp39 wheel
    # can be installed into. Nothing else uses conda.
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest"
        " | tar -xvj -C /usr/local bin/micromamba",
        f"micromamba create -y -p {UGROUND_ENV} -c conda-forge python=3.9",
    )
    .run_commands(
        f"git clone {UPSTREAM_REPO} {CODE}",
        f"cd {CODE} && git checkout --detach {UPSTREAM_COMMIT}",
    )
    # Run their build.sh rather than transcribing its pins here. It encodes an
    # exact, interdependent version set; retyping it is a transcription-error
    # surface for no benefit, and running it keeps this honest about what the
    # replication environment actually is.
    .run_commands(f"{WITH_ENV} cd {CODE} && chmod +x build.sh && ./build.sh")
    .run_commands(f"{WITH_ENV} pip install {FLASH_ATTN_WHEEL}")
    .run_commands(f"{WITH_ENV} cd {CODE} && pip install -e .")
    # Ours, for independent scoring, installed into the *container* Python.
    # It shares nothing with the upstream stack on purpose: the whole argument
    # for reimplementing the metrics is that they are an independent path.
    .pip_install("numpy>=1.26", "opencv-python-headless>=4.9", "huggingface-hub>=0.25")
)


@app.function(image=image, volumes={CACHE: cache}, timeout=900)
def probe() -> dict:
    """Does the environment actually work? CPU only, so it costs almost nothing.

    Checks what breaks first: the pinned 3.9 stack imports, the upstream package
    is importable at the pinned commit, and flash-attn loaded. Imports are probed
    inside UGROUND_ENV via subprocess, not in this process -- this process is the
    3.11 container Python and cannot see upstream's environment at all.

    A GPU check is deliberately absent: flash-attn importing is the signal worth
    having and it does not need a device, so this stays on CPU.
    """
    import json
    import subprocess
    import sys

    report: dict = {"container_python": sys.version.split()[0]}

    probe_src = (
        "import json,sys\n"
        "out={'upstream_python': sys.version.split()[0]}\n"
        "for m in ('torch','transformers','peft','deepspeed','flash_attn','cv2','UGround'):\n"
        "    try:\n"
        "        mod=__import__(m)\n"
        "        out[m]=getattr(mod,'__version__','imported')\n"
        "    except Exception as e:\n"
        "        out[m]='FAILED: %s: %s' % (type(e).__name__, e)\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run([UGROUND_PY, "-c", probe_src], capture_output=True, text=True, cwd=CODE)
    if proc.returncode == 0 and proc.stdout.strip():
        report.update(json.loads(proc.stdout.strip().splitlines()[-1]))
    else:
        report["upstream_env"] = f"FAILED rc={proc.returncode}: {proc.stderr[-800:]}"

    commit = subprocess.run(
        ["git", "-C", CODE, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    report["upstream_commit"] = commit.stdout.strip()
    report["commit_matches_pin"] = commit.stdout.strip() == UPSTREAM_COMMIT

    parse_check = f"import ast; ast.parse(open('{CODE}/train_ds.py').read())"
    entrypoint = subprocess.run([UGROUND_PY, "-c", parse_check], capture_output=True, text=True)
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
