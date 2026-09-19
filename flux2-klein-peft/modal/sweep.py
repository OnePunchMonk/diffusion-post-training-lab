"""Run the FLUX.2 [klein] PEFT sweep on Modal.

This is the compute path for the benchmark. Nothing in `flux2-klein-peft/` has
ever executed against real weights -- the flow-matching objective, the adapter
injection and the checkpoint round-trip are verified only by CPU tests and by
reading the diffusers/peft sources. So the entry points are ordered by cost,
and `smoke` exists to find out whether any of it works before six methods'
worth of GPU time is spent:

  modal run --detach modal/sweep.py::prepare   # CPU: one SynCD subject -> volume
  modal run --detach modal/sweep.py::smoke     # A100, ~10 steps, one method
  modal run --detach modal/sweep.py::sweep     # A100, six methods x 500 steps
  modal run --detach modal/sweep.py::evaluate  # A100, scores every checkpoint

A100-40GB rather than A10G: klein's transformer is ~8GB in bf16 and its Qwen3
text encoder is resident during training, so a 24GB card is tight enough that
an OOM twenty minutes into a run is likelier than the saving is worth.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("dptlab-klein-sweep")

try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:  # re-imported inside the container, where the mounts are baked
    REPO_ROOT = Path(__file__).resolve().parent

SUBJECTS = 1  # one subject; six methods on one subject is the cheapest real row
STEPS = 500

OUT = "/root/outputs"
DATA = "/root/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # ninja: without it peft's BOFT cannot build its CUDA extension, falls back
    # to a slow path, and *silently overrides* boft_n_butterfly_factor to 1 --
    # so the config would not mean what it says and BOFT would be benchmarked
    # at a different setting from the one recorded.
    .apt_install("git", "libgl1", "libglib2.0-0", "ninja-build")
    .pip_install("ninja")
    .pip_install(
        "torch",
        # Flux2KleinPipeline and AutoencoderKLFlux2 landed in 0.39.
        "diffusers>=0.39",
        "transformers>=4.44,<5",
        "accelerate>=0.33",
        # LoHa/LoKr/OFT/BOFT configs and their merge() implementations.
        "peft>=0.14",
        "safetensors>=0.4",
        "pyyaml>=6.0",
        "pillow>=10.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "huggingface-hub>=0.24",
        "datasets>=2.20",
        "torchvision",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(str(REPO_ROOT / "src" / "dptlab"), remote_path="/root/dptlab_src/dptlab")
    .add_local_dir(str(REPO_ROOT / "flux2-klein-peft" / "configs"), remote_path="/root/configs")
    .add_local_file(
        str(REPO_ROOT / "flux2-klein-peft" / "scripts" / "prepare_syncd.py"),
        "/root/prepare_syncd.py",
    )
)

outputs = modal.Volume.from_name("dptlab-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("dptlab-hf-cache", create_if_missing=True)
VOLUMES = {OUT: outputs, "/root/hf_cache": hf_cache, DATA: modal.Volume.from_name(
    "dptlab-klein-data", create_if_missing=True
)}
data_volume = VOLUMES[DATA]
HF_SECRET = modal.Secret.from_name("huggingface")


def _setup() -> None:
    import os
    import sys

    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/dptlab_src")
    os.chdir("/root")


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=3600)
def prepare(num_subjects: int = SUBJECTS) -> list[str]:
    """Slice SynCD onto the volume. CPU: ~1.8GB archive, no GPU rented for it."""
    import subprocess
    import sys
    from pathlib import Path as P

    _setup()
    root = P(DATA) / "syncd"
    if not (root / "subjects.json").exists():
        subprocess.run(
            [
                sys.executable, "/root/prepare_syncd.py",
                "--num-subjects", str(num_subjects),
                "--out", str(root),
                "--cache-dir", "/root/hf_cache",
            ],
            check=True,
        )
    data_volume.commit()
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"subjects: {subjects}")
    return subjects


def _train(method: str, subject: str, steps: int, tag: str = "klein-peft") -> dict:
    """One (method, subject) cell. Returns a small manifest.

    `tag` separates smoke output from real runs: the sweep skips a cell whose
    `final/run_manifest.json` exists, so a 10-step smoke written to the real
    path would silently be mistaken for a finished 500-step run.
    """
    import json
    import time
    from pathlib import Path as P

    from dptlab.training.common import TrainConfig
    from dptlab.training.lora import train_lora

    config = TrainConfig.from_yaml(f"/root/configs/{method}.yaml")
    config.dataset_path = f"{DATA}/syncd/{subject}"
    config.output_dir = f"{OUT}/{tag}/{method}/{subject}"
    config.max_train_steps = steps
    config.checkpointing_steps = max(steps, 1)

    started = time.perf_counter()
    checkpoint = train_lora(config)
    elapsed = time.perf_counter() - started

    manifest = json.loads((P(checkpoint) / "run_manifest.json").read_text())
    return {
        "method": method,
        "subject": subject,
        "checkpoint": str(checkpoint),
        "train_seconds": round(elapsed, 1),
        "trainable_parameters": manifest.get("trainable_parameters"),
        "steps": steps,
    }


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[HF_SECRET], timeout=3600)
def smoke(method: str = "lora", steps: int = 10) -> dict:
    """Ten steps of one method, to find out whether the path runs at all.

    Everything this exercises is unproven against real weights: the rectified
    flow objective, the VAE's batch-norm latent normalization, the packed-token
    layout, adapter injection, and the checkpoint round-trip. Ten steps costs
    cents and answers all of it.
    """
    _setup()
    subjects = _subject_names()
    result = _train(method, subjects[0], steps, tag="smoke")
    outputs.commit()
    print(result)
    return result


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[HF_SECRET], timeout=3600)
def train_cell(method: str, subject: str, steps: int, tag: str = "klein-peft") -> dict:
    """One cell, one container.

    Deliberately not a loop inside a single container. `load_frozen_pipe` puts
    a fresh ~8GB klein pipeline on the device per call and accelerate holds
    global state, so training several methods in one process accumulates
    pipelines until it OOMs -- which it did, after three methods, on a 40GB
    A100. The failure looked like "lokr, lora and oft are broken" and was
    entirely an artifact of the harness. One container per cell also lets the
    cells run in parallel for the same total GPU-seconds.
    """
    _setup()
    try:
        result = train_cell_body(method, subject, steps, tag)
    except Exception as exc:  # noqa: BLE001 - one bad cell must not sink the sweep
        result = {"method": method, "subject": subject, "FAILED": f"{type(exc).__name__}: {exc}"[:500]}
    outputs.commit()
    print(result)
    return result


def train_cell_body(method: str, subject: str, steps: int, tag: str) -> dict:
    return _train(method, subject, steps, tag=tag)


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[HF_SECRET], timeout=3600)
def smoke_all(steps: int = 10) -> list[dict]:
    """Ten steps of every method, before committing hours to the real sweep.

    LoRA working says nothing about the other five: the orthogonal methods take
    a different config field, a restricted target-module list, and -- like every
    non-LoRA method -- a peft-native checkpoint path rather than diffusers'.
    Each of those is a separate way to fail, and each costs cents to rule out.
    """
    import json

    _setup()
    from dptlab.training.peft_methods import list_methods

    subject = _subject_names()[0]
    results = []
    for method in list_methods():
        try:
            results.append(_train(method, subject, steps, tag="smoke"))
        except Exception as exc:  # noqa: BLE001 - reporting all six beats stopping at the first
            results.append({"method": method, "FAILED": f"{type(exc).__name__}: {exc}"[:400]})
        outputs.commit()

    print(json.dumps(results, indent=2, default=str))
    failed = [r for r in results if "FAILED" in r]
    print(f"\n{len(results) - len(failed)}/{len(results)} methods trained and saved")
    for r in failed:
        print(f"  {r['method']}: {r['FAILED']}")
    return results


def _subject_names() -> list[str]:
    from pathlib import Path as P

    root = P(DATA) / "syncd"
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not subjects:
        raise RuntimeError(f"No subjects under {root}. Run prepare first.")
    return subjects


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60 * 6)
def sweep(steps: int = STEPS, methods: str = "") -> list[dict]:
    """Fan every (method, subject) cell out to its own GPU container.

    This function itself takes no GPU: it only decides what to run and collects
    results. Cells whose `final/run_manifest.json` already exists are skipped,
    so a partial sweep resumes instead of paying twice.
    """
    import json
    from pathlib import Path as P

    _setup()
    from dptlab.training.peft_methods import list_methods

    wanted = methods.split(",") if methods else list_methods()
    subjects = _subject_names()

    todo, done = [], []
    for method in wanted:
        for subject in subjects:
            marker = P(OUT) / "klein-peft" / method / subject / "final" / "run_manifest.json"
            if marker.exists():
                print(f"skip {method}/{subject} (already trained)")
                done.append(json.loads(marker.read_text()))
            else:
                todo.append((method, subject, steps, "klein-peft"))

    print(f"{len(todo)} cells to train, {len(done)} already done")
    results = done + list(train_cell.starmap(todo)) if todo else done

    outputs.reload()
    print(json.dumps(results, indent=2, default=str))
    failed = [r for r in results if "FAILED" in r]
    print(f"\n{len(results) - len(failed)}/{len(results)} cells trained")
    for r in failed:
        print(f"  {r.get('method')}: {r['FAILED']}")
    return results


@app.local_entrypoint()
def main() -> None:
    print("Entry points: prepare -> smoke -> sweep -> evaluate")
