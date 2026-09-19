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


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[HF_SECRET], timeout=3600)
def eval_cell(method: str, subject: str, split: str = "heldout") -> dict:
    """Generate the subject's held-out prompts and score identity + prompt fidelity.

    Its own container for the same reason training cells get one: a klein
    pipeline plus CLIP plus DINO is several GB, and evaluating six methods in
    one process accumulates them until it OOMs.

    The two metrics point in opposite directions on purpose. CLIP-T rewards
    following the new context, which an adapter that learned nothing scores
    well on; DINO rewards reproducing *this* object, which an adapter that
    memorized the training shots scores well on. A method only wins if it
    moves both.
    """
    import json
    from pathlib import Path as P

    _setup()
    from PIL import Image

    from dptlab.eval.adapters.checkpoint import CheckpointAdapter
    from dptlab.eval.metrics.clip_score import CLIPScorer
    from dptlab.eval.metrics.subject_fidelity import CLIPImageScorer, DINOScorer

    subject_dir = P(DATA) / "syncd" / subject
    checkpoint = P(OUT) / "klein-peft" / method / subject / "final"
    out_dir = P(OUT) / "klein-peft" / method / subject / f"eval-{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = subject_dir / f"prompts_{split}.jsonl"
    rows = [json.loads(x) for x in prompt_file.read_text().splitlines() if x.strip()]
    prompts = [r["prompt"] for r in rows]
    ids = [r["id"] for r in rows]

    # References are the subject's own training images: the question DINO
    # answers is "is this the same object", not "is this a novel view".
    train = [json.loads(x) for x in (subject_dir / "metadata.jsonl").read_text().splitlines() if x.strip()]
    references = [Image.open(subject_dir / r["file_name"]).convert("RGB") for r in train]

    adapter = CheckpointAdapter(checkpoint_dir=str(checkpoint))
    images, latencies = [], []
    for i, prompt in enumerate(prompts):
        response = adapter.generate(prompt, seed=1234 + i)
        response.image.save(out_dir / f"{ids[i]}.png")
        images.append(response.image)
        latencies.append(response.latency_ms)

    manifest = json.loads((checkpoint / "run_manifest.json").read_text())
    result = {
        "method": method,
        "subject": subject,
        "split": split,
        "clip_t": round(CLIPScorer().compute(prompts, images).value, 4),
        "dino": round(DINOScorer().compute(images, references).value, 4),
        "clip_i": round(CLIPImageScorer().compute(images, references).value, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "n_prompts": len(prompts),
        "steps": manifest.get("final_step"),
        "trainable_parameters": manifest.get("trainable_parameters"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    outputs.commit()
    print(result)
    return result


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60 * 3)
def evaluate(methods: str = "", split: str = "heldout") -> list[dict]:
    """Score every trained cell, one container each, then print the table.

    `split` defaults to **heldout**. SynCD's own `prompts` are the prompts that
    *generated* the training images -- parallel arrays, not a split -- so
    scoring on them measures reconstruction. They remain available as
    `insample`, and the gap between the two is a per-method memorization
    measure worth reporting.
    """
    import json
    from pathlib import Path as P

    _setup()
    from dptlab.training.peft_methods import list_methods

    wanted = methods.split(",") if methods else list_methods()
    subjects = _subject_names()

    # Resumable, like the sweep: a container can drop (we lost one to a
    # heartbeat timeout), and re-scoring five finished cells to recover one is
    # five times the cost for no information.
    todo, results = [], []
    for m in wanted:
        for s in subjects:
            if not (P(OUT) / "klein-peft" / m / s / "final" / "run_manifest.json").exists():
                continue
            done_path = P(OUT) / "klein-peft" / m / s / f"eval-{split}" / "result.json"
            if done_path.exists():
                print(f"skip {m}/{s} ({split} already scored)")
                results.append(json.loads(done_path.read_text()))
            else:
                todo.append((m, s, split))

    print(f"{len(todo)} cells to evaluate on the {split} split, {len(results)} already done")
    if todo:
        results += list(eval_cell.starmap(todo))
    outputs.reload()

    print(json.dumps(results, indent=2, default=str))
    ok = [r for r in results if "clip_t" in r]
    ok.sort(key=lambda r: -r["dino"])
    print()
    header = f"{'method':<8}{'params':>12}{'steps':>7}{'CLIP-T':>9}{'DINO':>8}{'CLIP-I':>8}{'ms/img':>9}"
    print(header)
    print("-" * len(header))
    for r in ok:
        print(
            f"{r['method']:<8}{r['trainable_parameters']:>12,}{r['steps']:>7}"
            f"{r['clip_t']:>9.4f}{r['dino']:>8.4f}{r['clip_i']:>8.4f}{r['avg_latency_ms']:>9.0f}"
        )
    return results


@app.local_entrypoint()
def main() -> None:
    print("Entry points: prepare -> smoke -> sweep -> evaluate")
