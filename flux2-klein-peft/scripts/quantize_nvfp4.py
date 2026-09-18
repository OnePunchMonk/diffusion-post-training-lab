"""Quantize a merged export to NVFP4 (or FP8) and re-benchmark it.

NVFP4 is NVIDIA's 4-bit float microscaling format: weights are grouped into
16-element blocks, each block carrying its own FP8 scale, with a second
per-tensor scale on top. Blackwell executes it natively, which is where the
reported ~1.7x end-to-end speedups on FLUX come from -- on older hardware the
format still *stores* fine but dequantizes in software, so you get the memory
saving and none of the speed. This script therefore checks compute capability
and says which of the two you are getting rather than quietly benchmarking a
dequantization path.

**Order matters: quantize after merging, not before.** A PEFT adapter trains a
delta against the base weights it was fitted to; quantizing the base first
changes those weights underneath the adapter, and quantizing an adapter's own
low-rank factors wrecks them (they are small, high-dynamic-range, and their
product is what has to stay accurate). `flux2-klein-peft/scripts/merge_and_export.py` folds the
adapter in first, which also means this works identically for all six methods
-- by this point there is no adapter left, just a transformer.

  python flux2-klein-peft/scripts/merge_and_export.py --checkpoint .../final --out exports/klein-oft-s0
  python scripts/quantize_nvfp4.py --model exports/klein-oft-s0 --out exports/klein-oft-s0-nvfp4

The quality question this answers -- *does a 4-bit subject adapter still hold
the subject?* -- is the one that decides whether the sweep's winner survives
deployment, so `--eval-subject` re-runs the same DINO / CLIP-T metrics on the
quantized model and prints the deltas against the bf16 export.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# quant_type strings are "<weight>_<activation>"; a bare format means weight-only.
QUANT_TYPES = {
    "nvfp4": "NVFP4",
    "nvfp4-a": "NVFP4_NVFP4",  # weights and activations
    "fp8": "FP8",
    "fp8-a": "FP8_FP8",
    "int8": "INT8",
}

# Blackwell (sm_100) is the first architecture with native NVFP4 tensor cores.
_NVFP4_MIN_CAPABILITY = (10, 0)


def describe_hardware(quant: str) -> str:
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device: quantization will run but cannot be benchmarked here"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if not quant.startswith("nvfp4"):
        return f"{name} (sm_{major}{minor})"
    if (major, minor) >= _NVFP4_MIN_CAPABILITY:
        return f"{name} (sm_{major}{minor}) — native NVFP4, expect both memory and latency wins"
    return (
        f"{name} (sm_{major}{minor}) — pre-Blackwell, so NVFP4 weights dequantize in software: "
        "you get the ~4x memory saving, not the speedup. Do not report latency from this GPU."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="A merged export from flux2-klein-peft/scripts/merge_and_export.py.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="nvfp4", choices=sorted(QUANT_TYPES))
    ap.add_argument(
        "--calibrate-prompts",
        default=None,
        help="Prompt file for activation-aware calibration. Required in spirit for the *_a (activation) "
        "variants -- without a forward loop their activation scales are guesses.",
    )
    ap.add_argument(
        "--eval-subject",
        default=None,
        help="A prepared subject dir; re-runs DINO/CLIP-T on the quantized model and prints deltas.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from diffusers import NVIDIAModelOptConfig

    from dptlab.models.registry import get_model_spec

    model_dir = Path(args.model)
    manifest = json.loads((model_dir / "export_manifest.json").read_text())
    spec = get_model_spec(manifest["base_model"])
    quant_type = QUANT_TYPES[args.quant]

    logger.info("Hardware: %s", describe_hardware(args.quant))

    if args.quant.endswith("-a") and not args.calibrate_prompts:
        logger.warning(
            "Activation quantization without --calibrate-prompts: scales come from weights alone and "
            "activation outliers will clip. Weight-only (--quant %s) is the safer default.",
            args.quant.removesuffix("-a"),
        )

    quant_config = NVIDIAModelOptConfig(
        quant_type=quant_type,
        weight_only=not args.quant.endswith("-a"),
        # The final output projection and the time/guidance embeddings are tiny
        # relative to the transformer blocks but sit on every token's path, so
        # quantizing them costs quality for almost no memory.
        modules_to_not_convert=["proj_out", "time_text_embed", "x_embedder", "context_embedder"],
        forward_loop=_build_calibration_loop(args.calibrate_prompts) if args.calibrate_prompts else None,
    )

    pipeline_module, pipeline_cls = spec.pipeline_cls.rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(pipeline_module), pipeline_cls)

    logger.info("Loading %s and quantizing the transformer to %s", model_dir, quant_type)
    transformer_cls = _transformer_class(cls)
    transformer = transformer_cls.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )

    out = Path(args.out)
    transformer.save_pretrained(out / "transformer", safe_serialization=True)
    _copy_sibling_components(model_dir, out)

    (out / "export_manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "quantization": quant_type,
                "quantization_backend": "nvidia-modelopt",
                "weight_only": not args.quant.endswith("-a"),
                "calibrated": bool(args.calibrate_prompts),
                "quantized_from": str(model_dir),
            },
            indent=2,
        )
    )
    logger.info("Wrote quantized export to %s", out)

    if args.eval_subject:
        _compare(model_dir, out, Path(args.eval_subject), args.seed)


def _transformer_class(pipeline_cls):
    """The denoiser class this pipeline expects, read off its type annotations."""
    import inspect

    annotations = inspect.signature(pipeline_cls.__init__).parameters
    return annotations["transformer"].annotation


def _build_calibration_loop(prompts_path: str):
    """A forward loop modelopt runs to observe real activation ranges.

    Not implemented, and failing here rather than later is the point: modelopt
    hands the loop the bare transformer, but a meaningful calibration pass has
    to drive the whole pipeline (text encoder -> sampler -> transformer) so the
    activations it observes are the ones inference actually produces. Feeding
    it synthetic tensors would produce scales that look calibrated and are not.
    """
    raise NotImplementedError(
        f"Activation calibration from {prompts_path} needs the full pipeline in the loop, not the bare "
        "transformer. Use weight-only NVFP4 (the default) until that lands."
    )


def _copy_sibling_components(src: Path, dst: Path) -> None:
    """Carry the unquantized pipeline components over to the new export.

    Only the transformer is quantized: the VAE is small and its reconstruction
    error is not something you want compounded, and the text encoder is shared
    across every subject so quantizing it per-export saves nothing.
    """
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name in {"transformer", "export_manifest.json"}:
            continue
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _compare(bf16_dir: Path, quant_dir: Path, subject_dir: Path, seed: int) -> None:
    """Same prompts, same seeds, bf16 vs quantized."""
    import json as _json

    from PIL import Image

    from dptlab.eval.metrics.clip_score import CLIPScorer
    from dptlab.eval.metrics.subject_fidelity import DINOScorer

    rows = [_json.loads(x) for x in (subject_dir / "prompts_eval.jsonl").read_text().splitlines() if x.strip()]
    prompts = [r["prompt"] for r in rows]
    train = [_json.loads(x) for x in (subject_dir / "metadata.jsonl").read_text().splitlines() if x.strip()]
    references = [Image.open(subject_dir / r["file_name"]).convert("RGB") for r in train]

    clip_scorer, dino_scorer = CLIPScorer(), DINOScorer()
    results = {}
    for label, path in (("bf16", bf16_dir), ("quantized", quant_dir)):
        images, latencies = _generate(path, prompts, seed)
        results[label] = {
            "clip_t": clip_scorer.compute(prompts, images).value,
            "dino": dino_scorer.compute(images, references).value,
            "avg_latency_ms": sum(latencies) / len(latencies),
        }

    print(f"\n{'metric':<18}{'bf16':>12}{'quantized':>12}{'delta':>12}")
    for metric in ("clip_t", "dino", "avg_latency_ms"):
        a, b = results["bf16"][metric], results["quantized"][metric]
        print(f"{metric:<18}{a:>12.3f}{b:>12.3f}{b - a:>+12.3f}")


def _generate(model_dir: Path, prompts: list[str], seed: int):
    import torch
    from diffusers import DiffusionPipeline

    from dptlab.serve.backends import generation_defaults

    defaults = generation_defaults(model_dir)

    pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    import time

    images, latencies = [], []
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device=pipe.device).manual_seed(seed + i)
        started = time.perf_counter()
        image = pipe(prompt=prompt, generator=generator, **defaults).images[0]
        latencies.append((time.perf_counter() - started) * 1000)
        images.append(image.convert("RGB"))
    return images, latencies


if __name__ == "__main__":
    main()
