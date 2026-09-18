"""Merge a PEFT adapter into the base transformer and export a servable model.

**Why this exists.** Of the six methods in the sweep, exactly one -- plain LoRA
-- produces a checkpoint any inference server knows how to read. LoHa, LoKr,
OFT, BOFT and (in most servers) DoRA are unknown formats: SGLang Diffusion and
vLLM-Omni both expect either base weights or a LoRA-shaped delta, so pointing
them at an `adapter_model.safetensors` from one of those methods fails to load
or, worse, loads nothing and silently serves the base model.

Merging sidesteps the whole problem. Every PEFT method defines how its update
folds into `W`, so after merging there is no adapter left -- just a normal
FLUX.2 transformer with different weights, which every backend can serve. The
costs are real and worth stating: the export is full-size (a ~4B transformer,
not a few MB of adapter), and one merged model serves one subject, so you lose
the multi-adapter hot-swapping that makes LoRA cheap to serve at scale. For a
benchmark that's the right trade; for production with many subjects it is the
argument for LoRA that the score table alone won't show you.

  python flux2-klein-peft/scripts/merge_and_export.py \
      --checkpoint runs/klein-peft/oft/subject-0/ckpt/final \
      --out exports/klein-oft-subject-0

Writes a `diffusers`-layout directory (transformer + VAE + text encoder +
scheduler) plus `export_manifest.json`. See `dptlab.serve.backends` for the
serve commands that consume it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge_adapter_into(denoiser) -> int:
    """Fold every injected adapter layer into its base weight, in place.

    `peft`'s usual `merge_and_unload()` lives on `PeftModel`, but diffusers
    injects adapters directly into the transformer via `inject_adapter_in_model`
    -- there is no PeftModel wrapper to call it on. Every tuner layer still
    implements the same `merge()` contract, so walk the module tree and call it.

    `safe_merge=True` computes the merged weight and checks it for NaN/Inf
    before writing it back. That matters most for the orthogonal methods: OFT
    and BOFT build their update through a Cayley transform, and a poorly
    conditioned block produces a silently corrupted weight that would otherwise
    only surface as noise in generated images.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    merged = 0
    for module in denoiser.modules():
        if isinstance(module, BaseTunerLayer):
            module.merge(safe_merge=True)
            merged += 1
    return merged


def strip_adapter_layers(denoiser) -> None:
    """Replace each merged tuner layer with its plain base module.

    After `merge()` the adapter's own parameters are still attached and still
    serialized, so a saved checkpoint would carry dead `lora_A`/`oft_r`/...
    tensors that a loader may try to interpret. Swapping in `base_layer` leaves
    an ordinary `nn.Linear` tree that saves and loads like a stock model.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in list(denoiser.named_modules()):
        if not isinstance(module, BaseTunerLayer):
            continue
        parent_path, _, attr = name.rpartition(".")
        parent = denoiser.get_submodule(parent_path) if parent_path else denoiser
        setattr(parent, attr, module.get_base_layer())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="A dptlab checkpoint dir (has run_manifest.json).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--transformer-only",
        action="store_true",
        help="Export just the transformer. Smaller, but the serving backend must then be told where to "
        "get the VAE and text encoder from -- most are happier with a full pipeline directory.",
    )
    args = ap.parse_args()

    import torch

    from dptlab.models.registry import load_pipeline
    from dptlab.training.common import load_run_manifest
    from dptlab.training.peft_methods import load_peft_checkpoint

    manifest = load_run_manifest(args.checkpoint)
    model_key = manifest["config"]["model_key"]
    method = manifest.get("peft_method") or manifest["config"].get("peft_method", "lora")

    logger.info("Loading base %s to merge a %s adapter", model_key, method)
    pipe = load_pipeline(model_key, dtype=args.dtype, device="cpu")
    denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer

    load_peft_checkpoint(pipe, denoiser, Path(args.checkpoint), method, for_training=False)

    merged = merge_adapter_into(denoiser)
    strip_adapter_layers(denoiser)
    logger.info("Merged %d adapter layers", merged)
    if merged == 0:
        raise SystemExit(
            f"No adapter layers found after loading {args.checkpoint}. The checkpoint loaded nothing -- "
            "check that run_manifest.json's peft_method matches the weights on disk."
        )

    out = Path(args.out)
    if args.transformer_only:
        denoiser.save_pretrained(out / "transformer", safe_serialization=True)
    else:
        pipe.save_pretrained(out, safe_serialization=True)

    (out / "export_manifest.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(args.checkpoint),
                "base_model": model_key,
                "peft_method": method,
                "merged_layers": merged,
                "dtype": args.dtype,
                "trainable_parameters_before_merge": manifest.get("trainable_parameters"),
                "transformer_only": args.transformer_only,
                "torch_version": torch.__version__,
            },
            indent=2,
        )
    )
    logger.info("Exported servable model to %s", out)


if __name__ == "__main__":
    main()
