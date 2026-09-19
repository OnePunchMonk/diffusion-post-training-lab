"""Publish a checkpoint to the Hugging Face Hub instead of standing up a
persistent Modal endpoint.

Publishing to the Hub keeps checkpoints loadable without access to a running
Modal account. Every push writes a model card generated from the checkpoint's
`run_manifest.json` (so base model, recipe, PEFT method, and hyperparameters
are always documented) and, if an eval report exists, embeds the benchmark
numbers directly in the card.

The usage snippet in the card is method-aware, and that matters: only a plain
LoRA checkpoint loads with `pipe.load_lora_weights(repo_id)`. A LoHa / LoKr /
OFT / BOFT / DoRA adapter needs peft to inject matching layers first, and
diffusers' loader given one of those logs "no LoRA keys found" and serves the
base model -- a card promising `load_lora_weights` would send every downloader
straight into that trap.

Usage:
  python scripts/push_to_hub.py --checkpoint outputs/dpo-sdxl/final \
      --repo-id OnePunchMonk/dptlab-sdxl-dpo-v1 \
      --eval-report eval_results/dpo-sdxl/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dptlab.training.common import load_run_manifest

_MODEL_CARD_TEMPLATE = """---
license: {license}
base_model: {base_model_id}
tags:
  - diffusers
  - text-to-image
  - {recipe}
  - {peft_method}
  - dptlab
---

# {repo_name}

Post-trained with [`dptlab`](https://github.com/OnePunchMonk/diffusion-post-training-lab)
using the **{recipe}** recipe with a **{peft_method}** adapter, on top of
`{base_model_id}`.

{trainable_line}

## Training config

```json
{config_json}
```

## Benchmark

{benchmark_section}

## Usage

```python
{usage_snippet}
```
"""

_LORA_USAGE = """import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("{base_model_id}", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}")

image = pipe(prompt="your prompt here", num_inference_steps={steps}, guidance_scale={guidance}).images[0]"""

# Non-LoRA adapters are not a format diffusers' loader understands. peft has to
# inject the matching layer type first, which is what load_peft_checkpoint does.
_PEFT_USAGE = """import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

# `{peft_method}` is not a format `pipe.load_lora_weights()` can read -- it would
# log "no LoRA keys found" and silently serve the base model. Inject via peft.
from dptlab.training.peft_methods import load_peft_checkpoint  # pip install dptlab

pipe = DiffusionPipeline.from_pretrained("{base_model_id}", torch_dtype=torch.bfloat16).to("cuda")
denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer
load_peft_checkpoint(pipe, denoiser, snapshot_download("{repo_id}"), "{peft_method}", for_training=False)

image = pipe(prompt="your prompt here", num_inference_steps={steps}, guidance_scale={guidance}).images[0]"""

_LICENSE_BY_MODEL_KEY = {
    "sdxl": "openrail++",
    "flux-schnell": "apache-2.0",
    "flux-dev": "other",  # FLUX.1-dev non-commercial license — flag explicitly
    "flux2-klein-4b": "apache-2.0",
    "flux2-klein-9b": "apache-2.0",
}


def build_model_card(manifest: dict, repo_id: str, eval_report: dict | None) -> str:
    from dptlab.models.registry import get_model_spec
    from dptlab.training.peft_methods import get_method_spec

    config = manifest["config"]
    model_key = config["model_key"]
    peft_method = manifest.get("peft_method") or config.get("peft_method", "lora")
    spec = get_model_spec(model_key)

    benchmark_section = _benchmark_section(eval_report)

    trainable = manifest.get("trainable_parameters")
    trainable_line = (
        f"Trainable parameters: **{trainable / 1e6:.1f}M**." if trainable else ""
    )

    usage_template = _LORA_USAGE if get_method_spec(peft_method).diffusers_native else _PEFT_USAGE
    usage_snippet = usage_template.format(
        base_model_id=_base_model_id(model_key),
        repo_id=repo_id,
        peft_method=peft_method,
        steps=spec.default_inference_steps,
        guidance=spec.default_guidance_scale,
    )

    return _MODEL_CARD_TEMPLATE.format(
        license=_LICENSE_BY_MODEL_KEY.get(model_key, "other"),
        base_model_id=_base_model_id(model_key),
        recipe=config["recipe"],
        peft_method=peft_method,
        trainable_line=trainable_line,
        repo_name=repo_id.split("/")[-1],
        config_json=json.dumps(config, indent=2),
        benchmark_section=benchmark_section,
        usage_snippet=usage_snippet,
        repo_id=repo_id,
    )


def _benchmark_section(eval_report: dict | None) -> str:
    """Render whichever eval shape the caller has.

    Two exist: the original `dptlab eval` report (CLIP + aesthetic + win rate)
    and the PEFT sweep's per-cell result (CLIP-T + DINO + CLIP-I, split-aware).
    A card that silently omitted the numbers because the keys were unfamiliar
    would be worse than one that says which shape it got.
    """
    if not eval_report:
        return "_Not yet benchmarked — run `dptlab eval` and re-push to fill this in._"

    if "dino" in eval_report:  # PEFT sweep result
        split = eval_report.get("split", "unspecified")
        note = {
            "heldout": "Prompts describe settings no training image shows.",
            "insample": "SynCD's own generation prompts — these describe the **training** "
            "scenes, so this is a reconstruction measure, not generalization.",
        }.get(split, "")
        return (
            f"Split: **{split}** ({eval_report.get('n_prompts', '?')} prompts). {note}\n\n"
            f"| CLIP-T (prompt) | DINO (subject) | CLIP-I (subject) | Avg. latency (ms) |\n"
            f"|---|---|---|---|\n"
            f"| {eval_report['clip_t']:.4f} | {eval_report['dino']:.4f} | "
            f"{eval_report['clip_i']:.4f} | {eval_report['avg_latency_ms']:.0f} |\n\n"
            "CLIP-T and DINO pull in opposite directions: an adapter that learned nothing "
            "scores well on the first, one that memorized its training shots scores well on "
            "the second. Read them together.\n\n"
            "See `RESULTS.md` in the repo for all six methods and the confounds."
        )

    return (
        f"| CLIP score | Aesthetic score | Win rate vs. base | Avg. latency (ms) |\n"
        f"|---|---|---|---|\n"
        f"| {eval_report['clip_score']:.3f} | {eval_report['aesthetic_score']:.3f} | "
        f"{_fmt_winrate(eval_report.get('win_rate_vs_baseline'))} | "
        f"{eval_report['avg_latency_ms']:.0f} |\n\n"
        "See `MODELS.md` in the repo for the full leaderboard across checkpoints."
    )


def _base_model_id(model_key: str) -> str:
    from dptlab.models.registry import get_model_spec

    return get_model_spec(model_key).pretrained_id


def _fmt_winrate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (has run_manifest.json).")
    ap.add_argument("--repo-id", required=True, help="e.g. OnePunchMonk/dptlab-sdxl-dpo-v1")
    ap.add_argument("--eval-report", help="Path to an eval_results/.../report.json to embed in the model card.")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--caveat",
        default=None,
        help="A warning to put near the top of the card, e.g. a known defect in this checkpoint.",
    )
    args = ap.parse_args()

    from huggingface_hub import HfApi

    manifest = load_run_manifest(args.checkpoint)
    eval_report = json.loads(Path(args.eval_report).read_text()) if args.eval_report else None
    card_text = build_model_card(manifest, args.repo_id, eval_report)
    if args.caveat:
        # Above the fold: a caveat below the benchmark table is a caveat nobody reads.
        marker = f"# {args.repo_id.split('/')[-1]}"
        card_text = card_text.replace(marker, f"{marker}\n\n> ⚠️ **{args.caveat}**", 1)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Upload whichever weight files this method actually produced, plus the
    # adapter_config.json that peft needs to rebuild the right layer type.
    for name in ("lora_weights.safetensors", "adapter_model.safetensors", "adapter_config.json"):
        path = Path(args.checkpoint) / name
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=args.repo_id)

    api.upload_file(
        path_or_fileobj=str(Path(args.checkpoint) / "run_manifest.json"),
        path_in_repo="run_manifest.json",
        repo_id=args.repo_id,
    )
    api.upload_file(path_or_fileobj=card_text.encode(), path_in_repo="README.md", repo_id=args.repo_id)

    print(f"Pushed to https://huggingface.co/{args.repo_id}")
    print("Now run: python scripts/update_models_md.py "
          f"--repo-id {args.repo_id} --checkpoint {args.checkpoint} "
          f"{'--eval-report ' + args.eval_report if args.eval_report else ''}")


if __name__ == "__main__":
    main()
