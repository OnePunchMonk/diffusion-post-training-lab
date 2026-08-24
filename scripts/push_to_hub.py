"""Publish a checkpoint to the Hugging Face Hub instead of standing up a
persistent Modal endpoint.

Publishing to the Hub keeps checkpoints loadable with
`hub.load_lora_weights(repo_id)` without requiring access to a running Modal
account. Every push writes a model card generated from the checkpoint's
`run_manifest.json` (so base model, recipe, and hyperparameters are always
documented) and, if an eval report exists for that checkpoint, embeds the
benchmark numbers directly in the card.

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
  - lora
  - text-to-image
  - {recipe}
  - dptlab
---

# {repo_name}

Post-trained with [`dptlab`](https://github.com/OnePunchMonk/diffusion-post-training-lab)
using the **{recipe}** recipe on top of `{base_model_id}`.

## Training config

```json
{config_json}
```

## Benchmark

{benchmark_section}

## Usage

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("{base_model_id}", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}")

image = pipe(prompt="your prompt here").images[0]
```
"""

_LICENSE_BY_MODEL_KEY = {
    "sdxl": "openrail++",
    "flux-schnell": "apache-2.0",
    "flux-dev": "other",  # FLUX.1-dev non-commercial license — flag explicitly
}


def build_model_card(manifest: dict, repo_id: str, eval_report: dict | None) -> str:
    config = manifest["config"]
    model_key = config["model_key"]

    if eval_report:
        benchmark_section = (
            f"| CLIP score | Aesthetic score | Win rate vs. base | Avg. latency (ms) |\n"
            f"|---|---|---|---|\n"
            f"| {eval_report['clip_score']:.3f} | {eval_report['aesthetic_score']:.3f} | "
            f"{_fmt_winrate(eval_report.get('win_rate_vs_baseline'))} | {eval_report['avg_latency_ms']:.0f} |\n\n"
            f"See `MODELS.md` in the repo for the full leaderboard across checkpoints."
        )
    else:
        benchmark_section = "_Not yet benchmarked — run `dptlab eval` and re-push to fill this in._"

    return _MODEL_CARD_TEMPLATE.format(
        license=_LICENSE_BY_MODEL_KEY.get(model_key, "other"),
        base_model_id=_base_model_id(model_key),
        recipe=config["recipe"],
        repo_name=repo_id.split("/")[-1],
        config_json=json.dumps(config, indent=2),
        benchmark_section=benchmark_section,
        repo_id=repo_id,
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
    args = ap.parse_args()

    from huggingface_hub import HfApi

    manifest = load_run_manifest(args.checkpoint)
    eval_report = json.loads(Path(args.eval_report).read_text()) if args.eval_report else None
    card_text = build_model_card(manifest, args.repo_id, eval_report)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(Path(args.checkpoint) / "lora_weights.safetensors"),
        path_in_repo="lora_weights.safetensors",
        repo_id=args.repo_id,
    )
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
