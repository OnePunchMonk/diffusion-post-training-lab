"""Run dptlab training recipes on Modal GPUs.

This is the actual compute path for this project: local machines here don't
have a GPU, so every training run — including the first LoRA smoke test on
the toy concept dataset — happens on a Modal A10G container. Checkpoints
land on a persistent Modal volume so they survive between runs and can be
downloaded locally for `push_to_hub.py`.

Usage:
  modal run src/dptlab/serve/modal_train.py --recipe lora
  modal run src/dptlab/serve/modal_train.py --recipe dpo
  modal volume get dptlab-outputs lora-sdxl/final ./outputs/lora-sdxl/final
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("dptlab-train")

try:
    # src/dptlab/serve/modal_train.py -> repo root, three levels up.
    REPO_ROOT = Path(__file__).resolve().parents[3]
except IndexError:
    # Modal re-imports this module inside the remote container (a shallower
    # path there) just to hydrate the function definition; the image's
    # add_local_dir calls below have already been baked in by then, so this
    # value is unused remotely and only needs to not crash.
    REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "diffusers>=0.30",
        "transformers>=4.44",
        "accelerate>=0.33",
        "peft>=0.12",
        "safetensors>=0.4",
        "pyyaml>=6.0",
        "pillow>=10.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "huggingface-hub>=0.24",
        "torchvision",
        "sentencepiece",  # needed for FLUX's T5 tokenizer
    )
    .add_local_dir(str(REPO_ROOT / "src" / "dptlab"), remote_path="/root/dptlab_src/dptlab")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
    .add_local_dir(str(REPO_ROOT / "data"), remote_path="/root/data")
    .add_local_dir(str(REPO_ROOT / "prompts"), remote_path="/root/prompts")
)

outputs_volume = modal.Volume.from_name("dptlab-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("dptlab-hf-cache", create_if_missing=True)

_RECIPE_CONFIGS = {
    "lora": "configs/recipes/lora.yaml",
    "dpo": "configs/recipes/dpo.yaml",
    "grpo": "configs/recipes/grpo.yaml",
    "distill": "configs/recipes/distill.yaml",
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": outputs_volume, "/root/hf_cache": hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(recipe: str, max_train_steps: int | None = None):
    import logging
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/dptlab_src")
    os.chdir("/root")

    from dptlab.training.common import TrainConfig
    from dptlab.training.distill import train_distill
    from dptlab.training.dpo import train_dpo
    from dptlab.training.grpo import train_grpo
    from dptlab.training.lora import train_lora

    recipe_fns = {"lora": train_lora, "dpo": train_dpo, "grpo": train_grpo, "distill": train_distill}
    if recipe not in recipe_fns:
        raise ValueError(f"Unknown recipe {recipe!r}. Choose from {list(recipe_fns)}.")

    config = TrainConfig.from_yaml(_RECIPE_CONFIGS[recipe])
    config.output_dir = f"/root/outputs/{Path(config.output_dir).name}"
    if max_train_steps is not None:
        config.max_train_steps = max_train_steps

    output_dir = recipe_fns[recipe](config)
    outputs_volume.commit()
    print(f"Checkpoint written to volume dptlab-outputs at {output_dir}")
    return str(output_dir)


@app.local_entrypoint()
def main(recipe: str = "lora", max_train_steps: int | None = None):
    result = train.remote(recipe, max_train_steps)
    print(f"Done: {result}")
    print("Fetch it with: modal volume get dptlab-outputs "
          f"{Path(result).relative_to('/root/outputs')} ./outputs/{Path(result).relative_to('/root/outputs').parts[0]}")
