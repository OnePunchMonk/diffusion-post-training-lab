"""Run the eval harness against a Modal-trained checkpoint on a Modal GPU.

Mirrors modal_train.py: no local GPU available, so evaluation (CLIP score +
aesthetic score, both needing a GPU to be fast) runs here too, reading the
checkpoint straight from the `dptlab-outputs` volume and writing
`report.json` back to it for `scripts/push_to_hub.py` /
`scripts/update_models_md.py` to pick up locally.

Usage:
  modal run src/dptlab/serve/modal_eval.py --checkpoint-tag final --model-key sdxl
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("dptlab-eval")

REPO_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path(__file__).resolve().parent

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
    )
    .add_local_dir(str(REPO_ROOT / "src" / "dptlab"), remote_path="/root/dptlab_src/dptlab")
    .add_local_dir(str(REPO_ROOT / "prompts"), remote_path="/root/prompts")
)

outputs_volume = modal.Volume.from_name("dptlab-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 15,
    volumes={"/root/outputs": outputs_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate(model_key: str, checkpoint_name: str, checkpoint_tag: str, prompts_path: str, num_inference_steps: int):
    import logging
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/dptlab_src")
    os.chdir("/root")

    from dptlab.eval.adapters.checkpoint import CheckpointAdapter
    from dptlab.eval.runner import load_prompts, run_eval

    checkpoint_dir = f"/root/outputs/{checkpoint_name}/{checkpoint_tag}"
    adapter = CheckpointAdapter(model_key=model_key, checkpoint_dir=checkpoint_dir)
    prompts = load_prompts(prompts_path)

    report = run_eval(
        adapter,
        prompts,
        num_inference_steps=num_inference_steps,
        output_dir=checkpoint_dir,
    )
    outputs_volume.commit()
    return report.__dict__


@app.local_entrypoint()
def main(
    model_key: str = "sdxl",
    checkpoint_name: str = "lora-sdxl",
    checkpoint_tag: str = "final",
    prompts_path: str = "/root/prompts/smoke_eval.jsonl",
    num_inference_steps: int = 10,
):
    report = evaluate.remote(model_key, checkpoint_name, checkpoint_tag, prompts_path, num_inference_steps)
    print("Eval report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
