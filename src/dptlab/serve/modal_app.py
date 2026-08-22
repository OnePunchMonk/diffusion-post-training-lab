"""Modal deployment for dptlab checkpoints.

Scoping note (see README "vLLM vs Modal" section): vLLM serves LLMs and
some VLMs, not diffusion UNets/transformers, so there is no literal
"vLLM-serve SDXL". What we actually do:
  - Modal GPU containers do the diffusion sampling (this file).
  - vLLM serves an optional small LLM that rewrites terse user prompts into
    more detailed prompts before they hit the diffusion model (a real,
    common use of an LLM in a T2I pipeline) — see `PromptRewriter` below.

Deploy:
  modal deploy src/dptlab/serve/modal_app.py

The endpoint takes {"prompt": str, "checkpoint_tag": "base"|"lora"|"dpo"|
"grpo"|"distill", "num_inference_steps": int | None} and returns a PNG. The
distilled checkpoint is the one that actually gets deployed cheaply: it
defaults to 4 steps instead of 30, which is the whole point of Recipe C.
"""

from __future__ import annotations

import io

import modal

app = modal.App("dptlab-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "diffusers>=0.30", "transformers>=4.44", "accelerate", "peft", "safetensors", "pillow")
)

checkpoints_volume = modal.Volume.from_name("dptlab-checkpoints", create_if_missing=True)
CHECKPOINTS_ROOT = "/checkpoints"

# recipe -> default step count. This table is the deployment payoff of the
# distillation recipe: same GPU-second cost model, 6-12x fewer steps.
_DEFAULT_STEPS = {"base": 30, "lora": 30, "dpo": 30, "grpo": 30, "distill": 4}


@app.cls(image=image, gpu="A10G", volumes={CHECKPOINTS_ROOT: checkpoints_volume}, scaledown_window=120)
class DiffusionService:
    model_key: str = modal.parameter(default="sdxl")

    @modal.enter()
    def load(self):
        from dptlab.models.registry import load_pipeline

        self.pipe = load_pipeline(self.model_key, dtype="bfloat16", device="cuda")
        self._loaded_checkpoint_tag = "base"

    def _maybe_load_lora(self, checkpoint_tag: str):
        if checkpoint_tag == self._loaded_checkpoint_tag:
            return
        if checkpoint_tag == "base":
            self.pipe.unload_lora_weights()
        else:
            weights_path = f"{CHECKPOINTS_ROOT}/{self.model_key}-{checkpoint_tag}/final/lora_weights.safetensors"
            self.pipe.load_lora_weights(weights_path)
        self._loaded_checkpoint_tag = checkpoint_tag

    @modal.method()
    def generate(
        self,
        prompt: str,
        checkpoint_tag: str = "base",
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> bytes:
        import torch

        self._maybe_load_lora(checkpoint_tag)
        steps = num_inference_steps or _DEFAULT_STEPS.get(checkpoint_tag, 30)
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

        result = self.pipe(prompt=prompt, num_inference_steps=steps, generator=generator)
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def infer(payload: dict):
    from fastapi import Response

    service = DiffusionService(model_key=payload.get("model_key", "sdxl"))
    png_bytes = service.generate.remote(
        prompt=payload["prompt"],
        checkpoint_tag=payload.get("checkpoint_tag", "base"),
        num_inference_steps=payload.get("num_inference_steps"),
        seed=payload.get("seed"),
    )
    return Response(content=png_bytes, media_type="image/png")


# --- optional prompt-rewriting stage, actually served with vLLM ---

vllm_image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm>=0.5")

_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's short image prompt into a detailed, vivid text-to-image "
    "prompt. Keep the subject unchanged. Output only the rewritten prompt."
)


@app.cls(image=vllm_image, gpu="A10G", scaledown_window=120)
class PromptRewriter:
    """The one place vLLM actually fits this project: serving a small LLM
    (e.g. Qwen2.5-3B-Instruct) to expand terse prompts before diffusion
    sampling, using vLLM's PagedAttention batching for low-latency serving."""

    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16")

    @modal.method()
    def rewrite(self, prompt: str) -> str:
        from vllm import SamplingParams

        chat = [{"role": "system", "content": _REWRITE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        outputs = self.llm.chat(chat, SamplingParams(max_tokens=120, temperature=0.7))
        return outputs[0].outputs[0].text.strip()
