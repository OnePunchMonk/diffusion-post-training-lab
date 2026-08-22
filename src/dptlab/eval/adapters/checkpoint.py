"""Adapter that evaluates our own post-trained checkpoints.

Extends vlm-harness's `DiffusersAdapter` idea (a diffusers pipeline wrapped
behind the T2IAdapter protocol) with awareness of dptlab's checkpoint layout:
a directory containing `run_manifest.json` (written by
`training.common.save_run_manifest`) and `lora_weights.safetensors`. This is
what makes `dptlab-eval run --checkpoint outputs/lora/final` work without the
caller needing to know which base model or recipe produced that checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

from dptlab.eval.adapters.base import T2IResponse
from dptlab.models.registry import load_pipeline
from dptlab.training.common import load_run_manifest


class CheckpointAdapter:
    """Loads a base model (optionally + a LoRA checkpoint) as a T2IAdapter."""

    def __init__(
        self,
        model_key: str | None = None,
        checkpoint_dir: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
    ):
        if checkpoint_dir is not None:
            manifest = load_run_manifest(checkpoint_dir)
            model_key = model_key or manifest["config"]["model_key"]
            self.checkpoint_tag = manifest["config"]["recipe"]
        elif model_key is not None:
            self.checkpoint_tag = "base"
        else:
            raise ValueError("Must pass model_key (for the base model) or checkpoint_dir (for a fine-tuned run).")

        self._pipe = load_pipeline(model_key, dtype=dtype, device=device)
        self._model_key = model_key

        if checkpoint_dir is not None:
            weights_path = Path(checkpoint_dir) / "lora_weights.safetensors"
            self._pipe.load_lora_weights(str(weights_path))

    @property
    def model_id(self) -> str:
        return f"{self._model_key}:{self.checkpoint_tag}"

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
    ) -> T2IResponse:
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._pipe.device).manual_seed(seed)

        t0 = time.perf_counter()
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return T2IResponse(
            image=result.images[0].convert("RGB"),
            latency_ms=latency_ms,
            model_id=self.model_id,
            checkpoint_tag=self.checkpoint_tag,
            seed=seed,
            num_inference_steps=num_inference_steps,
        )
