"""Adapter that evaluates our own post-trained checkpoints.

Extends vlm-harness's `DiffusersAdapter` idea (a diffusers pipeline wrapped
behind the T2IAdapter protocol) with awareness of dptlab's checkpoint layout:
a directory containing `run_manifest.json` (written by
`training.common.save_run_manifest`) and the adapter weights. This is what
makes `dptlab-eval run --checkpoint outputs/lora/final` work without the caller
needing to know which base model, recipe, or PEFT method produced that
checkpoint -- including the non-LoRA methods, whose weights a stock diffusers
pipeline cannot load on its own (see `training.peft_methods`).
"""

from __future__ import annotations

import time
from pathlib import Path

from dptlab.eval.adapters.base import T2IResponse
from dptlab.models.registry import get_model_spec, load_pipeline
from dptlab.training.common import load_run_manifest
from dptlab.training.peft_methods import load_peft_checkpoint


class CheckpointAdapter:
    """Loads a base model (optionally + a PEFT checkpoint) as a T2IAdapter."""

    def __init__(
        self,
        model_key: str | None = None,
        checkpoint_dir: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
    ):
        peft_method = "lora"
        if checkpoint_dir is not None:
            manifest = load_run_manifest(checkpoint_dir)
            model_key = model_key or manifest["config"]["model_key"]
            # `peft_method` is only in manifests written after multi-method
            # support landed; older LoRA checkpoints have no such key.
            peft_method = manifest.get("peft_method") or manifest["config"].get("peft_method", "lora")
            self.checkpoint_tag = f"{manifest['config']['recipe']}:{peft_method}"
        elif model_key is not None:
            self.checkpoint_tag = "base"
        else:
            raise ValueError("Must pass model_key (for the base model) or checkpoint_dir (for a fine-tuned run).")

        self._pipe = load_pipeline(model_key, dtype=dtype, device=device)
        self._model_key = model_key
        self._spec = get_model_spec(model_key)

        if checkpoint_dir is not None:
            denoiser = self._pipe.unet if hasattr(self._pipe, "unet") else self._pipe.transformer
            load_peft_checkpoint(self._pipe, denoiser, Path(checkpoint_dir), peft_method, for_training=False)

    @property
    def model_id(self) -> str:
        return f"{self._model_key}:{self.checkpoint_tag}"

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
    ) -> T2IResponse:
        """Generate one image.

        Sampling defaults come from the ModelSpec rather than being hardcoded:
        FLUX.2 [klein] is step- and guidance-distilled, so running it at SDXL's
        30 steps / cfg 7.0 produces washed-out garbage and takes 7x as long. An
        explicit argument still wins.
        """
        import torch

        width = width or self._spec.default_resolution
        height = height or self._spec.default_resolution
        guidance_scale = guidance_scale if guidance_scale is not None else self._spec.default_guidance_scale
        num_inference_steps = num_inference_steps or self._spec.default_inference_steps

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._pipe.device).manual_seed(seed)

        t0 = time.perf_counter()
        kwargs = {}
        # FLUX.2's pipeline has no `negative_prompt` argument (it uses "" as the
        # negative and exposes only `negative_prompt_embeds`), so passing it
        # unconditionally is a TypeError on every klein run.
        if negative_prompt is not None and self._spec.family != "flux2":
            kwargs["negative_prompt"] = negative_prompt

        result = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
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
