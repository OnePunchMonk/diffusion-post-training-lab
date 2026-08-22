"""Model family registry.

SDXL and Flux have different pipeline classes, different conditioning (one vs.
two text encoders vs. T5+CLIP), and different LoRA target modules. Every
training recipe (LoRA / DPO / distill) and the eval harness go through this
registry instead of hardcoding a pipeline class, so adding a third base model
is a matter of adding one ModelSpec here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    pretrained_id: str
    pipeline_cls: str  # dotted path, resolved lazily to avoid importing diffusers at import time
    lora_target_modules: tuple[str, ...]
    default_resolution: int
    supports_guidance: bool = True
    notes: str = ""


_REGISTRY: dict[str, ModelSpec] = {
    "sdxl": ModelSpec(
        key="sdxl",
        pretrained_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls="diffusers.StableDiffusionXLPipeline",
        lora_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
        default_resolution=1024,
        supports_guidance=True,
        notes="Two text encoders (CLIP-L + OpenCLIP-bigG). Cheapest to iterate on; use for LoRA + DPO first.",
    ),
    "flux-schnell": ModelSpec(
        key="flux-schnell",
        pretrained_id="black-forest-labs/FLUX.1-schnell",
        pipeline_cls="diffusers.FluxPipeline",
        lora_target_modules=("to_k", "to_q", "to_v", "to_out.0", "proj_mlp", "proj_out"),
        default_resolution=1024,
        supports_guidance=False,  # schnell is guidance-distilled already, 1-4 steps
        notes="Apache-2.0, guidance-distilled. Good baseline for the distillation recipe since it's already few-step.",
    ),
    "flux-dev": ModelSpec(
        key="flux-dev",
        pretrained_id="black-forest-labs/FLUX.1-dev",
        pipeline_cls="diffusers.FluxPipeline",
        lora_target_modules=("to_k", "to_q", "to_v", "to_out.0", "proj_mlp", "proj_out"),
        default_resolution=1024,
        supports_guidance=True,
        notes="Non-commercial license. Use for LoRA/DPO quality comparisons only, not the deployed demo.",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return _REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown model key {key!r}. Available: {sorted(_REGISTRY)}") from e


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def load_pipeline(key: str, dtype: str = "bfloat16", device: str = "auto", **kwargs):
    """Resolve a ModelSpec to a loaded diffusers pipeline instance."""
    import importlib

    import torch

    spec = get_model_spec(key)
    module_path, cls_name = spec.pipeline_cls.rsplit(".", 1)
    pipeline_cls = getattr(importlib.import_module(module_path), cls_name)

    resolved_dtype = getattr(torch, dtype, torch.float32)
    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline_cls.from_pretrained(spec.pretrained_id, torch_dtype=resolved_dtype, **kwargs)
    return pipe.to(resolved_device)
