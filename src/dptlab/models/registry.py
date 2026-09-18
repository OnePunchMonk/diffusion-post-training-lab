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
    # Which denoising objective the training recipes use. SDXL is epsilon-
    # prediction over a DDPM schedule; the FLUX families are rectified flow,
    # and FLUX.2 additionally packs latents into a token sequence and carries
    # its own position ids, so the two can't share a training step.
    family: str = "sdxl"  # "sdxl" | "flux" | "flux2"
    # Square (d_in == d_out) projections only. OFT/BOFT learn an orthogonal
    # transform of the output space and need out_features divisible by the
    # block size, which the fused/MLP projections don't satisfy -- see
    # training/peft_methods.py.
    orthogonal_target_modules: tuple[str, ...] = ()
    default_inference_steps: int = 30
    default_guidance_scale: float = 7.0


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
        family="flux",
    ),
    "flux-dev": ModelSpec(
        key="flux-dev",
        pretrained_id="black-forest-labs/FLUX.1-dev",
        pipeline_cls="diffusers.FluxPipeline",
        lora_target_modules=("to_k", "to_q", "to_v", "to_out.0", "proj_mlp", "proj_out"),
        default_resolution=1024,
        supports_guidance=True,
        notes="Non-commercial license. Use for LoRA/DPO quality comparisons only, not the deployed demo.",
        family="flux",
    ),
    "flux2-klein-4b": ModelSpec(
        key="flux2-klein-4b",
        pretrained_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline_cls="diffusers.Flux2KleinPipeline",
        # FLUX.2 blocks: dual-stream blocks carry separate image (to_q/k/v,
        # to_out.0) and text (add_*_proj, to_add_out) projections; single-stream
        # blocks fuse qkv and the MLP input into one `to_qkv_mlp_proj`. Adapting
        # all of them is what the reference LoRA scripts do.
        lora_target_modules=(
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "to_qkv_mlp_proj",
            "proj_out",
        ),
        orthogonal_target_modules=("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"),
        default_resolution=1024,
        supports_guidance=False,  # step- and guidance-distilled: 4 steps at cfg 1.0
        family="flux2",
        default_inference_steps=4,
        default_guidance_scale=1.0,
        notes=(
            "Apache-2.0, ~13GB in bf16, single Qwen3 text encoder. The default target for the "
            "PEFT-method comparison: small enough that six methods fit one GPU-day."
        ),
    ),
    "flux2-klein-9b": ModelSpec(
        key="flux2-klein-9b",
        pretrained_id="black-forest-labs/FLUX.2-klein-9B",
        pipeline_cls="diffusers.Flux2KleinPipeline",
        lora_target_modules=(
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "to_qkv_mlp_proj",
            "proj_out",
        ),
        orthogonal_target_modules=("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"),
        default_resolution=1024,
        supports_guidance=False,
        family="flux2",
        default_inference_steps=4,
        default_guidance_scale=1.0,
        notes="Apache-2.0, ~29GB in bf16 (needs an A100/H100 or 4090 with offload). Quality ceiling for the 4B results.",
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
