"""Shared training utilities used by all three recipes (lora / dpo / distill).

Keeping this separate is what makes the three recipes comparable: they read
the same YAML config shape, log to the same run-tracking convention, and save
checkpoints in the same layout that the eval harness and Modal server expect.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class TrainConfig:
    model_key: str
    recipe: str  # "lora" | "dpo" | "distill"
    dataset_path: str
    output_dir: str
    resolution: int = 1024
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 2000
    lora_rank: int = 16
    lora_alpha: int = 16
    mixed_precision: str = "bf16"
    seed: int = 42
    checkpointing_steps: int = 500
    validation_prompts: list[str] = dataclasses.field(default_factory=list)
    validation_steps: int = 500
    extra: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        extra = {k: v for k, v in raw.items() if k not in known}
        base = {k: v for k, v in raw.items() if k in known}
        return cls(**base, extra=extra)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_manifest(output_dir: str | Path, config: TrainConfig, extra: dict | None = None) -> None:
    """Write run_manifest.json into the checkpoint dir.

    The eval harness and the Modal server both read this to know which base
    model + recipe a checkpoint came from, so a checkpoint directory is
    self-describing without needing a separate DB.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": dataclasses.asdict(config), **(extra or {})}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def load_run_manifest(checkpoint_dir: str | Path) -> dict:
    return json.loads((Path(checkpoint_dir) / "run_manifest.json").read_text())


def load_frozen_pipe(model_key: str, accelerator):
    """Load a pipeline in bf16 and move it to the accelerator's device.

    A10G-class GPUs (22-24GB) can't fit an SDXL UNet + VAE + two text
    encoders in fp32 (~15GB of weights alone) plus 1024px training
    activations. Frozen components never need fp32 precision, so they're
    loaded directly in bf16; LoRA parameters get upcast to fp32 separately
    (see `add_lora_adapter`) since low-rank adapters are small and benefit
    from full-precision optimizer updates. VAE slicing/tiling further caps
    activation memory during encode/decode at minimal quality cost.
    """
    from dptlab.models.registry import load_pipeline

    pipe = load_pipeline(model_key, dtype="bfloat16", device="cpu")
    pipe.to(accelerator.device)
    pipe.vae.requires_grad_(False)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.requires_grad_(False)
    return pipe


def add_lora_adapter(denoiser, lora_config) -> None:
    """Inject a fresh LoRA adapter and prep it for training (see
    `_finalize_trainable_adapter`)."""
    denoiser.requires_grad_(False)
    denoiser.add_adapter(lora_config)
    _finalize_trainable_adapter(denoiser)


def _finalize_trainable_adapter(denoiser) -> None:
    """Enable gradient checkpointing (trades compute for activation memory)
    and upcast the adapter's params to fp32 for stable optimizer updates on
    top of bf16 frozen weights. Shared by both the fresh-adapter path
    (`add_lora_adapter`) and the resume path (`load_lora_checkpoint`)."""
    import torch

    denoiser.enable_gradient_checkpointing()
    for param in denoiser.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)


def save_lora_checkpoint(pipe, denoiser, out_dir: str | Path, filename: str = "lora_weights.safetensors") -> None:
    """Save a LoRA adapter in the format `pipe.load_lora_weights()` actually
    reads.

    peft's own `get_peft_model_state_dict` produces keys like
    "base_model.model.<...>.lora_A.default.weight" -- diffusers' loader
    doesn't recognize that naming and silently loads nothing (logs "No LoRA
    keys associated ... found", not an error), so a checkpoint saved with
    plain `safetensors.save_file(get_peft_model_state_dict(...))` looks
    valid on disk but the base model runs unmodified at inference time.
    `convert_state_dict_to_diffusers` + the pipeline class's own
    `save_lora_weights` (the same call diffusers' official training scripts
    use) produces the format the loader expects.
    """
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    # get_peft_model_state_dict defaults to adapter_name="default", but a
    # denoiser resumed via pipe.load_lora_weights() (see load_lora_checkpoint)
    # gets whatever adapter name diffusers' loader assigned, not necessarily
    # "default" -- so look up the actual name instead of assuming it.
    adapter_name = next(iter(denoiser.peft_config))
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(denoiser, adapter_name=adapter_name))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"save_directory": str(out_dir), "safe_serialization": True, "weight_name": filename}
    if hasattr(pipe, "unet"):
        type(pipe).save_lora_weights(unet_lora_layers=state_dict, **save_kwargs)
    else:
        type(pipe).save_lora_weights(transformer_lora_layers=state_dict, **save_kwargs)


def load_lora_checkpoint(pipe, denoiser, weights_path: str | Path) -> None:
    """Resume from a checkpoint saved by `save_lora_checkpoint`.

    Uses `pipe.load_lora_weights()` (diffusers' own loader, which injects
    matching adapter layers straight from the checkpoint) rather than
    manually adding a fresh adapter and round-tripping through peft's
    get/set_peft_model_state_dict -- their key-naming convention doesn't
    match what diffusers' save/load path produces
    (`ValueError: Could not automatically infer state dict type`), so call
    this *instead of* `add_lora_adapter`, not after it.
    """
    pipe.load_lora_weights(str(weights_path))
    _finalize_trainable_adapter(denoiser)


def encode_conditioning(pipe, captions: list[str], resolution: int) -> tuple:
    """Text conditioning for the denoiser forward pass, model-agnostic.

    SDXL's UNet is conditioned on more than the text sequence embeddings: it
    also needs the pooled text embedding and a set of "add time ids"
    (original/crop/target size) via `added_cond_kwargs`, or every forward
    pass raises `TypeError: argument of type 'NoneType' is not iterable`
    inside `get_aug_embed`. FLUX's transformer takes only the sequence
    embeddings, so `added_cond_kwargs` is empty there. Returns
    (encoder_hidden_states, added_cond_kwargs).
    """
    if hasattr(pipe, "unet"):  # SDXL
        prompt_embeds, _neg_embeds, pooled_prompt_embeds, _neg_pooled = pipe.encode_prompt(
            prompt=captions, device=pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        add_time_ids = pipe._get_add_time_ids(
            (resolution, resolution),
            (0, 0),
            (resolution, resolution),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(pipe.device)
        add_time_ids = add_time_ids.repeat(prompt_embeds.shape[0], 1)
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        return prompt_embeds, added_cond_kwargs

    if hasattr(pipe, "encode_prompt"):  # FLUX and other single-conditioning pipelines
        prompt_embeds, *_ = pipe.encode_prompt(prompt=captions, prompt_2=captions, device=pipe.device)
        return prompt_embeds, {}

    raise NotImplementedError("Pipeline does not expose encode_prompt; add a model-specific encoder here.")
