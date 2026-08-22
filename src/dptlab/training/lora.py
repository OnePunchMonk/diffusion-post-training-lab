"""Recipe A: LoRA fine-tuning on a custom concept/style dataset.

Standard diffusers-style LoRA training loop: freeze the UNet/transformer,
inject LoRA adapters into the attention projections defined by the model's
ModelSpec, and train on (image, caption) pairs with the usual epsilon/
v-prediction denoising loss. This is the cheapest recipe and the one that
proves the end-to-end pipeline (data -> train -> checkpoint -> eval -> serve)
before the pricier DPO and distillation recipes reuse the same scaffolding.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from dptlab.data.dataset import ImageCaptionDataset
from dptlab.models.registry import get_model_spec
from dptlab.training.common import (
    TrainConfig,
    add_lora_adapter,
    encode_conditioning,
    load_frozen_pipe,
    save_run_manifest,
    set_seed,
)

logger = logging.getLogger(__name__)


def train_lora(config: TrainConfig) -> Path:
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from peft import LoraConfig
    from torch.utils.data import DataLoader

    set_seed(config.seed)
    spec = get_model_spec(config.model_key)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    pipe = load_frozen_pipe(config.model_key, accelerator)
    denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(spec.lora_target_modules),
        init_lora_weights="gaussian",
    )
    add_lora_adapter(denoiser, lora_config)

    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    dataset = ImageCaptionDataset(config.dataset_path, resolution=config.resolution)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    denoiser, optimizer, dataloader = accelerator.prepare(denoiser, optimizer, dataloader)

    global_step = 0
    max_epochs = math.ceil(config.max_train_steps / max(1, len(dataloader)))

    for _epoch in range(max_epochs):
        for batch in dataloader:
            with accelerator.accumulate(denoiser):
                pixel_values = batch["pixel_values"].to(dtype=pipe.vae.dtype)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states, added_cond_kwargs = encode_conditioning(
                    pipe, batch["caption"], config.resolution
                )
                model_pred = denoiser(
                    noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                ).sample

                target = noise if pipe.scheduler.config.prediction_type == "epsilon" else latents
                loss = F.mse_loss(model_pred.float(), target.float())

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0:
                    logger.info("step=%d loss=%.4f", global_step, loss.item())
                if global_step % config.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, denoiser, config, global_step)
                if global_step >= config.max_train_steps:
                    break
        if global_step >= config.max_train_steps:
            break

    output_dir = _save_checkpoint(accelerator, denoiser, config, global_step, final=True)
    save_run_manifest(output_dir, config, extra={"final_step": global_step})
    return output_dir


def _save_checkpoint(accelerator, denoiser, config: TrainConfig, step: int, final: bool = False) -> Path:
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    tag = "final" if final else f"step-{step}"
    out_dir = Path(config.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(denoiser)
        state_dict = get_peft_model_state_dict(unwrapped)
        save_file(state_dict, out_dir / "lora_weights.safetensors")
    return out_dir
