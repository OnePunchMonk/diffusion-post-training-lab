"""Recipe B: Diffusion-DPO preference alignment.

Reference: Wallace et al., "Diffusion Model Alignment Using Direct Preference
Optimization" (2023). Given (prompt, image_win, image_lose) preference triples,
we train a LoRA-adapted policy against a frozen reference copy of the same
weights, pushing the denoising loss lower on the winning image relative to the
reference and higher on the losing image, at a shared noise/timestep draw.

The optimization target is the preference margin rather than reconstruction
error, so this needs paired data (`scripts/build_preference_pairs.py`) and a
reference forward pass on both branches of every batch.
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

from dptlab.data.preference import PreferenceDataset
from dptlab.models.registry import get_model_spec
from dptlab.training.common import (
    TrainConfig,
    add_lora_adapter,
    encode_conditioning,
    load_frozen_pipe,
    save_lora_checkpoint,
    save_run_manifest,
    set_seed,
)

logger = logging.getLogger(__name__)


def train_dpo(config: TrainConfig) -> Path:
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from peft import LoraConfig
    from torch.utils.data import DataLoader

    set_seed(config.seed)
    spec = get_model_spec(config.model_key)
    beta = config.extra.get("dpo_beta", 5000.0)  # Diffusion-DPO uses a much larger beta than text DPO

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    pipe = load_frozen_pipe(config.model_key, accelerator)
    denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer

    # Frozen reference copy: DPO needs both the trainable policy and a fixed
    # reference to compute the implicit reward margin against.
    ref_denoiser = copy.deepcopy(denoiser)
    ref_denoiser.requires_grad_(False)
    ref_denoiser.eval()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(spec.lora_target_modules),
        init_lora_weights="gaussian",
    )
    add_lora_adapter(denoiser, lora_config)

    optimizer = torch.optim.AdamW([p for p in denoiser.parameters() if p.requires_grad], lr=config.learning_rate)

    dataset = PreferenceDataset(config.dataset_path, resolution=config.resolution)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    denoiser, ref_denoiser, optimizer, dataloader = accelerator.prepare(
        denoiser, ref_denoiser, optimizer, dataloader
    )

    global_step = 0
    max_epochs = math.ceil(config.max_train_steps / max(1, len(dataloader)))

    for _epoch in range(max_epochs):
        for batch in dataloader:
            with accelerator.accumulate(denoiser):
                latents_win = _encode_latents(pipe, batch["image_win"])
                latents_lose = _encode_latents(pipe, batch["image_lose"])

                noise = torch.randn_like(latents_win)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (latents_win.shape[0],), device=latents_win.device
                ).long()

                noisy_win = pipe.scheduler.add_noise(latents_win, noise, timesteps)
                noisy_lose = pipe.scheduler.add_noise(latents_lose, noise, timesteps)

                encoder_hidden_states, added_cond_kwargs = encode_conditioning(
                    pipe, batch["prompt"], config.resolution
                )

                pred_win = denoiser(noisy_win, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
                pred_lose = denoiser(noisy_lose, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
                with torch.no_grad():
                    ref_pred_win = ref_denoiser(
                        noisy_win, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample
                    ref_pred_lose = ref_denoiser(
                        noisy_lose, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample

                target = noise if pipe.scheduler.config.prediction_type == "epsilon" else latents_win

                pol_loss_win = F.mse_loss(pred_win.float(), target.float(), reduction="none").mean([1, 2, 3])
                pol_loss_lose = F.mse_loss(pred_lose.float(), target.float(), reduction="none").mean([1, 2, 3])
                ref_loss_win = F.mse_loss(ref_pred_win.float(), target.float(), reduction="none").mean([1, 2, 3])
                ref_loss_lose = F.mse_loss(ref_pred_lose.float(), target.float(), reduction="none").mean([1, 2, 3])

                # Implicit reward margin: how much more the policy improved on
                # the winning sample relative to the losing sample, vs. the
                # frozen reference (Wallace et al. Eq. 14).
                policy_margin = pol_loss_win - pol_loss_lose
                ref_margin = ref_loss_win - ref_loss_lose
                loss = -F.logsigmoid(-beta * (policy_margin - ref_margin)).mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0:
                    logger.info("step=%d dpo_loss=%.4f", global_step, loss.item())
                if global_step % config.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, pipe, denoiser, config, global_step)
                if global_step >= config.max_train_steps:
                    break
        if global_step >= config.max_train_steps:
            break

    output_dir = _save_checkpoint(accelerator, pipe, denoiser, config, global_step, final=True)
    save_run_manifest(output_dir, config, extra={"final_step": global_step, "dpo_beta": beta})
    return output_dir


def _encode_latents(pipe, pixel_values):
    import torch

    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values.to(dtype=pipe.vae.dtype)).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


def _save_checkpoint(accelerator, pipe, denoiser, config: TrainConfig, step: int, final: bool = False) -> Path:
    tag = "final" if final else f"step-{step}"
    out_dir = Path(config.output_dir) / tag
    if accelerator.is_main_process:
        save_lora_checkpoint(pipe, accelerator.unwrap_model(denoiser), out_dir)
    return out_dir
