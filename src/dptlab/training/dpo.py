"""Recipe B: Diffusion-DPO preference alignment.

Reference: Wallace et al., "Diffusion Model Alignment Using Direct Preference
Optimization" (2023). Given (prompt, image_win, image_lose) preference triples,
we train a LoRA-adapted policy model against a frozen reference copy of the
same weights, pushing the denoising loss lower on the winning image relative
to the reference model and higher on the losing image, at a shared noise/
timestep draw. This is the technique most worth discussing in an interview:
it is closer to modern LLM RLHF pipelines than plain supervised fine-tuning,
and it directly targets human-judged quality rather than reconstruction loss.
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

from dptlab.data.preference import PreferenceDataset
from dptlab.models.registry import get_model_spec, load_pipeline
from dptlab.training.common import TrainConfig, save_run_manifest, set_seed

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

    pipe = load_pipeline(config.model_key, dtype="float32", device="cpu")
    denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer

    # Frozen reference copy: DPO needs both the trainable policy and a fixed
    # reference to compute the implicit reward margin against.
    ref_denoiser = copy.deepcopy(denoiser)
    ref_denoiser.requires_grad_(False)
    ref_denoiser.eval()

    denoiser.requires_grad_(False)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(spec.lora_target_modules),
        init_lora_weights="gaussian",
    )
    denoiser.add_adapter(lora_config)

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

                encoder_hidden_states = _encode_prompts(pipe, batch["prompt"])

                pred_win = denoiser(noisy_win, timesteps, encoder_hidden_states).sample
                pred_lose = denoiser(noisy_lose, timesteps, encoder_hidden_states).sample
                with torch.no_grad():
                    ref_pred_win = ref_denoiser(noisy_win, timesteps, encoder_hidden_states).sample
                    ref_pred_lose = ref_denoiser(noisy_lose, timesteps, encoder_hidden_states).sample

                target = noise if pipe.scheduler.config.prediction_type == "epsilon" else latents_win

                pol_loss_win = F.mse_loss(pred_win.float(), target.float(), reduction="none").mean([1, 2, 3])
                pol_loss_lose = F.mse_loss(pred_lose.float(), target.float(), reduction="none").mean([1, 2, 3])
                ref_loss_win = F.mse_loss(ref_pred_win.float(), target.float(), reduction="none").mean([1, 2, 3])
                ref_loss_lose = F.mse_loss(ref_pred_lose.float(), target.float(), reduction="none").mean([1, 2, 3])

                # Implicit reward margin: how much more the policy improved on
                # the winning sample relative to the losing sample, vs. the
                # frozen reference. This is the Diffusion-DPO loss (Eq. 14 in
                # Wallace et al.), not vanilla denoising MSE.
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
                    _save_checkpoint(accelerator, denoiser, config, global_step)
                if global_step >= config.max_train_steps:
                    break
        if global_step >= config.max_train_steps:
            break

    output_dir = _save_checkpoint(accelerator, denoiser, config, global_step, final=True)
    save_run_manifest(output_dir, config, extra={"final_step": global_step, "dpo_beta": beta})
    return output_dir


def _encode_latents(pipe, pixel_values):
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


def _encode_prompts(pipe, captions: list[str]):
    if hasattr(pipe, "encode_prompt"):
        prompt_embeds, *_ = pipe.encode_prompt(prompt=captions, device=pipe.device)
        return prompt_embeds
    raise NotImplementedError("Pipeline does not expose encode_prompt; add a model-specific encoder here.")


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
