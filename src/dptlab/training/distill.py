"""Recipe C: step-distillation (LCM-style) for fast few-step inference.

Reference: Luo et al., "Latent Consistency Models" (2023). Trains a student
copy of the (optionally LoRA/DPO/GRPO-tuned) teacher to predict a consistency
function along the teacher's ODE trajectory, so the student converges in
4-8 steps instead of 25-50. This directly motivates the deployment story:
a distilled checkpoint is what actually makes a Modal endpoint cheap, since
inference cost scales ~linearly with denoising steps.
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

from dptlab.data.dataset import PromptOnlyDataset
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


def train_distill(config: TrainConfig) -> Path:
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from diffusers import LCMScheduler
    from peft import LoraConfig
    from torch.utils.data import DataLoader

    set_seed(config.seed)
    spec = get_model_spec(config.model_key)
    num_ddim_steps = config.extra.get("teacher_ddim_steps", 50)
    ema_decay = config.extra.get("ema_decay", 0.95)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    pipe = load_frozen_pipe(config.model_key, accelerator)
    teacher = pipe.unet if hasattr(pipe, "unet") else pipe.transformer
    teacher.requires_grad_(False)
    teacher.eval()

    # Student is a separate LoRA-adapted copy of the teacher (not the same
    # object!) so the teacher stays a fixed, unmodified supervision signal
    # throughout training. Target network is an EMA shadow of the student.
    student = copy.deepcopy(teacher)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(spec.lora_target_modules),
        init_lora_weights="gaussian",
    )
    add_lora_adapter(student, lora_config)
    target = copy.deepcopy(student).to(accelerator.device)
    target.requires_grad_(False)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=config.learning_rate)
    dataset = PromptOnlyDataset(config.dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    student, optimizer, dataloader = accelerator.prepare(student, optimizer, dataloader)

    global_step = 0
    max_epochs = math.ceil(config.max_train_steps / max(1, len(dataloader)))

    for _epoch in range(max_epochs):
        for batch in dataloader:
            with accelerator.accumulate(student):
                encoder_hidden_states, added_cond_kwargs = encode_conditioning(
                    pipe, batch["prompt"], config.resolution
                )
                latents = torch.randn(
                    (len(batch["prompt"]), 4, config.resolution // 8, config.resolution // 8),
                    device=accelerator.device,
                )

                # Sample two adjacent points on the teacher's PF-ODE trajectory
                # (skipping-step DDIM) and require student(t_n) ~= student(t_{n+1}),
                # i.e. the consistency property, rather than matching noise directly.
                idx = torch.randint(0, num_ddim_steps - 1, (1,)).item()
                t_n, t_np1 = _ddim_timesteps(pipe, num_ddim_steps)[idx : idx + 2]

                with torch.no_grad():
                    teacher_pred = teacher(
                        latents, t_np1, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample
                    x_prev = pipe.scheduler.step(teacher_pred, t_np1, latents).prev_sample
                    target_pred = target(
                        x_prev, t_n, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample

                student_pred = student(
                    latents, t_np1, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                ).sample
                loss = F.mse_loss(student_pred.float(), target_pred.float())

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    for t_param, s_param in zip(target.parameters(), accelerator.unwrap_model(student).parameters()):
                        t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0:
                    logger.info("step=%d consistency_loss=%.4f", global_step, loss.item())
                if global_step % config.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, student, config, global_step)
                if global_step >= config.max_train_steps:
                    break
        if global_step >= config.max_train_steps:
            break

    output_dir = _save_checkpoint(accelerator, student, config, global_step, final=True)
    save_run_manifest(output_dir, config, extra={"final_step": global_step, "recipe": "distill"})
    return output_dir


def _ddim_timesteps(pipe, num_steps: int):
    pipe.scheduler.set_timesteps(num_steps)
    return pipe.scheduler.timesteps


def _save_checkpoint(accelerator, student, config: TrainConfig, step: int, final: bool = False) -> Path:
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    tag = "final" if final else f"step-{step}"
    out_dir = Path(config.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(student)
        state_dict = get_peft_model_state_dict(unwrapped)
        save_file(state_dict, out_dir / "lora_weights.safetensors")
    return out_dir
