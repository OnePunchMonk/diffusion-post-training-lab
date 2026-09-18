"""Recipe A: parameter-efficient fine-tuning on a custom concept/style dataset.

Freeze the UNet/transformer, inject an adapter into the projections named by
the model's ModelSpec, and train on (image, caption) pairs. This is the
cheapest recipe and the one that proves the end-to-end pipeline (data -> train
-> checkpoint -> eval -> serve) before the pricier DPO and distillation
recipes reuse the same scaffolding.

Two axes are configurable and deliberately orthogonal:

- **Which adapter** (`peft_method`: lora / dora / loha / lokr / oft / boft),
  handled by `training.peft_methods`.
- **Which denoising objective** (epsilon for SDXL, rectified flow for
  FLUX.2 [klein]), handled by `training.objectives`.

The recipe is still called "lora" in the config for backwards compatibility
with existing checkpoints and the MODELS.md leaderboard.
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
    load_frozen_pipe,
    save_run_manifest,
    set_seed,
)
from dptlab.training.objectives import get_objective
from dptlab.training.peft_methods import (
    adapter_exists,
    build_peft_config,
    count_trainable_parameters,
    get_method_spec,
    load_peft_checkpoint,
    save_peft_checkpoint,
)

logger = logging.getLogger(__name__)


def train_lora(config: TrainConfig) -> Path:
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    set_seed(config.seed)
    spec = get_model_spec(config.model_key)
    method = get_method_spec(config.peft_method)
    objective = get_objective(spec)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    pipe = load_frozen_pipe(config.model_key, accelerator)
    denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer

    resume_from = config.extra.get("resume_from")
    if resume_from and adapter_exists(resume_from, config.peft_method):
        load_peft_checkpoint(pipe, denoiser, resume_from, config.peft_method)
        logger.info("Resumed %s adapter from %s", config.peft_method, resume_from)
    else:
        add_lora_adapter(denoiser, build_peft_config(config, spec))

    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    num_trainable = count_trainable_parameters(denoiser)
    logger.info(
        "method=%s model=%s trainable_params=%d (%.2fM)",
        config.peft_method,
        config.model_key,
        num_trainable,
        num_trainable / 1e6,
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    dataset = ImageCaptionDataset(config.dataset_path, resolution=config.resolution, use_masks=config.use_masks)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    denoiser, optimizer, dataloader = accelerator.prepare(denoiser, optimizer, dataloader)

    global_step = 0
    max_epochs = math.ceil(config.max_train_steps / max(1, len(dataloader)))

    for _epoch in range(max_epochs):
        for batch in dataloader:
            with accelerator.accumulate(denoiser):
                loss = objective.loss(pipe, denoiser, batch, config)

                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm:
                    accelerator.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0:
                    logger.info("step=%d loss=%.4f", global_step, loss.item())
                if global_step % config.checkpointing_steps == 0:
                    _save_checkpoint(accelerator, pipe, denoiser, config, global_step)
                if global_step >= config.max_train_steps:
                    break
        if global_step >= config.max_train_steps:
            break

    output_dir = _save_checkpoint(accelerator, pipe, denoiser, config, global_step, final=True)
    save_run_manifest(
        output_dir,
        config,
        extra={
            "final_step": global_step,
            "peft_method": config.peft_method,
            "trainable_parameters": num_trainable,
            "adapter_format": "diffusers" if method.diffusers_native else "peft",
        },
    )
    return output_dir


def _save_checkpoint(accelerator, pipe, denoiser, config: TrainConfig, step: int, final: bool = False) -> Path:
    tag = "final" if final else f"step-{step}"
    out_dir = Path(config.output_dir) / tag
    if accelerator.is_main_process:
        save_peft_checkpoint(pipe, accelerator.unwrap_model(denoiser), out_dir, config.peft_method)
    return out_dir
