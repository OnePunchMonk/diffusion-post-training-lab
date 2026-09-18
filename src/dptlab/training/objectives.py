"""Per-family denoising objectives.

SDXL and FLUX.2 disagree about almost every part of a training step -- how
latents are shaped, how noise is added, what the network predicts, what extra
conditioning the forward pass needs -- so the recipes call into one of these
instead of branching inline. Adding a model family means adding an objective
here; the LoRA/DoRA/LoHa/... loop above it doesn't change.

SDXL: DDPM forward process, epsilon (or v) prediction, 4-channel spatial
latents, `added_cond_kwargs` for the pooled embedding and size conditioning.

FLUX.2: rectified flow. Latents are 2x2-patchified, batch-norm standardized
using statistics stored on the VAE, and packed into a token sequence, with 4D
position ids carried alongside for both image and text tokens. The forward
process is the straight-line interpolant x_t = (1 - s) * x_0 + s * eps and the
network regresses the velocity eps - x_0, with s in [0, 1] passed directly as
the timestep.
"""

from __future__ import annotations

from typing import Protocol

from dptlab.models.registry import ModelSpec


class Objective(Protocol):
    def loss(self, pipe, denoiser, batch: dict, config) -> object: ...


class EpsilonObjective:
    """SDXL / any UNet on a DDPM schedule."""

    def loss(self, pipe, denoiser, batch: dict, config):
        import torch

        from dptlab.training.common import encode_conditioning

        pixel_values = batch["pixel_values"].to(device=pipe.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
        ).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            encoder_hidden_states, added_cond_kwargs = encode_conditioning(pipe, batch["caption"], config.resolution)

        model_pred = denoiser(
            noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        ).sample
        target = noise if pipe.scheduler.config.prediction_type == "epsilon" else latents
        return _masked_mse(model_pred, target, batch.get("mask"), spatial=True)


class Flux2FlowMatchObjective:
    """FLUX.2 [klein] / [dev]: rectified flow over packed latent tokens."""

    def loss(self, pipe, denoiser, batch: dict, config):
        import torch

        latents, latent_ids = self._encode_latents(pipe, batch["pixel_values"])

        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=list(batch["caption"]),
                device=pipe.device,
                max_sequence_length=config.extra.get("max_sequence_length", 512),
            )
        prompt_embeds = prompt_embeds.to(dtype=denoiser.dtype)

        sigmas = self._sample_sigmas(pipe, latents, config)
        noise = torch.randn_like(latents)
        # Straight-line interpolant: at s=1 the input is pure noise, at s=0 it
        # is the clean latent, matching the pipeline's sigma convention.
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents

        model_pred = denoiser(
            hidden_states=noisy.to(denoiser.dtype),
            timestep=sigmas.squeeze(-1).squeeze(-1).to(denoiser.dtype),
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        # The transformer returns one token per input token; with no reference
        # images the image tokens are all of them, but slice defensively so this
        # keeps working if reference conditioning is added later.
        model_pred = model_pred[:, : latents.shape[1]]

        return _masked_mse(model_pred, target, batch.get("mask"), spatial=False)

    @staticmethod
    def _encode_latents(pipe, pixel_values):
        """VAE-encode to the exact latent space the transformer was trained in.

        `AutoencoderKLFlux2` is not scaled by a single `scaling_factor` the way
        the SD VAEs are: the pipeline patchifies the raw latent into 2x2 blocks
        (4x the channels) and then standardizes it with running batch-norm
        statistics stored on `vae.bn`. Skipping either step leaves the latents
        several standard deviations off distribution and the adapter spends the
        whole run learning to undo the mismatch.
        """
        import torch
        from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

        pixel_values = pixel_values.to(device=pipe.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = retrieve_latents(pipe.vae.encode(pixel_values), sample_mode="sample")
            latents = pipe._patchify_latents(latents)
            mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = (latents - mean) / std

        latent_ids = pipe._prepare_latent_ids(latents).to(pipe.device)
        return pipe._pack_latents(latents).float(), latent_ids

    @staticmethod
    def _sample_sigmas(pipe, latents, config):
        """Draw one noise level per sample, shifted the way inference shifts it.

        Sampling sigma uniformly would spend most of the budget on timesteps the
        4-step distilled sampler never visits. Instead draw from a logit-normal
        (the SD3/FLUX training default, which concentrates on the mid-range
        where the velocity is hardest to predict) and push it through the same
        resolution-dependent shift the pipeline applies via `compute_empirical_mu`,
        so the training distribution matches the sampling distribution.
        """
        import torch
        from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

        batch_size = latents.shape[0]
        num_steps = config.extra.get("sampling_steps_for_shift", 4)
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_steps)

        u = torch.normal(
            mean=config.extra.get("logit_mean", 0.0),
            std=config.extra.get("logit_std", 1.0),
            size=(batch_size,),
            device=latents.device,
        )
        sigmas = torch.sigmoid(u)
        sigmas = pipe.scheduler.time_shift(mu, 1.0, sigmas)
        return sigmas.clamp(1e-4, 1.0).view(batch_size, 1, 1).to(latents.dtype)


def _masked_mse(pred, target, mask, spatial: bool):
    """MSE, optionally weighted toward the subject.

    `mask` comes from the SAM labeling pass (flux2-klein-peft/scripts/autolabel.py) and is a
    per-pixel subject probability. On 4-6 image subjects most of the frame is
    background the adapter shouldn't be memorizing, so weighting the loss toward
    the subject is the cheapest defence against the adapter baking in the
    background. Runs without masks are unaffected.
    """
    import torch.nn.functional as F

    pred = pred.float()
    target = target.float()
    if mask is None:
        return F.mse_loss(pred, target)

    mask = mask.to(pred.device).float()
    per_element = (pred - target) ** 2

    if spatial:
        weights = F.interpolate(mask, size=pred.shape[-2:], mode="area")
    else:
        # Packed tokens: one token per 2x2 latent patch, laid out row-major by
        # `_pack_latents`, so area-pooling the mask to the latent grid and
        # flattening in the same order lines the weights up with the tokens.
        side = round(pred.shape[1] ** 0.5)
        if side * side != pred.shape[1]:
            return F.mse_loss(pred, target)  # non-square latents: fall back
        weights = F.interpolate(mask, size=(side, side), mode="area")
        weights = weights.flatten(2).permute(0, 2, 1)

    # Never let the background go fully unsupervised -- a hard 0/1 mask makes
    # the model free to put anything outside the subject.
    weights = weights.clamp_min(0.1).expand_as(per_element)
    return (per_element * weights).sum() / weights.sum().clamp_min(1e-8)


_OBJECTIVES = {
    "sdxl": EpsilonObjective,
    # FLUX.1 is also rectified flow, but with a different (CLIP + T5) text
    # stack and latent packing than FLUX.2. Rather than silently running the
    # DDPM/epsilon step on it -- which trains, converges to nothing useful, and
    # looks fine in the loss curve -- leave it unregistered until it has its
    # own objective.
    "flux2": Flux2FlowMatchObjective,
}


def get_objective(spec: ModelSpec) -> Objective:
    try:
        return _OBJECTIVES[spec.family]()
    except KeyError as e:
        raise NotImplementedError(f"No training objective registered for model family {spec.family!r}") from e
