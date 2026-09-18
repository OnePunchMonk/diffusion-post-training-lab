"""PEFT method registry: one place that knows how each adapter family is
configured, counted, and serialized.

The point of this module is to make "which PEFT method?" a one-line config
change rather than a fork of the training loop, so the recipes can be compared
on equal footing. Three things differ between methods and all three live here:

1. **Config shape.** `r`/`lora_alpha` (LoRA, DoRA), `r`/`alpha` (LoHa, LoKr),
   `oft_block_size` (OFT), `boft_block_size` (BOFT). A single `lora_rank` knob
   in the YAML is mapped onto whichever field the method actually uses.

2. **Where they can be injected.** LoRA-style adapters are a low-rank *additive*
   update and go anywhere. OFT and BOFT instead learn a (block-)orthogonal
   matrix multiplying the output space, so they require `out_features` to be
   divisible by the block size; FLUX.2's fused `to_qkv_mlp_proj` and `proj_out`
   have out/in dims that don't factor cleanly, and injecting there fails at
   adapter-construction time. Hence `ModelSpec.orthogonal_target_modules`.

3. **Checkpoint format.** diffusers' `save_lora_weights`/`load_lora_weights`
   only understand LoRA keys -- pointing them at a LoHa or OFT state dict
   silently loads nothing (see the note in `common.save_lora_checkpoint`). So
   only plain LoRA takes the diffusers-native path; every other method is
   saved and reloaded through peft's own `get/set_peft_model_state_dict` plus
   an `adapter_config.json`, which round-trips exactly and stays readable by
   `PeftConfig.from_pretrained`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dptlab.models.registry import ModelSpec
    from dptlab.training.common import TrainConfig

ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
LORA_WEIGHTS_NAME = "lora_weights.safetensors"


@dataclass(frozen=True)
class PeftMethodSpec:
    key: str
    build: Callable[..., Any]
    #  True  -> checkpoint via pipe.save_lora_weights / pipe.load_lora_weights
    #  False -> checkpoint via peft state dict + adapter_config.json
    diffusers_native: bool
    orthogonal: bool
    notes: str


def _lora(rank: int, alpha: int, targets: list[str], extra: dict) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=extra.get("dropout", 0.0),
        init_lora_weights="gaussian",
    )


def _dora(rank: int, alpha: int, targets: list[str], extra: dict) -> Any:
    from peft import LoraConfig

    # DoRA is LoRA plus a learned per-column magnitude, so it reuses LoraConfig.
    # `init_lora_weights="gaussian"` is not supported alongside use_dora in peft
    # (the magnitude vector is initialized from the base weight norm, which
    # assumes the default zero-init B), so leave the init at its default.
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=extra.get("dropout", 0.0),
        use_dora=True,
    )


def _loha(rank: int, alpha: int, targets: list[str], extra: dict) -> Any:
    from peft import LoHaConfig

    return LoHaConfig(
        r=rank,
        alpha=alpha,
        target_modules=targets,
        module_dropout=extra.get("dropout", 0.0),
    )


def _lokr(rank: int, alpha: int, targets: list[str], extra: dict) -> Any:
    from peft import LoKrConfig

    return LoKrConfig(
        r=rank,
        alpha=alpha,
        target_modules=targets,
        module_dropout=extra.get("dropout", 0.0),
        decompose_both=extra.get("lokr_decompose_both", False),
        decompose_factor=extra.get("lokr_decompose_factor", -1),
    )


def _oft(rank: int, alpha: int, targets: list[str], extra: dict) -> Any:
    from peft import OFTConfig

    # OFT is parameterized by block size, not rank: r and oft_block_size are
    # mutually exclusive in peft (r=0 means "derive r from block size"). Block
    # size is the knob that actually trades parameters for expressiveness, so
    # that's what the config exposes.
    return OFTConfig(
        r=0,
        oft_block_size=extra.get("oft_block_size", 32),
        target_modules=targets,
        module_dropout=extra.get("dropout", 0.0),
        coft=extra.get("oft_coft", False),
        # Cayley-Neumann is an approximation of the matrix inverse in the Cayley
        # transform; it avoids a per-step dense inverse on every adapted layer,
        # which on a 4B transformer is the difference between OFT being
        # comparable to LoRA in step time and being several times slower.
        use_cayley_neumann=extra.get("oft_cayley_neumann", True),
    )


def _boft(rank: int, alpha: int, targets: list[str], extra: dict) -> Any:
    from peft import BOFTConfig

    return BOFTConfig(
        boft_block_size=extra.get("boft_block_size", 32),
        boft_n_butterfly_factor=extra.get("boft_n_butterfly_factor", 2),
        target_modules=targets,
        boft_dropout=extra.get("dropout", 0.0),
    )


_METHODS: dict[str, PeftMethodSpec] = {
    "lora": PeftMethodSpec(
        key="lora",
        build=_lora,
        diffusers_native=True,
        orthogonal=False,
        notes="Baseline. Only method whose checkpoints load into a stock diffusers pipeline unmodified.",
    ),
    "dora": PeftMethodSpec(
        key="dora",
        build=_dora,
        diffusers_native=False,
        orthogonal=False,
        notes="LoRA + learned magnitude. Usually beats LoRA at low rank; ~10-20% slower per step.",
    ),
    "loha": PeftMethodSpec(
        key="loha",
        build=_loha,
        diffusers_native=False,
        orthogonal=False,
        notes="Hadamard product of two low-rank pairs: higher effective rank for the same parameter count.",
    ),
    "lokr": PeftMethodSpec(
        key="lokr",
        build=_lokr,
        diffusers_native=False,
        orthogonal=False,
        notes="Kronecker factorization. Smallest checkpoints of the six; the community LyCORIS default for style.",
    ),
    "oft": PeftMethodSpec(
        key="oft",
        build=_oft,
        diffusers_native=False,
        orthogonal=True,
        notes="Block-orthogonal transform. Preserves pairwise angles in the weight space, so it drifts less "
        "from the base model's prior -- the reason it's worth testing against LoRA on 4-image subjects.",
    ),
    "boft": PeftMethodSpec(
        key="boft",
        build=_boft,
        diffusers_native=False,
        orthogonal=True,
        notes="Butterfly-factorized OFT: denser orthogonal transform for fewer parameters.",
    ),
}


def get_method_spec(key: str) -> PeftMethodSpec:
    try:
        return _METHODS[key]
    except KeyError as e:
        raise KeyError(f"Unknown PEFT method {key!r}. Available: {sorted(_METHODS)}") from e


def list_methods() -> list[str]:
    return sorted(_METHODS)


def target_modules_for(method: PeftMethodSpec, spec: ModelSpec) -> list[str]:
    """Which modules this method may be injected into on this model.

    Orthogonal methods fall back to the full LoRA target list only if the model
    doesn't declare a restricted set, so adding a new model without thinking
    about OFT still works (it will just fail loudly at injection if the dims
    don't factor, rather than silently adapting the wrong thing).
    """
    if method.orthogonal and spec.orthogonal_target_modules:
        return list(spec.orthogonal_target_modules)
    return list(spec.lora_target_modules)


def build_peft_config(config: TrainConfig, spec: ModelSpec) -> Any:
    method = get_method_spec(config.peft_method)
    return method.build(
        config.lora_rank,
        config.lora_alpha,
        target_modules_for(method, spec),
        config.extra,
    )


def count_trainable_parameters(denoiser) -> int:
    return sum(p.numel() for p in denoiser.parameters() if p.requires_grad)


def save_peft_checkpoint(pipe, denoiser, out_dir: str | Path, method_key: str) -> Path:
    """Save an adapter in whichever format its method can actually be reloaded from."""
    from dptlab.training.common import save_lora_checkpoint

    method = get_method_spec(method_key)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if method.diffusers_native:
        save_lora_checkpoint(pipe, denoiser, out_dir, filename=LORA_WEIGHTS_NAME)
        return out_dir

    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    adapter_name = next(iter(denoiser.peft_config))
    peft_config = denoiser.peft_config[adapter_name]
    state_dict = get_peft_model_state_dict(denoiser, adapter_name=adapter_name)
    # peft prefixes keys with the adapter name on save; strip it so the
    # checkpoint is adapter-name agnostic and set_peft_model_state_dict can
    # place it under whatever name the reloading process chose.
    state_dict = {k.replace(f".{adapter_name}.", "."): v.contiguous().cpu() for k, v in state_dict.items()}
    save_file(state_dict, out_dir / ADAPTER_WEIGHTS_NAME)
    peft_config.save_pretrained(str(out_dir))
    return out_dir


def load_peft_checkpoint(pipe, denoiser, checkpoint_dir: str | Path, method_key: str, for_training: bool = True) -> None:
    """Inverse of `save_peft_checkpoint`; used by both resume and eval.

    `for_training=False` skips the gradient-checkpointing + fp32-upcast prep.
    At eval time that prep is not just wasted work: gradient checkpointing
    re-runs each block's forward during inference for no benefit, and upcasting
    the adapter to fp32 on top of bf16 base weights forces a dtype cast on
    every adapted layer.
    """
    from dptlab.training.common import load_lora_checkpoint

    method = get_method_spec(method_key)
    checkpoint_dir = Path(checkpoint_dir)

    if method.diffusers_native:
        load_lora_checkpoint(pipe, denoiser, checkpoint_dir / LORA_WEIGHTS_NAME, for_training=for_training)
        return

    from peft import PeftConfig
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    from dptlab.training.common import _finalize_trainable_adapter

    peft_config = PeftConfig.from_pretrained(str(checkpoint_dir))
    denoiser.requires_grad_(False)
    denoiser.add_adapter(peft_config)
    incompatible = set_peft_model_state_dict(denoiser, load_file(checkpoint_dir / ADAPTER_WEIGHTS_NAME))
    if incompatible is not None and getattr(incompatible, "unexpected_keys", []):
        raise RuntimeError(
            f"Adapter checkpoint at {checkpoint_dir} has keys the injected {method_key} adapter doesn't: "
            f"{list(incompatible.unexpected_keys)[:5]}"
        )
    if for_training:
        _finalize_trainable_adapter(denoiser)


def adapter_exists(checkpoint_dir: str | Path, method_key: str) -> bool:
    d = Path(checkpoint_dir)
    name = LORA_WEIGHTS_NAME if get_method_spec(method_key).diffusers_native else ADAPTER_WEIGHTS_NAME
    return (d / name).exists()
