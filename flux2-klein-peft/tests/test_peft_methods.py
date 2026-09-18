"""Tests for the PEFT-method registry and the FLUX.2 wiring around it.

These are all CPU-only and weight-free: they cover the parts that decide
whether a sweep produces comparable numbers (config mapping, target-module
selection, checkpoint format routing), not the training maths, which needs a
GPU and real weights.
"""

from __future__ import annotations

import pytest

from dptlab.models.registry import get_model_spec, list_models
from dptlab.training.common import TrainConfig
from dptlab.training.peft_methods import (
    build_peft_config,
    get_method_spec,
    list_methods,
    target_modules_for,
)


def make_config(method: str, **overrides) -> TrainConfig:
    base = {
        "model_key": "flux2-klein-4b",
        "recipe": "lora",
        "dataset_path": "data/x",
        "output_dir": "out/x",
        "peft_method": method,
    }
    base.update(overrides)
    return TrainConfig(**base)


def test_klein_registered_with_distilled_defaults():
    spec = get_model_spec("flux2-klein-4b")
    assert "flux2-klein-4b" in list_models()
    assert spec.family == "flux2"
    # klein is step- and guidance-distilled; serving it at SDXL's defaults is
    # the single easiest way to get garbage out of a correct checkpoint.
    assert spec.supports_guidance is False
    assert spec.default_inference_steps == 4
    assert spec.default_guidance_scale == 1.0


@pytest.mark.parametrize("method", list_methods())
def test_every_method_builds_a_peft_config(method):
    config = build_peft_config(make_config(method), get_model_spec("flux2-klein-4b"))
    assert config.target_modules, f"{method} produced no target modules"


@pytest.mark.parametrize("method", list_methods())
def test_rank_knob_reaches_the_right_config_field(method):
    """`lora_rank` in YAML must land on whichever field the method actually uses."""
    config = build_peft_config(make_config(method, lora_rank=8, lora_alpha=8), get_model_spec("flux2-klein-4b"))
    if method in {"lora", "dora"}:
        assert config.r == 8 and config.lora_alpha == 8
    elif method in {"loha", "lokr"}:
        assert config.r == 8 and config.alpha == 8
    elif method == "oft":
        # OFT is parameterized by block size; r must stay 0 or peft rejects the
        # pair as mutually exclusive.
        assert config.r == 0 and config.oft_block_size > 0
    elif method == "boft":
        assert config.boft_block_size > 0


def test_orthogonal_methods_avoid_fused_projections():
    """OFT/BOFT need out_features divisible by the block size.

    FLUX.2's single-stream blocks fuse qkv and the MLP input into one
    projection whose dims don't factor cleanly, so injecting there fails at
    adapter construction. The restricted target list is what prevents that.
    """
    spec = get_model_spec("flux2-klein-4b")
    for method_key in ("oft", "boft"):
        targets = target_modules_for(get_method_spec(method_key), spec)
        assert "to_qkv_mlp_proj" not in targets
        assert "proj_out" not in targets
        assert "to_q" in targets

    lora_targets = target_modules_for(get_method_spec("lora"), spec)
    assert "to_qkv_mlp_proj" in lora_targets


def test_only_lora_claims_the_diffusers_native_format():
    """Everything else must round-trip through peft's own state dict.

    diffusers' loader silently loads nothing when handed keys it doesn't
    recognize, so a method wrongly marked diffusers-native would serve the base
    model while reporting a successful load.
    """
    assert get_method_spec("lora").diffusers_native is True
    for method in list_methods():
        if method != "lora":
            assert get_method_spec(method).diffusers_native is False, method


def test_unknown_method_names_the_alternatives():
    with pytest.raises(KeyError, match="loha"):
        get_method_spec("lohaa")


def test_flux1_has_no_objective_registered():
    """FLUX.1 is flow matching but not FLUX.2 -- it must fail loudly, not train."""
    from dptlab.training.objectives import get_objective

    with pytest.raises(NotImplementedError):
        get_objective(get_model_spec("flux-dev"))

    from dptlab.training.objectives import Flux2FlowMatchObjective

    assert isinstance(get_objective(get_model_spec("flux2-klein-4b")), Flux2FlowMatchObjective)
