from dptlab.models.registry import get_model_spec, list_models


def test_list_models_includes_sdxl_and_flux():
    models = list_models()
    assert "sdxl" in models
    assert "flux-schnell" in models
    assert "flux-dev" in models


def test_get_model_spec_unknown_key_raises():
    import pytest

    with pytest.raises(KeyError):
        get_model_spec("not-a-real-model")


def test_sdxl_spec_shape():
    spec = get_model_spec("sdxl")
    assert spec.pretrained_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "to_q" in spec.lora_target_modules
    assert spec.default_resolution == 1024
