"""Mask handling: the SAM output has to survive the trip from PNG to loss weight.

These are the parts of the labeling pass that are cheap to get subtly wrong and
expensive to notice — a mask that's misaligned with the latent grid, or one
whose weights leave the background fully unsupervised, degrades training quietly
rather than raising.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


def _no_torchvision() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchvision") is None

from PIL import Image

from dptlab.data.dataset import ImageCaptionDataset
from dptlab.eval.metrics.subject_fidelity import mask_to_neutral_background
from dptlab.training.objectives import _masked_mse

# ImageCaptionDataset builds its transforms with torchvision, which is in the
# `label` extra rather than the base install.
requires_torchvision = pytest.mark.skipif(_no_torchvision(), reason="torchvision not installed")


def make_dataset(tmp_path, with_mask: bool):
    Image.new("RGB", (128, 96), (200, 30, 30)).save(tmp_path / "0000.png")
    record = {"file_name": "0000.png", "caption": "a photo of sks thing"}
    if with_mask:
        (tmp_path / "masks").mkdir()
        mask = Image.new("L", (128, 96), 0)
        mask.paste(255, (32, 24, 96, 72))
        mask.save(tmp_path / "masks" / "0000.png")
        record["mask_file_name"] = "masks/0000.png"
    (tmp_path / "metadata.jsonl").write_text(json.dumps(record) + "\n")
    return tmp_path


@requires_torchvision
def test_mask_is_cropped_identically_to_the_image(tmp_path):
    """Image and mask must stay pixel-aligned through resize + crop.

    Applying independent random transforms to each is the classic way to get a
    mask that is subtly offset from its image, which silently supervises the
    wrong region.
    """
    root = make_dataset(tmp_path, with_mask=True)
    item = ImageCaptionDataset(str(root), resolution=64, use_masks=True)[0]

    assert item["mask"].shape[-2:] == item["pixel_values"].shape[-2:]
    assert item["mask"].max() == pytest.approx(1.0)


@requires_torchvision
def test_masks_are_ignored_unless_requested(tmp_path):
    root = make_dataset(tmp_path, with_mask=True)
    assert "mask" not in ImageCaptionDataset(str(root), resolution=64, use_masks=False)[0]


@requires_torchvision
def test_use_masks_is_a_noop_without_mask_files(tmp_path):
    """A dataset that never ran autolabel.py must still train."""
    root = make_dataset(tmp_path, with_mask=False)
    assert "mask" not in ImageCaptionDataset(str(root), resolution=64, use_masks=True)[0]


def test_unmasked_loss_is_plain_mse():
    pred, target = torch.zeros(1, 4, 8, 8), torch.ones(1, 4, 8, 8)
    assert _masked_mse(pred, target, None, spatial=True).item() == pytest.approx(1.0)


def test_mask_weights_the_subject_over_the_background():
    """Error inside the mask must count for more than error outside it."""
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0

    subject_error = torch.zeros(1, 4, 8, 8)
    subject_error[..., :4, :] = 1.0  # all error in the masked half
    background_error = torch.zeros(1, 4, 8, 8)
    background_error[..., 4:, :] = 1.0  # same magnitude, unmasked half

    target = torch.zeros(1, 4, 8, 8)
    assert (
        _masked_mse(subject_error, target, mask, spatial=True).item()
        > _masked_mse(background_error, target, mask, spatial=True).item()
    )


def test_background_is_never_fully_unsupervised():
    """Weights are floored at 0.1: a hard 0/1 mask would let the model put
    anything outside the subject."""
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0

    background_error = torch.zeros(1, 4, 8, 8)
    background_error[..., 4:, :] = 1.0
    assert _masked_mse(background_error, torch.zeros(1, 4, 8, 8), mask, spatial=True).item() > 0


def test_packed_token_mask_matches_the_latent_grid():
    """For FLUX.2 the mask is pooled to a square grid and flattened row-major,
    the same order `_pack_latents` uses."""
    mask = torch.ones(1, 1, 32, 32)
    pred, target = torch.zeros(1, 64, 16), torch.ones(1, 64, 16)  # 8x8 tokens
    assert _masked_mse(pred, target, mask, spatial=False).item() == pytest.approx(1.0)


def test_non_square_token_count_falls_back_to_plain_mse():
    mask = torch.ones(1, 1, 32, 32)
    pred, target = torch.zeros(1, 50, 16), torch.ones(1, 50, 16)  # not a square
    assert _masked_mse(pred, target, mask, spatial=False).item() == pytest.approx(1.0)


def test_neutral_background_replaces_only_the_unmasked_region():
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (0, 0, 8, 16))

    out = mask_to_neutral_background(image, mask)
    assert out.getpixel((2, 8)) == (255, 0, 0)  # kept
    assert out.getpixel((12, 8)) == (128, 128, 128)  # mid-grey, not white or black
