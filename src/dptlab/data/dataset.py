"""Datasets for the LoRA and GRPO/distill recipes.

Layout on disk (kept dead simple so a new concept dataset is just a folder):
  dataset_dir/
    metadata.jsonl   # {"file_name": "0001.png", "caption": "a photo of ..."}
    0001.png
    0002.png
    ...

`PromptOnlyDataset` reads a flat list of prompts (one per line, or a jsonl
with a "prompt" key) for recipes that only need text, not paired images
(GRPO, distillation).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    """(image, caption) pairs, optionally with a subject mask.

    A record may carry a `mask_file_name` written by `flux2-klein-peft/scripts/autolabel.py`
    (SAM subject segmentation). When `use_masks=True` the mask is loaded and
    cropped identically to the image so the two stay pixel-aligned -- the
    geometric transforms are therefore applied deterministically, not with
    independent random crops/flips.
    """

    def __init__(self, dataset_path: str, resolution: int = 1024, use_masks: bool = False):
        self.root = Path(dataset_path)
        self.resolution = resolution
        self.use_masks = use_masks
        self.records = [
            json.loads(line) for line in (self.root / "metadata.jsonl").read_text().splitlines() if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _geometry(self, interpolation):
        import torchvision.transforms as T

        return T.Compose(
            [
                T.Resize(self.resolution, interpolation=interpolation),
                T.CenterCrop(self.resolution),
            ]
        )

    def __getitem__(self, idx: int) -> dict:
        import torchvision.transforms as T

        record = self.records[idx]
        image = Image.open(self.root / record["file_name"]).convert("RGB")
        image = self._geometry(T.InterpolationMode.BILINEAR)(image)
        pixel_values = T.Normalize([0.5], [0.5])(T.ToTensor()(image))

        item = {"pixel_values": pixel_values, "caption": record["caption"]}

        mask_name = record.get("mask_file_name")
        if self.use_masks and mask_name:
            mask = Image.open(self.root / mask_name).convert("L")
            # Nearest keeps the mask binary through the resize; the loss
            # soft-clamps it anyway, but area-pooling a blurred mask later
            # would double-smooth the subject boundary.
            mask = self._geometry(T.InterpolationMode.NEAREST)(mask)
            item["mask"] = T.ToTensor()(mask)

        return item


class PromptOnlyDataset(Dataset):
    def __init__(self, dataset_path: str):
        path = Path(dataset_path)
        if path.suffix == ".jsonl":
            self.prompts = [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line.strip()]
        else:
            self.prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {"prompt": self.prompts[idx]}
