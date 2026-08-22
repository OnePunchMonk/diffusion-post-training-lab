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
    def __init__(self, dataset_path: str, resolution: int = 1024):
        self.root = Path(dataset_path)
        self.resolution = resolution
        self.records = [json.loads(line) for line in (self.root / "metadata.jsonl").read_text().splitlines()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        import torchvision.transforms as T

        record = self.records[idx]
        image = Image.open(self.root / record["file_name"]).convert("RGB")
        transform = T.Compose(
            [
                T.Resize(self.resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        return {"pixel_values": transform(image), "caption": record["caption"]}


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
