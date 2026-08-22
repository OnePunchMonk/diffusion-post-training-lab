"""Preference-pair dataset for the Diffusion-DPO recipe.

Layout:
  dataset_dir/
    metadata.jsonl   # {"prompt": "...", "win": "0001_win.png", "lose": "0001_lose.png"}
    0001_win.png
    0001_lose.png
    ...

Pairs can come from a human-labeled preference dataset (e.g. Pick-a-Pic-style)
or be auto-generated: sample two images per prompt from the base model and
rank them with `dptlab.eval.metrics` (see scripts/build_preference_pairs.py),
which mirrors the "verifier-driven preference generation" idea used in
Kimi k1.5 / DeepSeek-style RL pipelines to avoid needing human labels.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, dataset_path: str, resolution: int = 1024):
        self.root = Path(dataset_path)
        self.resolution = resolution
        self.records = [json.loads(line) for line in (self.root / "metadata.jsonl").read_text().splitlines()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        import torchvision.transforms as T

        record = self.records[idx]
        transform = T.Compose(
            [
                T.Resize(self.resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        win = transform(Image.open(self.root / record["win"]).convert("RGB"))
        lose = transform(Image.open(self.root / record["lose"]).convert("RGB"))
        return {"prompt": record["prompt"], "image_win": win, "image_lose": lose}
