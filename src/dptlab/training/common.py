"""Shared training utilities used by all three recipes (lora / dpo / distill).

Keeping this separate is what makes the three recipes comparable: they read
the same YAML config shape, log to the same run-tracking convention, and save
checkpoints in the same layout that the eval harness and Modal server expect.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class TrainConfig:
    model_key: str
    recipe: str  # "lora" | "dpo" | "distill"
    dataset_path: str
    output_dir: str
    resolution: int = 1024
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 2000
    lora_rank: int = 16
    lora_alpha: int = 16
    mixed_precision: str = "bf16"
    seed: int = 42
    checkpointing_steps: int = 500
    validation_prompts: list[str] = dataclasses.field(default_factory=list)
    validation_steps: int = 500
    extra: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        extra = {k: v for k, v in raw.items() if k not in known}
        base = {k: v for k, v in raw.items() if k in known}
        return cls(**base, extra=extra)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_manifest(output_dir: str | Path, config: TrainConfig, extra: dict | None = None) -> None:
    """Write run_manifest.json into the checkpoint dir.

    The eval harness and the Modal server both read this to know which base
    model + recipe a checkpoint came from, so a checkpoint directory is
    self-describing without needing a separate DB.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": dataclasses.asdict(config), **(extra or {})}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def load_run_manifest(checkpoint_dir: str | Path) -> dict:
    return json.loads((Path(checkpoint_dir) / "run_manifest.json").read_text())
