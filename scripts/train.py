"""Single entry point for all four recipes.

Usage:
  accelerate launch scripts/train.py --config configs/recipes/lora.yaml
  accelerate launch scripts/train.py --config configs/recipes/dpo.yaml
  accelerate launch scripts/train.py --config configs/recipes/grpo.yaml
  accelerate launch scripts/train.py --config configs/recipes/distill.yaml
"""

from __future__ import annotations

import argparse
import logging

from dptlab.training.common import TrainConfig
from dptlab.training.distill import train_distill
from dptlab.training.dpo import train_dpo
from dptlab.training.grpo import train_grpo
from dptlab.training.lora import train_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_RECIPES = {
    "lora": train_lora,
    "dpo": train_dpo,
    "grpo": train_grpo,
    "distill": train_distill,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a recipe YAML config.")
    args = ap.parse_args()

    config = TrainConfig.from_yaml(args.config)
    if config.recipe not in _RECIPES:
        raise ValueError(f"Unknown recipe {config.recipe!r}. Choose from {list(_RECIPES)}.")

    output_dir = _RECIPES[config.recipe](config)
    logging.info("Training complete. Checkpoint written to %s", output_dir)


if __name__ == "__main__":
    main()
