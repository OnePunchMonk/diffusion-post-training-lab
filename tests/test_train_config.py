from pathlib import Path

from dptlab.training.common import TrainConfig


def test_from_yaml_parses_known_and_extra_fields(tmp_path: Path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
recipe: dpo
model_key: sdxl
dataset_path: data/dpo_pairs
output_dir: outputs/dpo-sdxl
dpo_beta: 5000.0
"""
    )
    config = TrainConfig.from_yaml(config_path)
    assert config.recipe == "dpo"
    assert config.model_key == "sdxl"
    assert config.extra["dpo_beta"] == 5000.0
