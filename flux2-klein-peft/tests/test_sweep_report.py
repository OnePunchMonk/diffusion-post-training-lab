"""The sweep's report table is what gets pasted into MODELS.md, so it has to
survive partial sweeps: missing judge columns, single-subject cells, and
methods that haven't run yet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # flux2-klein-peft/scripts

from sweep_peft import report


def write_cell(root: Path, method: str, subject: str, **values) -> None:
    path = root / method / subject
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(values))


def test_report_shows_spread_across_subjects(tmp_path):
    write_cell(tmp_path, "lora", "s0", clip_t=0.30, dino=0.60, clip_i=0.80, trainable_parameters=10_000_000)
    write_cell(tmp_path, "lora", "s1", clip_t=0.32, dino=0.64, clip_i=0.82, trainable_parameters=10_000_000)

    table = report(tmp_path)
    assert "| lora |" in table
    assert "10.0M" in table
    # Two subjects must report a standard deviation: a bare mean is how a
    # difference smaller than the noise gets written up as a win.
    assert "±" in table


def test_single_subject_omits_meaningless_stdev(tmp_path):
    write_cell(tmp_path, "oft", "s0", clip_t=0.31, dino=0.62, clip_i=0.81)
    table = report(tmp_path)
    assert "0.310" in table
    assert "±" not in table


def test_judge_columns_appear_only_when_judged(tmp_path):
    write_cell(tmp_path, "lora", "s0", clip_t=0.30, dino=0.60, clip_i=0.80)
    assert "concept_preservation" not in report(tmp_path)

    write_cell(tmp_path, "dora", "s0", clip_t=0.31, dino=0.61, clip_i=0.81, concept_preservation=3.2, prompt_following=2.8)
    table = report(tmp_path)
    assert "concept_preservation" in table
    # The un-judged method still gets a row, with placeholders rather than a crash.
    lora_row = next(line for line in table.splitlines() if line.startswith("| lora |"))
    assert lora_row.endswith("| - | - |")
