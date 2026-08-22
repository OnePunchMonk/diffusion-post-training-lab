import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from update_models_md import _HEADER, _TABLE_END, _TABLE_START, build_row, upsert_row


def _empty_doc() -> str:
    return f"# Model leaderboard\n\n{_TABLE_START}\n{_HEADER}\n<div/>\n{_TABLE_END}\n"


def test_build_row_formats_fields():
    manifest = {"config": {"recipe": "dpo", "model_key": "sdxl"}}
    eval_report = {
        "clip_score": 0.271,
        "aesthetic_score": 5.83,
        "win_rate_vs_baseline": 0.62,
        "avg_latency_ms": 812.4,
        "metadata": {"num_inference_steps": 30},
    }
    row = build_row("OnePunchMonk/dptlab-sdxl-dpo-v1", manifest, eval_report, "first run")
    assert "dpo" in row
    assert "sdxl" in row
    assert "0.271" in row
    assert "62.0%" in row
    assert "first run" in row


def test_upsert_row_appends_new_and_replaces_existing():
    doc = _empty_doc()
    row_a = "| [a](https://huggingface.co/OnePunchMonk/a) | lora | sdxl | 0.2 | 5.0 | n/a | 500 | 30 | 2026-08-22 | |"
    doc = upsert_row(doc, "OnePunchMonk/a", row_a)
    assert row_a in doc

    row_a_v2 = "| [a](https://huggingface.co/OnePunchMonk/a) | lora | sdxl | 0.3 | 6.0 | n/a | 500 | 30 | 2026-08-23 | v2 |"
    doc = upsert_row(doc, "OnePunchMonk/a", row_a_v2)
    assert doc.count("OnePunchMonk/a") == 1
    assert row_a_v2 in doc
    assert row_a not in doc
