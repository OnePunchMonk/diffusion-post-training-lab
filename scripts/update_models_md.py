"""Append (or update) one row of MODELS.md from an eval report.

Keyed on repo_id, so re-running this after re-benchmarking a checkpoint you
already pushed replaces its row instead of duplicating it.

Usage:
  python scripts/update_models_md.py --repo-id OnePunchMonk/dptlab-sdxl-dpo-v1 \
      --checkpoint outputs/dpo-sdxl/final --eval-report eval_results/dpo-sdxl/report.json \
      --notes "first DPO run, 1k steps"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from dptlab.training.common import load_run_manifest

_MODELS_MD = Path(__file__).resolve().parent.parent / "MODELS.md"
_TABLE_START = "<!-- BEGIN MODELS_TABLE -->"
_TABLE_END = "<!-- END MODELS_TABLE -->"
_HEADER = "| Model | Recipe | Base | CLIP score | Aesthetic score | Win rate vs. base | Avg. latency (ms) | Steps | Date | Notes |"
_DIVIDER = "|---|---|---|---|---|---|---|---|---|---|"


def _fmt(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def build_row(repo_id: str, manifest: dict, eval_report: dict, notes: str) -> str:
    config = manifest["config"]
    win_rate = eval_report.get("win_rate_vs_baseline")
    win_rate_str = "n/a" if win_rate is None else f"{win_rate:.1%}"
    hub_link = f"[{repo_id.split('/')[-1]}](https://huggingface.co/{repo_id})"
    steps = eval_report.get("metadata", {}).get("num_inference_steps", "n/a")
    date = dt.datetime.now(tz=dt.timezone.utc).date().isoformat()

    return (
        f"| {hub_link} | {config['recipe']} | {config['model_key']} | "
        f"{_fmt(eval_report.get('clip_score'))} | {_fmt(eval_report.get('aesthetic_score'))} | "
        f"{win_rate_str} | {_fmt(eval_report.get('avg_latency_ms'), '.0f')} | {steps} | {date} | {notes} |"
    )


def upsert_row(models_md_text: str, repo_id: str, new_row: str) -> str:
    match = re.search(f"{_TABLE_START}\n(.*?)\n{_TABLE_END}", models_md_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {_TABLE_START}/{_TABLE_END} markers in MODELS.md")

    body = match.group(1)
    lines = [line for line in body.splitlines() if line.strip()]
    data_rows = [line for line in lines if line not in (_HEADER, _DIVIDER)]
    hub_slug = repo_id.split("/")[-1]
    data_rows = [row for row in data_rows if f"]({'https://huggingface.co/' + repo_id})" not in row or hub_slug not in row]
    data_rows = [row for row in data_rows if repo_id not in row]
    data_rows.append(new_row)

    new_table = "\n".join([_HEADER, _DIVIDER, *data_rows])
    return models_md_text[: match.start()] + f"{_TABLE_START}\n{new_table}\n{_TABLE_END}" + models_md_text[match.end():]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-report", required=True)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    manifest = load_run_manifest(args.checkpoint)
    eval_report = json.loads(Path(args.eval_report).read_text())

    row = build_row(args.repo_id, manifest, eval_report, args.notes)
    text = _MODELS_MD.read_text()
    _MODELS_MD.write_text(upsert_row(text, args.repo_id, row))
    print(f"Updated MODELS.md with row for {args.repo_id}")


if __name__ == "__main__":
    main()
