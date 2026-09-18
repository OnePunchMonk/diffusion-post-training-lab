"""Run the PEFT-method sweep: {methods} x {subjects}, then score every run.

The comparison this produces is the point of the whole exercise, so the driver
is deliberately boring: it trains one adapter per (method, subject) cell from
the same base config, evaluates each on that subject's held-out context prompts,
and writes one tidy row per cell. Nothing is averaged before it hits disk --
per-subject variance on 3-image subjects is large, and a table of means with no
spread is how a 0.01 difference gets reported as a win.

  cd flux2-klein-peft

  # 1. data
  python scripts/prepare_syncd.py --num-subjects 20 --out data/syncd-20
  python scripts/autolabel.py --dataset data/syncd-20/subject-0 --class-name "flip flops"

  # 2. sweep (resumable -- completed cells are skipped)
  python scripts/sweep_peft.py --data-root data/syncd-20 --out runs/klein-peft

  # 3. table
  python scripts/sweep_peft.py --data-root data/syncd-20 --out runs/klein-peft --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from dptlab.training.common import TrainConfig
from dptlab.training.peft_methods import list_methods

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Resolved relative to this file, not the cwd: the sweep is long-running and
# usually launched from the repo root, but the configs belong to this subproject.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def subjects_in(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir() and (p / "metadata.jsonl").exists())


def cell_dir(out_root: Path, method: str, subject: str) -> Path:
    return out_root / method / subject


def train_cell(method: str, subject_dir: Path, out_dir: Path, overrides: dict) -> Path:
    from dptlab.training.lora import train_lora

    config = TrainConfig.from_yaml(CONFIG_DIR / f"{method}.yaml")
    config.dataset_path = str(subject_dir)
    config.output_dir = str(out_dir)
    for key, value in overrides.items():
        setattr(config, key, value)
    return train_lora(config)


def eval_cell(checkpoint_dir: Path, subject_dir: Path, out_dir: Path, judge: bool, seed: int) -> dict:
    from PIL import Image

    from dptlab.eval.adapters.checkpoint import CheckpointAdapter
    from dptlab.eval.metrics.clip_score import CLIPScorer
    from dptlab.eval.metrics.subject_fidelity import (
        CLIPImageScorer,
        DINOScorer,
        mask_to_neutral_background,
    )

    rows = [json.loads(line) for line in (subject_dir / "prompts_eval.jsonl").read_text().splitlines() if line.strip()]
    prompts = [r["prompt"] for r in rows]
    ids = [r["id"] for r in rows]

    # References are the subject's training images -- the same photos the
    # adapter saw. That's intentional for subject fidelity (we're asking "is it
    # the same object"); it's why prompt following is scored separately, on
    # contexts that were held out.
    train_records = [
        json.loads(line) for line in (subject_dir / "metadata.jsonl").read_text().splitlines() if line.strip()
    ]
    references = []
    for record in train_records:
        image = Image.open(subject_dir / record["file_name"]).convert("RGB")
        if record.get("mask_file_name"):
            image = mask_to_neutral_background(image, Image.open(subject_dir / record["mask_file_name"]))
        references.append(image)

    adapter = CheckpointAdapter(checkpoint_dir=str(checkpoint_dir))
    images, latencies = [], []
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        response = adapter.generate(prompt, seed=seed + i)
        response.image.save(image_dir / f"{ids[i]}.png")
        images.append(response.image)
        latencies.append(response.latency_ms)

    clip_t = CLIPScorer().compute(prompts, images, sample_ids=ids)
    dino = DINOScorer().compute(images, references, sample_ids=ids)
    clip_i = CLIPImageScorer().compute(images, references, sample_ids=ids)

    result = {
        "clip_t": clip_t.value,
        "dino": dino.value,
        "clip_i": clip_i.value,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else float("nan"),
        "n_prompts": len(prompts),
        "per_sample": {"clip_t": clip_t.per_sample, "dino": dino.per_sample, "clip_i": clip_i.per_sample},
    }

    if judge:
        from dptlab.eval.metrics.dreambench_judge import DreamBenchJudge

        scores = DreamBenchJudge().compute(prompts, images, references[0], sample_ids=ids)
        result["concept_preservation"] = scores["concept_preservation"].value
        result["prompt_following"] = scores["prompt_following"].value
        result["per_sample"]["concept_preservation"] = scores["concept_preservation"].per_sample
        result["per_sample"]["prompt_following"] = scores["prompt_following"].per_sample

    return result


def report(out_root: Path) -> str:
    """Mean +/- std over subjects, per method. Written as Markdown for MODELS.md."""
    import statistics

    cells: dict[str, list[dict]] = {}
    for path in sorted(out_root.glob("*/*/result.json")):
        cells.setdefault(path.parent.parent.name, []).append(json.loads(path.read_text()))

    columns = ["clip_t", "dino", "clip_i", "concept_preservation", "prompt_following"]
    present = [c for c in columns if any(c in r for rows in cells.values() for r in rows)]

    header = "| method | params | subjects | " + " | ".join(present) + " |"
    divider = "|" + "---|" * (3 + len(present))
    lines = [header, divider]

    for method in sorted(cells):
        rows = cells[method]
        params = rows[0].get("trainable_parameters")
        cell_values = []
        for column in present:
            values = [r[column] for r in rows if column in r]
            if not values:
                cell_values.append("-")
            elif len(values) == 1:
                cell_values.append(f"{values[0]:.3f}")
            else:
                cell_values.append(f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}")
        params_str = f"{params / 1e6:.1f}M" if params else "-"
        lines.append(f"| {method} | {params_str} | {len(rows)} | " + " | ".join(cell_values) + " |")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Output of scripts/prepare_syncd.py.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*", default=list_methods())
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None, help="Override the configs, e.g. for a smoke run.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--judge", action="store_true", help="Also run the DreamBench++ Claude judge (costs money).")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    data_root = Path(args.data_root)

    if args.report_only:
        print(report(out_root))
        return

    subjects = subjects_in(data_root)[: args.max_subjects]
    if not subjects:
        raise SystemExit(f"No subjects with metadata.jsonl under {data_root}")

    overrides = {"max_train_steps": args.max_train_steps} if args.max_train_steps else {}

    for method in args.methods:
        for subject_dir in subjects:
            out_dir = cell_dir(out_root, method, subject_dir.name)
            result_path = out_dir / "result.json"
            if result_path.exists():
                logger.info("skip %s/%s (already done)", method, subject_dir.name)
                continue

            logger.info("train %s on %s", method, subject_dir.name)
            started = time.perf_counter()
            checkpoint = train_cell(method, subject_dir, out_dir / "ckpt", overrides)
            train_seconds = time.perf_counter() - started

            logger.info("eval %s on %s", method, subject_dir.name)
            result = eval_cell(checkpoint, subject_dir, out_dir, args.judge, args.seed)
            manifest = json.loads((checkpoint / "run_manifest.json").read_text())
            result.update(
                {
                    "method": method,
                    "subject": subject_dir.name,
                    "train_seconds": train_seconds,
                    "trainable_parameters": manifest.get("trainable_parameters"),
                    "model_key": manifest["config"]["model_key"],
                }
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2))

    print(report(out_root))


if __name__ == "__main__":
    main()
