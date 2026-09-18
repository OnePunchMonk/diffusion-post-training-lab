"""Step 1: print the eval command, then score its output independently.

Two modes, and the split is the point:

  --print-command   emits the DeepSpeed invocation to run inside the upstream
                    checkout, with our pinned paths substituted in.
  --score DIR       reads the masks that run wrote out and scores them with
                    *our* metric implementation against *our* rasterized ground
                    truth, then compares to the published number.

Running their script and reading the number it prints would only establish
that their code runs. Scoring their predicted masks with an independent metric
and independent ground truth is what makes a match evidence about the method
rather than about the harness.

Their eval script has three things that need overriding, none of them
interesting but all of them blocking: the GPU index is hardcoded
(`--include "localhost:5"`), the dataset directory is a sibling path
(`../dataset_sesame`), and the split is chosen through an interactive prompt.

  python scripts/eval_upstream.py --print-command
  python scripts/eval_upstream.py --score runs/eval-val --root data/reason_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asvl.data.reasonseg import ground_truth_masks, load_split, query_types
from asvl.metrics.reasonseg import compute_scores, missing_predictions
from asvl.upstream.pins import SAM_VIT_H_FILENAME, UPSTREAM_COMMIT, verdict

COMMAND_TEMPLATE = """\
# Run from inside {repo_dir} (upstream code, pinned at {commit}).
# Their launcher hardcodes GPU 5 and prompts for the split; both are overridden here.

deepspeed --include "localhost:{gpu}" --master_port {port} train_ds.py \\
  --model_key="AnchorSeg" \\
  --version="{weights}" \\
  --dataset_dir="{dataset_dir}" \\
  --vision_tower="{clip}" \\
  --vision_pretrained="{sam}" \\
  --val_dataset="ReasonSeg|val" \\
  --dataset="vqa" --sample_rates="1" \\
  --exp_name="{exp_name}" \\
  --seg_token_num=1 --num_classes_per_question=1 --batch_size=1 \\
  --image_feature_scale_num=1 --num_layers=33 \\
  --strategy="policy_walker" --mode=1 \\
  --baseline_type="ema" --baseline_beta=1.0 \\
  --separate_mm_projector --use_expand_question_list \\
  --eval_only --no_resume --eval_legacy --pad_train_clip_images \\
  --preprocessor_config='./configs/preprocessor_336.json'
"""


def print_command(args: argparse.Namespace) -> None:
    upstream = Path(args.upstream).resolve()
    print(
        COMMAND_TEMPLATE.format(
            repo_dir=upstream / "AnchorSeg",
            commit=UPSTREAM_COMMIT[:12],
            gpu=args.gpu,
            port=args.port,
            weights=args.weights or f"{upstream}/hf-cache  # snapshot dir from setup_upstream.py",
            dataset_dir=Path(args.dataset_dir).resolve(),
            clip=args.clip or f"{upstream}/hf-cache  # clip-vit-large-patch14-336 snapshot",
            sam=upstream / SAM_VIT_H_FILENAME,
            exp_name=args.exp_name,
        )
    )
    print(
        "Their dataloader expects ReasonSeg under "
        f"{Path(args.dataset_dir).resolve()}/reason_seg/ReasonSeg/val -- symlink our copy there:\n"
        f"  mkdir -p {args.dataset_dir}/reason_seg/ReasonSeg\n"
        f"  ln -s {Path(args.root).resolve()}/val {args.dataset_dir}/reason_seg/ReasonSeg/val\n"
    )
    print("Then score it independently:\n  python scripts/eval_upstream.py --score <output-dir>")


def score(args: argparse.Namespace) -> None:
    import numpy as np

    samples = load_split(args.root, args.split)
    targets = ground_truth_masks(samples)
    types = query_types(samples)

    prediction_dir = Path(args.score)
    predictions: dict[str, np.ndarray] = {}
    for path in sorted(prediction_dir.glob("*.npy")):
        predictions[path.stem] = np.load(path).astype(bool)
    for path in sorted(prediction_dir.glob("*.png")):
        from PIL import Image

        with Image.open(path) as image:
            predictions[path.stem] = np.array(image.convert("L")) > 127

    if not predictions:
        raise SystemExit(
            f"No .npy or .png masks under {prediction_dir}. Upstream's eval reports aggregate "
            "numbers rather than dumping per-sample masks, so it needs a small patch to write "
            "them out -- see docs/learning-doc.md."
        )

    absent = missing_predictions(predictions, targets)
    if absent:
        print(
            f"WARNING: {len(absent)} of {len(targets)} samples have no prediction; "
            "scored as empty."
        )
        print(f"         first few: {absent[:5]}")

    scores = compute_scores(predictions, targets, query_types=types)

    print(f"\n{'split':<28}{'gIoU':>8}{'cIoU':>8}{'n':>8}")
    print(scores.as_row(f"ReasonSeg {args.split}"))
    for name, subset in scores.by_query_type.items():
        print(subset.as_row(f"  {name} queries"))
    print()
    print(verdict(scores.giou * 100, scores.ciou * 100))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--root", default="data/reason_seg")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dataset-dir", default="data/upstream_layout")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", default="29500")
    ap.add_argument("--exp-name", default="asvl-repro-reasonseg-val")
    ap.add_argument("--print-command", action="store_true")
    ap.add_argument("--score", default=None, help="Directory of per-sample predicted masks.")
    args = ap.parse_args()

    if args.score:
        score(args)
    else:
        print_command(args)


if __name__ == "__main__":
    main()
