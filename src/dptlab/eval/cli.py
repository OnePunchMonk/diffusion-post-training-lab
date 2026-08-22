"""dptlab-eval CLI.

Examples:
  dptlab eval --model-key sdxl --prompts prompts/geneval_mini.jsonl
  dptlab eval --checkpoint outputs/dpo/final --baseline-model-key sdxl --prompts prompts/geneval_mini.jsonl
"""

from __future__ import annotations

import argparse
import json

from dptlab.eval.adapters.checkpoint import CheckpointAdapter
from dptlab.eval.runner import load_prompts, run_eval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dptlab")
    sub = p.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="Run the eval harness against a checkpoint.")
    ev.add_argument("--model-key", help="Base model key from the registry (e.g. sdxl, flux-schnell).")
    ev.add_argument("--checkpoint", help="Path to a post-trained checkpoint dir (has run_manifest.json).")
    ev.add_argument("--baseline-model-key", help="If set, also run this base model and report win rate vs it.")
    ev.add_argument("--prompts", required=True, help="Path to a prompt list (.txt or .jsonl with a 'prompt' key).")
    ev.add_argument("--num-inference-steps", type=int, default=30)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--output-dir", default="eval_results/run")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "eval":
        adapter = CheckpointAdapter(model_key=args.model_key, checkpoint_dir=args.checkpoint)
        baseline = CheckpointAdapter(model_key=args.baseline_model_key) if args.baseline_model_key else None
        prompts = load_prompts(args.prompts)

        report = run_eval(
            adapter,
            prompts,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            baseline_adapter=baseline,
            output_dir=args.output_dir,
        )
        print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
