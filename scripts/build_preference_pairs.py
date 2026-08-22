"""Auto-generate Diffusion-DPO preference pairs without human labels.

For each prompt, sample two images from the base model at different seeds
and rank them with the eval harness's own scorers (CLIP score + aesthetic
score) — the higher-scoring image becomes "win", the other becomes "lose".
This is the same "verifier as labeler" idea behind Kimi k1.5 / DeepSeek-style
RL data pipelines, applied to image preference data instead of human raters.

Usage:
  python scripts/build_preference_pairs.py \
      --model-key sdxl --prompts prompts/dpo_seed_prompts.txt \
      --out-dir data/dpo_pairs --samples-per-prompt 4
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from dptlab.eval.adapters.checkpoint import CheckpointAdapter
from dptlab.eval.metrics.aesthetic import AestheticScorer
from dptlab.eval.metrics.clip_score import CLIPScorer
from dptlab.eval.runner import load_prompts


def combined_score(clip_scorer: CLIPScorer, aesthetic_scorer: AestheticScorer, prompt: str, image) -> float:
    return 0.5 * clip_scorer.score(prompt, image) + 0.5 * (aesthetic_scorer.score(image) / 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--min-score-gap", type=float, default=0.15, help="Skip pairs that are too close to call.")
    args = ap.parse_args()

    adapter = CheckpointAdapter(model_key=args.model_key)
    clip_scorer = CLIPScorer()
    aesthetic_scorer = AestheticScorer()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for pair_idx, prompt in enumerate(load_prompts(args.prompts)):
        scored = []
        for seed in range(args.samples_per_prompt):
            resp = adapter.generate(prompt, seed=seed)
            score = combined_score(clip_scorer, aesthetic_scorer, prompt, resp.image)
            scored.append((score, resp.image))
        scored.sort(key=lambda t: t[0], reverse=True)

        best_score, best_img = scored[0]
        worst_score, worst_img = scored[-1]
        if best_score - worst_score < args.min_score_gap:
            continue  # ambiguous group, not a useful preference signal

        win_name, lose_name = f"{pair_idx:05d}_win.png", f"{pair_idx:05d}_lose.png"
        best_img.save(out_dir / win_name)
        worst_img.save(out_dir / lose_name)
        records.append({"prompt": prompt, "win": win_name, "lose": lose_name})

    (out_dir / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    print(f"Wrote {len(records)} preference pairs to {out_dir}")


if __name__ == "__main__":
    main()
