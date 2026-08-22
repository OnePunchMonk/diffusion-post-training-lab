"""Pairwise win-rate: fraction of prompts where checkpoint B beats checkpoint A.

The vlm-harness generative eval only ever reported per-model scalar metrics
(CLIPScore, FID, GenEval) independently per model. That's fine for LoRA
(does the fine-tune look like the target concept) but insufficient for DPO
and GRPO, whose entire point is "policy beats reference on paired samples at
matched prompts/seeds" — a claim you can only support with a paired
comparison, not two independent averages. This computes win rate using any
scalar metric (CLIP score, aesthetic score, or a weighted combination) as
the per-sample judge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class WinRateResult:
    win_rate_b_over_a: float
    n_pairs: int
    ties: int
    per_prompt: dict[str, str] = field(default_factory=dict)  # prompt -> "a" | "b" | "tie"


def compute_win_rate(
    prompts: list[str],
    images_a: list[Image.Image],
    images_b: list[Image.Image],
    judge: Callable[[str, Image.Image], float],
    tie_epsilon: float = 1e-3,
) -> WinRateResult:
    """`judge` scores one (prompt, image) pair, e.g. CLIPScorer.score or
    AestheticScorer.score with a bound prompt. Same seed should be used to
    generate images_a[i]/images_b[i] so the comparison isolates the checkpoint."""
    wins_b, ties, per_prompt = 0, 0, {}
    for prompt, img_a, img_b in zip(prompts, images_a, images_b):
        score_a = judge(prompt, img_a)
        score_b = judge(prompt, img_b)
        if abs(score_a - score_b) <= tie_epsilon:
            ties += 1
            per_prompt[prompt] = "tie"
        elif score_b > score_a:
            wins_b += 1
            per_prompt[prompt] = "b"
        else:
            per_prompt[prompt] = "a"

    n = len(prompts)
    decided = n - ties
    win_rate = wins_b / decided if decided else float("nan")
    return WinRateResult(win_rate_b_over_a=win_rate, n_pairs=n, ties=ties, per_prompt=per_prompt)
