"""Eval runner: generate images for a prompt set with one or two checkpoints,
score them, and write a report. Mirrors the shape of vlm-harness's
`engine/generative_runner.py` (adapter -> prompts -> images -> metrics ->
report), trimmed to the metrics dptlab actually ships (CLIPScore, aesthetic
score, pairwise win rate) instead of GenEval/FID/LLM-judge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dptlab.eval.adapters.base import T2IAdapter
from dptlab.eval.metrics.aesthetic import AestheticScorer
from dptlab.eval.metrics.clip_score import CLIPScorer
from dptlab.eval.metrics.winrate import compute_win_rate


@dataclass
class EvalReport:
    model_id: str
    n_prompts: int
    clip_score: float
    aesthetic_score: float
    avg_latency_ms: float
    win_rate_vs_baseline: float | None = None
    metadata: dict = field(default_factory=dict)


def load_prompts(prompts_path: str) -> list[str]:
    path = Path(prompts_path)
    if path.suffix == ".jsonl":
        return [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line.strip()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def run_eval(
    adapter: T2IAdapter,
    prompts: list[str],
    seed: int = 0,
    num_inference_steps: int = 30,
    baseline_adapter: T2IAdapter | None = None,
    output_dir: str | None = None,
) -> EvalReport:
    clip_scorer = CLIPScorer()
    aesthetic_scorer = AestheticScorer()

    images, latencies = [], []
    for i, prompt in enumerate(prompts):
        resp = adapter.generate(prompt, seed=seed + i, num_inference_steps=num_inference_steps)
        images.append(resp.image)
        latencies.append(resp.latency_ms)
        if output_dir:
            out = Path(output_dir) / "images"
            out.mkdir(parents=True, exist_ok=True)
            resp.image.save(out / f"{i:04d}.png")

    clip_result = clip_scorer.compute(prompts, images)
    aesthetic_result = aesthetic_scorer.compute(images)

    win_rate = None
    if baseline_adapter is not None:
        baseline_images = [
            baseline_adapter.generate(p, seed=seed + i, num_inference_steps=num_inference_steps).image
            for i, p in enumerate(prompts)
        ]
        wr = compute_win_rate(
            prompts, baseline_images, images,
            judge=lambda p, im: 0.5 * clip_scorer.score(p, im) + 0.5 * (aesthetic_scorer.score(im) / 10.0),
        )
        win_rate = wr.win_rate_b_over_a

    report = EvalReport(
        model_id=adapter.model_id,
        n_prompts=len(prompts),
        clip_score=clip_result.value,
        aesthetic_score=aesthetic_result.value,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else float("nan"),
        win_rate_vs_baseline=win_rate,
        metadata={"num_inference_steps": num_inference_steps, "seed": seed},
    )

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "report.json").write_text(json.dumps(asdict(report), indent=2))

    return report
