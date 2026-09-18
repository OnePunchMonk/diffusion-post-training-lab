"""DreamBench++ judge: Claude scores concept preservation and prompt following.

DINO and CLIPScore are cheap and reproducible but only weakly agree with human
judgement -- which is the finding DreamBench++ (Peng et al., ICLR 2025) is built
on. Its contribution is a pair of carefully-worded multimodal-LLM rubrics that
correlate with human raters far better than the embedding metrics do, scoring
each generation 0-4 on two independent axes:

- **Concept Preservation (CP)** -- reference image vs. generated image, judged
  on shape, colour, texture and (for living subjects) facial features.
- **Prompt Following (PF)** -- prompt vs. generated image, judged on relevance,
  accuracy, completeness and context.

The two must be read together. An adapter that ignores the subject scores 0 CP
and 4 PF; one that reproduces a training shot verbatim scores 4 CP and 0 PF.
Neither is good, and a single number would hide both.

The rubrics are fetched from the authors' repository at first use and cached, so
this stays faithful to the published wording rather than a paraphrase of it, and
picks up any upstream correction. Two deliberate deviations from the paper:

1. **Judge model.** The paper uses GPT-4o. This runs Claude instead, because
   that's what this project is set up to call. Absolute scores are therefore not
   comparable to numbers published against the GPT-4o judge -- only the ranking
   of methods *within one sweep* is meaningful. Keep one judge across a sweep.
2. **Chain of thought.** The paper's "internal thinking" variant preseeds an
   assistant turn; assistant prefill is rejected on current Claude models, so
   the reasoning is requested in the response and parsed off. This matches the
   paper's `full` (CoT) setting in substance.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

_RUBRIC_BASE = "https://raw.githubusercontent.com/yuangpeng/dreambench_plus/main/dreambench_plus/prompts"
_RUBRICS = {
    "concept_preservation": "user_prompt_subject_full.txt",
    "prompt_following": "user_prompt_text_full.txt",
}
_MAX_SCORE = 4
_JUDGE_SIZE = (512, 512)  # the resolution the benchmark judges at


@dataclass
class JudgeResult:
    metric_name: str
    value: float  # mean score, 0-4
    normalized: float  # mean / 4, for comparing against the [0,1] embedding metrics
    n_samples: int
    per_sample: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def load_rubric(axis: str, cache_dir: str | Path | None = None) -> str:
    """Fetch (and cache) one DreamBench++ rubric."""
    import urllib.request

    if axis not in _RUBRICS:
        raise KeyError(f"Unknown axis {axis!r}. Choose from {sorted(_RUBRICS)}.")

    cache = Path(cache_dir or Path.home() / ".cache" / "dptlab" / "dreambench_plus")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / _RUBRICS[axis]

    if not path.exists():
        with urllib.request.urlopen(f"{_RUBRIC_BASE}/{_RUBRICS[axis]}") as response:
            path.write_bytes(response.read())
    return path.read_text().strip()


def _encode(image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.convert("RGB").resize(_JUDGE_SIZE).save(buffer, format="JPEG", quality=95)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buffer.getvalue()).decode("utf-8"),
        },
    }


def _parse_score(text: str) -> float:
    """Pull the integer score out of the judge's reply.

    The rubric asks for a trailing `Score: N` line. Take the *last* match: the
    CP rubric's own text enumerates the 0-4 scale, and a judge that echoes part
    of the scale before answering would otherwise be read as scoring 0.
    """
    matches = re.findall(r"Score:\s*\[?([0-4])\]?", text)
    if not matches:
        raise ValueError(f"Judge reply had no parseable score:\n{text[:500]}")
    return float(matches[-1])


class DreamBenchJudge:
    def __init__(
        self,
        model: str = "claude-opus-5",
        client=None,
        cache_dir: str | Path | None = None,
        max_tokens: int = 1024,
    ):
        if client is None:
            import anthropic

            # Zero-arg construction resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
            # or an `ant auth login` profile -- don't require an env var here.
            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._cache_dir = cache_dir

    def _ask(self, rubric: str, content: list[dict]) -> float:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=rubric,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Judge declined to score: {response.stop_details}")
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_score(text)

    def score_concept_preservation(self, reference: Image.Image, generated: Image.Image) -> float:
        rubric = load_rubric("concept_preservation", self._cache_dir)
        return self._ask(rubric, [_encode(reference), _encode(generated)])

    def score_prompt_following(self, prompt: str, generated: Image.Image) -> float:
        rubric = load_rubric("prompt_following", self._cache_dir)
        return self._ask(rubric, [{"type": "text", "text": f"Text prompt: {prompt}"}, _encode(generated)])

    def compute(
        self,
        prompts: list[str],
        images: list[Image.Image],
        reference: Image.Image,
        sample_ids: list[str] | None = None,
    ) -> dict[str, JudgeResult]:
        ids = sample_ids or [str(i) for i in range(len(images))]

        cp = {sid: self.score_concept_preservation(reference, im) for sid, im in zip(ids, images)}
        pf = {sid: self.score_prompt_following(p, im) for sid, p, im in zip(ids, prompts, images)}

        return {
            "concept_preservation": self._result("concept_preservation", cp),
            "prompt_following": self._result("prompt_following", pf),
        }

    def _result(self, name: str, per_sample: dict[str, float]) -> JudgeResult:
        mean = sum(per_sample.values()) / len(per_sample) if per_sample else float("nan")
        return JudgeResult(
            metric_name=name,
            value=mean,
            normalized=mean / _MAX_SCORE,
            n_samples=len(per_sample),
            per_sample=per_sample,
            metadata={"judge_model": self._model, "rubric": "dreambench_plus", "scale": f"0-{_MAX_SCORE}"},
        )


def judge_available() -> bool:
    """Whether a judge run can be attempted without prompting for credentials."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()
