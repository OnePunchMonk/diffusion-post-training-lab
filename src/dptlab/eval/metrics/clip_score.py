"""CLIPScore: prompt-image alignment via CLIP embedding cosine similarity.

Reference: Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for
Image Captioning" (2021). Adapted from vlm-harness's `metrics/generative/clip_score.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class MetricResult:
    metric_name: str
    value: float
    n_samples: int
    per_sample: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class CLIPScorer:
    def __init__(self, model_id: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("pip install dptlab[eval]")

        self._torch = torch
        self._model_id = model_id or _DEFAULT_MODEL
        self._model = CLIPModel.from_pretrained(self._model_id)
        self._processor = CLIPProcessor.from_pretrained(self._model_id)
        self._model.eval()

    def score(self, prompt: str, image: Image.Image) -> float:
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        with self._torch.no_grad():
            out = self._model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos = float((img_emb * txt_emb).sum(dim=-1).item())
        return max(cos, 0.0) * 2.5

    def compute(
        self, prompts: list[str], images: list[Image.Image], sample_ids: list[str] | None = None
    ) -> MetricResult:
        ids = sample_ids or [str(i) for i in range(len(prompts))]
        per_sample = {sid: self.score(p, im) for sid, p, im in zip(ids, prompts, images)}
        avg = sum(per_sample.values()) / len(per_sample) if per_sample else float("nan")
        return MetricResult(
            metric_name="clip_score", value=avg, n_samples=len(prompts), per_sample=per_sample,
            metadata={"clip_model": self._model_id},
        )
