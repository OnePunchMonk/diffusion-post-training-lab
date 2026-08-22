"""LAION aesthetic predictor: a linear head on CLIP ViT-L/14 embeddings,
trained on human aesthetic ratings (Schuhmann et al., LAION-Aesthetics).

This is the metric vlm-harness's generative eval didn't have (it only shipped
CLIPScore, FID, and a GenEval-style compositional checker) — worth adding
because CLIPScore alone rewards literal prompt-following, not "does this
look good," which is exactly the axis Diffusion-DPO/GRPO are meant to move.
Reporting CLIPScore *and* aesthetic score side by side lets you show a
recipe traded one for the other (or improved both), which is a much better
eval story than a single number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

_HEAD_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
_CLIP_MODEL = "openai/clip-vit-large-patch14"


@dataclass
class MetricResult:
    metric_name: str
    value: float
    n_samples: int
    per_sample: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AestheticScorer:
    def __init__(self, device: str = "auto"):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("pip install dptlab[eval]")

        self._torch = torch
        self._device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip = CLIPModel.from_pretrained(_CLIP_MODEL).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(_CLIP_MODEL)
        self._head = self._load_head()

    def _load_head(self):
        from torch import nn
        from torch.hub import load_state_dict_from_url

        head = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        state_dict = load_state_dict_from_url(_HEAD_URL, map_location=self._device)
        head.load_state_dict(state_dict)
        return head.to(self._device).eval()

    def score(self, image: Image.Image) -> float:
        inputs = self._processor(images=[image], return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            emb = self._clip.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            score = self._head(emb.float())
        return float(score.item())

    def compute(self, images: list[Image.Image], sample_ids: list[str] | None = None) -> MetricResult:
        ids = sample_ids or [str(i) for i in range(len(images))]
        per_sample = {sid: self.score(im) for sid, im in zip(ids, images)}
        avg = sum(per_sample.values()) / len(per_sample) if per_sample else float("nan")
        return MetricResult(
            metric_name="aesthetic_score", value=avg, n_samples=len(images), per_sample=per_sample,
            metadata={"clip_model": _CLIP_MODEL, "head": "laion-improved-aesthetic-predictor"},
        )
