"""Subject fidelity: does the generated image show *this* object?

CLIPScore answers "does the image match the prompt", which is exactly the axis a
personalization adapter is allowed to sacrifice when it overfits. Scoring only
CLIPScore would rank a checkpoint that ignored the subject entirely above one
that learned it, so the sweep needs a second axis pointing the other way.

Two embedding metrics, both from the DreamBooth/DreamBench++ line of work:

- **DINO** (self-supervised ViT). The standard subject-fidelity metric, and the
  stricter of the two: DINO is trained without class labels, so its features
  discriminate between *instances* of a category rather than collapsing them.
  Two different backpacks score high on CLIP-I and low on DINO -- which is
  precisely the distinction personalization is being graded on.
- **CLIP-I** (CLIP image-image). Looser, more semantic; reported alongside DINO
  because it's what most papers quote and because a large CLIP-I/DINO gap is
  itself the signal that a method learned the category but not the instance.

Both are cosine similarities in [0, 1] against the subject's training images,
averaged over references. If a subject mask is supplied (from
`flux2-klein-peft/scripts/autolabel.py`), the reference is composited onto a neutral background
first, so the score measures the object rather than a shared backdrop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

_DEFAULT_DINO = "facebook/dino-vits16"
_DEFAULT_CLIP = "openai/clip-vit-base-patch32"


@dataclass
class MetricResult:
    metric_name: str
    value: float
    n_samples: int
    per_sample: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def mask_to_neutral_background(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Composite the masked subject onto mid-grey.

    Grey rather than white or black: the reference and the generation are both
    embedded by the same encoder, and a saturated background shifts the
    embedding more than a neutral one does.
    """
    mask = mask.convert("L").resize(image.size, Image.NEAREST)
    background = Image.new("RGB", image.size, (128, 128, 128))
    return Image.composite(image.convert("RGB"), background, mask)


class _EmbeddingScorer:
    """Shared cosine-similarity-against-references machinery."""

    metric_name = "subject_fidelity"

    def _embed(self, images: list[Image.Image]):  # pragma: no cover - subclass hook
        raise NotImplementedError

    def score(self, image: Image.Image, references: list[Image.Image]) -> float:
        embeddings = self._embed([image, *references])
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        sims = embeddings[0:1] @ embeddings[1:].T
        return float(sims.mean().item())

    def compute(
        self,
        images: list[Image.Image],
        references: list[Image.Image],
        sample_ids: list[str] | None = None,
    ) -> MetricResult:
        ids = sample_ids or [str(i) for i in range(len(images))]
        per_sample = {sid: self.score(im, references) for sid, im in zip(ids, images)}
        avg = sum(per_sample.values()) / len(per_sample) if per_sample else float("nan")
        return MetricResult(
            metric_name=self.metric_name,
            value=avg,
            n_samples=len(images),
            per_sample=per_sample,
            metadata={"model": self._model_id, "n_references": len(references)},
        )


class DINOScorer(_EmbeddingScorer):
    metric_name = "dino_score"

    def __init__(self, model_id: str | None = None):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError("pip install dptlab[eval]")

        self._torch = torch
        self._model_id = model_id or _DEFAULT_DINO
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id).eval()

    def _embed(self, images: list[Image.Image]):
        inputs = self._processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        # The CLS token, not mean-pooled patches: DINO's instance-level signal
        # lives in CLS, and pooling patches washes it toward category identity.
        return outputs.last_hidden_state[:, 0]


class CLIPImageScorer(_EmbeddingScorer):
    metric_name = "clip_i_score"

    def __init__(self, model_id: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("pip install dptlab[eval]")

        self._torch = torch
        self._model_id = model_id or _DEFAULT_CLIP
        self._processor = CLIPProcessor.from_pretrained(self._model_id)
        self._model = CLIPModel.from_pretrained(self._model_id).eval()

    def _embed(self, images: list[Image.Image]):
        inputs = self._processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with self._torch.no_grad():
            return self._model.get_image_features(**inputs)
