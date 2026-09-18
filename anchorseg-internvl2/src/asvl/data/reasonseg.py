"""ReasonSeg loading and polygon rasterization.

ReasonSeg annotations are LabelMe-style JSON, one file per image:

    {"text": ["the cesspit"], "is_sentence": false,
     "shapes": [{"label": "target", "shape_type": "polygon", "points": [[x, y], ...]}, ...]}

Three things in that format decide whether the ground-truth masks you score
against are the same ones the paper scored against:

1. **`label` is not always `"target"`.** Shapes labelled `"ignore"` mark
   regions that must be excluded from the target, not added to it. Treating
   every shape as a target inflates the ground-truth area, which *raises* IoU
   for an over-segmenting model and makes a broken replication look closer to
   the paper than it is.
2. **Multiple `target` polygons are one object**, not one object each. They are
   unioned into a single binary mask -- ReasonSeg targets are frequently
   disconnected (an object seen through railings, say).
3. **Image height and width are absent from the JSON** (`imageHeight` and
   `imageWidth` are null in this release), so the canvas size has to come from
   the image file. Guessing it from the polygon extent silently crops every
   mask to its own bounding box.

`is_sentence` is carried through because the paper reports results split by
query type: short phrases vs. long sentences. The long queries are the ones
that actually require reasoning, so an overall-only number hides the axis the
method is about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_LABEL = "target"
IGNORE_LABEL = "ignore"

# The upstream mirror of LISA's release, same 239/200/779 train/val/test split
# as the paper reports.
HF_DATASET_ID = "fcxfcx/ReasonSeg"
SPLIT_SIZES = {"train": 239, "val": 200, "test": 779}


@dataclass(frozen=True)
class ReasonSegSample:
    sample_id: str
    image_path: Path
    queries: list[str]
    is_sentence: bool
    shapes: list[dict]

    @property
    def query_type(self) -> str:
        return "long" if self.is_sentence else "short"


def rasterize(shapes: list[dict], height: int, width: int) -> np.ndarray:
    """Polygons -> one boolean mask, targets unioned then ignores subtracted.

    Uses pycocotools' RLE path rather than PIL's `ImageDraw.polygon`, because
    the two disagree on boundary pixels: PIL fills a polygon's outline
    inclusively, which adds roughly a one-pixel border. On ReasonSeg's many
    small targets that is a percent-level IoU difference -- enough to make a
    correct replication look like a miss.
    """
    from pycocotools import mask as mask_utils

    target = np.zeros((height, width), dtype=bool)
    ignore = np.zeros((height, width), dtype=bool)

    for shape in shapes:
        points = shape.get("points") or []
        if len(points) < 3:  # a polygon needs at least a triangle
            continue
        flat = [float(c) for point in points for c in point[:2]]
        rles = mask_utils.frPyObjects([flat], height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles)).astype(bool)

        if shape.get("label") == IGNORE_LABEL:
            ignore |= decoded
        else:
            target |= decoded

    return target & ~ignore


def load_split(root: str | Path, split: str = "val") -> list[ReasonSegSample]:
    """Read one split's annotations. Does not open the images."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} not found. Fetch it with "
            f"`python scripts/fetch_reasonseg.py --split {split}`."
        )

    samples = []
    for annotation_path in sorted(split_dir.glob("*.json")):
        record = json.loads(annotation_path.read_text())
        image_path = annotation_path.with_suffix(".jpg")
        if not image_path.exists():
            raise FileNotFoundError(f"{annotation_path} has no matching image at {image_path}")
        samples.append(
            ReasonSegSample(
                sample_id=annotation_path.stem,
                image_path=image_path,
                queries=list(record.get("text") or []),
                is_sentence=bool(record.get("is_sentence", False)),
                shapes=list(record.get("shapes") or []),
            )
        )

    expected = SPLIT_SIZES.get(split)
    if expected is not None and len(samples) != expected:
        raise ValueError(
            f"Loaded {len(samples)} samples from {split_dir}, expected {expected} for "
            f"ReasonSeg {split}. An incomplete download scores higher than a complete one, "
            "so this is a hard failure."
        )
    return samples


def ground_truth_masks(samples: list[ReasonSegSample]) -> dict[str, np.ndarray]:
    """Rasterize every sample's target mask at its image's native resolution."""
    from PIL import Image

    masks = {}
    for sample in samples:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        masks[sample.sample_id] = rasterize(sample.shapes, height, width)
    return masks


def query_types(samples: list[ReasonSegSample]) -> dict[str, str]:
    return {sample.sample_id: sample.query_type for sample in samples}
