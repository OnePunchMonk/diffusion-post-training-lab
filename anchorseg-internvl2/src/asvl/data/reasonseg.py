"""ReasonSeg loading and mask construction.

ReasonSeg annotations are LabelMe-style JSON, one file per image:

    {"text": ["the cesspit"], "is_sentence": false,
     "shapes": [{"label": "target", "points": [[x, y], ...]}, ...]}

This module reproduces LISA's `get_mask_from_json` (`utils/data_processing.py`)
and the evaluation's treatment of the result, because four details of it change
the score and none of them are inferable from the file format. Each was found
by reading the reference after `scripts/check_ground_truth.py` flagged five
empty masks in ReasonSeg val:

1. **The mask is trinary, not binary.** 0 = background, 1 = target, 255 =
   ignore. Evaluation passes `ignore_index=255`, so ignore pixels are excluded
   from *both* intersection and union -- they are not background. Subtracting
   the ignore region instead turns a prediction there into a false positive,
   when the reference makes it a no-op.

2. **Polygons are painted largest-area-first, and later paints overwrite
   earlier ones.** It is a painter's algorithm, not union-then-subtract. A small
   `target` inside a large `ignore` therefore ends up as *target*, because it is
   painted second. Union/subtract gives the opposite answer: in ReasonSeg val
   that silently emptied sample `914980029_4a7c8f579e_o`, whose target covers
   31828 px.

3. **`flag` shapes are dropped.** The reference calls them "meaningless
   deprecated annotations". They appear 62 times in val, always with fewer than
   3 points -- so a rule of "anything not `ignore` is a target" happens to
   survive only because a 2-point polygon has no area. Dropping them by label is
   the rule that actually holds.

4. **The polygon outline is painted inclusively** (`cv2.polylines` before
   `cv2.fillPoly`), adding roughly a one-pixel border. On ReasonSeg's small
   targets -- median target area is 6% of the frame -- that is a percent-level
   IoU difference, comparable to the gap between published methods. A stricter
   rasterizer is *not* more correct here; it is a different ground truth.

Four val samples carry no shapes at all and so are legitimately empty targets.
With `ignore_index` semantics and the empty-on-empty convention, those are
samples a model scores 1.0 on by predicting nothing -- 2% of the split, which
is larger than the tolerance a replication is judged against. That is worth
knowing before reading any gap as a modelling result.

`is_sentence` is carried through because the paper reports results split by
query type: short phrases vs. long sentences. The long queries are the ones
that require reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_LABEL = "target"
IGNORE_LABEL = "ignore"
FLAG_LABEL = "flag"  # deprecated markers; the reference drops them

# Mask values, matching LISA's get_mask_from_json and the ignore_index its
# evaluation passes.
BACKGROUND = 0
TARGET = 1
IGNORE = 255

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
    """Polygons -> a trinary uint8 mask (0 background, 1 target, 255 ignore).

    Faithful to LISA's `get_mask_from_json`: drop `flag` shapes, sort the rest
    by filled area descending, then paint each one -- outline included -- so
    smaller shapes overwrite larger ones. See the module docstring for why each
    of those matters.
    """
    import cv2

    mask = np.full((height, width), BACKGROUND, dtype=np.uint8)

    painted = []
    for shape in shapes:
        label = str(shape.get("label", "")).lower()
        if label == FLAG_LABEL:
            continue
        points = shape.get("points") or []
        if len(points) < 3:  # a polygon needs at least a triangle
            continue
        polygon = np.array([points], dtype=np.int32)

        # Area is measured the same way the shape will be painted, so the sort
        # order matches what actually lands on the canvas.
        probe = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(probe, polygon, True, 1, 1)
        cv2.fillPoly(probe, polygon, 1)
        painted.append((int(probe.sum()), polygon, label))

    # Largest first, so smaller shapes are painted last and win. This is what
    # puts a target that sits inside an ignore region back into the target.
    for _area, polygon, label in sorted(painted, key=lambda item: -item[0]):
        value = IGNORE if IGNORE_LABEL in label else TARGET
        cv2.polylines(mask, polygon, True, value, 1)
        cv2.fillPoly(mask, polygon, value)

    return mask


def target_mask(mask: np.ndarray) -> np.ndarray:
    """The positive class only."""
    return mask == TARGET


def ignore_mask(mask: np.ndarray) -> np.ndarray:
    """Pixels excluded from both intersection and union at evaluation time."""
    return mask == IGNORE


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
    """Trinary ground-truth mask per sample, at the image's native resolution."""
    from PIL import Image

    masks = {}
    for sample in samples:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        masks[sample.sample_id] = rasterize(sample.shapes, height, width)
    return masks


def query_types(samples: list[ReasonSegSample]) -> dict[str, str]:
    return {sample.sample_id: sample.query_type for sample in samples}
