"""gIoU and cIoU for ReasonSeg, implemented from the definitions.

This is deliberately *not* a call into the upstream evaluation code. If the
replication reused their metric implementation, then "we reproduced 67.20
gIoU" would only mean "we ran their script" -- any disagreement between their
definition and the one the paper describes would cancel out on both sides and
be invisible. Reimplementing from the stated definitions is what turns a
matching number into evidence.

The two metrics answer different questions and disagree loudly, which is why
the literature always reports both:

- **gIoU** ("generalized", though it is just the mean) averages per-sample IoU.
  Every image counts once, so a missed small object costs as much as a missed
  large one.
- **cIoU** ("cumulative") sums intersections and unions across the whole
  dataset before dividing. One large object can outweigh dozens of small ones,
  so cIoU is the metric that flatters a model that only finds big things.

A model can move one and not the other. AnchorSeg reports 67.20 gIoU / 75.15
cIoU on ReasonSeg val; if a replication matches one and misses the other, that
gap localizes the bug (systematically missing small objects moves gIoU far
more than cIoU).

Three conventions inherited from the LISA evaluation protocol
(`train_ds.py::validate`), all of which change the number materially and none
of which are arguable from the metric name alone:

1. **Ignore pixels are excluded from both intersection and union**, not counted
   as background. The reference calls `intersectionAndUnionGPU(..., K=2,
   ignore_index=255)` against a trinary ground-truth mask. Treating the ignore
   region as background turns a prediction there into a false positive, when
   the reference makes it a no-op.
2. **An empty prediction on an empty ground truth scores IoU 1.0**, not 0 or
   NaN -- the reference's `acc_iou[union_i == 0] += 1.0  # no-object target`.
   Four of ReasonSeg val's 200 samples carry no shapes at all, so this is worth
   2% of gIoU on the split that the replication is judged on, more than the
   tolerance itself.
3. **Masks are compared at full image resolution**, not at the model's working
   resolution. Predictions come back at SAM's 1024px and must be resized up
   before scoring, not the other way round -- downsampling the ground truth
   instead quietly shrinks the penalty for boundary error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMPTY_MATCH_IOU = 1.0


@dataclass
class ReasonSegScores:
    giou: float
    ciou: float
    n_samples: int
    # The paper breaks results out by query type; `is_sentence` in the
    # annotations is that split (short phrase vs. long sentence). Reporting
    # only the overall number hides the axis the method is actually about --
    # long queries are the ones that need reasoning.
    by_query_type: dict[str, ReasonSegScores] = field(default_factory=dict)
    per_sample_iou: dict[str, float] = field(default_factory=dict)

    def as_row(self, label: str = "") -> str:
        return f"{label:<28}{self.giou * 100:>8.2f}{self.ciou * 100:>8.2f}{self.n_samples:>8d}"


def single_iou(
    pred: np.ndarray, target: np.ndarray, ignore: np.ndarray | None = None
) -> tuple[float, int, int]:
    """Returns (iou, intersection, union) for one mask pair.

    `ignore` marks pixels excluded from the comparison entirely -- neither
    intersection nor union -- matching the reference's `ignore_index=255`.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Mask shape mismatch: prediction {pred.shape} vs ground truth {target.shape}"
        )

    pred = pred.astype(bool)
    target = target.astype(bool)

    if ignore is not None:
        if ignore.shape != target.shape:
            raise ValueError(
                f"Mask shape mismatch: ignore {ignore.shape} vs ground truth {target.shape}"
            )
        keep = ~ignore.astype(bool)
        pred = pred & keep
        target = target & keep

    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())

    if union == 0:
        # Both empty: the model correctly predicted nothing. See convention 2.
        return EMPTY_MATCH_IOU, 0, 0
    return intersection / union, intersection, union


def compute_scores(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    query_types: dict[str, str] | None = None,
    ignores: dict[str, np.ndarray] | None = None,
) -> ReasonSegScores:
    """Score a full split.

    `predictions` and `targets` are keyed by sample id; `ignores` optionally
    supplies each sample's excluded region. A sample present in `targets` but
    missing from `predictions` is scored as an empty prediction rather than
    skipped -- dropping it would let a crashed or truncated evaluation run
    report a higher score than a complete one.
    """
    missing = set(targets) - set(predictions)
    per_sample: dict[str, float] = {}
    total_intersection = 0
    total_union = 0

    for sample_id, target in targets.items():
        pred = predictions.get(sample_id)
        if pred is None:
            pred = np.zeros_like(target, dtype=bool)
        iou, intersection, union = single_iou(
            pred, target, (ignores or {}).get(sample_id)
        )
        per_sample[sample_id] = iou
        total_intersection += intersection
        total_union += union

    giou = float(np.mean(list(per_sample.values()))) if per_sample else float("nan")
    ciou = (total_intersection / total_union) if total_union > 0 else EMPTY_MATCH_IOU

    scores = ReasonSegScores(
        giou=giou,
        ciou=ciou,
        n_samples=len(per_sample),
        per_sample_iou=per_sample,
    )

    if query_types:
        for name in sorted(set(query_types.values())):
            subset = {k: v for k, v in targets.items() if query_types.get(k) == name}
            if subset:
                scores.by_query_type[name] = compute_scores(
                    {k: v for k, v in predictions.items() if k in subset},
                    subset,
                    ignores={k: v for k, v in (ignores or {}).items() if k in subset} or None,
                )

    if missing:
        scores.per_sample_iou = per_sample  # keep, but make the gap visible to callers
    return scores


def missing_predictions(
    predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> list[str]:
    """Sample ids the evaluation never produced a prediction for.

    Callers should treat a non-empty result as a failed run, not a scored one.
    """
    return sorted(set(targets) - set(predictions))
