"""Verify the ground-truth masks before any model is involved.

Every number in this replication is measured against these masks. If the
rasterization is wrong, the resulting gap from the paper looks exactly like a
modelling failure and will be debugged as one -- so it is worth the two minutes
to rule out first.

Checks:

- every sample rasterizes to a non-empty mask at the image's own resolution
- the mask area distribution is sane (ReasonSeg is small-object heavy; if the
  median target covers most of the frame, `ignore` shapes are being unioned in
  rather than subtracted)
- the short/long query split matches what the paper reports it as
- optionally, cross-checks against an independently published rendering of the
  same split (`Ricky06662/ReasonSeg_val`, which ships pre-rasterized boolean
  masks) -- two independent paths agreeing is much stronger evidence than one
  path looking plausible

  python scripts/check_ground_truth.py --root data/reason_seg --split val --cross-check
"""

from __future__ import annotations

import argparse

import numpy as np

from asvl.data.reasonseg import (
    ground_truth_masks,
    ignore_mask,
    load_split,
    query_types,
    target_mask,
)
from asvl.metrics.reasonseg import single_iou


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/reason_seg")
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--cross-check",
        action="store_true",
        help="Compare against Ricky06662/ReasonSeg_val's pre-rendered masks (~280MB download).",
    )
    args = ap.parse_args()

    samples = load_split(args.root, args.split)
    masks = ground_truth_masks(samples)
    types = query_types(samples)

    targets = {sid: target_mask(m) for sid, m in masks.items()}
    ignores = {sid: ignore_mask(m) for sid, m in masks.items()}

    empty = [sid for sid, m in targets.items() if not m.any()]
    fractions = np.array([m.mean() for m in targets.values()])
    ignore_fractions = np.array([m.mean() for m in ignores.values()])

    print(f"samples              {len(samples)}")
    print(f"empty masks          {len(empty)}" + (f"  {empty[:5]}" if empty else ""))
    print(f"target area fraction median {np.median(fractions):.3f}  mean {fractions.mean():.3f}")
    p10, p90 = np.percentile(fractions, 10), np.percentile(fractions, 90)
    print(f"                     p10 {p10:.3f}  p90 {p90:.3f}")
    n_with_ignore = int((ignore_fractions > 0).sum())
    print(f"ignore regions       {n_with_ignore} samples, mean area {ignore_fractions.mean():.3f}")
    counts = {t: sum(1 for v in types.values() if v == t) for t in sorted(set(types.values()))}
    print(f"query types          {counts}")
    print(f"queries per sample   {np.mean([len(s.queries) for s in samples]):.2f}")

    if empty:
        print(
            f"\nNOTE: {len(empty)} samples have an empty target. Under the reference's "
            "empty-on-empty convention a model scores 1.0 on each by predicting nothing, "
            f"which is worth {100 * len(empty) / len(samples):.1f} points of gIoU on this split."
        )

    if np.median(fractions) > 0.5:
        print(
            "\nWARNING: the median target covers over half the frame. ReasonSeg is small-object "
            "heavy, so this usually means `ignore` shapes are being unioned into the target "
            "instead of subtracted."
        )

    if args.cross_check:
        _cross_check(targets)


def _cross_check(ours: dict[str, np.ndarray]) -> None:
    """Agreement against an independently rasterized copy of the same split."""
    from datasets import load_dataset

    print("\nCross-checking against Ricky06662/ReasonSeg_val ...")
    dataset = load_dataset("Ricky06662/ReasonSeg_val", split="test")

    ious = []
    unmatched = 0
    for row in dataset:
        sample_id = str(row["image_id"]).rsplit("/", 1)[-1].rsplit(".", 1)[0]
        mine = ours.get(sample_id)
        if mine is None:
            unmatched += 1
            continue
        theirs = np.asarray(row["mask"], dtype=bool)
        if theirs.shape != mine.shape:
            theirs = theirs.T if theirs.T.shape == mine.shape else theirs
        if theirs.shape != mine.shape:
            unmatched += 1
            continue
        ious.append(single_iou(mine, theirs)[0])

    if not ious:
        print("  no samples matched by id -- the two releases name samples differently;")
        print("  compare a few by hand before trusting either.")
        return

    ious = np.array(ious)
    print(f"  matched {len(ious)} samples ({unmatched} unmatched)")
    print(f"  mean IoU {ious.mean():.4f}  min {ious.min():.4f}  below 0.99: {(ious < 0.99).sum()}")
    if ious.mean() < 0.98:
        print(
            "  WARNING: the two rasterizations disagree. Resolve this before evaluating a model -- "
            "a systematic ground-truth difference of this size is larger than the gap between "
            "published methods."
        )


if __name__ == "__main__":
    main()
