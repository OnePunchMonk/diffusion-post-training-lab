"""Why the cascade scored what it scored.

A gIoU number alone cannot distinguish "the method is weak" from "the
grounding stage returned nothing useful". This separates them, and on the
first run it was the difference between reporting a finding and reporting an
artifact: the median predicted box covered 84% of the image against a 6.4%
median target, because InternVL2-2B answers with the whole frame when it
cannot localize.

  python scripts/diagnose_cascade.py --replies results/cascade-internvl2-replies.json
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
from PIL import Image

from asvl.data.reasonseg import ground_truth_masks, load_split, target_mask

WHOLE_FRAME = [0.0, 0.0, 1000.0, 1000.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default="results/cascade-internvl2-replies.json")
    ap.add_argument("--root", default="data/reason_seg")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    replies = json.loads(open(args.replies).read())
    samples = {s.sample_id: s for s in load_split(args.root, args.split)}
    targets = {k: target_mask(v) for k, v in ground_truth_masks(list(samples.values())).items()}

    box_fracs, gt_fracs, recalls = [], [], []
    for sid, entry in replies.items():
        if not entry.get("box") or sid not in targets:
            continue
        with Image.open(samples[sid].image_path) as image:
            width, height = image.size
        x0, y0, x1, y1 = entry["box"]
        box_fracs.append((x1 - x0) * (y1 - y0) / (width * height))
        gt_fracs.append(targets[sid].mean())

        # What fraction of the target the box even contains. This is the ceiling
        # on any mask SAM can produce inside it, so a low value means the error
        # is upstream of SAM entirely.
        inside = np.zeros_like(targets[sid])
        inside[int(y0) : int(y1), int(x0) : int(x1)] = True
        recalls.append((inside & targets[sid]).sum() / max(targets[sid].sum(), 1))

    box_fracs, gt_fracs, recalls = map(np.array, (box_fracs, gt_fracs, recalls))

    print(f"{'boxes':<34}{len(box_fracs)}")
    print(f"{'predicted box area (median frac)':<34}{np.median(box_fracs):.3f}")
    print(f"{'ground-truth area (median frac)':<34}{np.median(gt_fracs):.3f}")
    ratio = np.median(box_fracs / np.maximum(gt_fracs, 1e-6))
    print(f"{'box / target area ratio (median)':<34}{ratio:.1f}x")
    print(f"{'target recall by box alone':<34}{np.median(recalls):.3f} median")
    print()

    # The degenerate answer: InternVL2 returns the whole frame when it cannot
    # ground the query. Counting it by query type is the actual experiment.
    print(f"{'query type':<10}{'n':>5}{'whole-frame boxes':>20}{'rate':>8}")
    for label, want in (("short", False), ("long", True)):
        subset = [r for r in replies.values() if r["is_sentence"] is want]
        degenerate = sum(1 for r in subset if r.get("raw") == WHOLE_FRAME)
        rate = degenerate / len(subset) if subset else float("nan")
        print(f"{label:<10}{len(subset):>5}{degenerate:>20}{rate:>8.0%}")

    lengths = np.array([len(r.get("reply") or "") for r in replies.values()])
    no_digits = sum(1 for r in replies.values() if not re.search(r"\d", r.get("reply") or ""))
    print()
    print(f"{'reply length (chars)':<34}median {np.median(lengths):.0f}, max {lengths.max()}")
    print(f"{'replies with no digits':<34}{no_digits}")
    if lengths.max() > 200:
        print(
            "  NOTE: replies echo the query before the box, so a long query eats the "
            "generation budget. Raise max_new_tokens before reading a long-query result."
        )


if __name__ == "__main__":
    main()
