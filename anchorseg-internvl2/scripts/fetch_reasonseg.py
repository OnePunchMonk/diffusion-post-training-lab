"""Fetch a ReasonSeg split from the Hugging Face mirror.

ReasonSeg was released via Google Drive by the LISA authors, which is awkward
to script and impossible to verify. `fcxfcx/ReasonSeg` mirrors it in the
original on-disk layout (one .jpg + one .json per sample) with the same
239/200/779 train/val/test counts the paper reports, so the split can be
fetched reproducibly and checked.

Step 1 of the replication only needs `val` -- 200 samples, about 1GB. The full
upstream training recipe additionally wants RefCOCO/+/g, ADE20K, COCO-Stuff and
LLaVA-Instruct-150k, which is roughly two orders of magnitude more data; don't
download those until step 2.

  python scripts/fetch_reasonseg.py --split val --out data/reason_seg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from asvl.data.reasonseg import HF_DATASET_ID, SPLIT_SIZES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=sorted(SPLIT_SIZES))
    ap.add_argument("--out", default="data/reason_seg")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # allow_patterns keeps this to one split: the test split alone is ~4x val,
    # and step 1 has no use for it.
    snapshot_dir = snapshot_download(
        HF_DATASET_ID,
        repo_type="dataset",
        allow_patterns=[f"{args.split}/*"],
        cache_dir=args.cache_dir,
    )

    source = Path(snapshot_dir) / args.split
    destination = out / args.split
    if destination.exists():
        print(f"{destination} already exists; leaving it alone.")
    else:
        # Copy rather than symlink into the HF cache: the upstream dataloaders
        # walk this tree and a cache eviction would break the split silently.
        shutil.copytree(source, destination)

    n_images = len(list(destination.glob("*.jpg")))
    n_annotations = len(list(destination.glob("*.json")))
    expected = SPLIT_SIZES[args.split]

    print(f"{destination}: {n_images} images, {n_annotations} annotations (expected {expected})")
    if n_images != expected or n_annotations != expected:
        raise SystemExit(
            "Count mismatch -- an incomplete split scores *higher* than a complete one "
            "(fewer hard samples), so this must be fixed before evaluating anything."
        )


if __name__ == "__main__":
    main()
