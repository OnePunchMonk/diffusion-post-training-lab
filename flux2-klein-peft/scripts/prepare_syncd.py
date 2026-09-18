"""Materialize a slice of SynCD into per-subject training sets.

**Use case.** Subject-driven personalization: given a handful of photos of one
specific object, teach the model that object so it can be re-rendered in new
contexts. It is the right probe for comparing PEFT methods because its two
failure modes are opposite and both visible -- too little capacity and the
subject doesn't stick (low concept preservation), too much and the adapter
memorizes the training shots and stops following the prompt (low prompt
following). A method only wins if it moves both.

**Dataset.** SynCD (Kumari et al., *Generating Multi-Image Synthetic Data for
Text-to-Image Customization*, ICCV 2025; MIT licence). ~90k objects with 2-3
images each, generated with FLUX so that the same object appears under
different lighting, background and pose. That multi-view property is what makes
it a sharper benchmark than the usual 4-6 photos of one object from one shoot:
background and pose are already decorrelated from identity in the training
data, so whatever identity leakage remains is attributable to the adapter
rather than to the dataset.

Each metadata record gives us both halves of a split for free:
  `category_description`  -- a plain description of the object, used as the
                             training caption for every image of that subject.
  `prompts`               -- the object placed in specific contexts, held out
                             and used as the evaluation prompts.

**Cost.** The full dataset is 17.5GB across 10 zips. This script pulls one
archive (~1.8GB) and slices N subjects out of it, which is enough for a
six-method sweep and keeps the download inside a coffee break.

  python scripts/prepare_syncd.py --num-subjects 20 --out data/syncd-20
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

REPO_ID = "nupurkmr9/syncd"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Root directory; one subdirectory per subject.")
    ap.add_argument("--num-subjects", type=int, default=20)
    ap.add_argument("--archive", default="archive_1.zip", help="Which of archive_1..10.zip to pull.")
    ap.add_argument(
        "--min-images",
        type=int,
        default=3,
        help="Skip subjects with fewer images than this. 3-image subjects give the adapter enough "
        "pose/lighting variation that identity and background actually separate.",
    )
    ap.add_argument("--category", default=None, help="Filter to one SynCD category (e.g. 'rigid').")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata_path = hf_hub_download(REPO_ID, "metadata.json", repo_type="dataset", cache_dir=args.cache_dir)
    records = json.loads(Path(metadata_path).read_text())
    # Records are positional: record i owns the directory "i/" inside whichever
    # archive holds it, so index by the directory name in `filenames` rather
    # than by list position (the two agree today, but the archives are the
    # authority on what actually exists locally).
    by_dir = {Path(r["filenames"][0]).parent.name: r for r in records}

    archive_path = hf_hub_download(REPO_ID, args.archive, repo_type="dataset", cache_dir=args.cache_dir)

    selected = 0
    with zipfile.ZipFile(archive_path) as zf:
        available = sorted({Path(n).parts[0] for n in zf.namelist() if n.endswith(".png")}, key=_sort_key)

        for dir_name in available:
            if selected >= args.num_subjects:
                break
            record = by_dir.get(dir_name)
            if record is None or len(record["filenames"]) < args.min_images:
                continue
            if args.category and record["category"] != args.category:
                continue

            subject_dir = out_root / f"subject-{dir_name}"
            images_ok = _extract_subject(zf, record, subject_dir)
            if not images_ok:
                shutil.rmtree(subject_dir, ignore_errors=True)
                continue

            selected += 1

    manifest = {
        "source": f"{REPO_ID}/{args.archive}",
        "paper": "https://arxiv.org/abs/2502.01720",
        "license": "MIT",
        "num_subjects": selected,
        "subjects": sorted(p.name for p in out_root.iterdir() if p.is_dir()),
    }
    (out_root / "subjects.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {selected} subjects under {out_root}")


def _sort_key(name: str):
    return (int(name) if name.isdigit() else 1 << 30, name)


def _extract_subject(zf: zipfile.ZipFile, record: dict, subject_dir: Path) -> bool:
    subject_dir.mkdir(parents=True, exist_ok=True)
    caption = record["category_description"]

    rows = []
    for i, member in enumerate(record["filenames"]):
        try:
            data = zf.read(member)
        except KeyError:
            return False
        file_name = f"{i:04d}.png"
        (subject_dir / file_name).write_bytes(data)
        rows.append({"file_name": file_name, "caption": caption})

    (subject_dir / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Held-out contexts: same object, contexts the adapter never saw during
    # training. These are what prompt-following is scored on.
    eval_rows = [
        {"prompt": p, "id": f"{subject_dir.name}-{i}", "reference_image": rows[0]["file_name"]}
        for i, p in enumerate(record["prompts"])
    ]
    (subject_dir / "prompts_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")

    (subject_dir / "subject.json").write_text(
        json.dumps(
            {
                "category": record["category"],
                "category_description": caption,
                "objaverse_id": record.get("objaverse_id"),
                "num_images": len(rows),
            },
            indent=2,
        )
    )
    return True


if __name__ == "__main__":
    main()
