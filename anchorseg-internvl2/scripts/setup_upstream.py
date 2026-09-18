"""Fetch everything step 1 needs: pinned code, released weights, frozen SAM/CLIP.

Step 1 evaluates upstream's released checkpoint on ReasonSeg val and checks the
result against their published number. Nothing is trained, so the download is
the checkpoint plus the two frozen components it was built on -- roughly 17GB,
versus the ~100GB+ of datasets the training recipe wants.

  python scripts/setup_upstream.py --out upstream

Afterwards, `python scripts/eval_upstream.py` prints the command to run inside
the checkout, because their evaluation entry point wants a GPU and a DeepSpeed
launch and there is nothing to gain from wrapping that in more Python.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from asvl.upstream.pins import (
    ANCHORSEG_7B_REPO,
    CLIP_VISION_TOWER,
    SAM_VIT_H_FILENAME,
    SAM_VIT_H_URL,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
)


def clone_pinned(out: Path) -> Path:
    """Clone at the exact commit, detached.

    Detached on purpose: a replication that silently follows `main` cannot
    attribute a drift in the number to their change or yours.
    """
    repo_dir = out / "AnchorSeg"
    if repo_dir.exists():
        print(f"{repo_dir} exists; leaving it. Delete it to re-clone.")
        return repo_dir

    subprocess.run(["git", "clone", UPSTREAM_REPO, str(repo_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--detach", UPSTREAM_COMMIT], check=True
    )
    print(f"Cloned {UPSTREAM_REPO} at {UPSTREAM_COMMIT[:12]} (detached)")
    return repo_dir


def fetch_weights(out: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    paths = {}

    print(f"Fetching {ANCHORSEG_7B_REPO} ...")
    paths["anchorseg"] = Path(snapshot_download(ANCHORSEG_7B_REPO, cache_dir=str(out / "hf-cache")))

    print(f"Fetching {CLIP_VISION_TOWER} ...")
    paths["clip"] = Path(snapshot_download(CLIP_VISION_TOWER, cache_dir=str(out / "hf-cache")))

    sam_path = out / SAM_VIT_H_FILENAME
    if sam_path.exists():
        print(f"{sam_path} exists; skipping.")
    else:
        print(f"Fetching SAM ViT-H ({SAM_VIT_H_URL}) ...")
        _download(SAM_VIT_H_URL, sam_path)
    paths["sam"] = sam_path

    return paths


def _download(url: str, destination: Path) -> None:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {read / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print()
    # Rename only after a complete read: a truncated .pth loads far enough to
    # produce garbage masks rather than failing outright.
    tmp.rename(destination)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="upstream")
    ap.add_argument("--skip-weights", action="store_true", help="Clone only (~50MB).")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    repo_dir = clone_pinned(out)
    if args.skip_weights:
        print("\nSkipped weights. Re-run without --skip-weights before evaluating.")
        return

    paths = fetch_weights(out)

    print("\nReady:")
    print(f"  code   {repo_dir}  @ {UPSTREAM_COMMIT[:12]}")
    for name, path in paths.items():
        print(f"  {name:<6} {path}")
    print("\nNext: python scripts/eval_upstream.py --upstream", out)


if __name__ == "__main__":
    main()
