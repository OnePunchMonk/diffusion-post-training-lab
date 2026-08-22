"""Generate a tiny synthetic concept dataset so the LoRA pipeline can be run
end-to-end without first going and sourcing real photos.

This draws a simple recurring "character" (a stack of colored geometric
shapes standing in for a toy robot) with randomized pose/background per
image, captioned with a rare token ("sks") the way DreamBooth-style LoRAs
are conventionally trained. The point is not that the resulting LoRA will
look good — 16 procedurally drawn images can't teach much — it's to prove
every stage of the pipeline (dataset -> train -> checkpoint -> eval ->
publish) runs correctly before spending real GPU time on a real dataset.

Usage:
  python scripts/make_toy_concept_dataset.py --out-dir data/concept_dataset --n 16

Swap in a real dataset later: same folder shape (images + metadata.jsonl
with "file_name"/"caption" keys), just with photographed data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

_BACKGROUNDS = [(240, 240, 245), (250, 235, 215), (225, 245, 235), (235, 235, 250)]
_BODY_COLORS = [(220, 60, 60), (60, 120, 220), (60, 180, 100), (230, 170, 40)]
_SETTINGS = ["on a wooden table", "on a grass field", "on a white studio backdrop", "on a city sidewalk"]


def _draw_robot(size: int, body_color: tuple[int, int, int], bg_color: tuple[int, int, int], seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    cx = size // 2 + rng.randint(-size // 10, size // 10)
    cy = size // 2 + rng.randint(-size // 10, size // 10)
    scale = rng.uniform(0.85, 1.15)

    head_r = int(size * 0.12 * scale)
    body_w, body_h = int(size * 0.28 * scale), int(size * 0.34 * scale)

    # body
    draw.rounded_rectangle(
        [cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2], radius=body_w // 6, fill=body_color
    )
    # head
    draw.ellipse([cx - head_r, cy - body_h // 2 - 2 * head_r, cx + head_r, cy - body_h // 2], fill=body_color)
    # eyes
    eye_r = max(2, head_r // 5)
    eye_y = cy - body_h // 2 - int(1.4 * head_r)
    draw.ellipse([cx - head_r // 2 - eye_r, eye_y - eye_r, cx - head_r // 2 + eye_r, eye_y + eye_r], fill="white")
    draw.ellipse([cx + head_r // 2 - eye_r, eye_y - eye_r, cx + head_r // 2 + eye_r, eye_y + eye_r], fill="white")
    # arms
    arm_w = max(3, body_w // 10)
    draw.line([cx - body_w // 2, cy - body_h // 4, cx - body_w, cy], fill=body_color, width=arm_w)
    draw.line([cx + body_w // 2, cy - body_h // 4, cx + body_w, cy], fill=body_color, width=arm_w)

    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/concept_dataset")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--token", default="sks robot toy")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(args.n):
        body_color = random.choice(_BODY_COLORS)
        bg_color = random.choice(_BACKGROUNDS)
        setting = random.choice(_SETTINGS)

        img = _draw_robot(args.resolution, body_color, bg_color, seed=i)
        file_name = f"{i:04d}.png"
        img.save(out_dir / file_name)
        records.append({"file_name": file_name, "caption": f"a photo of {args.token} {setting}"})

    (out_dir / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    print(f"Wrote {len(records)} images + metadata.jsonl to {out_dir}")


if __name__ == "__main__":
    main()
