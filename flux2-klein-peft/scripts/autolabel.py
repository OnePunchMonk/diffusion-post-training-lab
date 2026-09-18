"""Auto-label a concept dataset: InternVL2 captions + SAM subject masks.

Both models are *data tooling*, not training targets -- they run once on CPU or
a small GPU and their output is consumed by the training loop.

**InternVL2 (captioning).** The DreamBooth default caption is the same string
for every image ("a photo of sks backpack"). That's deliberate in the original
method, but it also means nothing in the caption distinguishes the photos, so
the adapter is free to entangle the subject with whatever the background
happens to be. Per-image captions that describe the *context* ("... on a wooden
table, daylight") give the model something other than the token to explain the
background with, which is the standard fix for background bleed. The subject
identifier is prepended, not generated, so the binding is still ours.

**SAM (masking).** Produces a subject mask per image, used two ways: to weight
the training loss toward the subject (`use_masks: true`), and at eval time to
crop the subject out before computing DINO subject-fidelity, so the score
measures the subject rather than a shared background.

  python scripts/autolabel.py --dataset data/db-backpack --identifier sks --class-name backpack

Rewrites metadata.jsonl in place, adding `mask_file_name` and replacing
`caption` (the original is kept as `caption_original`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CAPTION_INSTRUCTION = (
    "<image>\nDescribe this photo in one short sentence. Describe the setting, surface, lighting and "
    "background. Do not name or describe the main object itself."
)


def load_internvl(model_id: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = (
        AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    return model, tokenizer


def internvl_caption(model, tokenizer, image, device: str) -> str:
    """InternVL2's chat API takes pre-tiled pixel values, not a PIL image.

    The model card's `load_image` helper does dynamic aspect-ratio tiling; for
    single-subject product-style photos one 448px tile is enough and keeps the
    labeling pass fast, so we build the tensor directly rather than importing
    the model card's preprocessing.
    """
    import torch
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(torch.bfloat16).to(device)
    generation_config = {"max_new_tokens": 48, "do_sample": False}
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, CAPTION_INSTRUCTION, generation_config)
    return response.strip().replace("\n", " ")


def load_sam(model_id: str, device: str):
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).eval().to(device)
    return model, processor


def sam_subject_mask(model, processor, image, device: str):
    """Segment the main subject with a centre-box prompt.

    SynCD renders each object centred in frame, so a box covering the middle 80%
    is a reliable prompt and avoids putting a detector in the loop. SAM returns
    three candidate masks per prompt (roughly sub-part / part / whole); we take
    the highest-IoU-scored one, which for a centred object prompt is the whole
    object. Re-check this assumption before pointing the script at a dataset
    whose subjects are off-centre or partially occluded.
    """
    import torch

    width, height = image.size
    margin_x, margin_y = width * 0.1, height * 0.1
    box = [[[margin_x, margin_y, width - margin_x, height - margin_y]]]

    inputs = processor(image.convert("RGB"), input_boxes=box, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]  # (3, H, W)
    best = int(outputs.iou_scores.cpu().squeeze()[-1].argmax())
    return masks[best].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--identifier", default="sks")
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--caption-model", default="OpenGVLab/InternVL2-2B")
    ap.add_argument("--sam-model", default="facebook/sam-vit-base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-captions", action="store_true")
    ap.add_argument("--skip-masks", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.dataset)
    records = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines() if line.strip()]

    captioner = load_internvl(args.caption_model, device) if not args.skip_captions else None
    segmenter = load_sam(args.sam_model, device) if not args.skip_masks else None

    mask_dir = root / "masks"
    if segmenter:
        mask_dir.mkdir(exist_ok=True)

    for record in records:
        image = Image.open(root / record["file_name"]).convert("RGB")

        if captioner:
            context = internvl_caption(*captioner, image, device)
            record.setdefault("caption_original", record["caption"])
            record["caption"] = f"a photo of {args.identifier} {args.class_name}, {context}"
            logger.info("%s -> %s", record["file_name"], record["caption"])

        if segmenter:
            mask = sam_subject_mask(*segmenter, image, device)
            mask_name = f"masks/{Path(record['file_name']).stem}.png"
            Image.fromarray((np.asarray(mask) * 255).astype("uint8")).save(root / mask_name)
            record["mask_file_name"] = mask_name

    (root / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"Updated {len(records)} records in {root / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
