"""InternVL2 + SAM, training-free, on ReasonSeg val.

One inference pass over 200 images on a single A10G. Nothing is trained, so
this costs minutes rather than GPU-days -- which is the point: it is a real
experiment with a published number to sit beside, at a price that does not
need a budget conversation.

**The experiment.** LISA's Table 1 reports Grounded-SAM at 26.0 gIoU / 14.5
cIoU on ReasonSeg val: a cascade of text -> GroundingDINO box -> SAM. LISA
argues cascades score badly because the grounding stage can only *match* text
to objects, while ReasonSeg queries require inferring which object is meant.

Swap the grounding stage for InternVL2, which can reason, and hold the rest
fixed. If that explanation is right, the score should move -- and move more on
long queries than short ones. ReasonSeg's `is_sentence` flag separates those
two populations, so the prediction is checkable.

  modal run modal/cascade.py::run              # A10G, ~200 images
  python scripts/score_cascade.py              # scores it locally, independently

Masks land in the shared volume; scoring happens on your machine with the same
independent metrics the AnchorSeg replication uses. That separation is
deliberate: the run produces masks, not numbers.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "internvl2-sam-cascade"

VLM_REPO = "OpenGVLab/InternVL2-2B"   # MIT licence, ~4.4GB in bf16
SAM_REPO = "facebook/sam-vit-huge"    # the SAM the Grounded-SAM baseline uses
REASONSEG_DATASET = "fcxfcx/ReasonSeg"

CACHE = "/cache"
OUT = "/cache/cascade-internvl2"

# The box parser is shipped into the image rather than reimplemented here, so
# the code that runs on Modal is the same code the local tests cover. A second
# copy would be a second parser, and the tests would stop meaning anything.
CASCADE_SRC = Path(__file__).resolve().parent.parent / "src" / "asvl" / "cascade.py"
ASVL_SRC = Path(__file__).resolve().parent.parent / "src" / "asvl"

app = modal.App(APP_NAME)
cache = modal.Volume.from_name("anchorseg-cache", create_if_missing=True)

# A modern stack: nothing here has to match the 2023 pins the AnchorSeg
# replication is locked to, because nothing here is replicating that code.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.44.2",  # InternVL2's remote code targets 4.37-4.44
        "accelerate>=0.33",
        # 0.1.99, not 0.2.x: InternLM2's tokenizer.model fails to load on the
        # newer sentencepiece with "INTERNAL: piece must not include null
        # character". InternVL2's own requirements pin this version.
        "sentencepiece==0.1.99",
        "timm>=1.0",
        "einops>=0.8",
        "pillow>=10.0",
        "numpy>=1.26",
        "huggingface-hub>=0.25",
    )
    .add_local_file(CASCADE_SRC, "/root/asvl_cascade.py")
)

# Scoring runs on CPU with the same metric and rasterization code the local
# tests cover. Doing it here rather than locally avoids pulling ~600MB of
# full-resolution masks down; it is still independent of upstream's evaluator,
# which is what "independent" means here.
score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("numpy>=1.26", "opencv-python-headless>=4.9", "pillow>=10.0")
    .add_local_dir(ASVL_SRC, "/root/asvl")
)


@app.function(image=image, volumes={CACHE: cache}, gpu="A10G", timeout=5400)
def run(limit: int = 0, save_replies: bool = True) -> dict:
    """Ground every ReasonSeg val query with InternVL2, then segment with SAM."""
    import json
    import os
    import time

    import numpy as np
    import torch
    import torchvision.transforms as T
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, SamModel, SamProcessor

    os.makedirs(OUT, exist_ok=True)
    device = "cuda"

    data_dir = f"{CACHE}/reason_seg/ReasonSeg/val"
    if not os.path.isdir(data_dir):
        snapshot_download(
            REASONSEG_DATASET,
            repo_type="dataset",
            allow_patterns=["val/*"],
            local_dir=f"{CACHE}/reason_seg/ReasonSeg",
        )

    stems = sorted(f[:-5] for f in os.listdir(data_dir) if f.endswith(".json"))
    if limit:
        stems = stems[:limit]
    print(f"{len(stems)} samples")

    print(f"loading {VLM_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(VLM_REPO, trust_remote_code=True, use_fast=False)
    vlm = (
        AutoModel.from_pretrained(
            VLM_REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        .eval()
        .to(device)
    )

    print(f"loading {SAM_REPO}")
    sam_processor = SamProcessor.from_pretrained(SAM_REPO)
    sam = SamModel.from_pretrained(SAM_REPO, torch_dtype=torch.float32).eval().to(device)

    # One 448px tile. InternVL2's dynamic tiling would raise the token count
    # (and the cost) several-fold; single-tile is the cheap setting, and saying
    # so matters because it is a real handicap on small objects.
    transform = T.Compose(
        [
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    generation_config = {"max_new_tokens": 64, "do_sample": False}

    from asvl_cascade import GROUNDING_PROMPT, is_degenerate, parse_box

    replies: dict[str, dict] = {}
    started = time.perf_counter()

    for i, stem in enumerate(stems):
        record = json.loads(open(f"{data_dir}/{stem}.json").read())
        query = (record.get("text") or [""])[0]
        image_pil = Image.open(f"{data_dir}/{stem}.jpg").convert("RGB")
        width, height = image_pil.size

        pixel_values = transform(image_pil).unsqueeze(0).to(torch.bfloat16).to(device)
        with torch.no_grad():
            reply = vlm.chat(
                tokenizer, pixel_values, GROUNDING_PROMPT.format(query=query), generation_config
            )

        box = parse_box(reply, width, height)
        entry = {"query": query, "reply": reply, "is_sentence": bool(record.get("is_sentence"))}

        if box is None or is_degenerate(box, width, height):
            entry["box"] = None
            entry["reason"] = "unparseable" if box is None else "degenerate"
            # An empty mask is the honest prediction for "the model gave us
            # nothing", and scores 0 unless the target is also empty.
            np.save(f"{OUT}/{stem}.npy", np.zeros((height, width), dtype=bool))
        else:
            entry["box"] = box.as_xyxy()
            entry["scale"] = box.scale
            entry["raw"] = list(box.raw)

            inputs = sam_processor(
                image_pil, input_boxes=[[box.as_xyxy()]], return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = sam(**inputs, multimask_output=True)
            masks = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0]
            best = int(outputs.iou_scores.cpu().squeeze()[-1].argmax())
            np.save(f"{OUT}/{stem}.npy", masks[best].numpy().astype(bool))

        replies[stem] = entry
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(stems)}  {(time.perf_counter() - started):.0f}s")

    if save_replies:
        with open(f"{OUT}/replies.json", "w") as f:
            json.dump(replies, f, indent=2)

    cache.commit()

    parsed = sum(1 for v in replies.values() if v.get("box"))
    scales: dict[str, int] = {}
    for v in replies.values():
        if v.get("scale"):
            scales[v["scale"]] = scales.get(v["scale"], 0) + 1

    summary = {
        "samples": len(stems),
        "boxes_parsed": parsed,
        "no_box": len(stems) - parsed,
        "scales_seen": scales,
        "seconds": round(time.perf_counter() - started, 1),
        "out": OUT,
    }
    for key, value in summary.items():
        print(f"{key:<16} {value}")
    return summary


@app.function(image=image, volumes={CACHE: cache}, gpu="A10G", timeout=1800)
def smoke() -> dict:
    """Eight images, to check the prompt and the coordinate convention.

    Worth its own entrypoint because the failure this catches -- a box parsed
    at the wrong scale -- produces a full 200-image run that completes, costs
    real money, and scores like a broken method.
    """
    return run.local(limit=8)


@app.function(image=score_image, volumes={CACHE: cache}, timeout=1800)
def score() -> dict:
    """gIoU / cIoU against the reference ground truth, with the short/long split.

    The prediction masks came from the cascade; the ground truth is rasterized
    here by the same code the local test suite covers, honouring LISA's
    conventions (trinary mask, ignore excluded from intersection and union,
    empty-on-empty scores 1.0).
    """
    import json
    import sys

    import numpy as np

    sys.path.insert(0, "/root")
    from asvl.data.reasonseg import (
        ground_truth_masks,
        ignore_mask,
        load_split,
        query_types,
        target_mask,
    )
    from asvl.metrics.reasonseg import compute_scores, missing_predictions

    samples = load_split(f"{CACHE}/reason_seg/ReasonSeg", "val")
    trinary = ground_truth_masks(samples)
    targets = {k: target_mask(v) for k, v in trinary.items()}
    ignores = {k: ignore_mask(v) for k, v in trinary.items()}
    types = query_types(samples)

    predictions = {}
    import os

    for name in os.listdir(OUT):
        if name.endswith(".npy"):
            predictions[name[:-4]] = np.load(f"{OUT}/{name}").astype(bool)

    absent = missing_predictions(predictions, targets)
    scores = compute_scores(predictions, targets, query_types=types, ignores=ignores)

    result = {
        "giou": round(scores.giou * 100, 2),
        "ciou": round(scores.ciou * 100, 2),
        "n": scores.n_samples,
        "missing_predictions": len(absent),
        "by_query_type": {
            name: {"giou": round(v.giou * 100, 2), "ciou": round(v.ciou * 100, 2), "n": v.n_samples}
            for name, v in scores.by_query_type.items()
        },
    }

    replies_path = f"{OUT}/replies.json"
    if os.path.exists(replies_path):
        replies = json.loads(open(replies_path).read())
        result["boxes_parsed"] = sum(1 for v in replies.values() if v.get("box"))
        result["no_box"] = sum(1 for v in replies.values() if not v.get("box"))

    print(json.dumps(result, indent=2))
    print()
    print(f"{'method':<34}{'gIoU':>8}{'cIoU':>8}")
    print(f"{'InternVL2-2B + SAM (this run)':<34}{result['giou']:>8.1f}{result['ciou']:>8.1f}")
    for name, (giou, ciou) in {
        "Grounded-SAM (LISA Tab.1)": (26.0, 14.5),
        "OVSeg (LISA Tab.1)": (28.5, 18.6),
        "X-Decoder (LISA Tab.1)": (22.6, 17.9),
        "GRES (LISA Tab.1)": (22.4, 19.9),
    }.items():
        print(f"{name:<34}{giou:>8.1f}{ciou:>8.1f}")

    with open(f"{OUT}/scores.json", "w") as f:
        json.dump(result, f, indent=2)
    cache.commit()
    return result


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    run.remote(limit=limit)
    print("\nNext: python scripts/score_cascade.py --pull")
