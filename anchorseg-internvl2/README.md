# asvl — AnchorSeg replication, then an InternVL2 port

> A **self-contained subproject** of `diffusion-post-training-lab`. It shares no
> code with `dptlab` — different models, different task, different eval — so it
> has its own `pyproject.toml`, its own virtualenv, and its own test suite.
> Install and run it from inside this directory.

Replicating [**AnchorSeg: Language Grounded Query Banks for Reasoning
Segmentation**](https://arxiv.org/abs/2604.18562) (Qian et al., ACL 2026),
current state of the art on ReasonSeg, and then porting the method from its
LLaVA backbone to **InternVL2**.

**Task.** Reasoning segmentation: a free-form query that requires inference
rather than naming ("the thing propping the door open") in, a pixel mask out.
The model is a VLM that emits segmentation tokens whose hidden states are
projected into **SAM**'s prompt-embedding space; SAM's decoder produces the
mask and the whole thing trains end to end. SAM is a component of the model,
not a preprocessing step.

**What AnchorSeg changes.** LISA and its successors compress *what* to segment
and *where* to segment it into one `<SEG>` embedding. AnchorSeg replaces that
with an ordered bank of latent reasoning tokens plus a separate spatial anchor
token, and adds a Token–Mask Cycle Consistency objective to align token-level
predictions with pixel supervision across resolutions.

**Target numbers** (upstream, LLaVA-1.5-7B, ReasonSeg val): **gIoU 67.20 /
cIoU 75.15**.

## Plan

| Step | What | Status |
|---|---|---|
| 1 | Reproduce the **eval** from released weights — no training | data verified, GPU run pending |
| 2 | Reproduce the **training** on LLaVA-1.5-7B | not started |
| 3 | Port the method to an **InternVL2** backbone | not started |
| 4 | Full fine-tune vs. LoRA ablation on the InternVL2 variant | not started |

Steps 1-2 are replication; 3-4 are the contribution. The order matters: if
step 1 doesn't match, nothing downstream is interpretable, and step 1 costs
about an hour of GPU rather than the ~$100 of a training run.

## Why the metrics are reimplemented here

`src/asvl/metrics/reasonseg.py` computes gIoU and cIoU from the definitions
rather than calling upstream's evaluator. If the replication reused their
metric code, "we reproduced 67.20" would only mean "we ran their script" — any
disagreement between their implementation and the paper's stated definition
would cancel on both sides and stay invisible. An independent implementation is
what makes a matching number evidence.

Same reasoning for `src/asvl/data/reasonseg.py`, and it has already paid for
itself. Running `scripts/check_ground_truth.py` on the real split surfaced 5
empty masks out of 200; reading LISA's `get_mask_from_json` afterwards showed
**four** mismatches with the reference — the mask is trinary (ignore is `255`,
excluded from intersection *and* union, not background); polygons are painted
largest-area-first so a small target inside a large ignore stays a target;
`flag` shapes are dropped as deprecated; and the outline is painted inclusively,
a percent-level IoU difference on a split whose median target is 6% of the frame.

Every one of those is silent, and every one would have surfaced three steps
later as "our InternVL2 port underperforms". That is the argument for doing
step 1 first. Details in [`docs/learning-doc.md`](docs/learning-doc.md).

## Quickstart

```bash
cd anchorseg-internvl2
uv venv --python 3.12 && uv pip install -e ".[dev]"
python -m pytest                 # metrics + rasterizer, no weights needed

# ReasonSeg val: 200 images, ~1GB
python scripts/fetch_reasonseg.py --split val --out data/reason_seg

# sanity-check the ground truth before involving a model at all
python scripts/check_ground_truth.py --root data/reason_seg --split val
```

## Hardware

Upstream trains on **one GPU with LoRA r=8** (`--include "localhost:5"`,
`--lora_r=8`, batch 2 × grad-accum 10), not a multi-GPU full fine-tune. Their
recipe is 120 epochs × 500 steps ≈ 60k steps — roughly 40-70 h on an A100.

Step 4's full fine-tune is a deliberate departure from the paper, not part of
the replication, and should only be attempted once steps 1-3 line up —
otherwise a gap can't be attributed.

## Layout

```
src/asvl/
  data/reasonseg.py      annotation loading + polygon rasterization
  metrics/reasonseg.py   independent gIoU / cIoU
  upstream/pins.py       pinned commit/revisions + published targets & verdict
scripts/
  setup_upstream.py      clone at the pinned commit, fetch weights + SAM + CLIP
  eval_upstream.py       print the eval command; score its masks independently
  fetch_reasonseg.py     pull the split from the HF mirror
  check_ground_truth.py  verify masks before any model is involved
tests/                   numpy-only, no GPU, no downloads
docs/learning-doc.md     why it's built this way
```

## Credits

Method and released code/weights: [rui-qian/AnchorSeg](https://github.com/rui-qian/AnchorSeg)
(ACL 2026), which builds on READ, LISA, GSVA, SESAME, PixelLM, LLaVA and SAM.
ReasonSeg is from [LISA](https://github.com/dvlab-research/LISA); the split
used here comes from the [fcxfcx/ReasonSeg](https://huggingface.co/datasets/fcxfcx/ReasonSeg)
mirror (239/200/779, matching the paper).

MIT.
