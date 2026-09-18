# learning-doc

Notes on replicating AnchorSeg and porting it to InternVL2 — what the method
does, how the harness is built, and the decisions that will look arbitrary
later if they aren't written down.

---

## 1. The task and why SAM is inside the model

**Reasoning segmentation**: the query doesn't name the target. "The thing
propping the door open" requires inferring *which* object, then localizing it.
Ordinary referring segmentation ("the red mug") only needs the second half.

The architecture everyone uses descends from LISA (CVPR 2024):

```
image ─┬─► CLIP ViT ──► VLM (LLaVA / InternVL2) ──► hidden state of [SEG] token
       │                                                     │
       │                                          projection to SAM prompt space
       │                                                     ▼
       └─► SAM image encoder (frozen) ─────────────► SAM mask decoder ──► mask
```

SAM is a **component of the model**, not preprocessing: the gradient from the
mask loss flows back through SAM's decoder, through the projection, into the
VLM. That's the whole trick — "embedding as mask." SAM's image encoder stays
frozen (it's 632M params and already good); the mask decoder is tiny (~4M) and
is trained.

## 2. What AnchorSeg changes

LISA's `<SEG>` is a single embedding carrying both *what* to segment and
*where*. AnchorSeg's claim is that this conflation is the bottleneck, and
replaces it with:

- an ordered **bank of latent reasoning tokens** — intermediate semantic states
- a separate **segmentation anchor token** — explicit spatial grounding
- **Token–Mask Cycle Consistency (TMCC)** — a bidirectional objective aligning
  token-level predictions with pixel supervision across resolutions

Reported on ReasonSeg val with LLaVA-1.5-7B: **gIoU 67.20 / cIoU 75.15**.

### Why this paper and not the other SOTA claim

**STAMP-2B** posts a higher headline average on the reasoning-segmentation
survey tables, but it *removes SAM* — masks are predicted non-autoregressively
as a fill-in-the-blank over image patches. Fine work, wrong shape for a project
whose premise is SAM as a component.

**DR²Seg** (Jan 2026) keeps SAM (SAM2/SAM3) but is a self-rewarding RL method
over Qwen2.5-VL. RL replications are substantially flakier — reward hacking,
rollout variance, and a much wider band of "did I reproduce it?" — so it's a
bad first replication even though it's newer.

AnchorSeg: SOTA, keeps SAM, code *and* weights released, ACL 2026.

## 3. The single most important design decision here

**The metrics and ground truth are reimplemented, not imported.**

If this repo called upstream's evaluator, then "we reproduced 67.20 gIoU" would
mean exactly one thing: *their code runs on my machine*. Any disagreement
between their implementation and the definition in the paper would appear on
both sides of the comparison and cancel. That's not a replication, it's a smoke
test.

So `src/asvl/metrics/reasonseg.py` computes gIoU/cIoU from the definitions, and
`src/asvl/data/reasonseg.py` rasterizes ground truth from the annotation
polygons. Two independent paths agreeing is evidence; one path agreeing with
itself is not.

Both are numpy-only — no torch, no GPU, no 14GB download — so they're testable
today, which is why there are 21 tests in a repo that has never run a model.

### gIoU vs cIoU, and why both

- **gIoU** = mean of per-sample IoU. Every image counts once.
- **cIoU** = (Σ intersections) / (Σ unions) over the dataset. Pixels count once.

A model that only finds large objects scores well on cIoU and badly on gIoU.
There's a test pinning exactly this: one large object right, one small object
missed → gIoU 0.50, cIoU 0.99.

This matters for debugging, not just reporting. If the replication misses on
gIoU but matches cIoU, the gap is concentrated in small objects — which points
at mask resolution or rasterization, *not* at the model. `pins.verdict()`
encodes that inference so the reader doesn't have to rederive it.

### What actually happened: I got the ground truth wrong

The first version of this repo reasoned about the annotation format from first
principles and wrote what seemed obviously right: union the `target` polygons,
subtract the `ignore` polygons, rasterize with pycocotools because PIL's
inclusive outline adds a spurious one-pixel border.

Then `scripts/check_ground_truth.py` ran on the real split and reported **5
empty masks out of 200**. Reading LISA's `get_mask_from_json` afterwards showed
four separate mismatches with the reference — each silent, each of which would
have surfaced later as a modelling gap and been debugged as one:

| | I had | Reference | Consequence |
|---|---|---|---|
| Mask values | binary | **trinary** — 0 bg, 1 target, **255 ignore** | Eval passes `ignore_index=255`; ignore pixels leave *both* intersection and union. As background, a prediction there is a false positive instead of a no-op. |
| Combination | union targets, subtract ignores | **painter's algorithm, largest area first** | A small target inside a large ignore ends up as *target*, painted second. Subtracting erases it — this emptied sample `914980029`, whose target covers 31828 px. |
| `flag` label | treated as a target | **dropped** ("meaningless deprecated annotations") | 62 occurrences in val. Mine survived only because all 62 have fewer than 3 points, so the degenerate-shape guard caught them by luck. |
| Boundary | pycocotools RLE, exclusive | `cv2.polylines` **then** `cv2.fillPoly` — **inclusive** | ~1px border. With a median target area of 6% of the frame, that is percent-level IoU — comparable to the gap between published methods. |

The fourth is the instructive one. My original note argued pycocotools was
*more correct* than PIL's inclusive fill. Wrong frame: there is no more-correct
rasterizer here. The ground truth is whatever the reference paints, and a
stricter one is a **different benchmark**, not a better-measured one.

This is the entire argument for step 1, in one finding. Every one of these bugs
is silent, and every one would have surfaced as "our InternVL2 port
underperforms" three steps and several hundred dollars later.

### What the real split looks like

```
samples              200
empty masks          4         <- genuinely shapeless annotations
target area          median 0.064   p10 0.005   p90 0.304
ignore regions       76 samples, mean area 0.024
query types          long 113, short 87
queries per sample   1.72
```

- **Small objects dominate.** Median target is 6% of the frame, p10 is 0.5%.
  This is why gIoU and cIoU diverge, and why a boundary-level rasterization
  difference is not a rounding error.
- **The 4 empty targets are worth 2.0 points of gIoU.** Under the reference's
  `acc_iou[union_i == 0] += 1.0`, a model scores 1.0 on each by predicting
  nothing — *larger than the ±1.0 tolerance* the replication is judged against.
  Whether an implementation honours that convention decides the verdict alone.

### Conventions the tests pin

- **Ignore excluded from both intersection and union**, per `ignore_index=255`.
- **Empty-on-empty scores IoU 1.0**, per `acc_iou[union_i == 0] += 1.0`.
- **Masks compared at full image resolution.** `single_iou` raises on a shape
  mismatch rather than silently resizing — a silent resize is how an entire eval
  ends up scored at SAM's 1024px, understating boundary error.
- **Missing predictions score as empty, not skipped.** Otherwise a job that
  crashed after its 50 easiest images reports a *better* number than one that
  finished.

## 4. The plan, and why this order

| Step | | Cost |
|---|---|---|
| 1 | Reproduce the **eval** from released weights | ~1 h GPU, ~17GB download |
| 2 | Reproduce the **training** on LLaVA-1.5-7B | ~40-70 h A100, ~100GB+ data |
| 3 | Port the method to **InternVL2** | training cost again |
| 4 | Full-FT vs LoRA ablation on the InternVL2 variant | multi-GPU |

Step 1 first because it's ~1% of the cost of step 2 and it validates the data
pipeline, the metrics, the mask resolution convention and the checkpoint. If it
doesn't match, nothing downstream is interpretable — a step-3 gap could be the
backbone port, or it could be that the ground truth was wrong the whole time.

## 5. What upstream actually does (relevant to step 4)

From `scripts/7b_reason_seg_val/train_anchorseg_llava1.5_ema.sh`:

```
deepspeed --include "localhost:5" train_ds.py \
  --lora_r=8 --batch_size=2 --grad_accumulation_steps=10 \
  --epochs=120 --steps_per_epoch=500 --lr=3e-4 \
  --vision_pretrained=sam_vit_h_4b8939.pth
```

**One GPU. LoRA r=8.** SAM ViT-H frozen. 60k steps, effective batch 20.

This matters: SOTA here was *not* reached by full fine-tuning. Full-FT of
InternVL2 is therefore an ablation the paper didn't run, not part of the
replication — and it should come last, because a gap in step 3 is already hard
enough to attribute without also having changed the training scope.

Rough numbers for step 4, optimizer states only (bf16 weights + bf16 grads +
fp32 master + Adam m/v = 16 B/param):

| | params | states |
|---|---|---|
| InternVL2-2B full FT | 2.2B | ~35 GB |
| InternVL2-8B full FT | 8.1B | ~130 GB |
| LoRA r=8 + projection + SAM decoder | ~0.03B | ~0.5 GB |

35GB before activations means ZeRO-3 across 4×A100-80G, or one H100 with CPU
offload. Also budget a **language-ability regression check**: full fine-tuning a
VLM on segmentation data is where it forgets how to hold a conversation, which
is the documented reason the field uses LoRA here — not just cost.

## 6. Friction in upstream's harness

Nothing deep, but all of it blocks a first run:

- GPU index hardcoded (`--include "localhost:5"`).
- Dataset dir is a sibling path (`../dataset_sesame`) with a layout that differs
  from where our fetch script puts things; `eval_upstream.py --print-command`
  prints the symlink to bridge it.
- The eval script picks the split through an **interactive prompt**
  (`select_datasets_interactive`), which doesn't survive a batch job.
- `--dataset="vqa" --sample_rates="1"` in the *eval* script looks wrong but
  isn't — with `--eval_only` the training dataset is unused, and it just has to
  parse.
- **Their eval reports aggregates, not per-sample masks.** Independent scoring
  needs the masks, so step 1 requires a small patch to dump them. That patch is
  the only upstream modification in the replication and should be kept minimal
  and recorded here when written.

## 7. Open

- Step 1 has not been run. Every number in this repo is a target, not a result.
- The per-sample mask dump patch (§6) isn't written yet.
- The InternVL2 port (step 3) is unscoped: its image tokenization is
  tiling-based and its hidden sizes differ from LLaVA's, so the anchor/reasoning
  token projection needs rewiring, not just a config swap.
- Whether TMCC's cross-resolution alignment assumes LLaVA's fixed 336px grid is
  an open question that will decide how much of step 3 is mechanical.
