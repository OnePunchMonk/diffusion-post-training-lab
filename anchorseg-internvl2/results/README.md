# Results

## InternVL2-2B + SAM, training-free, on ReasonSeg val

One inference pass, 200 images, ~8 min on an A10G. Nothing trained.

| Method | gIoU | cIoU | Trained? |
|---|---|---|---|
| **InternVL2-2B + SAM (this run)** | **19.5** | **7.4** | no |
| Grounded-SAM (LISA Tab. 1) | 26.0 | 14.5 | no |
| OVSeg (LISA Tab. 1) | 28.5 | 18.6 | no |
| X-Decoder (LISA Tab. 1) | 22.6 | 17.9 | no |
| GRES (LISA Tab. 1) | 22.4 | 19.9 | no |
| LISA-7B (LISA Tab. 1) | ~52 | — | yes |

By query type:

| | n | gIoU | cIoU |
|---|---|---|---|
| short (phrase) | 87 | 24.0 | 13.5 |
| long (sentence) | 113 | 16.0 | 4.9 |

### The hypothesis, and what happened

LISA argues that cascades score badly on ReasonSeg because the grounding stage
can only *match* text to objects, while the queries require inferring which
object is meant. That predicts a reasoning-capable grounding stage should help,
and help most on long queries.

**It did neither.** The cascade scored below Grounded-SAM, and the long/short
gap runs the wrong way: 16.0 vs 24.0 gIoU.

### Why — the mechanism, not the number

`scripts/diagnose_cascade.py`:

```
predicted box area (median frac)  0.842
ground-truth area (median frac)   0.064
box / target area ratio (median)  6.4x
target recall by box alone        1.000 median

query type    n   whole-frame boxes    rate
short        87                  30     34%
long        113                  61     54%
```

**91 of 200 boxes are exactly `[0, 0, 1000, 1000]`** — the whole frame. That is
InternVL2-2B's way of declining to localize, and it does it on 54% of long
queries against 34% of short ones. The median predicted box covers 84% of the
image against a 6.4% median target.

So the failure is not "the VLM grounds the wrong object". It is "the VLM
declines to ground at all, more often as the query gets harder", and SAM is
then prompted with a box containing the entire image. That also explains why
cIoU (7.4) collapses further than gIoU (19.5): whole-frame predictions inflate
the union enormously, and cIoU is a pixel-weighted metric.

The box-recall figure makes the same point from the other side: the median box
contains 100% of the target, which sounds excellent and only means the boxes
are too big to be wrong.

### Confounds — read before quoting this

1. **Single 448px tile.** InternVL2's dynamic aspect-ratio tiling was disabled
   to keep the run cheap. Its published grounding results (RefCOCO etc.) use
   tiling, so this is a handicapped configuration and the whole-frame rate may
   drop substantially with it enabled. **This is the next experiment and it is
   cheap.**
2. **2B is the smallest variant.** Grounding is a capability that scales; 8B
   may behave differently.
3. **`max_new_tokens=64`.** Replies echo the query before emitting the box and
   reach 244 characters, close to the budget. Long queries are the ones at
   risk, and they are also the ones that scored worst — so raise this before
   drawing conclusions about the long/short split specifically.

Given (1) and (3) both bite hardest on exactly the population that scored
worst, the long/short gap should be treated as **not yet established**. The
whole-frame behaviour itself is solid: it is visible in the raw replies.

### Reproducing

```bash
modal run --detach modal/cascade.py::smoke   # 8 images, checks the coordinate convention
modal run --detach modal/cascade.py::run     # 200 images, ~8 min A10G
modal run --detach modal/cascade.py::score   # CPU, independent metrics
python scripts/diagnose_cascade.py
```

`cascade-internvl2-replies.json` holds every raw model reply, so any of the
above can be re-checked without re-running the GPU pass.
