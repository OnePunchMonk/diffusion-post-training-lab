# Ideas and open work

A running checklist from the 2026-09-18/19 working session, mirrored as
[issue #10](https://github.com/OnePunchMonk/diffusion-post-training-lab/issues/10).
Status is what the repo can defend today, not what has a file for it.

Ordered by the thing that unblocks the most: **the repo has almost no measured
results.** Every harness below is written and tested; two have produced a
number. That ratio is the problem worth fixing first.

---

## 1. FLUX.2 [klein] PEFT sweep — `flux2-klein-peft/`

The harness is complete and unrun. This is the highest-value gap because six
methods, one subject is ~2 GPU-hours and turns a "benchmark" PR into a
benchmark.

- [x] **One-subject sweep run and evaluated** — six adapters trained, scored on
      three held-out prompts. See `flux2-klein-peft/results/`. LoRA leads both
      axes; treat the ordering as a hypothesis, n=1
- [ ] **Publish the adapters to the Hub** — `scripts/push_to_hub.py` is already
      method-aware (non-LoRA methods get a peft-injection snippet, since
      `load_lora_weights` would silently serve the base model)
- [ ] **First real `MODELS.md` rows** — currently one row, an SDXL toy LoRA
      from August
- [ ] Scale to ~5 subjects once one works (~$15-25). Per-subject variance on
      3-image subjects is large, so a single subject is a smoke test, not a
      benchmark
- [ ] **Measure the OFT/BOFT confound.** They adapt strictly fewer modules than
      LoRA — FLUX.2's fused `to_qkv_mlp_proj` doesn't factor for orthogonal
      blocks — so "OFT lost" may mean "OFT with a smaller surface lost".
      Report trainable-parameter counts beside every score

## 2. Quantization — `flux2-klein-peft/scripts/quantize_nvfp4.py`

Written, never executed.

- [ ] **Run NVFP4 weight-only on a merged export**, then A/B DINO and CLIP-T
      against bf16. The question that matters: *does a 4-bit subject adapter
      still hold the subject?*
- [ ] **Activation calibration** — currently raises `NotImplementedError` on
      purpose. modelopt hands the loop a bare transformer, but a meaningful
      pass has to drive the whole pipeline (text encoder → sampler →
      transformer). Feeding it synthetic tensors gives scales that look
      calibrated and aren't
- [ ] Note: NVFP4 needs Blackwell for the speedup. Pre-Blackwell you get the
      ~4× memory saving and software dequantization — the script says which
      you're getting

## 3. Prism ML Bonsai — extreme quantization of the *same* model

[Bonsai Image 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit)
is a ternary (1.58-bit) and binary (1-bit) quantization of FLUX.2 Klein 4B,
Apache 2.0. Transformer: 1.21 GB / 0.93 GB against 7.75 GB FP16. MLX build runs
on Apple Silicon — **free, locally, on the M5 Pro.**

- [ ] **Run it locally** and generate the repo's eval prompt set. Zero spend
- [ ] **Quality-vs-size curve**: FP16 → NVFP4 → ternary → binary, on CLIP-T /
      aesthetic / latency. This is the "plot with a cost axis" the repo lacks
- [ ] **The real experiment: does an FP16-trained LoRA survive on a 1.58-bit
      base?** `quantize_nvfp4.py` asserts "quantize after merging, never
      before" because an adapter is a delta against the weights it was fitted
      to. Bonsai hands you a pre-quantized base — the opposite order — so this
      directly tests a claim already written into this repo. Either answer is
      worth having
- [ ] Their FP16-retained list (modulation, embedders, norms, output
      projections) is nearly identical to the `modules_to_not_convert` in our
      NVFP4 script. Worth diffing properly

## 4. InternVL2 + SAM — `anchorseg-internvl2/`

- [x] **Training-free cascade on ReasonSeg val: 19.5 gIoU / 7.4 cIoU**
      (Grounded-SAM baseline 26.0 / 14.5). See `results/README.md`
- [x] Diagnosed *why*: 91/200 boxes are the whole frame — InternVL2-2B declines
      to localize, on 54% of long queries vs 34% of short
- [ ] **Rerun with dynamic tiling enabled and `max_new_tokens=256`.** Both
      confounds bite hardest on the population that scored worst, so the
      long/short gap is not yet established. Same ~$1
- [ ] Try **LFM2-VL-450M** (Liquid AI) as a 4× smaller grounding stage. Note
      Liquid AI has *no* image diffusion models — their one "Diffusion" entry
      is a masked **text** diffusion encoder
- [ ] Try InternVL2-8B to separate "2B is too small" from "cascades can't do this"

## 5. AnchorSeg replication — `anchorseg-internvl2/`

Environment verified on Modal; data cached. Steps 2-4 are expensive and parked.

- [ ] **Patch upstream to dump per-sample masks.** Their eval prints aggregates
      only, and independent scoring needs the masks. This is the one upstream
      modification the replication requires and the only thing blocking step 1
- [ ] Step 1: reproduce gIoU 67.20 / cIoU 75.15 from released weights (~1 h GPU)
- [ ] Step 2: reproduce the *training* on LLaVA-1.5-7B (~40-70 GPU-h, ~$100).
      Consider a partial run (20 of 120 epochs, ~$20) as a weaker control
- [ ] Step 3: port the method to an InternVL2 backbone — the actual
      contribution. The real work is that AnchorSeg's anchor token assumes
      LLaVA's fixed 336px grid, while InternVL2 tiles dynamically
- [ ] Step 4: full-FT vs LoRA ablation. Note upstream trains on **one GPU with
      LoRA r=8** — full FT is an ablation the paper never ran, and needs a
      language-ability regression check

## 6. Existing recipes with known defects

Both are labelled "do not cite" in their own docstrings. They are also the two
techniques most named in efficiency job descriptions.

- [ ] **Fix `distill.py`** (LCM-style step distillation) — "known correctness
      bugs"
- [ ] **Fix `grpo.py`** — "NOT A CORRECT GRPO IMPLEMENTATION". The docstring
      already names the defect precisely: a real diffusion GRPO surrogate needs
      the log-probability ratio, which means treating sampling as an SDE and
      summing per-step Gaussian log-densities along the trajectory. Converting
      that diagnosis into an implementation is bounded work with a visible
      before/after
- [ ] **Implement CFG distillation** — absent entirely, despite the repo
      registering two guidance-distilled models and documenting why klein needs
      `guidance_scale=1.0`

## 7. Systems / profiling — absent

Nothing in the repo shows where time goes in a forward pass.

- [ ] **Profile a klein forward pass** at 512 and 1024: attention vs MLP vs VAE,
      compute- vs memory-bound. Cheap, and every efficiency claim needs this
      baseline to measure against
- [ ] Re-profile after quantization; publish the delta
- [ ] One fused Triton kernel for whatever the profile says is the bottleneck,
      benchmarked honestly including when it loses
- [ ] Serving: the SGLang / vLLM-Omni launch specs in `serve/backends.py` have
      never been run. No measured throughput or latency
- [ ] No video model anywhere

## 8. Papers to try

- [ ] **[VoT: Vision-of-Thought](https://www.alphaxiv.org/abs/2609.07815)**
      (Sun et al., Sept 2026) — a discrete visual-thinking layer between a VLM
      and a diffusion transformer, three-branch MoT, GenEval 0.91 vs
      FLUX.1-dev 0.82. Built on Mogao-14B, so full replication is expensive;
      the tokenizer-alignment idea may be testable at klein scale

## 9. Direction: a vision-models playground

The pieces already here — SAM, InternVL2, DINO as a metric — point at a wider
scope than diffusion post-training: one repo covering **DINO, SAM, Depth
Anything, VLMs, image generation and video generation** behind a shared eval
harness.

- [ ] Decide whether that lives here or in a new repo. `dptlab`'s stated scope
      is diffusion post-training and it is already straining
- [ ] A shared adapter/eval protocol across families, so a new model is a
      `ModelSpec` rather than a fork
- [ ] **Depth Anything** — absent entirely
- [ ] **DINO** is only a metric today; DINOv2/v3 as a backbone is a different
      piece of work

## 10. Housekeeping

- [ ] `MODELS.md` says it's a build artifact but has one stale row
- [ ] Modal: always `--detach`. A non-detached run dies when the local client
      disconnects — it truncated a 2.4GB download and killed two runs here
- [ ] Modal volumes are inside the free-storage tier; deleting them saves $0
