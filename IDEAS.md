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
      three held-out prompts — [`flux2-klein-peft/RESULTS.md`](flux2-klein-peft/RESULTS.md).
      LoRA leads both axes; treat the ordering as a hypothesis, n=1
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

## 10. Post-training methods not yet covered

The repo has four recipes (LoRA, DPO, GRPO, LCM distillation). The space is much wider, and most of these are a new module against the existing `TrainConfig` / `ModelSpec` / eval scaffolding rather than a fork. Grouped by axis; **(cheap)** means it fits the current one-subject, sub-hour budget.

### Preference optimization beyond DPO

- [ ] **SPO** (Step-aware Preference Optimization) — preference at each denoising step rather than on the final image, which is the obvious objection to Diffusion-DPO
- [ ] **Diffusion-KTO** — pointwise "good/bad" labels instead of pairs. Much cheaper to label, and `build_preference_pairs.py` already throws away unpaired samples
- [ ] **SPIN-Diffusion** — self-play, no human labels. Connects to the `self-improving-diffusion` framework
- [ ] **ReFL / DRaFT / AlignProp** — backprop a reward straight through the sampling chain. A genuinely different mechanism from both DPO and GRPO, and the one most likely to expose memory limits

### RL beyond the current (broken) GRPO

- [ ] **DDPO** — sampling as a multi-step MDP. The foundation `grpo.py`'s docstring points at
- [ ] **DPOK** — KL-regularized RL fine-tuning
- [ ] **Flow-GRPO** — the SDE formulation with per-step Gaussian log-densities that the GRPO fix needs
- [ ] **DanceGRPO** — unified across image and video

### Distillation beyond LCM

- [ ] **Progressive distillation** — the origin; useful as a baseline the others are measured against
- [ ] **Consistency Distillation / CTM**
- [ ] **ReFlow / InstaFlow** — straighten the trajectory, then distill. Directly relevant: klein is already rectified-flow
- [ ] **ADD / LADD** (adversarial diffusion distillation) — where pure regression distillation plateaus
- [ ] **DMD / DMD2** — distribution matching rather than trajectory matching. Current strong baseline
- [ ] **SiD** (score identity distillation)
- [ ] **CFG distillation** — already tracked in §6; belongs to this family

### Inference-time acceleration (training-free, so **cheap**)

These need no training at all and produce a latency number immediately — the thing the repo most lacks.

- [ ] **DeepCache / TeaCache / FORA** — cache and reuse block outputs across steps **(cheap)**
- [ ] **Token merging (ToMe) for diffusion** **(cheap)**
- [ ] **Attention slicing / sparse attention at inference** **(cheap)**

### Structural compression

- [ ] **Depth pruning of DiT blocks** — drop transformer layers and distil to recover
- [ ] **Width/channel pruning**
- [ ] **Knowledge distillation into a smaller student architecture** (as opposed to fewer steps)

### Adaptation methods that are not PEFT

- [ ] **Textual Inversion** — learn an embedding, touch no weights. The cheapest personalization baseline, and a fair comparison point the sweep currently lacks **(cheap)**
- [ ] **Full DreamBooth fine-tune** — the other end of the same axis
- [ ] **ControlNet / T2I-Adapter / IP-Adapter** — conditioning rather than concept adaptation; a different axis the `ModelSpec` abstraction does not yet express
- [ ] **REPA** (representation alignment) — a training accelerator, not a post-training method, but cheap and it composes
- [ ] **Adapter merging/composition** — TIES, DARE. Natural follow-on now that six adapters for one subject exist

### Data and reward side

- [ ] **Train a reward model** rather than reusing CLIP+aesthetic. `grpo.py` and `build_preference_pairs.py` both lean on eval metrics as rewards, which is convenient and circular
- [ ] **Synthetic preference data** generation and its failure modes

### Safety / control

- [ ] **Concept erasure** (ESD, UCE) and **machine unlearning** for diffusion — post-training that removes a capability rather than adding one. Entirely absent, and a distinct evaluation problem

## 11. Step distillation — ideas

`distill.py` is one LCM-style recipe, marked "do not cite". The family is much
larger, and klein is an unusually good testbed: it is **already** step- and
guidance-distilled, so every experiment here is "can we distil a distilled
model further, and what breaks first".

| Method | Idea | Why it's on the list |
|---|---|---|
| **Progressive Distillation** | Halve the steps, repeatedly | The origin. Needed as the baseline others are measured against |
| **Consistency Models / LCM** | Map any point on the ODE trajectory to its endpoint | What `distill.py` attempts |
| **CTM** (Consistency Trajectory) | Interpolates between consistency and diffusion objectives | Fixes LCM's quality ceiling at 1-2 steps |
| **TCD** | Trajectory consistency, with a tunable stochasticity knob | Practical; widely used with SDXL |
| **ReFlow / InstaFlow** | Straighten the trajectory first, *then* distil | Directly relevant: klein is rectified flow already |
| **PeRFlow** | Piecewise rectified flow | Cheaper reflow, works as a plug-in |
| **ADD / LADD** | Add a discriminator to the distillation loss | Where pure regression distillation plateaus |
| **DMD / DMD2** | Match distributions, not trajectories | Current strong baseline; read closely |
| **SiD** | Score identity distillation, data-free | No teacher samples needed |
| **Shortcut models** | Condition on step size; one model, any budget | Removes the "one student per step count" problem |

Experiments that fit this repo specifically:

- [ ] **LCM-LoRA as a sweep method.** Distillation *as an adapter* — this is the
      one that belongs in `flux2-klein-peft` directly. It makes "which PEFT
      method distils best" a question, and the harness already exists
- [ ] **Distil a subject adapter and the step count jointly.** Does a
      personalized klein survive being pushed from 4 steps to 2?
- [ ] **Step count vs subject fidelity curve** at 4/3/2/1 steps on the existing
      six adapters. Cheap — inference only, no training
- [ ] Fix `distill.py`'s documented bugs before adding anything (§6)

## 12. CFG distillation — ideas

Classifier-free guidance costs **two forward passes per step**. CFG distillation
folds it into one. Named in the JD, absent from this repo, and the repo already
registers two guidance-distilled models without implementing the technique.

- [ ] **Guided Distillation** (Meng et al. 2023) — the original: the student
      takes the guidance weight `w` as an input, so one model covers the whole
      guidance range
- [ ] **Guidance embedding** (the FLUX.1-dev / klein approach) — guidance as a
      conditioning vector baked in at training time. Worth reading klein's own
      implementation, since it is sitting in the registry
- [ ] **Does a LoRA re-introduce guidance dependence?** klein is
      guidance-distilled and served at `guidance_scale=1.0`. If a subject
      adapter trained at cfg 1.0 behaves differently under cfg > 1, that is a
      measurable and slightly alarming result, and it costs one inference sweep
      **(cheap)**
- [ ] **CFG only on some timesteps** — guidance matters far more early than
      late. Training-free, gives an immediate latency win, and is a good
      first efficiency number **(cheap)**
- [ ] **CFG++ / rescaled guidance** — not distillation, but the same axis and
      nearly free to try
- [ ] Implement it on SDXL first, where CFG still genuinely costs 2×, rather
      than on klein where it has already been removed

## 13. Quantization — ideas

`quantize_nvfp4.py` exists and has never run. The interesting question for this
repo is not "can we quantize klein" — Prism ML already did, to 1.58 bits — but
**how quantization and adapters interact**, which is where the two threads meet.

### Post-training quantization

- [ ] **SmoothQuant** — activation outliers are the whole problem; migrate
      difficulty from activations to weights
- [ ] **GPTQ / AWQ** — weight-only, calibration-based
- [ ] **Q-Diffusion / PTQ4DM / TFMQ-DiT** — diffusion-specific. The key insight
      is **per-timestep calibration**: activation ranges shift enormously across
      the denoising trajectory, so one calibration set is wrong for most of it
- [ ] **Sensitivity analysis** — which blocks tolerate low bits? Prism ML keeps
      modulation, embedders, norms and output projections in FP16, which is
      almost exactly our `modules_to_not_convert`. Measuring *why* would be a
      real contribution **(cheap-ish)**

### Formats

- [ ] **NVFP4 / MXFP8** — microscaling, native on Blackwell. Script written
- [ ] **INT8 / FP8** — the boring baselines the fancy formats must beat
- [ ] **Ternary / binary** — Prism ML's Bonsai (§3). 1.58 bits at GenEval 0.723

### Quantization × adapters — the actual research question

- [ ] **QLoRA / QA-LoRA / LoftQ, ported to diffusion.** In LLM-land the answer
      to "does an FP16 adapter survive a quantized base" is *train the adapter
      against the quantized base*. Nobody has systematically done this for
      diffusion PEFT, and this repo now has six adapters and a quantization
      script
- [ ] **SVDQuant** — absorbs activation outliers into a **low-rank branch**.
      Literally quantization plus LoRA in one method, and the closest existing
      work to where these two threads meet. Read this before designing anything
- [ ] **Order matters, and we assert it without evidence.**
      `quantize_nvfp4.py` says "quantize after merging, never before". Bonsai
      offers a pre-quantized base, so the claim is testable: merge→quantize vs
      adapt-on-quantized, same subject, same metrics
- [ ] **Quantize the six existing adapters** and re-score. Which PEFT method is
      most robust to 4-bit? The orthogonal methods might behave very differently
      from the low-rank ones, since they cannot change singular values
- [ ] **Activation calibration** driving the whole pipeline (§3, currently
      raises `NotImplementedError` rather than faking it)

### Interactions worth measuring

- [ ] **Quantization × caching** (DeepCache/TeaCache) — do the savings compose
      or collide?
- [ ] **Quantization × step count** — does a 4-step model tolerate 4-bit as well
      as a 30-step one? There are fewer steps to average away the error

## 14. Housekeeping

- [ ] `MODELS.md` says it's a build artifact but has one stale row
- [ ] Modal: always `--detach`. A non-detached run dies when the local client
      disconnects — it truncated a 2.4GB download and killed two runs here
- [ ] Modal volumes are inside the free-storage tier; deleting them saves $0
