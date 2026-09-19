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
- [ ] **Swap the grounding stage for [LFM2.5-VL](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
      (Liquid AI).** The strongest single follow-up to the cascade result, for
      four reasons that all happen to line up:
  - **It actually grounds.** Bounding-box prediction is a *new capability in
    2.5* (LFM2-VL did not have it), and it scores **81.28 on RefCOCO-M**. Our
    failure mode was InternVL2-2B returning the whole frame on 46% of samples;
    a model with a real grounding number is the clean way to separate "cascades
    can't reason" from "InternVL2-2B specifically won't localize"
  - **Native tiling** — 512×512 non-overlapping patches plus a thumbnail for
    global context. That removes the biggest confound in our run, where we
    disabled InternVL2's dynamic tiling to keep costs down
  - **Output is normalized [0, 1]** JSON:
    `[{"label": ..., "bbox": [x1, y1, x2, y2]}]`. Our `parse_box` already
    infers and handles that scale (`SCALE_UNIT`), so the parser needs no change
    — though the JSON-array format does need a branch
  - **MLX builds at 4/5/6/8-bit and bf16** → the grounding stage runs **free on
    the M5 Pro**. No GPU spend at all **(free)**
  - Sizes: 450M (4× smaller than InternVL2-2B), 1.6B, 3B — so it also gives a
    scaling curve for free
  - Licence is **LFM1.0**, not Apache. Worth reading before anything is
    published on top of it
- [ ] Note Liquid AI has *no* image diffusion models — their one "Diffusion"
      entry is a masked **text** diffusion encoder (`LFM2.5-Encoder-350M-Diffusion`)
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

## 8. Direction: a vision-models playground

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

## 9. Post-training methods not yet covered

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

## 10. Step distillation — ideas

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

## 11. CFG distillation — ideas

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

## 12. Quantization — ideas

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

## 13. Paper survey — candidates to try

Surveyed 2026-09-19. Ranked by *fit to what this repo already has*, not by how
recent they are. A paper earns a place here by being testable at klein scale
with released code, or by answering a question already written into the repo.

Cost tags: **(free)** runs on the M5 Pro · **(cheap)** ≲1 GPU-hour ·
**(£££)** needs real training budget.

### Tier 1 — directly answers an open question in this repo

- [ ] **[FourTune](https://hanlab.mit.edu/projects/fourtune)** — *Towards Fully
      4-Bit Efficient Post-Training for Diffusion Models*
      ([arXiv 2607.05711](https://arxiv.org/abs/2607.05711), 2026; Xue, Min,
      Li, Zhang, Xi, Zhang, Agrawala, Zhu, Han, Lin, Li — MIT / CMU / Stanford
      / Berkeley, the SVDQuant group).
      End-to-end **W4A4G4**: weights, activations *and gradients* in 4 bits.
      A triple-branch pipeline augments LoRA with a **frozen numerical
      stabilizer** for quantization-sensitive outliers, plus block-wise
      quantization and fused kernels for quantized backprop.
      **2.25× less memory and 2.27× more training throughput than BF16 LoRA**
      on FLUX.1-dev 12B, at full-precision quality.
      **Why this is the most relevant paper in the survey for us:** every other
      quantization entry here is about *inference*. This one makes *training*
      cheaper, which is the binding constraint on this repo — it would roughly
      halve the cost of the 20-subject sweep and let it run on a smaller card.
      It is also validated on exactly our three axes: customization, RL and
      distillation. Code link exists on the project page but release is
      unconfirmed — check before scoping. **(cheap to try, if the code is out)**
- [ ] **[SVDQuant](https://arxiv.org/abs/2411.05007)** (ICLR 2025 Spotlight) —
      absorbs activation outliers into a high-precision **low-rank branch**,
      then quantizes the rest to 4 bits. 3.5× memory and 3.0× speedup on
      FLUX.1 12B on a 16GB 4090.
      **Why it matters:** it "seamlessly supports off-the-shelf
      LoRAs without re-quantization" — the LoRA branch fuses into the low-rank
      branch by slightly raising the rank. That is a *direct answer* to the
      question in §13 and §3 about whether an FP16-trained adapter survives a
      quantized base: SVDQuant's answer is yes, by construction. Our six klein
      adapters are exactly the input it wants. Engine
      ([Nunchaku](https://github.com/mit-han-lab/nunchaku)) is released. **(cheap)**
- [ ] **[LoRaQ](https://arxiv.org/html/2604.18117v1)** — optimized low-rank
      approximation for 4-bit quantization. Read against SVDQuant; the
      difference between them is the thing to understand
- [ ] **[LoraQuant](https://arxiv.org/html/2510.26690v1)** — mixed-precision
      quantization *of the LoRA itself*, to ultra-low bits. The complement:
      SVDQuant quantizes the base and keeps the adapter high-precision, this
      squeezes the adapter. Together they bracket the design space
- [ ] **[GPTQ-intrinsic LoRA](https://arxiv.org/pdf/2606.01412)** — near-optimal
      joint low-precision quantization with low-rank adaptation
- [ ] **[OrbitQuant](https://arxiv.org/pdf/2607.02461)** — data-agnostic
      quantization for image **and video** diffusion transformers. Data-agnostic
      matters: it sidesteps the per-timestep calibration problem that
      `quantize_nvfp4.py` currently refuses to fake

### Tier 2 — step distillation, testable at klein scale

klein is already 4-step and rectified-flow, so these are all "distil a
distilled model further, and see what breaks".

- [ ] **[SANA-Sprint](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf)**
      (ICCV 2025) — one-step via continuous-time consistency distillation.
      SOTA GenEval at 0.1s vs 1.1s on H100. The strongest single reference
- [ ] **[pi-Flow](https://arxiv.org/pdf/2510.14974)** — policy-based few-step
      generation via imitation distillation. A different framing from both
      consistency and distribution matching
- [ ] **[One-Step Flow](https://arxiv.org/pdf/2412.09465)** (ICLR 2026) —
      noise-augmented conditional rectified flow to widen the teacher's support
- [ ] **[Self-Corrected Flow Distillation](https://www.researchgate.net/publication/390709870_Self-Corrected_Flow_Distillation_for_Consistent_One-Step_and_Few-Step_Image_Generation)**
      — consistency across one- and few-step regimes
- [ ] **[Few-Step Diffusion Sampling Through Instance-Aware Discretizations](https://arxiv.org/pdf/2603.17671)**
      — picks the step schedule per instance. **Training-free**, so this is the
      cheapest real efficiency win available **(cheap)**
- [ ] **[A Decomposable Probe for Few-Step Diffusion Models](https://arxiv.org/pdf/2607.03256)**
      — prompt / latent / score selectivity across backbone families and
      distillation paradigms. Not a method: an *analysis* toolkit, and the kind
      of thing that makes a benchmark repo more than a leaderboard **(cheap)**
- [ ] **Score identity Distillation (SiD)** — data-free. Note: code and
      checkpoints not released as of this survey, only promised

### Tier 3 — flow matching itself

klein is rectified flow, and `objectives.py` implements the interpolant and
the logit-normal timestep schedule by hand. These are the papers that would
justify or change those choices.

- [ ] **[Shortcut models](https://arxiv.org/abs/2410.12557)** (Frans et al.,
      ICLR 2025) — condition on step size; one model serves any step budget.
      Removes the "one student per step count" problem entirely
- [ ] **MeanFlow** (Geng et al., NeurIPS 2025) and **Improved MeanFlows** —
      one-step generative modelling via average velocity. Plus 2026 follow-ups:
      **Overcoming the curvature bottleneck in MeanFlow**, **Terminal Velocity
      Matching**
- [ ] **[Isokinetic Flow Matching](https://arxiv.org/pdf/2604.04491)** —
      pathwise straightening. Straighter paths are *why* few-step works, so
      this is upstream of every distillation method above
- [ ] **[Curriculum Sampling](https://arxiv.org/pdf/2603.12517)** — a two-phase
      timestep curriculum for efficient flow-matching training. Directly
      applicable: our `_sample_sigmas` uses a fixed logit-normal, and a
      curriculum is a small, testable change **(cheap)**
- [ ] **[On Variance Reduction in Learning Mean Flows](https://arxiv.org/pdf/2605.09235)**
- [ ] **[Order-Optimal Sample Complexity of Rectified Flows](https://arxiv.org/abs/2601.20250)**
      — theory; useful for knowing how much data the sweep actually needs
- [ ] **[MIT 6.S184 lecture notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)**
      — the clean introduction to flow matching and diffusion. Read first if
      any of the above feels shaky

### Tier 4 — bigger swings, budget permitting

- [ ] **[VoT: Vision-of-Thought](https://www.alphaxiv.org/abs/2609.07815)**
      (Sept 2026) — discrete visual-thinking layer between a VLM and a
      diffusion transformer; three-branch MoT. GenEval **0.91** vs FLUX.1-dev
      0.82. Built on Mogao-14B, so replication is **(£££)** — but the
      tokenizer-alignment idea might be testable at klein scale
- [ ] **[AnchorSeg](https://arxiv.org/abs/2604.18562)** (ACL 2026) — already
      scaffolded in `anchorseg-internvl2/`. **(£££)** for training, **(cheap)**
      for the eval-only step

### Reading order if time is short

1. **FourTune** — it makes training cheaper, which is the binding constraint
2. **SVDQuant** — it answers a question we have already written down twice
3. **MIT lecture notes** §flow matching — grounds everything in Tier 3
4. **SANA-Sprint** — the distillation reference point
5. **Instance-aware discretizations** + **Curriculum Sampling** — the two
   cheapest things here that produce a number

## 14. Housekeeping

- [ ] `MODELS.md` says it's a build artifact but has one stale row
- [ ] Modal: always `--detach`. A non-detached run dies when the local client
      disconnects — it truncated a 2.4GB download and killed two runs here
- [ ] Modal volumes are inside the free-storage tier; deleting them saves $0
