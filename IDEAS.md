# Roadmap and reference guide

The working roadmap for `dptlab` and its two subprojects, mirrored as
[issue #10](https://github.com/OnePunchMonk/diffusion-post-training-lab/issues/10).
It is also meant to be read cold: every open item says what the concept is,
where it touches the code, what "done" looks like, and what to read. Sections
§A–§D are reference material the checklist leans on.

**Status is what the repo can defend today**, not what has a file for it.
Last full revision: 2026-09-25, against `main` @ `810b48a`.

**Legend.** `[x]` done · `[ ]` open · **(free)** runs locally on the M5 Pro ·
**(cheap)** ≲1 GPU-hour · **(£££)** needs real training budget ·
**Done when** = the acceptance criterion; an item without one is a reading
task.

## Contents

- [0. Where things stand](#0-where-things-stand)
- [1. Critical path](#1-critical-path--what-to-do-next-and-why)
- [2. Bugs found, not yet fixed](#2-bugs-found-not-yet-fixed)
- [3. Finish the klein benchmark](#3-finish-the-klein-benchmark)
- [4. Quantization](#4-quantization)
- [5. Step distillation](#5-step-distillation)
- [6. Classifier-free guidance and CFG distillation](#6-classifier-free-guidance-and-cfg-distillation)
- [7. Preference optimization and RL](#7-preference-optimization-and-rl)
- [8. Training-free inference acceleration](#8-training-free-inference-acceleration)
- [9. Structural compression](#9-structural-compression)
- [10. Adaptation beyond PEFT; adapter composition](#10-adaptation-beyond-peft-adapter-composition)
- [11. Rewards, data, safety](#11-rewards-data-safety)
- [12. Systems, profiling, serving](#12-systems-profiling-serving)
- [13. InternVL2 + SAM cascade](#13-internvl2--sam-cascade--anchorseg-internvl2)
- [14. AnchorSeg replication](#14-anchorseg-replication--anchorseg-internvl2)
- [15. Direction: a vision-models playground](#15-direction-a-vision-models-playground)
- [16. Paper survey (2026-09-19)](#16-paper-survey--candidates-to-try-2026-09-19)
- [17. Housekeeping and runbook](#17-housekeeping-and-runbook)
- [A. Concepts primer](#a-concepts-primer)
- [B. Metrics reference](#b-metrics-reference)
- [C. Code map](#c-code-map)
- [D. Bibliography](#d-bibliography)

---

## 0. Where things stand

### Done

- [x] **FLUX.2 [klein] training path runs end to end** against real weights:
      rectified-flow objective, VAE batch-norm latents, 2×2 packed tokens,
      adapter injection, checkpoint round-trip for all six PEFT methods
- [x] **Six adapters trained and evaluated** on SynCD subject-5, on two
      splits — [`flux2-klein-peft/RESULTS.md`](flux2-klein-peft/RESULTS.md)
- [x] **Eval split fixed.** SynCD's `prompts` are the prompts that *generated*
      the training images, not held-out contexts. Six recontextualization
      templates now form the held-out split; both are reported
- [x] **Six adapters on the Hub**, `OnePunchMonk101010/dptlab-klein-<method>-subject5`;
      `push_to_hub.py` writes a peft-injection snippet for non-LoRA methods,
      because `load_lora_weights` silently serves the base model for them
- [x] **First real `MODELS.md` rows**: six klein rows, held-out split
- [x] **Training-free InternVL2-2B + SAM cascade on ReasonSeg val:
      19.5 gIoU / 7.4 cIoU**, with the mechanism diagnosed (§13)
- [x] **AnchorSeg environment verified on Modal**, weights and data cached

### Current numbers: klein, subject-5, 167 optimizer steps, 512px

| method | params | in CLIP-T | in DINO | **out CLIP-T** | **out DINO** | out CLIP-I | DINO drop |
|---|---|---|---|---|---|---|---|
| **dora** | 14.5M | 0.9507 | 0.4775 | 0.9572 | **0.4424** | **0.7167** | +0.0351 |
| lora ⚠️ | 13.8M | 0.9648 | 0.5060 | 0.9646 | 0.4320 | 0.7062 | **+0.0740** |
| lokr | 0.9M | 0.9249 | 0.3197 | 0.9502 | 0.2946 | 0.6665 | +0.0251 |
| loha | 13.8M | 0.9636 | 0.3050 | **0.9710** | 0.2907 | 0.6637 | +0.0143 |
| oft | 1.4M | 0.8815 | 0.3142 | 0.9176 | 0.2750 | 0.6484 | +0.0392 |
| boft | 3.0M | 0.8945 | 0.3342 | 0.8727 | 0.2680 | 0.6291 | +0.0662 |

**n = 1 subject, 6 held-out prompts: this validates the pipeline. It is not a
benchmark.** Read it as a hypothesis for §3:

- **The ranking flips between splits.** In-sample LoRA leads DINO; held out,
  DoRA leads, because LoRA's DINO drops twice as far (0.074 vs 0.035). At this
  budget LoRA's in-sample lead was mostly memorization of training scenes.
- **LoHa** has the best held-out prompt following and nearly the worst subject
  fidelity. Its small drop means it barely learned the subject, not that it
  generalized.
- ⚠️ **The LoRA row may be a partially loaded adapter** (bug §2.2).

### Current numbers: ReasonSeg val, 200 images, training-free

| method | gIoU | cIoU |
|---|---|---|
| InternVL2-2B + SAM (ours) | 19.5 | 7.4 |
| Grounded-SAM (LISA Tab. 1) | 26.0 | 14.5 |
| LISA-7B (trained) | ~52 | — |

- [ ] **The cascade code and results are only on the unmerged branch
      `anchorseg-internvl2-replication`** (`4501f70`): `modal/cascade.py`,
      `scripts/diagnose_cascade.py`, `results/`. `main` references
      `results/README.md` but does not contain it. **Merge it.**

### Not measured anywhere yet

Latency breakdowns, throughput, memory, any quantized model, any result from
`dpo.py` on a real preference set, anything from `grpo.py` / `distill.py`, any
video model. The repo has more harness than measurement. Most items below are
ranked by how quickly they turn harness into a number.

---

## 1. Critical path — what to do next, and why

Ordered so each step makes the next one's result trustworthy.

| # | Step | Why this position | Cost |
|---|---|---|---|
| 1 | Fix the step-count bug (§2.1) and the LoRA checkpoint question (§2.2) | Every later number depends on both | **(free)** + tests |
| 2 | Merge the cascade branch (§0) | Results referenced from `main` should exist on `main` | **(free)** |
| 3 | BOFT with `CUDA_HOME` set, re-run one cell (§2.3) | Otherwise BOFT is benchmarked at a setting nobody chose | **(cheap)** |
| 4 | 20-subject sweep at a true 500 steps (§3) | Turns a smoke test into a benchmark | ~40 A100-h, see §3 |
| 5 | Profile a klein forward pass (§12) | The baseline every efficiency claim needs | **(cheap)** |
| 6 | Step-count × fidelity curve on existing adapters (§5) | Inference-only, first efficiency-vs-quality curve | **(cheap)** |
| 7 | NVFP4 weight-only on merged exports (§4) | First quantization number; answers "does 4-bit hold the subject" | **(cheap)** |
| 8 | Bonsai locally (§4.3) | Quality-vs-size curve at zero spend | **(free)** |
| 9 | LFM2.5-VL grounding stage (§13) | Separates "cascades can't reason" from "this VLM won't localize" | **(free)** |
| 10 | Fix `grpo.py` as Flow-GRPO (§7) | The most-cited technique the repo has broken | **(cheap)** to validate |

---

## 2. Bugs found, not yet fixed

Each needs a regression test that fails before the fix.

### 2.1 `max_train_steps` delivers the wrong number of optimizer steps

- [ ] **Fix, in all four recipes** (`lora.py:90`, `dpo.py:79`, `grpo.py:79`, `distill.py:81`)

**Symptom.** Configs ask for 500 steps; the klein sweep delivered 167.

**Root cause.** The epoch count ignores gradient accumulation:

```python
max_epochs = math.ceil(config.max_train_steps / max(1, len(dataloader)))
```

`global_step` only advances when `accelerator.sync_gradients` is true. With 3
images, batch 1 and `gradient_accumulation_steps: 4`, `len(dataloader) == 3`
and `max_epochs == ceil(500 / 3) == 167`. accelerate also forces a sync at the
end of every dataloader pass, so each epoch is exactly **one** optimizer step
built from **3** micro-batches, not 4. Two silent errors: 167 steps instead
of 500, and an effective batch of 3 instead of 4.

**Fix.** Count optimizer steps per epoch, not micro-batches:

```python
updates_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
max_epochs = math.ceil(config.max_train_steps / updates_per_epoch)
```

Better, loop on `while global_step < max_train_steps` over a cycling iterator,
which also removes the short-final-accumulation problem on tiny datasets.
Factor it into `training/common.py` so the four copies can't drift again.

**Done when.** A CPU test with a toy module, a 3-item dataset, accumulation 4
and `max_train_steps=10` counts exactly 10 `optimizer.step()` calls, and the
run manifest records `final_step == max_train_steps`.

### 2.2 The LoRA checkpoint may be incomplete

- [ ] **Investigate before quoting any LoRA number**

Loading `dptlab-klein-lora-subject5` warns that the PEFT config names target
modules missing from the state dict: the attention projections of
`transformer_blocks.0`–`.4` (the dual-stream blocks). LoRA is the only method
on the diffusers-native `save_lora_weights` path, so the likely culprit is a
key-name conversion that drops the dual-stream keys on save or load.

**Done when.** A test saves, reloads and compares every `lora_A`/`lora_B`
tensor by name for a tiny FLUX.2-shaped model; and the reloaded Hub
checkpoint's parameter count equals the 13,813,760 trained. If keys were lost,
re-evaluate LoRA and update `MODELS.md` and `RESULTS.md`.

### 2.3 BOFT silently runs at `boft_n_butterfly_factor=1`

- [ ] **Fix and fail loudly**

peft's BOFT needs a CUDA extension compiled at first use. Without `CUDA_HOME`
(ninja alone isn't enough) it logs a warning and falls back to factor 1, which
is a single block-diagonal factor, i.e. plain OFT at a different block size.
BOFT was benchmarked at a setting nobody chose.

**Fix.** Set `CUDA_HOME` in the Modal image (use a `devel` CUDA base, not
`runtime`). After adapter construction, assert the realized
`n_butterfly_factor` equals the config and raise if not. Record the realized
config in `run_manifest.json`.

### 2.4 OFT/BOFT adapt fewer modules than the LoRA family

- [ ] **Report adaptation surface beside every score, or find a factorization**

FLUX.2's single-stream blocks fuse q, k, v and the MLP input into
`to_qkv_mlp_proj`, whose output dim doesn't divide into orthogonal blocks
cleanly, and `proj_out` isn't square. `ModelSpec.orthogonal_target_modules`
therefore excludes them. "OFT lost" partly means "OFT with a smaller surface
lost".

Options: (a) add a *surface-matched* LoRA cell restricted to
`orthogonal_target_modules`, which isolates the method from the surface; (b)
pick a block size dividing both dims of the fused projection; (c) report
"modules adapted / total" as a column. (a) is cheapest and most informative.

### 2.5 `distill.py` has three documented correctness bugs

- [ ] **Fix before any distillation work** (§5)

From the `KNOWN-WRONG` comment in the training loop:

1. **Inputs are pure noise at every timestep.** `latents = torch.randn(...)`
   is then fed to the teacher at an arbitrary `t_{n+1}`. Consistency
   distillation needs a *real* latent noised to `t_{n+1}`; as written every
   step trains at the σ_max marginal. Fix: encode real (or teacher-generated)
   images and `add_noise` to `t_{n+1}`.
2. **The loss compares ε-predictions at two different timesteps.** The
   consistency function is over x̂₀. Convert both to x̂₀ (or use the LCM
   `c_skip`/`c_out` parametrization) before the distance.
3. **Hardcoded SD latent geometry** `(4, res//8, res//8)`. Wrong for FLUX
   (16 channels, packed tokens). Route through `objectives.py`.

Also: the teacher step uses the default guidance, so this is not yet
*guided* consistency distillation (LCM distils the CFG-combined teacher with a
sampled `w`). See §5 and §6.

### 2.6 `grpo.py` is not a policy gradient

- [ ] **Fix as Flow-GRPO** (§7.2 has the derivation)

Group sampling and advantage normalization are right; the update is not:

1. **No log-prob ratio, no clipping.** The loss is `advantage × denoising
   MSE`. For negative advantage, minimizing it *maximizes* reconstruction
   error, which is unbounded below.
2. **The "KL" is the same MSE again**, so it only rescales the objective by
   `1 + kl_coeff`. A real KL needs a frozen reference policy.
3. Minor: sampling uses `num_inference_steps=25` for every model, including
   4-step klein.

---

## 3. Finish the klein benchmark

The sweep answers: *at a matched budget, which PEFT method best trades subject
fidelity against prompt fidelity on FLUX.2 [klein] 4B?* Background on the
methods is in §A.2.

- [x] Publish the six adapters to the Hub
- [x] First real `MODELS.md` rows
- [ ] **Scale to ~20 subjects** from `data/syncd-20/`
- [ ] **Re-run at a true 500 steps** once §2.1 lands
- [ ] **Two seeds on a subset** (e.g. 5 subjects × 6 methods × seed 43) to
      measure seed variance against subject variance
- [ ] **Surface-matched LoRA cell** (§2.4)
- [ ] **Parameter-matched sweep.** Params differ 16× (LoKr 0.9M, DoRA
      14.5M). Add a second budget point per method, e.g. ~1M and ~14M
      trainable, so "method" and "capacity" can be separated
- [ ] **Textual Inversion cell** as a zero-weight-change floor (§10)
- [ ] **DreamBench++ judge** on the final 20-subject run only (opt-in, costs
      API money)

### Protocol for the 20-subject run

- **Fixed:** base, resolution 512, lr 1e-4, batch 1 × accum 4, 500 optimizer
  steps, logit-normal σ with 4-step shift, masks on, seed 42, 4-step cfg 1.0
  sampling at eval.
- **Eval:** held-out split (6 templates) plus in-sample, 2 images per prompt.
- **Report per method:** mean over subjects of held-out DINO / CLIP-I /
  CLIP-T, with **95% bootstrap CIs over subjects**; the in→out DINO drop; and
  **paired per-subject differences against LoRA** (sign test or Wilcoxon).
  With 6 methods and 20 subjects, differences under ~0.01 DINO are unlikely to
  survive. Say so rather than ranking them.
- **Plot:** DINO vs CLIP-T, one point per method, CI whiskers on both axes.
  The Pareto front is the result. A single ranked column is not.
- **Also record:** trainable params, checkpoint MB, train wall time, peak
  memory, eval latency, and whether the method can be served un-merged (§12).

### Cost

Measured: ~7 min per cell at 167 steps on an A100-40GB, cells in parallel.
20 subjects × 6 methods at 167 steps ≈ **14 A100-h**; at a true 500 steps
roughly **3× the training share**, call it 30–40 A100-h. Price it from the
current Modal rate. Always `smoke_all` first.

**Done when.** `RESULTS.md` has a 20-subject table with CIs and the Pareto
plot. `MODELS.md` rows are regenerated with `n=20`. The §2 bugs are closed or
their effect is quantified in the table.

---

## 4. Quantization

The interesting question here is not "can klein be quantized"; Prism ML has
already taken it to 1.58 bits. It is **how quantization and adapters
interact**, which is where the two threads of this repo meet. Background on
formats is in §A.4.

### 4.1 Run what exists — `flux2-klein-peft/scripts/quantize_nvfp4.py`

- [ ] **Run NVFP4 weight-only on a merged export**, then A/B DINO and CLIP-T
      against bf16: *does a 4-bit subject adapter still hold the subject?*
      Pipeline: `merge_and_export.py` → `quantize_nvfp4.py --eval-subject`
- [ ] **Same for FP8 and INT8**, the baselines NVFP4 must beat
- [ ] Note the hardware: NVFP4 compute needs Blackwell (sm_100). Before that
      you get the ~3.5× memory saving and software dequantization with no
      speedup. The script reports which one you're getting. Quote it
- [ ] **Activation calibration.** It currently raises `NotImplementedError`
      on purpose. modelopt hands the loop a bare transformer, but real
      calibration has to drive the whole pipeline (text encoder → sampler →
      transformer) over real prompts, at the timesteps the 4-step sampler
      visits. Synthetic tensors produce scales that look calibrated and aren't

**Done when.** A table with rows bf16 / FP8 / INT8 / NVFP4-W and columns
transformer GB, latency (with GPU named), held-out DINO, CLIP-T and Δ vs bf16,
for at least DoRA and one orthogonal method.

### 4.2 Quantization × adapters: the research question

- [ ] **Order matters, and the repo asserts it without evidence.**
      `quantize_nvfp4.py` says "quantize after merging, never before", because
      an adapter is a delta against the weights it was fitted to. Test it:
      (a) merge → quantize vs (b) quantize base → apply FP16 adapter vs (c)
      train the adapter against the quantized base (QLoRA-style). Same
      subject, same metrics
- [ ] **Which PEFT method survives 4-bit best?** Quantize all six merged
      exports and re-score. Orthogonal methods preserve singular values of W₀
      and low-rank methods change them, so their quantization error could
      behave quite differently. A cheap, novel-looking result
- [ ] **QLoRA / QA-LoRA / LoftQ ported to diffusion.** In LLM work, the fix
      for "FP16 adapter on a quantized base" is to train against the
      quantized base (QLoRA), make the adapter's merge quantization-aware
      (QA-LoRA), or initialise the adapter to absorb quantization error
      (LoftQ). No one has done this systematically for diffusion PEFT
- [ ] **SVDQuant** is literally quantization plus a low-rank branch, and
      claims off-the-shelf LoRAs fuse in without re-quantization. Read it
      before designing any of the above (§16 Tier 1)

### 4.3 Prism ML Bonsai: extreme quantization of the same model

[Bonsai Image 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit)
comes in ternary (1.58-bit) and binary (1-bit) versions of FLUX.2 Klein 4B,
licensed Apache 2.0. The transformer is 1.21 GB (ternary) and 0.93 GB (binary),
against 7.75 GB at FP16. Its GenEval at 1.58-bit is 0.723. The MLX build runs
on Apple Silicon.

- [ ] **Run locally**, generate the repo's eval prompt set **(free)**
- [ ] **Quality-vs-size curve**: FP16 → FP8 → NVFP4 → ternary → binary on
      CLIP-T / aesthetic / latency. The plot with a cost axis the repo lacks
- [ ] **Does an FP16-trained adapter survive a 1.58-bit base?** This is
      ordering (b) in §4.2, taken to the extreme. Either answer is worth having
- [ ] **Diff their FP16-retained modules** (modulation, embedders, norms,
      output projections) against our `modules_to_not_convert`
      (`proj_out`, `time_text_embed`, `x_embedder`, `context_embedder`)

### 4.4 Methods to know (reference list; details in §D)

| Family | Methods | One-line idea |
|---|---|---|
| Weight-only PTQ | GPTQ, AWQ | Second-order / activation-aware rounding of weights with a small calibration set |
| W+A PTQ | SmoothQuant | Migrate activation outliers into weights with a per-channel scale |
| Diffusion PTQ | Q-Diffusion, PTQ4DM, TFMQ-DM | **Per-timestep calibration**: activation ranges shift across the trajectory, so one calibration set is wrong for most of it |
| Low-rank + quant | SVDQuant, LoRaQ | Keep a high-precision low-rank branch to absorb outliers; quantize the residual |
| Quantized adapters | QLoRA, QA-LoRA, LoftQ, LoraQuant | Adapter trained on / aware of / initialised from a quantized base; or quantize the adapter itself |
| 4-bit training | FourTune | W4A4G4: 4-bit weights, activations and gradients in LoRA training |
| Data-free | OrbitQuant | No calibration data, which sidesteps the per-timestep problem |

- [ ] **Sensitivity analysis**: quantize one block type at a time and measure
      Δ DINO / Δ CLIP-T. Explains *why* the retained-module lists look the way
      they do **(cheap-ish)**
- [ ] **Quantization × step count**: a 4-step model has fewer steps to average
      error away. Does it tolerate 4-bit worse than a 30-step one?
- [ ] **Quantization × caching** (§8): do the savings compose or collide?

---

## 5. Step distillation

**Why it matters.** Latency scales roughly linearly with sampler steps, so
step count is the largest single serving-cost lever. klein is an unusually
good testbed because it is **already** step- and guidance-distilled (4 steps,
cfg 1.0). Every experiment here asks "can a distilled model be distilled
further, and what breaks first?". Theory is in §A.3.

### 5.1 Taxonomy

| Method | Matches | Idea | Why it's on the list |
|---|---|---|---|
| Progressive Distillation | trajectory | Student does 2 teacher steps in 1; halve repeatedly | The origin; baseline |
| Consistency Models / **LCM** | trajectory | Map any point on the PF-ODE to its endpoint; self-consistency along the path | What `distill.py` attempts |
| LCM-LoRA | trajectory | LCM trained as a LoRA; a portable "acceleration adapter" | Distillation *as* PEFT: fits the sweep |
| CTM | trajectory | Learn jumps between any two times *s < t* | Fixes LCM's 1–2-step quality ceiling |
| TCD | trajectory | Trajectory consistency + tunable stochasticity at sampling | Practical, widely used with SDXL |
| ReFlow / InstaFlow | trajectory | Re-train on (noise, teacher-output) pairs to straighten paths, then distil | klein is rectified flow already |
| PeRFlow | trajectory | Piecewise reflow per time window | Cheaper reflow, plug-in |
| ADD / LADD | distribution | Distillation + adversarial loss (LADD: discriminator in latent space on teacher features) | Where regression plateaus; SD3-Turbo, FLUX-schnell lineage |
| DMD / DMD2 | distribution | Minimise KL(student ‖ data) via the difference of two score functions (real vs fake) | Current strong baseline |
| SiD | distribution | Score identity distillation, data-free | No teacher samples needed |
| Shortcut / MeanFlow | trajectory | Condition on step size / learn average velocity; one model, any budget | Removes "one student per step count" |
| SANA-Sprint | trajectory | Continuous-time consistency, one step | Strongest single 2025 reference |

### 5.2 Experiments that fit this repo

- [ ] **Fix `distill.py`** (§2.5) before adding anything
- [ ] **Step count vs subject fidelity** at 4/3/2/1 steps on the six existing
      adapters. Inference only **(cheap)**. The first efficiency-vs-quality
      curve in the repo
- [ ] **LCM-LoRA as a sweep method**, which makes "which PEFT method distils
      best" a question the harness already answers
- [ ] **Distil a subject adapter and the step count jointly.** Does a
      personalized klein survive 4 → 2 steps?
- [ ] **Stack two adapters**: subject adapter + acceleration adapter. Does
      naive addition work, or do they interfere (§10 merging)?
- [ ] **Instance-aware step schedules** (§16 Tier 2), a training-free
      schedule search **(cheap)**

**Done when.** A curve of steps → (DINO, CLIP-T, latency) for ≥ 2 methods,
plus one working distilled student evaluated on the same held-out split.

---

## 6. Classifier-free guidance and CFG distillation

**Concept.** CFG runs the denoiser twice per step, conditional and
unconditional, and extrapolates:

```
ε̂ = ε(x_t, ∅) + w · (ε(x_t, c) − ε(x_t, ∅))
```

That doubles cost per step. **CFG distillation** trains a student to produce
ε̂ in one pass. **Guidance embedding** (FLUX.1-dev, klein) feeds `w` as a
conditioning input baked in at training time. klein goes further: it is
guidance-distilled to cfg 1.0, so CFG is gone entirely at serving.

- [ ] **Guided Distillation** (Meng et al.): the student takes `w` as input,
      one model for the whole guidance range. Two-stage: CFG → student, then
      progressive step distillation
- [ ] **Read klein's guidance handling** in diffusers' `Flux2KleinPipeline`
      and write down what "guidance-distilled" means concretely there
- [ ] **Does a LoRA re-introduce guidance dependence?** Sweep cfg ∈ {1, 1.5,
      2, 3} on the six adapters. If a cfg-1.0-trained adapter behaves
      differently at cfg > 1, that is measurable and slightly alarming
      **(cheap)**
- [ ] **Guidance interval**: CFG only on middle timesteps (Kynkäänniemi
      et al.) — guidance helps most early/mid and hurts diversity late.
      Training-free, an immediate latency win on SDXL **(cheap)**
- [ ] **CFG++ / rescaled guidance** (Lin et al. φ-rescale): same axis, nearly
      free
- [ ] **Implement on SDXL first**, where CFG still genuinely costs 2×. Then
      combine with LCM (guided consistency distillation) to fix §2.5's missing
      `w`

**Done when.** An SDXL student at 1 forward per step matches the cfg-7 teacher
within a stated CLIP-T / aesthetic tolerance at half the latency, measured.

---

## 7. Preference optimization and RL

The LLM stack is SFT → preference optimization → RL. The repo has one
working preference method (Diffusion-DPO) and one broken RL method (GRPO).

### 7.1 Diffusion-DPO, as implemented (`dpo.py`)

For a preference pair (x_w, x_l) under prompt c, noise both at a shared
(t, ε) and compare the policy's denoising error to a frozen reference's:

```
Δ_w = ‖ε − ε_θ(x_w,t,c)‖² − ‖ε − ε_ref(x_w,t,c)‖²
Δ_l = ‖ε − ε_θ(x_l,t,c)‖² − ‖ε − ε_ref(x_l,t,c)‖²
L   = −log σ(−β · (Δ_w − Δ_l))
```

β is huge (the repo uses 5000, paper range 2000–5000) because per-sample MSE
differences are tiny. Pairs come from `build_preference_pairs.py`: sample
twice, rank with CLIP + aesthetic, keep pairs above a margin.

- [ ] **Run DPO on SDXL with a real preference set** (Pick-a-Pic v2 subset)
      and report paired win rate vs the base with `compute_win_rate`. No DPO
      result exists yet
- [ ] **Port DPO to klein**: the loss needs velocity targets through
      `objectives.py` instead of ε

### 7.2 Why GRPO needs log-probabilities, and how Flow-GRPO gets them

A deterministic ODE sampler has no density per step, so there is no π_θ to
form a ratio with. The fix (DDPO for DDPM, Flow-GRPO for rectified flow) is to
sample with an **SDE** that has the same marginals as the ODE. Then each step
is Gaussian:

```
x_{t−Δ} ~ N( μ_θ(x_t, t, c),  σ_t² Δ I )
log π_θ(x_{t−Δ} | x_t) = −‖x_{t−Δ} − μ_θ(x_t,t,c)‖² / (2 σ_t² Δ) + const
```

For rectified flow, μ_θ is the Euler step plus a score-correction term. The
score is recoverable from the velocity, since x̂₀ = x_t − t·v and
ε̂ = x_t + (1−t)·v. Flow-GRPO gives the exact drift; σ_t is a noise-level
hyperparameter.

GRPO on top of this:

```
A_i  = (r_i − mean(r)) / std(r)            # per prompt, group of G
ρ_i,k = π_θ(x_{k−1}|x_k) / π_old(x_{k−1}|x_k)   # per trajectory step k
L    = −E[ min(ρ A, clip(ρ, 1±ε) A) ] + β · KL(π_θ ‖ π_ref)
```

With equal per-step variances, the per-step KL is closed form:
‖μ_θ − μ_ref‖² / (2σ_t²Δ). That is what `kl_coeff` should multiply.

- [ ] **Implement it**: store trajectories and per-step log-probs at sampling
      time; recompute with grad at update time; clip; KL against a frozen
      reference (or LoRA-disabled forward, which is free)
- [ ] **Sanity checks before any claim**: reward rises on the training
      prompts; ratio stays near 1 in the first inner epoch; KL grows smoothly;
      held-out prompts don't regress. Check for **reward hacking**: aesthetic
      scorers love saturated, over-smoothed images, so look at them
- [ ] Flow-GRPO's efficiency tricks: train on a subset of timesteps, fewer
      denoising steps at train time than at eval

### 7.3 Methods not yet covered

| Method | Signal | Mechanism | Note |
|---|---|---|---|
| **SPO** | step-level preference | Preference judged at each denoising step, not only the final image | The obvious objection to Diffusion-DPO |
| **Diffusion-KTO** | pointwise good/bad | Kahneman–Tversky utility, no pairs | `build_preference_pairs.py` discards unpaired samples KTO could use |
| **SPIN-Diffusion** | self-play | Model's own samples as losers vs data | No labels |
| **ReFL / DRaFT / AlignProp** | differentiable reward | Backprop the reward through (part of) the sampling chain | Different from DPO and RL; stresses memory (gradient checkpointing, truncated backprop) |
| **DDPO** | scalar reward | Sampling as an MDP, policy gradient with per-step log-probs | Foundation of §7.2 |
| **DPOK** | scalar reward | Policy gradient + KL regularization to the pretrained model | |
| **Flow-GRPO** | scalar reward | GRPO with the ODE→SDE conversion | The fix for `grpo.py` |
| **DanceGRPO** | scalar reward | GRPO unified across image and video, several reward models | |

- [ ] **(cheap)** candidates: KTO on the unpaired samples we already
      generate; DRaFT-K (backprop through only the last K steps) on klein,
      which only has 4 steps anyway

---

## 8. Training-free inference acceleration

No training, and a latency number immediately: the thing the repo most lacks.
Profile first (§12) so the baseline exists.

- [ ] **DeepCache / FORA / TeaCache**: reuse block outputs across adjacent
      steps. DeepCache caches high-level UNet features; FORA caches DiT
      attention/MLP outputs; TeaCache decides *when* to reuse from the change
      in timestep-modulated inputs. With only 4 steps on klein, test on SDXL
      too, where there is redundancy to exploit **(cheap)**
- [ ] **Token merging (ToMe for SD)**: merge redundant tokens before
      attention, unmerge after **(cheap)**
- [ ] **Attention slicing, SDPA backends, sparse/windowed attention**:
      memory vs speed, measured **(cheap)**
- [ ] **`torch.compile`** of the transformer, with and without CUDA graphs.
      The boring baseline most "speedups" should be compared against
      **(cheap)**

**Done when.** A table of method → latency (GPU named, batch 1 and 4, 512 and
1024) → Δ held-out metrics, against an un-accelerated compiled baseline.

---

## 9. Structural compression

- [ ] **Depth pruning of DiT blocks**: drop transformer blocks (e.g. by
      block-influence score), then distil to recover. TinyFusion learns which
      to drop
- [ ] **Width / channel / head pruning**
- [ ] **Architecture distillation into a smaller student** (BK-SDM-style
      block removal + feature distillation), as distinct from fewer steps

(£££) for anything that needs recovery training at klein scale; the
block-influence measurement alone is **(cheap)**.

---

## 10. Adaptation beyond PEFT; adapter composition

- [ ] **Textual Inversion**: learn a token embedding, touch no weights. The
      floor of the personalization axis and a fair comparison point the sweep
      lacks **(cheap)**. Note klein's text encoder is Qwen3, not CLIP
- [ ] **Full DreamBooth fine-tune** (with prior-preservation loss): the
      ceiling of the same axis **(£££)**
- [ ] **ControlNet / T2I-Adapter / IP-Adapter**: *conditioning* (structure,
      image prompt) rather than concept adaptation. `ModelSpec` has no way to
      express an extra conditioning branch yet; that is the design work
- [ ] **REPA** (representation alignment): align DiT hidden states to a
      frozen DINOv2 during training. A training accelerator that composes with
      everything, and DINO is already a dependency
- [ ] **Adapter merging**: task arithmetic, TIES (trim, elect sign, merge),
      DARE (drop and rescale). Natural follow-on: merge two subjects' adapters,
      or subject + LCM-LoRA (§5.2). Measure interference with both subjects'
      DINO

---

## 11. Rewards, data, safety

- [ ] **Train a reward model** rather than reusing CLIP + aesthetic.
      `grpo.py` and `build_preference_pairs.py` both use eval metrics as
      rewards, which is convenient and circular: optimize the metric, then
      report the metric. At minimum, **evaluate with a held-out scorer**
      (ImageReward, PickScore or HPSv2) that training never saw
- [ ] **Synthetic preference data failure modes**: margin threshold vs pair
      quality; scorer bias (saturation, centred subjects); diversity collapse
- [ ] **Concept erasure** (ESD, UCE) and **unlearning** for diffusion:
      post-training that removes a capability. It needs its own eval: erased
      concept gone, neighbouring concepts intact, robust to paraphrase and
      adversarial prompts

---

## 12. Systems, profiling, serving

Nothing in the repo shows where time goes in a forward pass.

- [ ] **Profile a klein forward pass** at 512 and 1024 with `torch.profiler`
      (+ Nsight Systems for a timeline): text encoder vs transformer vs VAE
      decode; within the transformer, attention vs MLP vs modulation/norm.
      Classify each as compute- or memory-bound with a roofline estimate
      (FLOPs / bytes moved vs GPU peak). 1024 has 4× the tokens, so attention
      grows ~16× and MLP ~4×; check where the crossover sits **(cheap)**
- [ ] **Re-profile after quantization and caching**; publish the delta
- [ ] **One fused Triton kernel** for whatever the profile names (candidates:
      modulation + norm fusion, RoPE application, dequant-matmul), benchmarked
      honestly, including the shapes where it loses to cuBLAS/FlashAttention
- [ ] **Run the serving specs** in `serve/backends.py` (SGLang Diffusion,
      vLLM-Omni). Never executed. Measure throughput (images/s) and p50/p95
      latency vs batch size
- [ ] **Serving asymmetry as a result.** No inference server loads non-LoRA
      PEFT adapters, so five of six methods must be served merged: one full
      model copy per subject, versus a few MB of hot-swappable delta for LoRA.
      Put "servable un-merged?" and "per-subject serving cost" in the §3 table
- [ ] **Multi-GPU training**: branch `multi-gpu-training` exists but is behind
      `main` and predates klein. Rebase or retire it
- [ ] **No video model anywhere** (see §15)

---

## 13. InternVL2 + SAM cascade — `anchorseg-internvl2/`

**Setup.** InternVL2-2B is asked for a box, and SAM segments within that box.
There is no training. ReasonSeg val has 200 images with short (phrase) and
long (sentence) queries.

**Hypothesis.** LISA argues that cascades fail because the grounding stage
can only match text, not reason, so a reasoning VLM should help, most of all
on long queries. **It did neither.** Long queries scored 16.0 gIoU and short
queries 24.0.

**Mechanism.** 91/200 boxes are exactly `[0,0,1000,1000]`: the model declines
to localize and returns the whole frame. It does this on 54% of long queries
against 34% of short ones. The median predicted box covers 84% of the image,
while the median target covers 6.4%. The failure is upstream of SAM. It also
explains why cIoU (7.4) collapses further than gIoU (19.5): whole-frame masks
inflate the pixel-weighted union (§B).

- [x] Cascade run and diagnosed (branch `anchorseg-internvl2-replication`)
- [ ] **Merge that branch** (§0)
- [ ] **Rerun with dynamic tiling on and `max_new_tokens=256`.** Tiling was
      disabled, and replies echo the query before the box, reaching 244 chars
      against a 64-token budget. Both confounds bite hardest on long queries,
      so the long/short gap is **not yet established**. ~$1
- [ ] **Swap the grounding stage for [LFM2.5-VL](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)**
      (Liquid AI). The strongest single follow-up, for four reasons that line up:
  - **It actually grounds.** Box prediction is new in 2.5, and it scores
    **81.28 on RefCOCO-M**. That separates "cascades can't reason" from
    "InternVL2-2B won't localize"
  - **Native tiling**: 512×512 patches plus a global thumbnail. That removes
    our biggest confound
  - **Output is normalized [0,1] JSON** `[{"label":…, "bbox":[x1,y1,x2,y2]}]`.
    `parse_box` already handles that scale (`SCALE_UNIT`); the JSON-array
    format needs one branch
  - **MLX builds (4/5/6/8-bit, bf16)**, so it runs locally **(free)**. Sizes
    450M / 1.6B / 3B give a scaling curve for free
  - Licence is **LFM1.0**, not Apache. Read it before publishing on top
- [ ] **InternVL2-8B**, to separate "2B is too small" from "cascades can't do
      this"
- [ ] **Report the whole-frame rate as a first-class metric** for every
      grounding model tried. It is the mechanism, so it should be in the table
- [ ] Note: Liquid AI has no image diffusion models. Their "Diffusion" entry
      is a masked **text** diffusion encoder (`LFM2.5-Encoder-350M-Diffusion`)

**Done when.** A table with rows = grounding model × tiling on/off and columns
= gIoU, cIoU, short/long split, whole-frame rate.

---

## 14. AnchorSeg replication — `anchorseg-internvl2/`

AnchorSeg (ACL 2026) is a reasoning-segmentation model in the LISA line: an
LLM emits a special token whose hidden state prompts a SAM-style mask decoder.
AnchorSeg's change is an *anchor* token tied to the spatial grid. Upstream
trains on one GPU with LoRA r=8 on LLaVA-1.5-7B. See
`anchorseg-internvl2/docs/learning-doc.md`.

- [ ] **Patch upstream to dump per-sample masks.** Their eval prints
      aggregates only, and independent scoring needs masks. This is the one
      required upstream modification and the only step-1 blocker
- [ ] **Step 1**: reproduce gIoU 67.20 / cIoU 75.15 from released weights
      (~1 GPU-h). **Done when** our scorer on their masks is within ±0.5 of
      the paper
- [ ] **Step 2**: reproduce training on LLaVA-1.5-7B (~40–70 GPU-h, ~$100).
      A partial run (20 of 120 epochs, ~$20) is a weaker but real control
- [ ] **Step 3**: port to an InternVL2 backbone. This is the actual
      contribution. The hard part is that the anchor assumes LLaVA's fixed
      336px / 24×24 grid while InternVL2 tiles dynamically, so the anchor
      needs per-tile positions or a thumbnail-only anchor
- [ ] **Step 4**: full-FT vs LoRA ablation, which the paper never ran. It
      needs a language-ability regression check (e.g. a VQA subset)

---

## 15. Direction: a vision-models playground

The pieces here (SAM, InternVL2, DINO as a metric) point at a wider scope than
diffusion post-training: **DINO, SAM, Depth Anything, VLMs, image and video
generation** behind one shared eval harness.

- [ ] **Decide here vs a new repo.** `dptlab`'s stated scope is diffusion
      post-training, and it is already straining. Recommendation: new repo,
      importing `dptlab.eval` as a dependency
- [ ] **Shared adapter/eval protocol** across families, so a new model is a
      `ModelSpec` rather than a fork. Generalise `T2IAdapter` to task-typed
      adapters (generate, segment, depth, caption)
- [ ] **Depth Anything (V2)**: absent. Cheap entry point: zero-shot depth +
      use depth as a ControlNet condition for klein (§10)
- [ ] **DINOv2/v3 as a backbone**, not only a metric (and as REPA's target,
      §10)
- [ ] **A video model**: Wan 2.x or LTX-Video at the small end; DanceGRPO and
      OrbitQuant already cover video

---

## 16. Paper survey — candidates to try (2026-09-19)

Ranked by *fit to what this repo already has*, not by recency. A paper earns a
place by being testable at klein scale with released code, or by answering a
question already written into the repo. Entries from 2026 are as surveyed on
2026-09-19 and have not been re-verified since.

### Tier 1 — directly answers an open question here

- [ ] **[FourTune](https://hanlab.mit.edu/projects/fourtune)**, *Towards Fully
      4-Bit Efficient Post-Training for Diffusion Models*
      ([arXiv 2607.05711](https://arxiv.org/abs/2607.05711); the SVDQuant
      group, MIT/CMU/Stanford/Berkeley). End-to-end **W4A4G4**. LoRA is
      augmented with a **frozen numerical stabilizer** for outliers, plus
      block-wise quantization and fused kernels for quantized backprop. It
      reports **2.25× less memory and 2.27× throughput vs BF16 LoRA** on
      FLUX.1-dev 12B at full-precision quality. **Why first:** every other
      quantization entry targets *inference*, but this one makes *training*
      cheaper, which is the binding constraint here (§3 cost). It is validated
      on customization, RL and distillation, our three axes. Code release is
      unconfirmed, so check before scoping. **(cheap, if the code is out)**
- [ ] **[SVDQuant](https://arxiv.org/abs/2411.05007)** (ICLR 2025 Spotlight).
      It absorbs activation outliers into a high-precision low-rank branch and
      quantizes the rest to 4 bits: 3.5× memory, 3.0× speed on FLUX.1 12B on a
      16 GB 4090. It claims **off-the-shelf LoRAs fuse in without
      re-quantization**, which directly answers §4.2. Our six adapters are
      exactly its input. Engine: [Nunchaku](https://github.com/mit-han-lab/nunchaku).
      **(cheap)**
- [ ] **[LoRaQ](https://arxiv.org/html/2604.18117v1)**: an optimized low-rank
      approximation for 4-bit. Read it against SVDQuant; the difference is the
      point
- [ ] **[LoraQuant](https://arxiv.org/html/2510.26690v1)**: mixed-precision
      quantization *of the LoRA itself*. It complements SVDQuant, and together
      they bracket the design space
- [ ] **[GPTQ-intrinsic LoRA](https://arxiv.org/pdf/2606.01412)**: joint
      low-precision quantization with low-rank adaptation
- [ ] **[OrbitQuant](https://arxiv.org/pdf/2607.02461)**: data-agnostic
      quantization for image and video DiTs. It sidesteps the per-timestep
      calibration that `quantize_nvfp4.py` refuses to fake

### Tier 2 — step distillation, testable at klein scale

- [ ] **[SANA-Sprint](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf)**
      (ICCV 2025): one step via continuous-time consistency, SOTA GenEval at
      0.1 s vs 1.1 s on H100. The reference point
- [ ] **[pi-Flow](https://arxiv.org/pdf/2510.14974)**: policy-based few-step
      generation via imitation distillation
- [ ] **[One-Step Flow](https://arxiv.org/pdf/2412.09465)** (ICLR 2026):
      noise-augmented conditional rectified flow to widen teacher support
- [ ] **[Self-Corrected Flow Distillation](https://www.researchgate.net/publication/390709870_Self-Corrected_Flow_Distillation_for_Consistent_One-Step_and_Few-Step_Image_Generation)**:
      consistency across one- and few-step regimes
- [ ] **[Instance-Aware Discretizations](https://arxiv.org/pdf/2603.17671)**:
      per-instance step schedules, **training-free**. The cheapest real
      efficiency win **(cheap)**
- [ ] **[A Decomposable Probe for Few-Step Diffusion Models](https://arxiv.org/pdf/2607.03256)**:
      prompt / latent / score selectivity across distillation paradigms. An
      analysis toolkit, the kind of thing that makes a benchmark more than a
      leaderboard **(cheap)**
- [ ] **SiD**: data-free. Code and checkpoints were not released at survey
      time

### Tier 3 — flow matching itself

klein is rectified flow, and `objectives.py` hand-implements the interpolant
and the logit-normal + shifted timestep sampling (`_sample_sigmas`). These
would justify or change those choices.

- [ ] **[Shortcut models](https://arxiv.org/abs/2410.12557)** (ICLR 2025):
      condition on step size; one model, any budget
- [ ] **MeanFlow** (NeurIPS 2025), **Improved MeanFlows**, and 2026
      follow-ups (*Overcoming the curvature bottleneck in MeanFlow*, *Terminal
      Velocity Matching*): one step via average velocity
- [ ] **[Isokinetic Flow Matching](https://arxiv.org/pdf/2604.04491)**:
      pathwise straightening, upstream of every distillation method
- [ ] **[Curriculum Sampling](https://arxiv.org/pdf/2603.12517)**: a two-phase
      timestep curriculum. It is a small, testable change to `_sample_sigmas`
      **(cheap)**
- [ ] **[On Variance Reduction in Learning Mean Flows](https://arxiv.org/pdf/2605.09235)**
- [ ] **[Order-Optimal Sample Complexity of Rectified Flows](https://arxiv.org/abs/2601.20250)**:
      theory on how much data the sweep needs
- [ ] **[MIT 6.S184 lecture notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf)**:
      the clean introduction. Read first if §A.1 feels shaky

### Tier 4 — bigger swings

- [ ] **[VoT: Vision-of-Thought](https://www.alphaxiv.org/abs/2609.07815)**
      (Sept 2026): a discrete visual-thinking layer between a VLM and a DiT,
      with a three-branch MoT. GenEval 0.91 vs FLUX.1-dev 0.82. It is built on
      Mogao-14B **(£££)**, but the tokenizer-alignment idea may be testable at
      klein scale
- [ ] **[AnchorSeg](https://arxiv.org/abs/2604.18562)** (ACL 2026): §14

### Reading order if time is short

1. **FourTune**: it makes training cheaper, the binding constraint
2. **SVDQuant**: it answers a question written down twice already
3. **MIT 6.S184** flow-matching chapters: they ground Tier 3
4. **Flow-GRPO**: the fix for `grpo.py`
5. **SANA-Sprint**, then **DMD2**: the distillation reference points
6. **Instance-aware discretizations** + **Curriculum Sampling**: the two
   cheapest things that produce a number

---

## 17. Housekeeping and runbook

- [ ] **Always `modal run --detach`.** A non-detached run dies when the local
      client disconnects. It truncated a 2.4 GB download and killed two runs
- [ ] **`smoke_all` before `sweep`.** It found two harness bugs on first run:
      six pipelines in one container OOM after three methods, and a 10-step
      smoke written to the real output path made the sweep skip LoRA as
      "already trained"
- [ ] `sweep` and `evaluate` are resumable. Rely on it rather than re-running
      finished cells
- [ ] Modal volumes sit inside the free-storage tier, so deleting them saves
      $0
- [ ] Configs point at `data/syncd-20/subject-0`; the published run is
      subject-5. Make the subject a CLI arg of `modal/sweep.py`, not a YAML
      edit
- [ ] Stale branches: `flux2-klein-peft-benchmark` (merged via
      [#8](https://github.com/OnePunchMonk/diffusion-post-training-lab/pull/8)),
      `multi-gpu-training` (§12). Delete or rebase

---

## A. Concepts primer

### A.1 Diffusion vs rectified flow

**DDPM / ε-prediction (SDXL).** Forward process
`x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε`, network predicts ε (or v), loss
`‖ε − ε_θ(x_t,t,c)‖²`. Latents: 4 channels, spatial, VAE scaling factor.
Implemented as `EpsilonObjective`.

**Rectified flow (FLUX.1, FLUX.2, SD3).** Straight-line interpolant with σ ∈
[0,1]:

```
x_σ = (1 − σ) x₀ + σ ε          target velocity  v = ε − x₀
L   = ‖v_θ(x_σ, σ, c) − (ε − x₀)‖²
```

Sampling integrates the ODE `dx/dσ = v_θ` from σ=1 to 0. A perfectly straight
path needs one Euler step, so few-step samplers and distillation are about
straightness (ReFlow, §5). Implemented as `Flux2FlowMatchObjective`.

**Timestep sampling.** Uniform σ wastes budget on noise levels the sampler
barely visits. The repo draws `σ = sigmoid(u)` with `u ~ N(logit_mean,
logit_std)` (logit-normal, SD3), then applies the scheduler's resolution-
dependent **time shift** with μ computed for the *eval* step count
(`sampling_steps_for_shift: 4`). That aligns the training σ distribution with
where the 4-step sampler evaluates.

**FLUX.2 latent specifics.** VAE latents are batch-norm standardized with
statistics stored on the VAE, 2×2-patchified, and packed into a token
sequence with 4D position ids for image and text tokens. klein 4B uses a
single Qwen3 text encoder. There are dual-stream blocks (separate
image/text projections: `to_q/k/v`, `add_*_proj`) and single-stream blocks
(fused `to_qkv_mlp_proj`).

### A.2 The six PEFT methods

W₀ ∈ ℝ^{d_out×d_in} is frozen; only the adapter trains.

| Method | Update | Knob | Property |
|---|---|---|---|
| **LoRA** | W₀ + (α/r)·BA, B∈ℝ^{d_out×r}, A∈ℝ^{r×d_in} | r, α | Rank ≤ r; the only format servers load |
| **DoRA** | m ⊙ (W₀ + BA) / ‖W₀ + BA‖_col | r, α | Separates direction (LoRA) from per-column magnitude m. Closer to full-FT learning dynamics |
| **LoHa** | W₀ + (B₁A₁) ⊙ (B₂A₂) | r | Hadamard product: rank up to r² for 2× LoRA params |
| **LoKr** | W₀ + C ⊗ (BA) | r, factor | Kronecker product: very few params, full-rank-capable. Smallest checkpoints |
| **OFT** | R·W₀ with R block-diagonal orthogonal (Cayley: R = (I+Q)(I−Q)⁻¹, Q skew) | block size | Preserves pairwise angles between neurons ("hyperspherical energy"), so it can't change singular values |
| **BOFT** | R = product of log₂-many butterfly-sparse orthogonal factors | block size, n_butterfly_factor | Dense orthogonal transform from few params. Needs the CUDA extension (§2.3) |

**Practical consequences here.** (1) Only LoRA round-trips through diffusers'
`save/load_lora_weights`. Everything else uses peft state dicts plus
`adapter_config.json` (`peft_methods.py`); pointing diffusers at them silently
loads nothing. (2) OFT/BOFT need `d_out` divisible by the block size, hence
the reduced target list (§2.4). (3) Matched rank ≠ matched parameters ≠
matched capacity (§3).

### A.3 Distillation vocabulary

- **PF-ODE**: the deterministic ODE whose marginals match the diffusion.
  Trajectory-matching methods distil *along* it.
- **Consistency function** f(x_t, t) → x₀, constant along one trajectory.
  Parametrized f = c_skip(t)·x_t + c_out(t)·F_θ(x_t, t) so f(x, ε_min) = x.
  Loss: `d(f_θ(x_{t_{n+1}}), f_{θ⁻}(x̂_{t_n}))`, where x̂_{t_n} is one teacher
  ODE step from x_{t_{n+1}} and θ⁻ is an EMA target.
- **Distribution matching** (DMD): gradient of KL(p_student ‖ p_data) =
  difference of a frozen "real" score and an online "fake" score evaluated on
  student samples.
- **Adversarial** (ADD/LADD): add a discriminator loss so the student's
  samples are sharp even where regression averages modes.

### A.4 Numeric formats

| Format | Bits | Scaling | Notes |
|---|---|---|---|
| BF16 | 16 | none | Training default; klein 4B transformer ≈ 7.75 GB |
| FP8 (E4M3/E5M2) | 8 | per-tensor or per-channel | Hopper+ native |
| INT8 | 8 | per-channel weights, per-token activations | SmoothQuant territory |
| MXFP8 / MXFP4 | 8 / 4 | 32-element blocks, E8M0 (power-of-2) scale | OCP microscaling standard |
| **NVFP4** | 4 (E2M1) | 16-element blocks, FP8 E4M3 scale + FP32 per-tensor | Blackwell native; ~4.5 effective bits/weight |
| Ternary | 1.58 | {−1, 0, +1} × scale | Bonsai; BitNet-style |
| Binary | 1 | {−1, +1} × scale | Bonsai |

**W / A / G notation.** W4A16 = 4-bit weights, 16-bit activations (weight-
only: memory win, speed only with a fused dequant kernel). W4A4 = both
(compute win on hardware with 4-bit tensor cores). G4 = 4-bit gradients
(FourTune). **Why diffusion is harder than LLMs:** activation statistics
depend on the timestep, so a calibration set drawn at one σ is wrong at
others.

---

## B. Metrics reference

| Metric | Measures | Computation | Pitfall |
|---|---|---|---|
| **CLIP-T** (CLIPScore) | prompt fidelity | cosine(CLIP image emb, CLIP text emb) | Saturates. Rewards an adapter that learned nothing |
| **DINO** | subject fidelity (primary) | mean cosine of DINO ViT-S/16 CLS embeddings, generated vs reference images | Rewards memorized backgrounds unless eval is held-out (§0) |
| **CLIP-I** | subject fidelity | as DINO with CLIP image embeddings | Less identity-specific than DINO: category-level similarity scores high |
| **DINO drop** | memorization | in-sample DINO − held-out DINO | Small drop can mean "learned nothing" (LoHa) |
| **Aesthetic** | "looks good" | LAION linear head on CLIP ViT-L/14 | Biased to saturated, centred images. Hackable as a reward |
| **Win rate** | paired preference | same prompt + seed, A vs B, scorer or judge picks | The right test for DPO/GRPO "policy beats reference" |
| **DreamBench++ CP / PF** | human-aligned, 0–4 | multimodal LLM judge with the authors' rubrics | Costs money; judge variance, so average over repeats |
| **gIoU** | segmentation | mean over images of per-image IoU | Every image weighs equally |
| **cIoU** | segmentation | Σ intersection / Σ union over the whole set | Pixel-weighted: large masks and whole-frame predictions dominate |
| **GenEval** | compositional T2I | object detector checks counts, colors, positions | Used for Bonsai (0.723) and VoT (0.91) comparisons |
| **FID** | distribution quality | Fréchet distance of Inception features | Needs thousands of samples; meaningless at our n |

**Reading the klein table.** CLIP-T and DINO pull in opposite directions, so
only the pair is meaningful. Report held-out, the drop, CIs over subjects,
and trainable params in the same row.

---

## C. Code map

| Concern | File |
|---|---|
| Config shape for all recipes | `src/dptlab/training/common.py` (`TrainConfig`) |
| Base-model registry | `src/dptlab/models/registry.py` (`ModelSpec`: sdxl, flux-schnell, flux-dev, flux2-klein-4b/9b) |
| Denoising objectives | `src/dptlab/training/objectives.py` (`EpsilonObjective`, `Flux2FlowMatchObjective`, `_sample_sigmas`) |
| PEFT method registry | `src/dptlab/training/peft_methods.py` |
| Recipes | `training/lora.py`, `dpo.py`, `grpo.py` ⚠️, `distill.py` ⚠️ |
| Data | `src/dptlab/data/dataset.py`, `preference.py` |
| Eval harness | `src/dptlab/eval/` (`CheckpointAdapter`, `AestheticScorer`, `compute_win_rate`, `subject_fidelity.py`, `dreambench_judge.py`) |
| Serving | `src/dptlab/serve/backends.py` (SGLang, vLLM-Omni; unrun) |
| Hub + leaderboard | `scripts/push_to_hub.py`, `scripts/update_models_md.py`, `MODELS.md` |
| Preference pairs | `scripts/build_preference_pairs.py` |
| klein sweep | `flux2-klein-peft/` (`configs/`, `modal/sweep.py`, `scripts/prepare_syncd.py`, `autolabel.py`, `merge_and_export.py`, `quantize_nvfp4.py`, `RESULTS.md`, `learning-doc.md`) |
| Reasoning segmentation | `anchorseg-internvl2/` (`src/asvl/`, `scripts/`, `modal/step1_eval.py`, `docs/learning-doc.md`); cascade on branch `anchorseg-internvl2-replication` |

---

## D. Bibliography

arXiv IDs are recorded from memory and were not re-checked in the
2026-09-25 revision; confirm one before citing it. 2026 papers are linked
in §16.

**Foundations.** DDPM, Ho et al. 2020 (2006.11239) · DDIM, Song et al. 2020
(2010.02502) · Score SDE, Song et al. 2021 (2011.13456) · EDM, Karras et al.
2022 (2206.00364) · Flow Matching, Lipman et al. 2022 (2210.02747) ·
Rectified Flow, Liu et al. 2022 (2209.03003) · SD3 / scaling rectified flow,
Esser et al. 2024 (2403.03206) · CFG, Ho & Salimans 2022 (2207.12598)

**PEFT and personalization.** LoRA, Hu et al. 2021 (2106.09685) · DoRA, Liu
et al. 2024 (2402.09353) · FedPara/LoHa (2108.06098) · KronA (2212.10650) ·
LyCORIS (2309.14859) · OFT, Qiu et al. 2023 (2306.07280) · BOFT, Liu et al.
2023 (2311.06243) · DreamBooth (2208.12242) · Textual Inversion (2208.01618) ·
SynCD, Kumari et al. (2502.01720) · DreamBench++ (2406.16855)

**Preference and RL.** Diffusion-DPO, Wallace et al. 2023 (2311.12908) ·
SPO (2406.04314) · Diffusion-KTO (2404.04465) · SPIN-Diffusion (2402.10210) ·
ImageReward/ReFL (2304.05977) · DRaFT (2309.17400) · AlignProp (2310.03739) ·
DDPO, Black et al. 2023 (2305.13301) · DPOK (2305.16381) · GRPO/DeepSeekMath
(2402.03300) · Flow-GRPO (2505.05470) · DanceGRPO (2505.07818)

**Distillation.** Progressive Distillation (2202.00512) · Consistency Models
(2303.01469) · LCM (2310.04378) · LCM-LoRA (2311.05556) · CTM (2310.02279) ·
TCD (2402.19159) · InstaFlow (2309.06380) · PeRFlow (2405.07510) · ADD
(2311.17042) · LADD (2403.12015) · DMD (2311.18828) · DMD2 (2405.14867) · SiD
(2404.04057) · Shortcut models (2410.12557) · MeanFlow (2505.13447) · Guided
distillation, Meng et al. (2210.03142)

**Guidance.** CFG++ (2406.08070) · Noise schedules / guidance rescale, Lin
et al. (2305.08891) · Guidance interval, Kynkäänniemi et al. (2404.07724)

**Acceleration and compression.** DeepCache (2312.00858) · FORA (2407.01425)
· TeaCache (2411.19108) · ToMe (2210.09461), ToMe for SD (2303.17604) · BK-SDM
(2305.15798) · TinyFusion (2412.01199)

**Quantization.** GPTQ (2210.17323) · AWQ (2306.00978) · SmoothQuant
(2211.10438) · Q-Diffusion (2302.04304) · PTQ4DM (2211.15736) · TFMQ-DM
(2311.16503) · QLoRA (2305.14314) · QA-LoRA (2309.14717) · LoftQ (2310.08659)
· SVDQuant (2411.05007)

**Composition, conditioning, safety.** TIES-Merging (2306.01708) · DARE
(2311.03099) · ControlNet (2302.05543) · T2I-Adapter (2302.08453) · IP-Adapter
(2308.06721) · REPA (2410.06940) · ESD (2303.07345) · UCE (2308.14761)

**Vision backbones and segmentation.** DINOv2 (2304.07193) · SAM
(2304.02643) · Depth Anything (2401.10891), V2 (2406.09414) · LISA
(2308.00692) · InternVL 1.5 / InternVL2 (2404.16821) · CLIPScore (2104.08718)
