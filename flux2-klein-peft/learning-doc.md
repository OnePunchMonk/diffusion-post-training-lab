# learning-doc

Notes on the FLUX.2 [klein] PEFT benchmark: what each piece is actually doing,
why it's built the way it is, and the things that cost time to work out. This
is the "why" companion to the README's "how".

---

## 1. The question

The repo's four recipes vary the **objective** (denoising, preference, RL,
distillation) with the adapter fixed at LoRA. This work varies the other axis:
same objective, same data, same budget, six different PEFT adapters.

LoRA is the default everywhere, mostly by inertia — it was first, it's in every
tutorial, and diffusers has native support for it. That's a bad reason to
believe it's the best choice. The methods that came after it make specific,
testable claims:

| Method | Update to `W` | The claim |
|---|---|---|
| LoRA | `W + BA` | Low-rank is enough |
| DoRA | decomposes into direction + magnitude | LoRA conflates the two; separating them helps at low rank |
| LoHa | `(B₁A₁) ⊙ (B₂A₂)` | Higher effective rank per parameter than a single low-rank pair |
| LoKr | Kronecker product | Even more parameter-efficient factorization |
| OFT | `R·W`, `R` block-orthogonal | *Don't* change the singular values — preserve the base model's prior |
| BOFT | `R` butterfly-factorized | Denser orthogonal transform, fewer parameters |

OFT and BOFT are the interesting ones, because they're doing something
categorically different. LoRA-family methods *add* a delta; orthogonal methods
*rotate* the output space. A rotation can't change the magnitude of anything,
which is a strong constraint — the hypothesis is that it prevents the kind of
drift that makes an adapter forget how to follow prompts. Whether that survives
contact with a 4-step distilled model on three training images is exactly the
kind of thing you have to measure.

### Why a benchmark, and not just "try LoRA and ship it"

Two failure modes in personalization point in opposite directions:

- **Underfit**: the subject doesn't stick. Prompt following is fine, subject
  fidelity is bad.
- **Overfit**: the adapter memorizes the training photos. Subject fidelity is
  great, prompt following collapses.

A single score hides both. Any method can win on either axis by failing the
other. That's why the eval has two metrics pointing in opposite directions, and
why the README insists on reporting them together.

---

## 2. Dataset and use case

**Task**: subject-driven personalization. Few photos of one object → a model
that can put that object anywhere.

**Train**: [SynCD](https://huggingface.co/datasets/nupurkmr9/syncd) (ICCV 2025,
MIT). ~90k objects, 2-3 images each, generated with FLUX so each object appears
under different lighting, background and pose.

The multi-view property is the reason to prefer it over the classic 4-6-photos-
from-one-shoot datasets. In a single-shoot dataset, identity and background are
correlated *in the data* — so when an adapter reproduces the background you
can't tell whether that's the adapter's fault or the dataset's. SynCD already
decorrelates them, which means residual background bleed is attributable to the
method under test. That's the whole reason it makes a better benchmark
substrate, not just a newer one.

Each record hands you a train/eval split for free:

- `category_description` → the training caption (same for every image of a
  subject, which is the DreamBooth convention).
- `prompts` → the same object in specific contexts, held out for eval.

**Eval**: three cheap metrics plus an optional expensive one.

| Axis | Metric | Why |
|---|---|---|
| Subject | **DINO** | Self-supervised, no class labels → discriminates *instances*, not categories. Two different backpacks score high on CLIP-I and low on DINO. That gap is the measurement. |
| Subject | CLIP-I | Looser, semantic. Reported because papers quote it and because a large CLIP-I/DINO gap is itself a signal. |
| Prompt | CLIPScore | Did it follow the new context. |
| Both | **DreamBench++** CP/PF | Human-aligned LLM rubrics, 0-4 each. Opt-in — it costs money per generation. |

---

## 3. Things that cost time

### 3.1 FLUX.2 is not SDXL with a different pipeline class

The existing training loop was written for SDXL: DDPM `add_noise`, epsilon
target, 4-channel spatial latents, `added_cond_kwargs`. Every one of those is
wrong for FLUX.2, and — this is the dangerous part — **none of them raise**.
Run the SDXL loop on klein and you get a training run that converges to a loss
that looks fine and a model that learned nothing.

Four separate things had to change (`src/dptlab/training/objectives.py`):

**Rectified flow, not DDPM.** The forward process is a straight line:

```
x_t = (1 - σ)·x₀ + σ·ε        target = ε - x₀
```

σ ∈ [0,1] is passed directly as the timestep (the pipeline passes `t/1000`
where `t = σ·1000`, so the transformer sees σ either way).

**Timestep sampling has to match the sampler.** Sampling σ uniformly wastes the
budget on noise levels a 4-step distilled sampler never visits. So: draw from a
logit-normal (concentrates on the mid-range where velocity is hardest to
predict) and push it through the *same* resolution-dependent shift the pipeline
applies via `compute_empirical_mu(image_seq_len, num_steps)`. Train and sample
on the same distribution or the adapter learns to fix a mismatch nobody asked
about.

**The VAE has no `scaling_factor`.** This one is a genuine trap. Every SD-family
VAE scales latents by a scalar `vae.config.scaling_factor`. `AutoencoderKLFlux2`
does not. It:

1. patchifies the latent into 2×2 blocks (4× the channels, half the H/W),
2. standardizes with **running batch-norm statistics stored on `vae.bn`**,
3. packs the result into a token sequence.

Skip steps 1-2 and the latents sit several σ off-distribution. The model trains.
It just spends the entire run learning to undo your normalization error.

**Position ids are explicit.** `_prepare_latent_ids` and `_prepare_text_ids`
produce 4D `(T, H, W, L)` coordinates for image and text tokens, passed as
`img_ids` / `txt_ids`. There's no implicit spatial structure for the model to
recover — the sequence is flat.

Plus: one Qwen3 text encoder (not CLIP+T5, not CLIP-L+bigG), no pooled
embedding, and `guidance=None` because klein is guidance-distilled.

**FLUX.1 is deliberately left unregistered.** It's also rectified flow, but with
a different text stack and latent packing. Mapping it to the SDXL objective
"so it runs" would produce exactly the silent-failure mode above. Better to
raise `NotImplementedError` than to train something worthless.

### 3.2 Only LoRA has a checkpoint format

`denoiser.add_adapter(config)` accepts any `PeftConfig` — diffusers just calls
`inject_adapter_in_model`. So training all six methods is genuinely a one-line
change.

**Saving them is not.** `pipe.save_lora_weights` / `load_lora_weights` only
understand LoRA keys. Hand the loader a LoHa or OFT state dict and it logs
`"No LoRA keys associated ... found"` — *not an error* — and serves the base
model. The checkpoint looks valid on disk. The eval numbers come back as the
base model's numbers. You would conclude that OFT doesn't work.

So `peft_methods.py` routes by method: LoRA takes the diffusers-native path,
everything else round-trips through `get_peft_model_state_dict` +
`adapter_config.json`. The adapter name is stripped from the keys on save so a
checkpoint isn't bound to whatever name the saving process happened to use.

This same fact reappears at serving time (§4) and in the model card (the usage
snippet is method-aware, because a card promising `load_lora_weights` would send
every downloader into the silent-base-model trap).

### 3.3 OFT/BOFT can't go everywhere

Orthogonal methods learn a block-diagonal `R` multiplying the output space, so
they need `out_features` divisible by the block size. FLUX.2's single-stream
blocks fuse qkv *and* the MLP input into one `to_qkv_mlp_proj` whose dims don't
factor cleanly — injection fails at adapter-construction time.

Hence `ModelSpec.orthogonal_target_modules`: OFT/BOFT get the square attention
projections only. This is a real confound in the comparison and should be stated
when reporting results — OFT is adapting strictly fewer modules than LoRA, so
"OFT lost" might mean "OFT with a smaller surface lost."

Also: `use_cayley_neumann=True` by default. OFT builds `R` through a Cayley
transform, which needs a matrix inverse; the Neumann series approximates it. On a
4B transformer that's the difference between OFT being comparable to LoRA in
step time and being several times slower.

### 3.4 Matched budgets are not matched ranks

`r=16` means different things per method. LoHa's update is a Hadamard product of
*two* low-rank pairs — same `r`, roughly double the parameters. LoKr's parameter
count is dominated by `decompose_factor`, not `r` at all. OFT and BOFT don't have
a rank; they have a block size.

The configs pick per-method knobs to land near a common trainable-parameter
budget, and the sweep records `trainable_parameters` per run so the table can
show it. Comparing methods at equal `r` rather than equal budget would be
comparing nothing.

### 3.5 Variance

Three training images per subject. Per-subject variance is large. The sweep
therefore writes one row per `(method, subject)` cell and only aggregates at
report time, with a standard deviation. A table of bare means across 20 subjects
is how a 0.01 difference gets written up as a win.

---

## 4. Serving

The constraint that shapes everything: **no inference server loads a non-LoRA
PEFT adapter.** SGLang Diffusion and vLLM-Omni both expect base weights or a
LoRA-shaped delta.

So `merge_and_export.py` folds the adapter into the transformer. After merging
there's no adapter left — just a FLUX.2 transformer with different weights,
which anything can serve. Implementation notes:

- `peft`'s `merge_and_unload()` lives on `PeftModel`, but diffusers injects
  adapters directly with no wrapper. Every tuner layer still implements
  `merge()`, so walk the module tree.
- `safe_merge=True` checks the merged weight for NaN/Inf. Matters most for
  OFT/BOFT — a poorly conditioned Cayley transform silently corrupts weights and
  only shows up as noise in generated images.
- After merging, the adapter parameters are still attached and still get
  serialized. Swap each tuner layer for its `base_layer` or the checkpoint
  carries dead tensors a loader may try to interpret.

**The trade is the interesting part, and the score table won't show it.** LoRA
serves as a few MB of delta, hot-swappable per request. Every other method costs
a full model copy per subject. If two methods tie on quality, that's what decides
deployment. Worth writing down next to the numbers.

**Backends**: SGLang Diffusion has day-0 FLUX.2 support and is the default.
vLLM-Omni's FLUX.2 recipes and diffusion-LoRA support were still landing
upstream — check its startup log actually names your directory before trusting
a served image.

(The README previously said "vLLM serves LLMs, not diffusion transformers."
That was true when written. It isn't anymore — both vLLM-Omni and SGLang
Diffusion serve DiTs directly.)

---

## 5. NVFP4

NVFP4 groups weights into 16-element blocks, each with its own FP8 scale, plus a
per-tensor scale. Blackwell runs it natively.

**Order matters: quantize *after* merging.** A PEFT adapter is a delta against
the base weights it was fitted to. Quantize the base first and those weights
shift underneath the adapter. Quantize the adapter's own factors and you wreck
them — they're small, high-dynamic-range, and it's their *product* that has to
stay accurate. Post-merge there's no adapter, so the question disappears and the
same path works for all six methods.

**Hardware honesty.** Pre-Blackwell, NVFP4 weights still store fine but
dequantize in software: you get the ~4× memory saving and none of the speed.
`quantize_nvfp4.py` reports compute capability and says which you're getting, so
latency numbers from the wrong GPU don't end up in a table.

What's left unbuilt: activation quantization needs a calibration forward loop
driving the *whole* pipeline (text encoder → sampler → transformer), because
modelopt hands the loop a bare transformer and feeding it synthetic tensors
produces scales that look calibrated and aren't. Weight-only until that's done —
the script raises rather than faking it.

The question worth answering: *does a 4-bit subject adapter still hold the
subject?* `--eval-subject` re-runs DINO/CLIP-T on the quantized model and prints
deltas against bf16.

---

## 6. Auto-labeling: InternVL2 + SAM

Both are data tooling — one pass, small GPU, only their output reaches training.

**InternVL2** rewrites captions to describe the *context* (surface, lighting,
background) and is explicitly told not to describe the subject. The reasoning:
with one identical caption per subject, nothing in the text explains the
background, so the adapter is free to bind background into the subject token.
Giving the background its own words is the standard fix. The identifier is
prepended by us, not generated, so the binding stays ours.

**SAM** segments the subject with a centre-box prompt (SynCD objects are
centred, so this is reliable without a detector in the loop; SAM returns three
candidates roughly sub-part/part/whole, take the highest IoU score). Used twice:

1. Weight the training loss toward the subject (`use_masks: true`). The weight
   is clamped to a floor of 0.1 rather than 0 — a hard mask leaves the
   background fully unsupervised, i.e. the model may put anything there.
2. Composite references onto neutral **grey** before DINO scoring, so subject
   fidelity measures the object rather than a shared backdrop. Grey, not
   white/black: a saturated background moves the embedding more.

For packed latents the mask is area-pooled to the latent grid and flattened in
the same row-major order `_pack_latents` uses, so weights line up with tokens.

---

## 7. Judge

DreamBench++'s contribution is the *rubric wording* — carefully constructed
prompts that correlate with human raters far better than embedding metrics.
`dreambench_judge.py` fetches the authors' published rubrics at first use and
caches them, rather than paraphrasing, so the wording stays faithful and picks
up upstream corrections.

Two deliberate deviations, both stated in the module and the README:

1. **Claude, not GPT-4o.** Absolute scores are therefore not comparable to
   published numbers — only rankings *within one sweep*. Keep one judge across a
   sweep.
2. **Reasoning in the response, not via prefill.** The paper preseeds an
   assistant turn; assistant prefill is rejected on current Claude models.

Parser detail worth knowing: take the **last** `Score: N` match, not the first.
The concept-preservation rubric enumerates the 0-4 scale in its own text, so a
judge that restates part of the rubric before answering would otherwise be read
as scoring 0. There's a test for exactly this.

---

## 8. Where the weights live

Checkpoints go to the Hugging Face Hub, not into this repo. `push_to_hub.py`
uploads whichever weight files the method produced (`lora_weights.safetensors`
*or* `adapter_model.safetensors` + `adapter_config.json`), plus
`run_manifest.json`, plus a generated model card whose usage snippet matches the
method. `MODELS.md` is the leaderboard; `sweep_peft.py --report-only` emits the
table.

---

## 9. Open / not done

- **No results yet.** Everything above is the harness. The sweep hasn't been
  run — the numbers in `MODELS.md` are from the earlier SDXL LoRA validation run.
- **OFT/BOFT adapt fewer modules than LoRA** (§3.3). Confound; state it when
  reporting.
- **Activation quantization** needs the pipeline-in-the-loop calibration (§5).
- **FLUX.1** has no objective registered (§3.1).
- **`self-improving-diffusion` integration**: that framework's `src/runtime/`
  denoiser backends aren't implemented yet — its executor is a deterministic
  control-plane test double, not an image generator. A dptlab checkpoint can't
  be served through it until a real runtime backend exists. The natural bridge
  when it does: `ModelRef(kind, revision)` → a merged export directory, since
  that's already the format every other backend consumes.
