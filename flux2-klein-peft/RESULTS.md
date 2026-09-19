# Results — FLUX.2 [klein] PEFT sweep

Produced by the scripts in this directory. Raw per-cell JSON in
[`results/`](results/). Run 2026-09-19, Modal A100-40GB, one container per cell.

## Subject-5, 167 optimizer steps

Two splits, because the first version of this evaluation was wrong and fixing
it produced a better measurement than the fix alone.

- **in-sample** — SynCD's own `prompts`. The dataset is synthetic and
  `prompts[i]` is the prompt that *generated* `filenames[i]`, so these describe
  the training scenes. A reconstruction measure.
- **held out** — six recontextualization templates applied to the subject's
  description (snowy forest, swimming pool, neon street, beach at sunset, Van
  Gogh oil painting, held in a hand). Settings no training image shows.

| method | params | in CLIP-T | in DINO | **out CLIP-T** | **out DINO** | out CLIP-I | DINO drop |
|---|---|---|---|---|---|---|---|
| **dora** | 14,489,728 | 0.9507 | 0.4775 | 0.9572 | **0.4424** | **0.7167** | +0.0351 |
| lora | 13,813,760 | 0.9648 | 0.5060 | 0.9646 | 0.4320 | 0.7062 | **+0.0740** |
| lokr | 878,976 | 0.9249 | 0.3197 | 0.9502 | 0.2946 | 0.6665 | +0.0251 |
| loha | 13,813,760 | 0.9636 | 0.3050 | **0.9710** | 0.2907 | 0.6637 | +0.0143 |
| oft | 1,428,480 | 0.8815 | 0.3142 | 0.9176 | 0.2750 | 0.6484 | +0.0392 |
| boft | 3,041,280 | 0.8945 | 0.3342 | 0.8727 | 0.2680 | 0.6291 | +0.0662 |

**n = 1 subject. This validates the pipeline; it is not a benchmark.** There is
no spread to report. Treat the ordering as a hypothesis for a ~20-subject run.

### The ranking flips between splits

In-sample, LoRA leads subject fidelity (0.5060 vs DoRA's 0.4775). Held out,
**DoRA leads** (0.4424 vs 0.4320), because LoRA's DINO drops more than twice as
far — **0.0740 against 0.0351**.

Read plainly: at this budget LoRA's apparent advantage was substantially
memorization of the training scenes, and DoRA's decomposition into direction
and magnitude carried more of what it learned into unseen settings. That is the
claim DoRA's paper makes, and it is exactly the distinction an in-sample-only
evaluation cannot see.

LoHa is the other informative cell — the *best* held-out prompt following
(0.9710) with nearly the worst subject fidelity, and the smallest drop of any
method (+0.0143). Smallest drop is not a virtue here: it barely learned the
subject, so it had little to lose.

### How the eval was wrong, and how it surfaced

The first version scored only SynCD's `prompts` and called them held out.
`prepare_syncd.py` said they were "contexts the adapter never saw". They are
the generation prompts of the training images — parallel arrays, not a split.

No metric caught it. It surfaced in about a minute of looking at the images
next to the training data: eval prompt 0 asked for "a gnome ... on a decorative
plate on a kitchen counter with a sink and stove", and training image `0000.png`
*is* a gnome on a decorative plate on a kitchen counter with a sink and stove.

Both splits are now generated and both are reported, since the gap between them
is a per-method memorization measure worth more than either number alone.

### Confounds, in order of how much they matter

1. **n = 1 subject, 6 held-out prompts.** Everything above could reorder.
2. **167 optimizer steps, not the 500 the configs request.** `lora.py` converts
   steps to epochs without accounting for `gradient_accumulation_steps`. All
   six cells got the identical 167, so the comparison is internally fair, but
   the label is wrong and 167 steps on 3 images is very little training.
3. **OFT and BOFT adapt strictly fewer modules** — FLUX.2's fused
   `to_qkv_mlp_proj` does not factor for orthogonal blocks, so they are
   excluded by `ModelSpec.orthogonal_target_modules`. "OFT lost" partly means
   "OFT with a smaller adaptation surface lost".
4. **BOFT ran with `boft_n_butterfly_factor=1`, not the configured 2.** peft
   could not build its CUDA extension (`CUDA_HOME` unset even with ninja) and
   downgrades silently.
5. **LoRA's checkpoint may be incomplete.** Loading it warns that the PEFT
   config contains target modules absent from the state dict — the attention
   projections of `transformer_blocks.0` through `.4`. Unresolved; if real, the
   LoRA rows are of a partially-loaded adapter. **Investigate before quoting
   LoRA's numbers.**

### Cost

| stage | GPU | time |
|---|---|---|
| `prepare` | none | ~4 min |
| `smoke_all` (10 steps × 6) | A100-40GB | ~5 min |
| `sweep` (167 steps × 6, parallel) | A100-40GB × 6 | ~7 min wall |
| `evaluate` × 2 splits | A100-40GB × 6 | ~6 min wall |

Well under an hour of billed GPU for everything on this page.

### Reproducing

```bash
modal run --detach modal/sweep.py::prepare
modal run --detach modal/sweep.py::smoke_all   # 10 steps x 6, cents — do this first
modal run --detach modal/sweep.py::sweep
modal run --detach modal/sweep.py::evaluate    # defaults to the held-out split
```

`smoke_all` exists because the expensive failure is a full sweep that completes
and scores like a broken method. It found two harness bugs on its first run: six
pipelines in one container OOM after three methods (each cell now gets its own),
and a 10-step smoke written to the real output path made the sweep skip LoRA as
"already trained".

Both `sweep` and `evaluate` are resumable — a container dropped to a heartbeat
timeout mid-eval, and re-scoring five finished cells to recover one is five
times the cost for no information.

Use `--detach`. A non-detached run dies when the local client disconnects.
