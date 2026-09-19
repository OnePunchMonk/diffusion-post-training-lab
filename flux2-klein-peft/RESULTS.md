# Results — FLUX.2 [klein] PEFT sweep

Produced by the scripts in this directory. Raw per-cell JSON is in
[`results/`](results/); `results/sweep-subject-5.json` is the merged file.

Run on 2026-09-19, Modal A100-40GB, one container per cell.

## Subject-5, 167 optimizer steps, 3 held-out prompts

Sorted by DINO (subject fidelity).

| method | trainable params | CLIP-T | DINO | CLIP-I | ms/img |
|---|---|---|---|---|---|
| **lora** | 13,813,760 | **0.9648** | **0.5060** | **0.7540** | 2686 |
| dora | 14,489,728 | 0.9507 | 0.4775 | 0.7459 | 2369 |
| boft | 3,041,280 | 0.8945 | 0.3342 | 0.7017 | 2719 |
| lokr | 878,976 | 0.9249 | 0.3197 | 0.7013 | 1454 |
| oft | 1,428,480 | 0.8815 | 0.3142 | 0.6992 | 2858 |
| loha | 13,813,760 | 0.9636 | 0.3050 | 0.7110 | 1478 |

**This is one subject and three prompts. It is a pipeline-validation run, not a
benchmark.** No standard deviation exists because there is nothing to take it
over. Treat the ordering as a hypothesis to test at ~20 subjects, not a result.

### What it does and does not say

- **LoRA leads on both axes.** That is worth noting precisely because the newer
  methods are the ones with a story: at this budget and step count, the
  baseline is not beaten.
- **LoHa is the interesting cell.** Near-top CLIP-T (0.9636) with the *lowest*
  DINO (0.3050) — good prompt following, weakest subject identity. That is the
  underfit corner: the adapter is barely changing the model, so the base
  model's prompt behaviour survives intact and the subject never lands. It is
  exactly why both metrics are reported.
- **Parameter count is not the story.** LoKr at 0.88M is within noise of OFT at
  1.43M and BOFT at 3.04M, while LoHa at 13.8M is bottom on DINO.

### Confounds, in order of how much they matter

1. **n = 1 subject, 3 prompts.** Everything above could reorder.
2. **167 optimizer steps, not the 500 the configs request.** `lora.py` converts
   steps to epochs without accounting for `gradient_accumulation_steps`, so
   `max_train_steps` overstates by roughly the accumulation factor. All six
   cells got the identical 167, so the comparison is internally fair — but the
   number is not what the config says, and 167 steps on 3 images is very
   little training. See the tracking issue.
3. **OFT and BOFT adapt strictly fewer modules** than the LoRA-family methods.
   FLUX.2's fused `to_qkv_mlp_proj` does not factor for orthogonal blocks, so
   they are excluded by `ModelSpec.orthogonal_target_modules`. "OFT lost" here
   partly means "OFT with a smaller adaptation surface lost".
4. **BOFT ran with `boft_n_butterfly_factor=1`, not the configured 2.** peft
   could not build its CUDA extension (`CUDA_HOME` unset even with ninja
   installed) and silently downgrades. BOFT was benchmarked at a setting nobody
   chose.

### Cost

| stage | GPU | time |
|---|---|---|
| `prepare` (SynCD slice) | none | ~4 min |
| `smoke_all` (10 steps × 6) | A100-40GB | ~5 min |
| `sweep` (167 steps × 6, parallel) | A100-40GB × 6 | ~7 min wall |
| `evaluate` (6 cells, parallel) | A100-40GB × 6 | ~3 min wall |

Well under an hour of billed GPU time for the whole thing.

### Reproducing

```bash
modal run --detach modal/sweep.py::prepare
modal run --detach modal/sweep.py::smoke_all   # 10 steps x 6, cents — do this first
modal run --detach modal/sweep.py::sweep
modal run --detach modal/sweep.py::evaluate
```

`smoke_all` exists because the expensive failure mode is a full sweep that
completes and scores like a broken method. It found two harness bugs on its
first outing: six pipelines in one container OOM after three methods (each
cell now gets its own container), and a 10-step smoke written to the real
output path made the sweep skip LoRA as "already trained".

Use `--detach`. A non-detached run dies when the local client disconnects.
