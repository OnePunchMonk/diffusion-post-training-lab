# flux2-klein-peft

Six PEFT methods, one base model, one task: **does the choice of adapter
actually matter for subject-driven personalization on FLUX.2 [klein]?**

The parent repo's four recipes vary the *objective* (denoising, preference, RL,
distillation) with the adapter fixed at LoRA. This subproject varies the other
axis — same objective, same data, same budget, six different adapters.

Read [`learning-doc.md`](learning-doc.md) for the reasoning and the traps.

## Layout

```
flux2-klein-peft/
  learning-doc.md        why it's built this way; the failure modes that don't raise
  configs/               _base.yaml + one per method (lora, dora, loha, lokr, oft, boft)
  scripts/
    prepare_syncd.py     slice SynCD into per-subject training sets
    autolabel.py         InternVL2 context captions + SAM subject masks
    sweep_peft.py        {methods} x {subjects} train + eval, resumable
    merge_and_export.py  fold any adapter into base weights so a server can load it
    quantize_nvfp4.py    NVFP4/FP8 a merged export, then A/B the quality
  tests/                 CPU-only; no weights, no GPU
```

The importable library code lives in the `dptlab` package rather than here,
because it extends modules that already exist there:

| Module | Sits next to | Does |
|---|---|---|
| `dptlab/training/peft_methods.py` | `lora.py`, `common.py` | method registry: config shape, legal target modules, checkpoint format |
| `dptlab/training/objectives.py` | `lora.py` | epsilon (SDXL) vs. rectified flow (FLUX.2) training steps |
| `dptlab/eval/metrics/subject_fidelity.py` | `clip_score.py` | DINO + CLIP-I |
| `dptlab/eval/metrics/dreambench_judge.py` | `aesthetic.py` | DreamBench++ rubrics, Claude judge |
| `dptlab/serve/backends.py` | `modal_app.py` | SGLang Diffusion / vLLM-Omni launch specs |

## Quickstart

```bash
pip install -e ".[train,eval,label]"    # add ,judge for the LLM judge; ,quant for NVFP4
cd flux2-klein-peft

# 1. data — one SynCD archive (~1.8GB), sliced to 20 subjects
python scripts/prepare_syncd.py --num-subjects 20 --out data/syncd-20

# 2. optional: per-image context captions + subject masks
python scripts/autolabel.py --dataset data/syncd-20/subject-0 --class-name "flip flops"

# 3. smoke test one cell before paying for the grid
python scripts/sweep_peft.py --data-root data/syncd-20 --out runs/smoke \
    --max-subjects 1 --max-train-steps 20

# 4. the sweep (resumable; skips completed method x subject cells)
python scripts/sweep_peft.py --data-root data/syncd-20 --out runs/klein-peft

# 5. table for MODELS.md
python scripts/sweep_peft.py --data-root data/syncd-20 --out runs/klein-peft --report-only
```

**Budget it first.** 6 methods x 20 subjects x 500 steps at 512px is roughly
30-40 GPU-hours. Start at `--max-subjects 5` (~8 GPU-h) and check the methods
separate at all before committing to the full grid.

## Serving

```bash
# no server loads a LoHa/LoKr/OFT/BOFT adapter -- merge it into the weights first
python scripts/merge_and_export.py \
    --checkpoint runs/klein-peft/oft/subject-0/ckpt/final \
    --out exports/klein-oft-subject-0

python -c "from dptlab.serve.backends import build_serve_spec; \
           print(build_serve_spec('sglang', 'exports/klein-oft-subject-0').render())"

# optional: NVFP4, and check the subject survived 4 bits
python scripts/quantize_nvfp4.py --model exports/klein-oft-subject-0 \
    --out exports/klein-oft-subject-0-nvfp4 --eval-subject data/syncd-20/subject-0
```

## Status

Harness complete, **no sweep has been run yet** — there are no results to
report. The code is verified by CPU-only tests and by reading the diffusers /
peft sources; the flow-matching step, adapter injection and merge paths have
not been exercised against real weights. Step 3 above is the first thing to run
on a GPU box.

Credits: [SynCD](https://huggingface.co/datasets/nupurkmr9/syncd) (ICCV 2025,
MIT) · [DreamBench++](https://huggingface.co/papers/2406.16855) (ICLR 2025) ·
[FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0)
