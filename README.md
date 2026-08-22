# dptlab — Diffusion Post-Training Lab

Post-train open-weight text-to-image diffusion models (SDXL, FLUX) with four
recipes, evaluate them with a shared harness, and publish each checkpoint to
the [Hugging Face Hub](https://huggingface.co) with an auto-generated model
card. [`MODELS.md`](MODELS.md) is the running leaderboard: every pushed
checkpoint gets a benchmarked row. Built as a from-scratch study of the
post-training stack for image generation models — the same shape as modern
LLM post-training (SFT → preference optimization → RL → distillation),
applied to diffusion.

## Recipes

| Recipe | File | Idea | Reference |
|---|---|---|---|
| **LoRA fine-tune** | `src/dptlab/training/lora.py` | Teach a new concept/style via low-rank adapters on the denoiser's attention projections. | DreamBooth / LoRA |
| **Diffusion-DPO** | `src/dptlab/training/dpo.py` | Preference alignment from (win, lose) image pairs, using a frozen reference copy — the diffusion analogue of RLHF's DPO. | Wallace et al. 2023 |
| **GRPO** | `src/dptlab/training/grpo.py` | Group-relative RL: sample a group of images per prompt, score each with the eval harness itself as the reward, optimize relative to the group mean. No reference model, no human labels. | DeepSeek GRPO / Flow-GRPO-style diffusion RL |
| **Step-distillation** | `src/dptlab/training/distill.py` | Consistency-distill a (possibly DPO/GRPO-tuned) teacher into a 4-step student — this is what actually makes the Modal endpoint cheap. | Latent Consistency Models |

Every recipe reads the same `TrainConfig` YAML shape (`configs/recipes/*.yaml`),
goes through the same `models/registry.py` (so SDXL and FLUX are one-line
swaps), and writes checkpoints with a `run_manifest.json` that the eval
harness and the Hub-publishing script both read to know which base model +
recipe produced them.

### GRPO: the eval harness as the reward model

The GRPO recipe's reward function is literally `dptlab.eval.metrics` — the
same CLIP/aesthetic scorers used to *evaluate* checkpoints also *train* them.
`scripts/build_preference_pairs.py` uses the same idea to auto-generate DPO
preference pairs without human labels: sample twice, rank with the scorers,
keep the pair if the gap is large enough. This mirrors the verifier-driven RL
data pipelines behind DeepSeek-R1 and Kimi k1.5, adapted to image generation.

## Eval harness

`src/dptlab/eval/` — adapted from and credits
[OnePunchMonk/vlm-harness](https://github.com/OnePunchMonk/vlm-harness)'s
generative-adapter design (`T2IAdapter` protocol, `DiffusersAdapter`), extended
for this project's needs:

- **`CheckpointAdapter`** — loads a base model + optional LoRA checkpoint from
  our `run_manifest.json` layout, so evaluating any checkpoint is one line.
- **`AestheticScorer`** — LAION aesthetic predictor (linear head on CLIP
  ViT-L/14). The upstream harness only shipped CLIPScore/FID/GenEval, which
  reward prompt-following, not "does this look good" — the axis DPO/GRPO are
  actually trying to move. Reporting both side by side is the point.
- **`compute_win_rate`** — paired comparison (same prompt/seed, checkpoint A
  vs. B), because DPO and GRPO's claim is "policy beats reference," which two
  independent scalar averages can't actually support.

```bash
dptlab eval --checkpoint outputs/dpo-sdxl/final \
    --baseline-model-key sdxl \
    --prompts prompts/geneval_mini.jsonl
```

## Publishing: Hugging Face Hub, not a hosted endpoint

Rather than standing up a Modal endpoint that only works while your account
is paying for it, every checkpoint gets pushed to the Hub as its own repo
with LoRA weights, `run_manifest.json`, and an auto-generated model card
(base model, recipe, hyperparameters, and — once benchmarked — the same
numbers that land in `MODELS.md`). That's the actual deliverable: a
`diffusers`-loadable checkpoint anyone can pull, not an endpoint only you can
hit.

```bash
# train -> benchmark -> publish -> record
accelerate launch scripts/train.py --config configs/recipes/dpo.yaml
dptlab eval --checkpoint outputs/dpo-sdxl/final --baseline-model-key sdxl \
    --prompts prompts/geneval_mini.jsonl --output-dir eval_results/dpo-sdxl
python scripts/push_to_hub.py --checkpoint outputs/dpo-sdxl/final \
    --repo-id OnePunchMonk/dptlab-sdxl-dpo-v1 --eval-report eval_results/dpo-sdxl/report.json
python scripts/update_models_md.py --repo-id OnePunchMonk/dptlab-sdxl-dpo-v1 \
    --checkpoint outputs/dpo-sdxl/final --eval-report eval_results/dpo-sdxl/report.json
```

See [`MODELS.md`](MODELS.md) for the leaderboard this produces and the full
publish loop.

### Optional: Modal serving reference

`src/dptlab/serve/modal_app.py` is kept as a reference implementation for
anyone who *does* want a live endpoint (`modal deploy src/dptlab/serve/modal_app.py`),
loading checkpoints straight from the Hub instead of a Modal volume. It also
carries a scoping note worth keeping regardless of deployment target: vLLM
serves LLMs (and some VLMs), not diffusion UNets/transformers — there's no
literal "vLLM-serve SDXL." The one place vLLM legitimately fits a T2I
pipeline is serving a small prompt-rewriting LLM (`PromptRewriter` in that
file), which is what it's used for there.

## Repo layout

```
src/dptlab/
  models/registry.py       # SDXL / FLUX ModelSpec registry
  data/                     # dataset classes for each recipe
  training/                 # lora.py, dpo.py, grpo.py, distill.py, common.py
  eval/
    adapters/                # T2IAdapter protocol + CheckpointAdapter
    metrics/                 # clip_score, aesthetic, winrate
    runner.py, cli.py
  serve/modal_app.py        # optional: reference Modal endpoint (+ vLLM prompt rewriter)
configs/recipes/*.yaml       # one config per recipe
scripts/
  train.py                    # accelerate launch scripts/train.py --config ...
  build_preference_pairs.py   # auto-generate DPO pairs via the eval scorers
  push_to_hub.py               # publish a checkpoint + model card to the Hub
  update_models_md.py          # record its benchmark row in MODELS.md
tests/
MODELS.md                      # the leaderboard
```

## Status

Scaffolding is in place end-to-end (config → training loop → checkpoint →
eval → Hub publish → `MODELS.md` row) for all four recipes against SDXL and
FLUX. What's not done yet: an actual training run, real preference/concept
datasets, and a filled-in `MODELS.md` — next steps are running LoRA on a
small concept dataset first to validate the pipeline, then DPO/GRPO on top.

## License

MIT. Eval-harness design adapted from
[OnePunchMonk/vlm-harness](https://github.com/OnePunchMonk/vlm-harness).
