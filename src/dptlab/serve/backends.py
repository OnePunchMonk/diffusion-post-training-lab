"""Serving backends for a post-trained FLUX.2 [klein] checkpoint.

There is one fact that shapes this whole module: **no inference server loads a
non-LoRA PEFT adapter.** SGLang Diffusion and vLLM-Omni accept base weights or
a LoRA-shaped delta; LoHa, LoKr, OFT and BOFT are formats they have never heard
of. So five of the six methods in the sweep can only be served *merged* --
`flux2-klein-peft/scripts/merge_and_export.py` folds the adapter into the transformer and writes
an ordinary diffusers directory, which every backend below can serve.

That asymmetry is itself a benchmark result. LoRA can be served as a few MB of
delta hot-swapped per request; every other method costs a full model copy per
subject. If two methods score the same, that difference decides which one you
would actually deploy.

Backends:

- **SGLang Diffusion** -- the default. Day-0 FLUX.2 support, step-level
  continuous batching and an OpenAI-compatible image endpoint, which is what
  makes it a drop-in for the Modal server this repo already has.
- **vLLM-Omni** -- vLLM's diffusion module. Also does step-level continuous
  batching; its FLUX.2 recipes and diffusion LoRA support were still landing as
  of this writing, so treat it as the second option and check the launch log
  actually reports the merged weights.

Both are launched as subprocesses rather than imported: they pin CUDA, kernel
and driver versions that have no business being constraints on `pip install
dptlab`, and both expose HTTP as the real interface anyway.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from dptlab.models.registry import get_model_spec


@dataclass
class ServeSpec:
    """A resolved launch plan. Rendered to a command; not executed here."""

    backend: str
    model_path: str
    args: list[str] = field(default_factory=list)
    port: int = 30000
    notes: str = ""

    @property
    def command(self) -> list[str]:
        return self.args

    def render(self) -> str:
        return " ".join(shlex.quote(a) for a in self.args)


def _export_manifest(model_path: str | Path) -> dict:
    path = Path(model_path) / "export_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{model_path} has no export_manifest.json. Serving backends need a merged model -- "
            "run flux2-klein-peft/scripts/merge_and_export.py on the checkpoint first."
        )
    return json.loads(path.read_text())


def sglang_spec(model_path: str | Path, port: int = 30000, tp_size: int = 1, **kwargs) -> ServeSpec:
    """`sglang serve` on a merged export."""
    manifest = _export_manifest(model_path)
    spec = get_model_spec(manifest["base_model"])

    args = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
    ]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]

    return ServeSpec(
        backend="sglang",
        model_path=str(model_path),
        args=args,
        port=port,
        notes=(
            f"{manifest['base_model']} + merged {manifest['peft_method']}. Request it at "
            f"{spec.default_inference_steps} steps / guidance {spec.default_guidance_scale} -- klein is "
            "step- and guidance-distilled and the server does not know that from the weights alone."
        ),
    )


def vllm_omni_spec(model_path: str | Path, port: int = 8000, tp_size: int = 1, **kwargs) -> ServeSpec:
    """`vllm-omni serve` on a merged export."""
    manifest = _export_manifest(model_path)

    args = [
        "vllm-omni",
        "serve",
        str(model_path),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
    ]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]

    return ServeSpec(
        backend="vllm-omni",
        model_path=str(model_path),
        args=args,
        port=port,
        notes=(
            f"{manifest['base_model']} + merged {manifest['peft_method']}. FLUX.2 recipes and diffusion "
            "LoRA were still in-flight upstream; confirm the startup log names this directory and not a "
            "silently substituted base checkpoint before trusting any served image."
        ),
    )


_BACKENDS = {"sglang": sglang_spec, "vllm-omni": vllm_omni_spec}


def list_backends() -> list[str]:
    return sorted(_BACKENDS)


def build_serve_spec(backend: str, model_path: str | Path, **kwargs) -> ServeSpec:
    try:
        builder = _BACKENDS[backend]
    except KeyError as e:
        raise KeyError(f"Unknown backend {backend!r}. Available: {list_backends()}") from e
    return builder(model_path, **kwargs)


def generation_defaults(model_path: str | Path) -> dict:
    """Sampling parameters a client should send for this exported model.

    Returned rather than baked into the server command because both backends
    take these per-request, and a client that omits them gets the generic
    diffusion defaults (dozens of steps, guidance ~7) which are wrong for a
    distilled model in both quality and latency.
    """
    manifest = _export_manifest(model_path)
    spec = get_model_spec(manifest["base_model"])
    return {
        "num_inference_steps": spec.default_inference_steps,
        "guidance_scale": spec.default_guidance_scale,
        "height": spec.default_resolution,
        "width": spec.default_resolution,
    }
