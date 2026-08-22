"""Text-to-image adapter protocol.

Adapted from OnePunchMonk/vlm-harness's `adapters/generative/base.py`. Kept
as its own protocol (not shared with a text-in/text-out LLM adapter) because
the input/output shape and cost model are different enough to not be worth
forcing into one interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass
class T2IResponse:
    image: Image.Image
    latency_ms: float = 0.0
    model_id: str = ""
    checkpoint_tag: str = "base"  # "base" | "lora" | "dpo" | "grpo" | "distill"
    seed: int | None = None
    num_inference_steps: int = 0
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class T2IAdapter(Protocol):
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
    ) -> T2IResponse: ...

    @property
    def model_id(self) -> str: ...
