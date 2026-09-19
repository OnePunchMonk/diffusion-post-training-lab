"""Pinned upstream artifacts.

A replication that tracks a moving `main` is not a replication: if the number
drifts you cannot tell whether your change or theirs caused it. Everything the
step-1 evaluation touches is pinned to an exact commit or revision here, and
the published numbers being targeted are recorded alongside so the comparison
is checked in code rather than eyeballed against a README.
"""

from __future__ import annotations

from dataclasses import dataclass

# github.com/rui-qian/AnchorSeg, the ACL 2026 release.
UPSTREAM_REPO = "https://github.com/rui-qian/AnchorSeg.git"
UPSTREAM_COMMIT = "0d8e3f098b763aeb035baadf85649b94c7e1e721"  # 2026-08-12, "init AnchorSeg"

# Released checkpoint for the ReasonSeg val result.
ANCHORSEG_7B_REPO = "rui-qian/hf-AnchorSeg-7b_reason_seg_val_llava1.5_ema"

# Frozen components the model is built on. The SAM checkpoint is the original
# ViT-H release -- the upstream training script names this exact file, and
# swapping in a different SAM variant changes the mask decoder the projection
# layer was trained against.
SAM_VIT_H_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_VIT_H_FILENAME = "sam_vit_h_4b8939.pth"
CLIP_VISION_TOWER = "openai/clip-vit-large-patch14-336"


@dataclass(frozen=True)
class PublishedResult:
    """A number from the paper/repo that a replication run is checked against."""

    name: str
    giou: float
    ciou: float
    source: str
    # How close counts as reproduced. Evaluation of these models is not bitwise
    # deterministic (autoregressive decoding, fused kernels, batch composition),
    # so demanding an exact match would fail on noise; a whole point of gIoU is
    # far more than run-to-run variance, so a miss that large is a real bug.
    tolerance: float = 1.0


# LISA Table 1, ReasonSeg val (overall). These are the training-free and
# specialist baselines the cascade experiment sits beside. Grounded-SAM is the
# direct comparison: same shape (text -> box -> SAM), different grounding stage.
LISA_TABLE1_VAL = {
    "OVSeg": (28.5, 18.6),
    "GRES": (22.4, 19.9),
    "X-Decoder": (22.6, 17.9),
    "Grounded-SAM": (26.0, 14.5),
}
GROUNDED_SAM_VAL = PublishedResult(
    name="Grounded-SAM / ReasonSeg val (LISA Table 1)",
    giou=26.0,
    ciou=14.5,
    source="https://arxiv.org/abs/2308.00692 Table 1",
    # Wider than the AnchorSeg tolerance on purpose: this is not a replication
    # of Grounded-SAM, it is a different grounding stage in the same cascade.
    # The number is a reference point, not a target to hit.
    tolerance=99.0,
)

REASONSEG_VAL_7B = PublishedResult(
    name="AnchorSeg-LLaVA-1.5-7B / ReasonSeg val",
    giou=67.20,
    ciou=75.15,
    source="https://github.com/rui-qian/AnchorSeg README results table",
)


def verdict(
    measured_giou: float,
    measured_ciou: float,
    target: PublishedResult = REASONSEG_VAL_7B,
) -> str:
    """Plain-language outcome for a replication run."""
    d_giou = measured_giou - target.giou
    d_ciou = measured_ciou - target.ciou
    within = abs(d_giou) <= target.tolerance and abs(d_ciou) <= target.tolerance

    lines = [
        f"target   gIoU {target.giou:6.2f}   cIoU {target.ciou:6.2f}   ({target.name})",
        f"measured gIoU {measured_giou:6.2f}   cIoU {measured_ciou:6.2f}",
        f"delta    gIoU {d_giou:+6.2f}   cIoU {d_ciou:+6.2f}   (tolerance +/-{target.tolerance})",
        "",
        "REPRODUCED" if within else "NOT REPRODUCED",
    ]

    if not within:
        # Which metric moved localizes the bug, so say so rather than making
        # the reader rederive it.
        if abs(d_giou) > target.tolerance >= abs(d_ciou):
            lines.append(
                "gIoU is off while cIoU holds: the gap is concentrated in small objects. "
                "Suspect mask resolution or the polygon rasterization before the model."
            )
        elif abs(d_ciou) > target.tolerance >= abs(d_giou):
            lines.append(
                "cIoU is off while gIoU holds: a few large objects dominate the difference. "
                "Suspect a handful of catastrophic failures rather than a systematic shift."
            )
        else:
            lines.append(
                "Both metrics are off by a similar amount: suspect the data split, the "
                "checkpoint, or the prompt template rather than the mask pipeline."
            )
    return "\n".join(lines)
