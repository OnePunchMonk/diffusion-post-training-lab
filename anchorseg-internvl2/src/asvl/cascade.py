"""Training-free cascade: InternVL2 grounds the query, SAM turns it into a mask.

The experiment this supports: **LISA's Table 1 reports Grounded-SAM at 26.0
gIoU / 14.5 cIoU on ReasonSeg val** -- a zero-training cascade of
text -> GroundingDINO box -> SAM. LISA's argument for why cascades score badly
there is that the grounding stage can only *match* text to objects, while
ReasonSeg queries require inferring which object is meant ("the thing propping
the door open" names nothing).

That argument makes a testable prediction. Swap the grounding stage for a VLM
that can reason and keep everything else the same: if LISA is right about *why*
cascades fail, the number should move, and it should move more on long queries
than on short ones. ReasonSeg's `is_sentence` flag splits exactly those two
populations, so the prediction is checkable rather than rhetorical.

Nothing here is trained. The cost is one inference pass over 200 images.

The fragile part is not the models, it is the **coordinate convention**. A box
parser that assumes the wrong normalization produces masks that are plausible,
non-empty, wrong, and score around 5 gIoU -- which reads as "the method
doesn't work" rather than "the parser is broken". So the scale is *inferred
from the observed values and reported*, never assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# InternVL2's documented referring-expression prompt.
GROUNDING_PROMPT = (
    "<image>\nPlease provide the bounding box coordinates of the region this "
    "sentence describes: <ref>{query}</ref>"
)

# Coordinate conventions a VLM might emit, in the order we test for them.
SCALE_PIXEL = "pixel"
SCALE_THOUSAND = "0-1000"
SCALE_UNIT = "0-1"


@dataclass
class ParsedBox:
    """A box in absolute pixel coordinates, plus how we got there."""

    x0: float
    y0: float
    x1: float
    y1: float
    scale: str
    raw: tuple[float, float, float, float]

    def as_xyxy(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def area_fraction(self) -> float:
        return abs((self.x1 - self.x0) * (self.y1 - self.y0))


@dataclass
class CascadeStats:
    """What the run actually did, so a bad number can be attributed."""

    n_samples: int = 0
    n_parsed: int = 0
    n_unparseable: int = 0
    n_degenerate: int = 0
    scales_seen: dict[str, int] = field(default_factory=dict)
    unparseable_examples: list[str] = field(default_factory=list)

    def note_scale(self, scale: str) -> None:
        self.scales_seen[scale] = self.scales_seen.get(scale, 0) + 1

    def summary(self) -> str:
        lines = [
            f"samples            {self.n_samples}",
            f"boxes parsed       {self.n_parsed}",
            f"unparseable        {self.n_unparseable}",
            f"degenerate boxes   {self.n_degenerate}",
            f"coordinate scales  {self.scales_seen or '{}'}",
        ]
        if len(self.scales_seen) > 1:
            lines.append(
                "  WARNING: more than one scale inferred across the run. The inference is "
                "per-sample and a mixed result usually means small boxes are being read as "
                "0-1 when they are 0-1000. Check before trusting the score."
            )
        for example in self.unparseable_examples[:3]:
            lines.append(f"  unparseable: {example[:120]!r}")
        return "\n".join(lines)


def infer_scale(values: tuple[float, float, float, float], width: int, height: int) -> str:
    """Which coordinate convention these four numbers are in.

    Inferred rather than assumed, and returned so the caller can report it.
    The three conventions are separated by magnitude:

    - any value above 1000 cannot be 0-1000 or 0-1, so it is pixels
    - all values at or below 1 is the unit square
    - otherwise 0-1000, which is what InternVL2 documents

    The ambiguous case is a genuinely tiny box in pixel coordinates on a large
    image, which looks like the unit square. It is rare enough to accept and
    frequent enough to be worth reporting, hence `scales_seen`.
    """
    largest = max(values)
    if largest > 1000:
        return SCALE_PIXEL
    if largest <= 1.0:
        return SCALE_UNIT
    # A box in pixels on a small image can land in 0-1000 too. Prefer the
    # documented convention, but fall back to pixels when 0-1000 would put the
    # box outside the image entirely.
    if largest > max(width, height):
        return SCALE_THOUSAND
    return SCALE_THOUSAND


def parse_box(text: str, width: int, height: int) -> ParsedBox | None:
    """Pull the first bounding box out of a VLM reply, in pixel coordinates.

    Handles the `<box>[[x1, y1, x2, y2]]</box>` form InternVL2 documents, and
    falls back to the first four numbers in the reply -- models drift from
    their own template, and a rigid parser silently turns a correct prediction
    into a miss.
    """
    box_match = re.search(r"<box>\s*(.*?)\s*</box>", text, re.S)
    search_space = box_match.group(1) if box_match else text

    numbers = re.findall(r"-?\d+\.?\d*", search_space)
    if len(numbers) < 4:
        return None

    raw = tuple(float(n) for n in numbers[:4])
    scale = infer_scale(raw, width, height)

    if scale == SCALE_UNIT:
        x0, y0, x1, y1 = (raw[0] * width, raw[1] * height, raw[2] * width, raw[3] * height)
    elif scale == SCALE_THOUSAND:
        x0, y0, x1, y1 = (
            raw[0] / 1000 * width,
            raw[1] / 1000 * height,
            raw[2] / 1000 * width,
            raw[3] / 1000 * height,
        )
    else:
        x0, y0, x1, y1 = raw

    # Models sometimes emit the corners in the other order.
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    x0 = max(0.0, min(x0, width))
    x1 = max(0.0, min(x1, width))
    y0 = max(0.0, min(y0, height))
    y1 = max(0.0, min(y1, height))

    return ParsedBox(x0, y0, x1, y1, scale=scale, raw=raw)


def is_degenerate(box: ParsedBox, width: int, height: int, min_fraction: float = 1e-4) -> bool:
    """A box with no usable area.

    Prompting SAM with a zero-area box returns an arbitrary mask rather than an
    error, so this has to be caught here or it becomes a silent wrong answer.
    """
    return box.area_fraction < min_fraction * width * height
