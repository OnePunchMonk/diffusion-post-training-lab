"""Box parsing for the InternVL2 -> SAM cascade.

The whole experiment routes through this function. A parser that picks the
wrong coordinate convention produces masks that are plausible, non-empty and
wrong, scoring a few gIoU -- indistinguishable from "the method doesn't work"
unless it is pinned here.
"""

from __future__ import annotations

from asvl.cascade import (
    SCALE_PIXEL,
    SCALE_THOUSAND,
    SCALE_UNIT,
    is_degenerate,
    parse_box,
)

W, H = 640, 480


def test_documented_internvl2_format():
    box = parse_box("<box>[[100, 200, 300, 400]]</box>", W, H)
    assert box is not None
    assert box.scale == SCALE_THOUSAND
    # 0-1000 normalized against the image, not raw pixels.
    assert box.x0 == 64.0 and box.y0 == 96.0
    assert box.x1 == 192.0 and box.y1 == 192.0


def test_prose_around_the_box_is_ignored():
    reply = "The region described is here: <box>[[10, 20, 900, 950]]</box>. Hope that helps!"
    box = parse_box(reply, W, H)
    assert box is not None and box.x1 > box.x0


def test_falls_back_when_the_model_drops_its_own_tags():
    """Models drift from their template; a rigid parser turns a correct
    prediction into a miss."""
    box = parse_box("The bounding box is [250, 250, 750, 750].", W, H)
    assert box is not None
    assert box.scale == SCALE_THOUSAND


def test_unit_square_coordinates():
    box = parse_box("<box>[[0.1, 0.2, 0.5, 0.6]]</box>", W, H)
    assert box is not None
    assert box.scale == SCALE_UNIT
    assert box.x0 == 64.0 and box.y1 == 288.0


def test_pixel_coordinates_on_a_large_image():
    box = parse_box("<box>[[100, 200, 1800, 1400]]</box>", 2000, 1500)
    assert box is not None
    assert box.scale == SCALE_PIXEL
    assert box.x1 == 1800.0


def test_reversed_corners_are_normalized():
    box = parse_box("<box>[[800, 900, 100, 200]]</box>", W, H)
    assert box is not None
    assert box.x0 < box.x1 and box.y0 < box.y1


def test_coordinates_are_clamped_to_the_image():
    """An out-of-frame box must be clipped, not passed to SAM as-is."""
    box = parse_box("<box>[[-50, -50, 1200, 1200]]</box>", W, H)
    assert box is not None
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x1 <= W and box.y1 <= H


def test_refusal_or_empty_reply_returns_none():
    assert parse_box("I cannot identify that region in the image.", W, H) is None
    assert parse_box("", W, H) is None
    assert parse_box("<box>[[10, 20]]</box>", W, H) is None  # too few numbers


def test_degenerate_box_is_caught():
    """SAM returns an arbitrary mask for a zero-area prompt rather than an
    error, so a collapsed box has to be rejected before it reaches SAM."""
    collapsed = parse_box("<box>[[500, 500, 500, 500]]</box>", W, H)
    assert collapsed is not None
    assert is_degenerate(collapsed, W, H)

    real = parse_box("<box>[[100, 100, 900, 900]]</box>", W, H)
    assert real is not None
    assert not is_degenerate(real, W, H)


def test_raw_values_are_kept_for_diagnosis():
    """When a run scores badly, the first question is what the model emitted."""
    box = parse_box("<box>[[100, 200, 300, 400]]</box>", W, H)
    assert box is not None
    assert box.raw == (100.0, 200.0, 300.0, 400.0)
