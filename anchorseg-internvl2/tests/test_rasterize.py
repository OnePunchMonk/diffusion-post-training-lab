"""Mask construction, checked against LISA's `get_mask_from_json`.

These masks are what every number in the replication is scored against. Each
test below corresponds to a way the obvious implementation differs from the
reference -- all four were found by running scripts/check_ground_truth.py on
the real split and then reading the reference, not by reasoning about the
format.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from asvl.data.reasonseg import (
    BACKGROUND,
    IGNORE,
    TARGET,
    ignore_mask,
    rasterize,
    target_mask,
)


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_single_target_polygon():
    mask = rasterize([{"label": "target", "points": square(2, 2, 10, 10)}], 20, 20)
    assert mask[5, 5] == TARGET
    assert mask[18, 18] == BACKGROUND


def test_mask_is_trinary_not_binary():
    """Ignore is its own value, not background.

    Evaluation passes ignore_index=255 and excludes those pixels from both
    intersection and union. Collapsing them to background turns a prediction
    there into a false positive, which the reference treats as a no-op.
    """
    shapes = [
        {"label": "target", "points": square(0, 0, 6, 6)},
        {"label": "ignore", "points": square(10, 10, 16, 16)},
    ]
    mask = rasterize(shapes, 20, 20)
    assert set(np.unique(mask)) <= {BACKGROUND, TARGET, IGNORE}
    assert mask[12, 12] == IGNORE
    assert ignore_mask(mask)[12, 12]
    assert not target_mask(mask)[12, 12]


def test_small_target_inside_large_ignore_stays_target():
    """The painter's algorithm, largest area first.

    This is the case that a union-then-subtract implementation gets backwards.
    It silently emptied ReasonSeg val sample 914980029_4a7c8f579e_o, whose
    target covers 31828 px.
    """
    shapes = [
        {"label": "ignore", "points": square(0, 0, 20, 20)},
        {"label": "target", "points": square(5, 5, 10, 10)},
    ]
    mask = rasterize(shapes, 24, 24)
    assert mask[7, 7] == TARGET, "small target inside a large ignore was erased"
    assert mask[2, 2] == IGNORE


def test_paint_order_does_not_depend_on_input_order():
    """Sorting is by area, so shuffling the annotation order changes nothing."""
    a = {"label": "ignore", "points": square(0, 0, 20, 20)}
    b = {"label": "target", "points": square(5, 5, 10, 10)}
    assert np.array_equal(rasterize([a, b], 24, 24), rasterize([b, a], 24, 24))


def test_flag_shapes_are_dropped_by_label():
    """`flag` is a deprecated marker, not a region.

    It appears 62 times in ReasonSeg val, always with fewer than 3 points -- so
    a rule of "anything not ignore is a target" survives by luck alone. A flag
    with real area must still be dropped.
    """
    shapes = [
        {"label": "flag", "points": square(0, 0, 20, 20)},
        {"label": "target", "points": square(2, 2, 5, 5)},
    ]
    mask = rasterize(shapes, 24, 24)
    assert mask[15, 15] == BACKGROUND, "a flag polygon was painted as a target"
    assert mask[3, 3] == TARGET


def test_outline_is_painted_inclusively():
    """The reference paints the outline before filling, adding ~1px of border.

    On ReasonSeg's small targets (median 6% of frame) that is a percent-level
    IoU difference. A stricter rasterizer is a *different* ground truth, not a
    more correct one.
    """
    mask = rasterize([{"label": "target", "points": square(5, 5, 9, 9)}], 20, 20)
    # The boundary row/column the polygon passes through is included.
    assert mask[5, 5] == TARGET
    assert mask[9, 9] == TARGET


def test_degenerate_shapes_are_skipped():
    shapes = [
        {"label": "target", "points": [[0, 0], [1, 1]]},  # a line, not a polygon
        {"label": "target", "points": []},
        {"label": "target", "points": square(2, 2, 6, 6)},
    ]
    assert target_mask(rasterize(shapes, 20, 20)).sum() > 0


def test_no_shapes_yields_an_empty_target():
    """Four val samples genuinely have no shapes; they must not crash, and they
    are scored as empty targets."""
    mask = rasterize([], 20, 20)
    assert not target_mask(mask).any()
    assert not ignore_mask(mask).any()


def test_canvas_size_comes_from_the_caller_not_the_polygons():
    mask = rasterize([{"label": "target", "points": square(2, 2, 6, 6)}], 50, 80)
    assert mask.shape == (50, 80)
