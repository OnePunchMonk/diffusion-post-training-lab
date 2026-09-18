"""Polygon rasterization: the ground-truth masks the whole replication is
scored against. If these are wrong, every downstream number is wrong in a way
that looks like a modelling result."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from asvl.data.reasonseg import rasterize


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_single_target_polygon():
    mask = rasterize([{"label": "target", "points": square(0, 0, 10, 10)}], 20, 20)
    assert mask[:10, :10].all()
    assert not mask[15:, 15:].any()


def test_multiple_targets_union_into_one_object():
    """Disconnected target polygons are one object, not several.

    ReasonSeg targets are often split by an occluder. Treating each polygon as
    its own object would change both the mask and the sample count.
    """
    shapes = [
        {"label": "target", "points": square(0, 0, 4, 4)},
        {"label": "target", "points": square(10, 10, 14, 14)},
    ]
    mask = rasterize(shapes, 20, 20)
    assert mask[1, 1] and mask[12, 12]


def test_ignore_regions_are_subtracted_not_added():
    """The failure mode this guards against inflates the ground truth, which
    *raises* IoU for an over-segmenting model and makes a broken replication
    look closer to the paper than it is."""
    shapes = [
        {"label": "target", "points": square(0, 0, 10, 10)},
        {"label": "ignore", "points": square(0, 0, 5, 5)},
    ]
    mask = rasterize(shapes, 20, 20)
    assert not mask[2, 2], "ignore region leaked into the target"
    assert mask[7, 7], "ignore region removed too much"


def test_degenerate_shapes_are_skipped():
    shapes = [
        {"label": "target", "points": [[0, 0], [1, 1]]},  # a line, not a polygon
        {"label": "target", "points": []},
        {"label": "target", "points": square(0, 0, 4, 4)},
    ]
    assert rasterize(shapes, 20, 20).sum() > 0


def test_canvas_size_comes_from_the_caller_not_the_polygons():
    """Annotations carry no imageHeight/imageWidth in this release, so the size
    must come from the image file. Deriving it from the polygon extent would
    crop every mask to its own bounding box."""
    mask = rasterize([{"label": "target", "points": square(2, 2, 6, 6)}], 50, 80)
    assert mask.shape == (50, 80)
    assert mask.sum() == pytest.approx(16, abs=8)


def test_mask_is_boolean():
    mask = rasterize([{"label": "target", "points": square(0, 0, 4, 4)}], 10, 10)
    assert mask.dtype == np.bool_
