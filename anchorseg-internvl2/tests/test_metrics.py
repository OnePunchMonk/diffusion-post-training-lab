"""gIoU/cIoU behaviour, including the two conventions that are easy to get
wrong and impossible to notice from an aggregate number."""

from __future__ import annotations

import numpy as np
import pytest

from asvl.metrics.reasonseg import compute_scores, missing_predictions, single_iou


def box(h=10, w=10, y0=0, y1=5, x0=0, x1=5):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_perfect_prediction():
    assert single_iou(box(), box())[0] == pytest.approx(1.0)


def test_disjoint_prediction():
    assert single_iou(box(x0=0, x1=5), box(x0=5, x1=10))[0] == pytest.approx(0.0)


def test_half_overlap():
    # pred covers rows 0-5, target rows 0-10, same columns -> 50/100
    assert single_iou(box(y1=5), box(y1=10))[0] == pytest.approx(0.5)


def test_empty_prediction_on_empty_target_scores_one():
    """Correct abstention must not be punished.

    The reference does this explicitly: `acc_iou[union_i == 0] += 1.0`. Four of
    ReasonSeg val's 200 samples carry no shapes at all, so this convention is
    worth 2% of gIoU on the split the replication is judged on -- more than the
    tolerance itself.
    """
    empty = np.zeros((10, 10), dtype=bool)
    assert single_iou(empty, empty)[0] == pytest.approx(1.0)


def test_nonempty_prediction_on_empty_target_scores_zero():
    assert single_iou(box(), np.zeros((10, 10), dtype=bool))[0] == pytest.approx(0.0)


def test_shape_mismatch_is_an_error_not_a_silent_resize():
    """A prediction left at SAM's working resolution must fail loudly.

    Silently resizing here is how a whole evaluation ends up scored at the
    wrong resolution, which understates boundary error.
    """
    with pytest.raises(ValueError, match="shape mismatch"):
        single_iou(np.zeros((1024, 1024), dtype=bool), np.zeros((480, 640), dtype=bool))


def test_giou_and_ciou_diverge_on_size_imbalance():
    """The reason both are reported.

    One large object segmented perfectly and one small object missed entirely:
    gIoU says 0.5 (half the images failed), cIoU says ~0.99 (almost all the
    pixels were right). A model that only finds big things is flattered by
    cIoU, which is exactly what gIoU is there to catch.
    """
    targets = {"big": box(100, 100, 0, 100, 0, 100), "small": box(100, 100, 0, 2, 0, 2)}
    predictions = {"big": targets["big"], "small": np.zeros((100, 100), dtype=bool)}

    scores = compute_scores(predictions, targets)
    assert scores.giou == pytest.approx(0.5)
    assert scores.ciou > 0.99


def test_missing_prediction_counts_as_empty_not_skipped():
    """A truncated run must score lower than a complete one, not equal.

    If missing samples were skipped, a job that crashed after its easiest 50
    images would report a better number than one that finished.
    """
    targets = {"a": box(), "b": box()}
    complete = compute_scores({"a": box(), "b": box()}, targets)
    truncated = compute_scores({"a": box()}, targets)

    assert complete.giou == pytest.approx(1.0)
    assert truncated.giou == pytest.approx(0.5)
    assert missing_predictions({"a": box()}, targets) == ["b"]


def test_ignore_pixels_are_excluded_from_both_intersection_and_union():
    """The reference passes ignore_index=255; ignore is not background.

    A prediction that falls entirely inside the ignore region must be a no-op,
    not a false positive.
    """
    target = box(20, 20, 0, 5, 0, 5)
    ignore = box(20, 20, 10, 15, 10, 15)

    pred_clean = target.copy()
    pred_with_ignored_extra = target | ignore

    assert single_iou(pred_clean, target, ignore)[0] == pytest.approx(1.0)
    assert single_iou(pred_with_ignored_extra, target, ignore)[0] == pytest.approx(1.0)
    # Without the ignore mask the same prediction is punished as a false positive.
    assert single_iou(pred_with_ignored_extra, target)[0] < 0.6


def test_ignore_can_make_a_sample_empty_and_therefore_score_one():
    """If ignore covers the whole target, union is 0 and the sample scores 1.0
    rather than dividing by zero."""
    target = box(20, 20, 0, 5, 0, 5)
    assert single_iou(target, target, ignore=target)[0] == pytest.approx(1.0)


def test_ignore_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        single_iou(box(), box(), ignore=np.zeros((4, 4), dtype=bool))


def test_compute_scores_threads_ignores_through():
    targets = {"a": box(20, 20, 0, 5, 0, 5)}
    ignores = {"a": box(20, 20, 10, 15, 10, 15)}
    predictions = {"a": targets["a"] | ignores["a"]}

    assert compute_scores(predictions, targets, ignores=ignores).giou == pytest.approx(1.0)
    assert compute_scores(predictions, targets).giou < 0.6


def test_query_type_breakdown():
    """The paper splits short vs. long queries; long ones need the reasoning."""
    targets = {"s1": box(), "l1": box()}
    predictions = {"s1": box(), "l1": np.zeros((10, 10), dtype=bool)}
    types = {"s1": "short", "l1": "long"}

    scores = compute_scores(predictions, targets, query_types=types)
    assert scores.by_query_type["short"].giou == pytest.approx(1.0)
    assert scores.by_query_type["long"].giou == pytest.approx(0.0)
    assert scores.by_query_type["short"].n_samples == 1
