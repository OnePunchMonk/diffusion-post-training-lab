"""The replication verdict. This is the thing that decides whether step 1
passed, so its logic should not itself be guesswork."""

from __future__ import annotations

from asvl.upstream.pins import REASONSEG_VAL_7B, verdict


def test_exact_match_reproduces():
    out = verdict(67.20, 75.15)
    assert "REPRODUCED" in out and "NOT REPRODUCED" not in out


def test_small_drift_still_reproduces():
    """Decoding is not bitwise deterministic; demanding equality would fail on
    noise."""
    assert "NOT REPRODUCED" not in verdict(66.9, 75.4)


def test_large_gap_fails():
    assert "NOT REPRODUCED" in verdict(52.0, 60.0)


def test_giou_only_gap_points_at_small_objects():
    """A gIoU-only miss localizes to small objects, which is a ground-truth or
    resolution bug far more often than a model one."""
    out = verdict(REASONSEG_VAL_7B.giou - 5, REASONSEG_VAL_7B.ciou)
    assert "small objects" in out


def test_ciou_only_gap_points_at_large_failures():
    out = verdict(REASONSEG_VAL_7B.giou, REASONSEG_VAL_7B.ciou - 5)
    assert "large objects" in out


def test_both_off_points_away_from_the_mask_pipeline():
    out = verdict(REASONSEG_VAL_7B.giou - 5, REASONSEG_VAL_7B.ciou - 5)
    assert "data split" in out
