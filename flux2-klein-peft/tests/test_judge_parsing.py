"""The DreamBench++ judge's score parser.

The rubric text itself enumerates the 0-4 scale, so a judge that restates part
of the rubric before answering is the realistic failure mode -- taking the
first match would silently record a 0 for a good generation.
"""

from __future__ import annotations

import pytest

from dptlab.eval.metrics.dreambench_judge import _parse_score


def test_plain_score():
    assert _parse_score("Score: 3") == 3.0


def test_bracketed_score():
    assert _parse_score("Analysis: close match.\nScore: [4]") == 4.0


def test_takes_the_last_score_not_the_first():
    reply = "Recall the scale: Score: 0 means no resemblance.\nAnalysis: strong.\nScore: 3"
    assert _parse_score(reply) == 3.0


def test_unparseable_reply_raises():
    with pytest.raises(ValueError, match="no parseable score"):
        _parse_score("I cannot evaluate these images.")
