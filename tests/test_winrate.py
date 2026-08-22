from PIL import Image

from dptlab.eval.metrics.winrate import compute_win_rate


def _blank(color):
    return Image.new("RGB", (8, 8), color)


def test_win_rate_all_b_wins():
    prompts = ["a", "b", "c"]
    images_a = [_blank((0, 0, 0))] * 3
    images_b = [_blank((255, 255, 255))] * 3

    # judge scores image_b higher every time regardless of prompt
    judge = lambda prompt, image: 1.0 if image.getpixel((0, 0)) == (255, 255, 255) else 0.0

    result = compute_win_rate(prompts, images_a, images_b, judge=judge)
    assert result.win_rate_b_over_a == 1.0
    assert result.n_pairs == 3
    assert result.ties == 0


def test_win_rate_ties_are_excluded_from_denominator():
    prompts = ["a"]
    images_a = [_blank((0, 0, 0))]
    images_b = [_blank((0, 0, 0))]
    judge = lambda prompt, image: 0.5  # identical score -> tie

    result = compute_win_rate(prompts, images_a, images_b, judge=judge)
    assert result.ties == 1
    import math

    assert math.isnan(result.win_rate_b_over_a)
