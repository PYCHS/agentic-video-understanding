"""Tests for paired Video-MME significance utilities."""

import pytest

from evaluation.paired_stats import (
    exact_mcnemar_p_value,
    paired_bootstrap_ci,
    summarize_paired_results,
)


def test_rejects_empty_or_mismatched_vectors():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        exact_mcnemar_p_value([True], [True, False])


def test_bootstrap_seed_is_reproducible():
    baseline = [True, False, True, False, False, True]
    adaptive = [True, True, False, True, False, True]

    first = paired_bootstrap_ci(baseline, adaptive, samples=500, seed=17)
    second = paired_bootstrap_ci(baseline, adaptive, samples=500, seed=17)

    assert first == second


def test_mcnemar_no_discordance_returns_one():
    baseline = [True, False, True, False]
    adaptive = baseline.copy()

    baseline_only, adaptive_only, p_value = exact_mcnemar_p_value(
        baseline, adaptive
    )

    assert (baseline_only, adaptive_only) == (0, 0)
    assert p_value == 1.0


def test_mcnemar_matches_exact_two_sided_binomial_tail():
    # Four discordant pairs all favor adaptive:
    # 2 * P[Binomial(4, 0.5) <= 0] = 2 / 16 = 0.125.
    baseline = [False, False, False, False]
    adaptive = [True, True, True, True]

    baseline_only, adaptive_only, p_value = exact_mcnemar_p_value(
        baseline, adaptive
    )

    assert (baseline_only, adaptive_only) == (0, 4)
    assert p_value == pytest.approx(0.125)


def test_summary_reports_adaptive_minus_baseline_accuracy():
    baseline = [True, False, False, True]
    adaptive = [True, True, False, True]

    stats = summarize_paired_results(
        baseline, adaptive, bootstrap_samples=200, seed=3
    )

    assert stats.accuracy_delta == pytest.approx(0.25)
    assert stats.discordant_a_only == 0
    assert stats.discordant_b_only == 1
    assert stats.ci_low <= stats.accuracy_delta <= stats.ci_high


@pytest.mark.parametrize("bad", [0, 1, .5, "False", None, float("nan")])
@pytest.mark.parametrize("function", [paired_bootstrap_ci, exact_mcnemar_p_value,
                                     summarize_paired_results])
def test_rejects_non_boolean_correctness(bad, function):
    for baseline, adaptive in [([bad], [True]), ([False], [bad])]:
        with pytest.raises(ValueError, match="must be bool"):
            function(baseline, adaptive)


@pytest.mark.parametrize("settings", [
    {"samples": True}, {"samples": 1.5}, {"samples": 0},
    {"seed": None}, {"seed": True}, {"seed": "128"},
    {"confidence": True}, {"confidence": "0.95"},
    {"confidence": float("nan")}, {"confidence": 1},
])
def test_rejects_invalid_bootstrap_settings(settings):
    with pytest.raises(ValueError):
        paired_bootstrap_ci([False], [True], **settings)


def test_default_seed_matches_documented_protocol():
    baseline = [True, False, False, True]
    adaptive = [False, True, True, True]
    assert paired_bootstrap_ci(baseline, adaptive) == paired_bootstrap_ci(
        baseline, adaptive, seed=128)
    assert summarize_paired_results(baseline, adaptive) == summarize_paired_results(
        baseline, adaptive, seed=128)


def test_video_bootstrap_preserves_clusters_and_question_weighting(monkeypatch):
    from evaluation.paired_stats import paired_video_bootstrap_ci
    # Exhaust all ordered draws of two videos: AA, AB, BA, BB.
    # A has three wins; B has one loss. Mixed draws must be +0.5, not zero.
    class Exhaustive:
        def __init__(self, seed):
            self.indices = iter([0, 0, 0, 1, 1, 0, 1, 1])
        def randrange(self, n):
            assert n == 2
            return next(self.indices)
    monkeypatch.setattr("evaluation.paired_stats.random.Random", Exhaustive)
    result = paired_video_bootstrap_ci([False]*3+[True], [True]*3+[False],
                                      ["A"]*3+["B"], samples=4, confidence=.5)
    assert result.accuracy_delta == .5
    assert (result.ci_low, result.ci_high) == (.5, .5)
    assert (result.video_count, result.question_count) == (2, 4)


def test_video_bootstrap_is_reproducible_after_row_reordering():
    from evaluation.paired_stats import paired_video_bootstrap_ci
    a, b, ids = [True, False, False], [False, True, True], ["z", "a", "z"]
    result = paired_video_bootstrap_ci(a, b, ids, samples=100)
    assert result == paired_video_bootstrap_ci(a[::-1], b[::-1], ids[::-1], samples=100)
    swapped = paired_video_bootstrap_ci(b, a, ids, samples=100)
    assert swapped.accuracy_delta == -result.accuracy_delta
    assert (swapped.ci_low, swapped.ci_high) == (-result.ci_high, -result.ci_low)


@pytest.mark.parametrize("ids", [["one", "one"], ["one"], ["one", ""], ["one", None]])
def test_video_bootstrap_rejects_unusable_clusters(ids):
    from evaluation.paired_stats import paired_video_bootstrap_ci
    with pytest.raises(ValueError):
        paired_video_bootstrap_ci([True, False], [False, True], ids)
