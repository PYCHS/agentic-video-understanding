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
