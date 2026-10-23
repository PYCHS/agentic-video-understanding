"""Independent SciPy reference checks; synthetic tables, not study results."""
import pytest

scipy_stats = pytest.importorskip("scipy.stats")
from evaluation.paired_stats import exact_mcnemar_p_value


@pytest.mark.parametrize("discordant", [1, 2, 3, 10, 20, 50, 100])
def test_mcnemar_matches_scipy_for_every_split(discordant):
    for baseline_only in range(discordant + 1):
        adaptive_only = discordant - baseline_only
        # Concordant successes/failures must not change the discordant test.
        baseline = [True]*baseline_only + [False]*adaptive_only + [True, False]
        adaptive = [False]*baseline_only + [True]*adaptive_only + [True, False]
        a, b, actual = exact_mcnemar_p_value(baseline, adaptive)
        expected = scipy_stats.binomtest(baseline_only, discordant, p=.5,
                                        alternative="two-sided").pvalue
        assert (a, b) == (baseline_only, adaptive_only)
        assert actual == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("baseline,adaptive", [
    ([True, False, False], [False, True, False]),
    ([False, False, True], [True, True, True]),
    ([True, False, True], [True, False, True]),
])
def test_paired_bootstrap_against_exhaustive_numpy_reference(monkeypatch, baseline, adaptive):
    from itertools import product
    import numpy as np
    from evaluation.paired_stats import paired_bootstrap_ci

    # Enumerate every ordered resample. Keep the same indices for both methods.
    indices = np.array(list(product(range(3), repeat=3)))
    baseline_draws = np.array(baseline, dtype=float)[indices].mean(axis=1)
    adaptive_draws = np.array(adaptive, dtype=float)[indices].mean(axis=1)
    reference = adaptive_draws - baseline_draws
    # 27 draws and 90% confidence avoid integer-rank boundary conventions.
    expected = np.quantile(reference, [.05, .95], method="inverted_cdf")

    class Exhaustive:
        def __init__(self, seed):
            self.indices = iter(indices.ravel().tolist())
        def randrange(self, n):
            assert n == 3
            return next(self.indices)

    monkeypatch.setattr("evaluation.paired_stats.random.Random", Exhaustive)
    actual = paired_bootstrap_ci(baseline, adaptive, samples=27, confidence=.9)
    assert actual == pytest.approx(expected)
