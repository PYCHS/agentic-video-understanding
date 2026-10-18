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
