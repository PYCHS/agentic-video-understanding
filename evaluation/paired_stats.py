"""Paired significance utilities for matched Video-MME evaluations.

Inputs are per-question correctness vectors from real evaluation runs.
This module intentionally contains no example or placeholder measurements.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairedStats:
    accuracy_delta: float
    ci_low: float
    ci_high: float
    discordant_a_only: int
    discordant_b_only: int
    mcnemar_p_value: float


def _validate(a: Sequence[bool], b: Sequence[bool]) -> None:
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("correctness vectors must be non-empty and equally sized")
    if any(type(value) is not bool for vector in (a, b) for value in vector):
        raise ValueError("correctness values must be bool, not scores or labels")


def paired_bootstrap_ci(
    baseline: Sequence[bool],
    adaptive: Sequence[bool],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 128,
) -> tuple[float, float]:
    """Bootstrap a CI for adaptive accuracy minus baseline accuracy."""
    _validate(baseline, adaptive)
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(confidence) not in (int, float) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    diffs = [int(y) - int(x) for x, y in zip(baseline, adaptive)]
    n = len(diffs)
    rng = random.Random(seed)
    draws = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )

    alpha = (1.0 - confidence) / 2.0
    lo = max(0, min(samples - 1, int(math.floor(alpha * samples))))
    hi = max(0, min(samples - 1, int(math.ceil((1.0 - alpha) * samples)) - 1))
    return draws[lo], draws[hi]


def exact_mcnemar_p_value(
    baseline: Sequence[bool], adaptive: Sequence[bool]
) -> tuple[int, int, float]:
    """Return discordant counts and the two-sided exact McNemar p-value."""
    _validate(baseline, adaptive)
    baseline_only = sum(x and not y for x, y in zip(baseline, adaptive))
    adaptive_only = sum(y and not x for x, y in zip(baseline, adaptive))
    n = baseline_only + adaptive_only
    if n == 0:
        return baseline_only, adaptive_only, 1.0

    k = min(baseline_only, adaptive_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return baseline_only, adaptive_only, min(1.0, 2.0 * tail)


def summarize_paired_results(
    baseline: Sequence[bool],
    adaptive: Sequence[bool],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 128,
) -> PairedStats:
    """Compute draft paired accuracy statistics; the protocol is not yet frozen."""
    _validate(baseline, adaptive)
    delta = sum(int(y) - int(x) for x, y in zip(baseline, adaptive)) / len(baseline)
    ci_low, ci_high = paired_bootstrap_ci(
        baseline, adaptive, samples=bootstrap_samples, seed=seed
    )
    a_only, b_only, p_value = exact_mcnemar_p_value(baseline, adaptive)
    return PairedStats(delta, ci_low, ci_high, a_only, b_only, p_value)
