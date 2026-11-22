"""Statistics used in the report: bootstrap confidence intervals for
means, Wilson intervals for success rates, and GPU-time cost estimates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion. Preferred over the
    normal approximation for small n or rates near 0/1 (common here: 10
    reps per cell)."""
    if n == 0:
        return Interval(point=0.0, low=0.0, high=0.0)
    z = _z_score(confidence)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    low = (center - margin) / denom
    high = (center + margin) / denom
    return Interval(point=phat, low=max(0.0, low), high=min(1.0, high))


def _z_score(confidence: float) -> float:
    # Covers the confidence levels this project actually uses; avoids a
    # scipy dependency for one lookup.
    table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if confidence not in table:
        raise ValueError(f"unsupported confidence level {confidence}; use one of {list(table)}")
    return table[confidence]


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    if not values:
        return Interval(point=0.0, low=0.0, high=0.0)
    if len(values) == 1:
        return Interval(point=values[0], low=values[0], high=values[0])

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = 1 - confidence
    low_idx = int((alpha / 2) * n_resamples)
    high_idx = int((1 - alpha / 2) * n_resamples) - 1
    point = sum(values) / n
    return Interval(point=point, low=means[low_idx], high=means[min(high_idx, n_resamples - 1)])


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def gpu_cost_usd(gpu_seconds: float, *, gpu_hourly_usd: float = 0.50) -> float:
    """Estimated cost, always presented as an estimate (PLAN.md §7)."""
    return gpu_seconds / 3600.0 * gpu_hourly_usd


def paired_mean_difference(a: list[float], b: list[float]) -> float:
    """Mean of (a_i - b_i) for a paired comparison (e.g. codemode vs
    classic, same task, same repetition index). Both lists must be the
    same length and already aligned by the caller."""
    if len(a) != len(b):
        raise ValueError("paired lists must have the same length")
    if not a:
        return 0.0
    return sum(x - y for x, y in zip(a, b, strict=True)) / len(a)
