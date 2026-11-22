import pytest

from cmbench.metrics import (
    bootstrap_mean_ci,
    gpu_cost_usd,
    median,
    paired_mean_difference,
    wilson_interval,
)


def test_wilson_interval_perfect_score() -> None:
    interval = wilson_interval(10, 10)
    assert interval.point == 1.0
    assert interval.high == pytest.approx(1.0)
    assert interval.low < 1.0  # Wilson never collapses to a point at n=10


def test_wilson_interval_zero_score() -> None:
    interval = wilson_interval(0, 10)
    assert interval.point == 0.0
    assert interval.low == pytest.approx(0.0)
    assert interval.high > 0.0


def test_wilson_interval_empty_sample() -> None:
    interval = wilson_interval(0, 0)
    assert interval.point == 0.0
    assert interval.low == 0.0
    assert interval.high == 0.0


def test_wilson_interval_half() -> None:
    interval = wilson_interval(5, 10)
    assert interval.point == 0.5
    assert interval.low < 0.5 < interval.high


def test_wilson_unsupported_confidence_raises() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=0.80)


def test_bootstrap_mean_ci_contains_true_mean_for_constant_values() -> None:
    interval = bootstrap_mean_ci([42.0] * 20)
    assert interval.point == 42.0
    assert interval.low == pytest.approx(42.0)
    assert interval.high == pytest.approx(42.0)


def test_bootstrap_mean_ci_is_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0, 2.5, 3.3, 1.1, 7.0, 2.2]
    a = bootstrap_mean_ci(values, seed=42)
    b = bootstrap_mean_ci(values, seed=42)
    assert a == b


def test_bootstrap_mean_ci_single_value() -> None:
    interval = bootstrap_mean_ci([5.0])
    assert interval.point == 5.0
    assert interval.low == 5.0
    assert interval.high == 5.0


def test_bootstrap_mean_ci_empty() -> None:
    interval = bootstrap_mean_ci([])
    assert interval.point == 0.0


def test_median_even_and_odd() -> None:
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median([]) == 0.0


def test_gpu_cost_usd() -> None:
    assert gpu_cost_usd(3600, gpu_hourly_usd=0.50) == pytest.approx(0.50)
    assert gpu_cost_usd(1800, gpu_hourly_usd=0.50) == pytest.approx(0.25)


def test_paired_mean_difference() -> None:
    a = [10.0, 20.0, 30.0]
    b = [8.0, 18.0, 33.0]
    assert paired_mean_difference(a, b) == pytest.approx((2 + 2 - 3) / 3)


def test_paired_mean_difference_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_mean_difference([1.0, 2.0], [1.0])


def test_paired_mean_difference_empty() -> None:
    assert paired_mean_difference([], []) == 0.0
