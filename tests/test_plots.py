from pathlib import Path

import pandas as pd
import pytest

from cmbench.metrics import wilson_interval
from cmbench.report import plots

MODELS = ["fake-model-a", "fake-model-b"]
CORE_TASKS = plots.CORE_TASKS


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "trial_id": "x",
        "model_id": MODELS[0],
        "mode": "classic",
        "task_id": "T1",
        "tool_count": 12,
        "success": True,
        "failure_type": None,
        "detail": None,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "num_requests": 1,
        "latency_total_s": 1.0,
        "gpu_seconds": 1.0,
    }
    base.update(overrides)
    return base


def _synthetic_trials() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    trial = 0
    for model_id in MODELS:
        for task_id in CORE_TASKS:
            for mode in ("classic", "codemode"):
                for rep in range(4):
                    trial += 1
                    # gpt-oss-style: model A fails T4 in classic mode, so
                    # that cell exercises the "not comparable" hatching.
                    success = not (model_id == MODELS[0] and task_id == "T4" and mode == "classic")
                    rows.append(
                        _row(
                            trial_id=f"t{trial}",
                            model_id=model_id,
                            mode=mode,
                            task_id=task_id,
                            success=success,
                            failure_type=None if success else "context_overflow",
                            detail=None if success else "Input length (45925) exceeds",
                            prompt_tokens=100 + rep * 10,
                        )
                    )
            # tool-bloat variant, T3/T5 only
            if task_id in ("T3", "T5"):
                for mode in ("classic", "codemode"):
                    for rep in range(2):
                        trial += 1
                        rows.append(
                            _row(
                                trial_id=f"t{trial}",
                                model_id=model_id,
                                mode=mode,
                                task_id=task_id,
                                tool_count=60,
                                prompt_tokens=300 + rep * 10,
                            )
                        )
        # T4p: classic-only complementary probe, no codemode counterpart
        for _rep in range(3):
            trial += 1
            rows.append(
                _row(
                    trial_id=f"t{trial}",
                    model_id=model_id,
                    mode="classic",
                    task_id=plots.PAGINATED_PROBE_TASK,
                    success=False,
                    failure_type="context_overflow",
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trials() -> pd.DataFrame:
    return _synthetic_trials()


def test_wilson_matches_metrics_wilson_interval() -> None:
    ci = wilson_interval(7, 10)
    point, low, high = plots.wilson(7, 10)
    assert (point, low, high) == (ci.point, ci.low, ci.high)


def test_median_iqr_on_empty_series_is_zero() -> None:
    assert plots._median_iqr(pd.Series([], dtype=float)) == (0.0, 0.0, 0.0)


def test_median_iqr_computes_median_and_quartile_spread() -> None:
    med, lo, hi = plots._median_iqr(pd.Series([10.0, 20.0, 30.0, 40.0]))
    assert med == 25.0
    assert lo == pytest.approx(7.5)
    assert hi == pytest.approx(7.5)


def test_write_all_creates_exactly_the_6_required_figures(
    tmp_path: Path, trials: pd.DataFrame
) -> None:
    outputs = plots.write_all(trials, tmp_path)

    assert set(outputs) == {
        "input_tokens_by_task",
        "gpu_time_per_trial",
        "success_rate_by_task",
        "latency_by_task",
        "tool_count_effect",
        "failure_breakdown",
    }
    for path in outputs.values():
        assert path.exists() and path.stat().st_size > 0


def test_success_rate_figure_includes_t4p_without_crashing(
    tmp_path: Path, trials: pd.DataFrame
) -> None:
    out_path = tmp_path / "success.png"
    plots.fig_success_rate_with_ci(trials, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_supplementary_figures_render_without_trial_data(tmp_path: Path) -> None:
    diagram_path = tmp_path / "setup.png"
    plots.fig_setup_diagram(diagram_path)
    assert diagram_path.exists() and diagram_path.stat().st_size > 0


def test_context_limit_scatter_renders(tmp_path: Path, trials: pd.DataFrame) -> None:
    out_path = tmp_path / "context.png"
    plots.fig_context_limit_scatter(trials, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_request_size_from_detail_parses_input_length() -> None:
    assert plots._request_size_from_detail("Error: Input length (45925) exceeds max") == 45925
    assert plots._request_size_from_detail(None) is None
    assert plots._request_size_from_detail("no number here") is None


def test_request_size_from_detail_handles_nan_from_a_csv_round_trip() -> None:
    # A missing `detail` becomes a float NaN, not None, once it has gone
    # through a pandas DataFrame/CSV round-trip -- NaN is truthy, so a
    # naive `detail or ""` fallback would crash re.search on a float.
    assert plots._request_size_from_detail(float("nan")) is None
