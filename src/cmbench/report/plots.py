"""The 6 required figures (CLAUDE.md §6), in the blog's visual style
(CLAUDE.md §7): 1600px-wide PNG, white background, Inter font (DejaVu
Sans fallback -- Inter isn't installed on this machine/CI), classic =
slate (#64748b), codemode = amber (#d97706), success = green
(#16a34a), failure = red (#dc2626), per-model colors from the 4-color
set (#7c3aed, #0d9488, ...), footer "samiraitabbou.hashnode.dev".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

CLASSIC = "#64748b"
CODEMODE = "#d97706"
SUCCESS = "#16a34a"
FAILURE = "#dc2626"
MODEL_COLORS = ["#7c3aed", "#0d9488", "#2563eb", "#db2777"]
FAILURE_TYPE_COLORS = {
    "wrong_answer": "#dc2626",
    "context_overflow": "#7c3aed",
    "tool_error": "#d97706",
    "max_turns": "#64748b",
    "provider_error": "#db2777",
    "timeout": "#2563eb",
}

plt.rcParams["font.family"] = "Inter" if "Inter" in {f.name for f in matplotlib.font_manager.fontManager.ttflist} else "DejaVu Sans"  # noqa: E501
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def _finish(fig: Figure, title: str, subtitle: str, out_path: Path) -> None:
    fig.suptitle(title, x=0.045, y=0.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.045, 0.915, subtitle, ha="left", fontsize=12, color="#64748b")
    fig.text(0.98, 0.015, "samiraitabbou.hashnode.dev", ha="right", fontsize=9, color="#94a3b8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)


def _grouped_bar(
    ax: Axes,
    tasks: list[str],
    classic_vals: list[float],
    codemode_vals: list[float],
    *,
    ylabel: str,
    annotate_fmt: str = "{:.0f}",
) -> None:
    x = range(len(tasks))
    width = 0.35
    bars_c = ax.bar(
        [i - width / 2 for i in x], classic_vals, width, label="classic", color=CLASSIC
    )
    bars_m = ax.bar(
        [i + width / 2 for i in x], codemode_vals, width, label="codemode", color=CODEMODE
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks)
    ax.set_ylabel(ylabel)
    for bars in (bars_c, bars_m):
        ax.bar_label(bars, fmt=annotate_fmt, fontsize=8, padding=2, color="#334155")
    ax.legend(frameon=False, loc="upper right")


def fig_input_tokens_by_task(cells: pd.DataFrame, out_path: Path) -> None:
    """1. Median input (prompt) tokens per task, classic vs codemode,
    one subplot per model (main 12-tool series only)."""
    main = cells[cells["tool_count"] == 12]
    models = sorted(main["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = main[main["model_id"] == model_id].set_index(["task_id", "mode"])
        tasks = sorted(sub.index.get_level_values("task_id").unique())
        classic_vals = [sub.loc[(t, "classic"), "median_prompt_tokens"] for t in tasks]
        codemode_vals = [sub.loc[(t, "codemode"), "median_prompt_tokens"] for t in tasks]
        _grouped_bar(ax, tasks, classic_vals, codemode_vals, ylabel="median input tokens")
        ax.set_title(model_id, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _finish(
        fig,
        "Input tokens per task",
        "Median prompt tokens, classic vs Code Mode -- main series, 12 tools",
        out_path,
    )


def fig_gpu_time_per_trial(cells: pd.DataFrame, out_path: Path) -> None:
    """2. Median GPU time per successful trial, classic vs codemode, one
    subplot per model (main 12-tool series only)."""
    main = cells[cells["tool_count"] == 12]
    models = sorted(main["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = main[main["model_id"] == model_id].set_index(["task_id", "mode"])
        tasks = sorted(sub.index.get_level_values("task_id").unique())
        classic_vals = [sub.loc[(t, "classic"), "median_gpu_seconds"] for t in tasks]
        codemode_vals = [sub.loc[(t, "codemode"), "median_gpu_seconds"] for t in tasks]
        _grouped_bar(
            ax,
            tasks,
            classic_vals,
            codemode_vals,
            ylabel="median GPU seconds",
            annotate_fmt="{:.1f}",
        )
        ax.set_title(model_id, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _finish(
        fig,
        "GPU time per trial",
        "Median GPU-seconds per trial (all trials; 0s for immediate failures) "
        "-- main series, 12 tools",
        out_path,
    )


def fig_success_rate_with_ci(cells: pd.DataFrame, out_path: Path) -> None:
    """3. Success rate with Wilson 95% CI, classic vs codemode, one
    subplot per model (main 12-tool series only)."""
    main = cells[cells["tool_count"] == 12]
    models = sorted(main["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = main[main["model_id"] == model_id].set_index(["task_id", "mode"])
        tasks = sorted(sub.index.get_level_values("task_id").unique())
        x = range(len(tasks))
        width = 0.35
        for offset, mode, color in ((-1, "classic", CLASSIC), (1, "codemode", CODEMODE)):
            vals = [sub.loc[(t, mode), "success_rate"] for t in tasks]
            lows = [sub.loc[(t, mode), "success_rate_low"] for t in tasks]
            highs = [sub.loc[(t, mode), "success_rate_high"] for t in tasks]
            err_low = [max(0.0, v - lo) for v, lo in zip(vals, lows, strict=True)]
            err_high = [max(0.0, hi - v) for v, hi in zip(vals, highs, strict=True)]
            xs = [i + offset * width / 2 for i in x]
            ax.bar(xs, vals, width, color=color, label=mode)
            ax.errorbar(xs, vals, yerr=[err_low, err_high], fmt="none", ecolor="#0f172a", capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("success rate")
        ax.set_title(model_id, fontsize=13)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _finish(
        fig,
        "Success rate by task",
        "Classic vs Code Mode, 95% Wilson confidence intervals -- main series, 12 tools, n=10/cell",
        out_path,
    )


def fig_latency(cells: pd.DataFrame, out_path: Path) -> None:
    """4. Median end-to-end latency per task, classic vs codemode, one
    subplot per model (main 12-tool series only)."""
    main = cells[cells["tool_count"] == 12]
    models = sorted(main["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = main[main["model_id"] == model_id].set_index(["task_id", "mode"])
        tasks = sorted(sub.index.get_level_values("task_id").unique())
        classic_vals = [sub.loc[(t, "classic"), "median_latency_s"] for t in tasks]
        codemode_vals = [sub.loc[(t, "codemode"), "median_latency_s"] for t in tasks]
        _grouped_bar(
            ax,
            tasks,
            classic_vals,
            codemode_vals,
            ylabel="median latency (s)",
            annotate_fmt="{:.0f}",
        )
        ax.set_title(model_id, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _finish(
        fig,
        "End-to-end latency per trial",
        "Median wall-clock seconds per trial (includes failures) -- main series, 12 tools",
        out_path,
    )


def fig_tool_count_effect(tool_bloat: pd.DataFrame, out_path: Path) -> None:
    """5. Effect of tool count (12 vs 60) on median input tokens, for
    T3/T5, one subplot per model, bars colored by mode."""
    models = sorted(tool_bloat["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = tool_bloat[tool_bloat["model_id"] == model_id]
        labels = [f"{row.task_id}\n{row.mode}" for row in sub.itertuples()]
        colors = [CLASSIC if row.mode == "classic" else CODEMODE for row in sub.itertuples()]
        x = range(len(labels))
        width = 0.35
        b12 = ax.bar(
            [i - width / 2 for i in x],
            sub["median_prompt_tokens_12tools"],
            width,
            label="12 tools",
            color=list(colors),
            alpha=0.55,
        )
        b60 = ax.bar(
            [i + width / 2 for i in x],
            sub["median_prompt_tokens_60tools"],
            width,
            label="60 tools",
            color=colors,
        )
        ax.bar_label(b12, fmt="{:.0f}", fontsize=7, padding=2, color="#334155")
        ax.bar_label(b60, fmt="{:.0f}", fontsize=7, padding=2, color="#334155")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("median input tokens")
        ax.set_title(model_id, fontsize=13)
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _finish(
        fig,
        "Effect of tool count (H6)",
        "Median input tokens, 12 vs 60 tool definitions in context -- T3/T5 only, lighter=12 tools",
        out_path,
    )


def fig_failure_breakdown(failures: pd.DataFrame, out_path: Path) -> None:
    """6. Failure-type breakdown, one subplot per model, bars colored by
    failure type, split by mode."""
    models = sorted(failures["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = failures[failures["model_id"] == model_id]
        pivot = sub.pivot_table(
            index="mode", columns="failure_type", values="count", aggfunc="sum", fill_value=0
        )
        modes = list(pivot.index)
        bottom = [0.0] * len(modes)
        for failure_type in sorted(pivot.columns):
            vals = list(pivot[failure_type])
            ax.bar(
                modes,
                vals,
                bottom=bottom,
                color=FAILURE_TYPE_COLORS.get(failure_type, "#94a3b8"),
                label=failure_type,
            )
            bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
        ax.set_ylabel("failed trials (count)")
        ax.set_title(model_id, fontsize=13)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _finish(
        fig,
        "Failure types",
        "Count of failed trials by failure type -- all 140 trials/model (both tool-count variants)",
        out_path,
    )


def write_all(
    cells: pd.DataFrame,
    tool_bloat: pd.DataFrame,
    failures: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Generate every figure into `out_dir`. Returns {name: path}."""
    outputs = {
        "input_tokens_by_task": out_dir / "01_input_tokens_by_task.png",
        "gpu_time_per_trial": out_dir / "02_gpu_time_per_trial.png",
        "success_rate_by_task": out_dir / "03_success_rate_by_task.png",
        "latency_by_task": out_dir / "04_latency_by_task.png",
        "tool_count_effect": out_dir / "05_tool_count_effect.png",
        "failure_breakdown": out_dir / "06_failure_breakdown.png",
    }
    fig_input_tokens_by_task(cells, outputs["input_tokens_by_task"])
    fig_gpu_time_per_trial(cells, outputs["gpu_time_per_trial"])
    fig_success_rate_with_ci(cells, outputs["success_rate_by_task"])
    fig_latency(cells, outputs["latency_by_task"])
    fig_tool_count_effect(tool_bloat, outputs["tool_count_effect"])
    fig_failure_breakdown(failures, outputs["failure_breakdown"])
    return outputs
