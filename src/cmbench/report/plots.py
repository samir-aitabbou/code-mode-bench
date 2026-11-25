"""The 6 required figures (CLAUDE.md §6), plus 2 supplementary ones
specific to this project's T4/T4p story (an architecture diagram and a
context-limit scatter), in one shared publication style: STIX serif,
the colour-blind-safe Okabe-Ito palette (classic MCP = grey #9a9a9a,
Code Mode = blue #0072B2), 95% Wilson confidence intervals on success
rates, median + interquartile-range error bars on tokens/latency/GPU
time, and a hatched, white-filled bar wherever a cell's success rate is
below 50% -- its median comes mostly from aborted trials there, not
completed ones, so it is drawn but flagged as not a like-for-like
comparison (ANALYSE-CLAUDE.md point C).

Every color/statistic/drawing helper here is shared, on purpose, with
`scripts/paper_figures.py`: that script imports from this module
instead of redefining its own copies, so there is exactly one figure
style and one set of statistics in the repository, not two.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from cmbench.metrics import wilson_interval

matplotlib.use("Agg")

# T4p is a classic-only complementary probe (ANALYSE-CLAUDE.md point A,
# DEVIATIONS.md); it has no Code Mode counterpart, so every
# classic-vs-codemode comparison figure excludes it and only the
# success-rate and context-limit figures show it on its own.
PAGINATED_PROBE_TASK = "T4p"
CORE_TASKS = ["T1", "T2", "T3", "T4", "T5", "T6"]
TOOL_BLOAT_TASKS = ["T3", "T5"]
LOW_SUCCESS_THRESHOLD = 0.5  # below this, a cell's median is "not comparable" (point C)

CLASSIC = "#9a9a9a"
CODEMODE = "#0072B2"
FAILURE_TYPE_COLORS = {
    "wrong_answer": "#E69F00",
    "context_overflow": "#D55E00",
    "tool_error": "#CC79A7",
    "max_turns": "#F0E442",
    "provider_error": "#56B4E9",
    "timeout": "#999999",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "hatch.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    """(point, low, high) success rate with a 95% Wilson interval --
    the exact same statistic as `results/summary/by_cell.csv`
    (`cmbench.metrics.wilson_interval`), so a figure and its CSV never
    disagree."""
    ci = wilson_interval(successes, n)
    return ci.point, ci.low, ci.high


def save(fig: Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _median_iqr(values: pd.Series) -> tuple[float, float, float]:
    """(median, lower error, upper error) for an errorbar spanning the
    interquartile range -- these are per-trial distributions (tokens,
    latency, GPU time), not proportions, so IQR is the right spread,
    not a Wilson interval."""
    if values.empty:
        return 0.0, 0.0, 0.0
    med = float(values.median())
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    return med, max(0.0, med - q1), max(0.0, q3 - med)


def _not_comparable_handle() -> Rectangle:
    return Rectangle((0, 0), 1, 1, fc="white", ec="#555555", hatch="////", lw=0.6)


def _bar_with_hatch(
    ax: Axes, x: float, height: float, width: float, color: str, *, low_success: bool, **kw: Any
) -> None:
    if low_success:
        ax.bar(
            x,
            height,
            width,
            facecolor="white",
            edgecolor=color,
            hatch="////",
            linewidth=0.6,
            zorder=2,
            **kw,
        )
    else:
        ax.bar(x, height, width, facecolor=color, edgecolor="black", linewidth=0.5, zorder=2, **kw)


def _style_axes(ax: Axes) -> None:
    ax.yaxis.grid(True, linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)


def _grouped_median_iqr_figure(trials: pd.DataFrame, value_col: str, ylabel: str) -> Figure:
    """Shared layout for figures 1 (input tokens), 2 (GPU time) and 4
    (latency): one subplot per model, classic vs Code Mode grouped
    bars over T1-T6, median + IQR, hatched where success < 50%."""
    main = trials[(trials["tool_count"] == 12) & (trials["task_id"].isin(CORE_TASKS))]
    models = sorted(main["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 3.2))
    axes = axes if hasattr(axes, "__len__") else [axes]
    width = 0.36
    for ax, model_id in zip(axes, models, strict=True):
        x = np.arange(len(CORE_TASKS))
        for offset, mode, color in ((-1, "classic", CLASSIC), (1, "codemode", CODEMODE)):
            for i, task_id in enumerate(CORE_TASKS):
                cell = main[
                    (main["model_id"] == model_id)
                    & (main["mode"] == mode)
                    & (main["task_id"] == task_id)
                ]
                med, lo, hi = _median_iqr(cell[value_col])
                low_success = bool(len(cell) == 0 or cell["success"].mean() < LOW_SUCCESS_THRESHOLD)
                xi = i + offset * width / 2
                _bar_with_hatch(ax, xi, med, width, color, low_success=low_success)
                ax.errorbar(
                    [xi],
                    [med],
                    yerr=[[lo], [hi]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.6,
                    capsize=1.8,
                    zorder=3,
                )
        ax.set_xticks(list(x))
        ax.set_xticklabels(CORE_TASKS)
        ax.set_ylabel(ylabel)
        ax.set_title(model_id, loc="left", fontweight="bold")
        _style_axes(ax)
    handles = [
        Rectangle((0, 0), 1, 1, fc=CLASSIC, ec="black", lw=0.5),
        Rectangle((0, 0), 1, 1, fc=CODEMODE, ec="black", lw=0.5),
        _not_comparable_handle(),
    ]
    fig.legend(
        handles,
        ["Classic MCP", "Code Mode", "Success rate < 50% (not comparable)"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def fig_input_tokens_by_task(trials: pd.DataFrame, out_path: Path) -> None:
    """1. Median input (prompt) tokens per task, classic vs Code Mode,
    one subplot per model (main 12-tool series, T1-T6 only)."""
    save(_grouped_median_iqr_figure(trials, "prompt_tokens", "Median input tokens (IQR)"), out_path)


def fig_gpu_time_per_trial(trials: pd.DataFrame, out_path: Path) -> None:
    """2. Median GPU time per trial, classic vs Code Mode, one subplot
    per model (main 12-tool series, T1-T6 only)."""
    save(
        _grouped_median_iqr_figure(trials, "gpu_seconds", "Median GPU time per trial, s (IQR)"),
        out_path,
    )


def fig_success_rate_with_ci(trials: pd.DataFrame, out_path: Path) -> None:
    """3. Success rate with 95% Wilson CI, classic vs Code Mode, one
    subplot per model (main 12-tool series, T1-T6), plus T4p appended
    as a 7th, classic-only category (the paginated-invoices probe,
    ANALYSE-CLAUDE.md point A / REPORT.md §4.5)."""
    main = trials[(trials["tool_count"] == 12) & (trials["task_id"].isin(CORE_TASKS))]
    models = sorted(main["model_id"].unique())
    labels = [*CORE_TASKS, PAGINATED_PROBE_TASK]
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 3.2), sharey=True)
    axes = axes if hasattr(axes, "__len__") else [axes]
    width = 0.36
    for ax, model_id in zip(axes, models, strict=True):
        for mode, color, lab in (
            ("classic", CLASSIC, "Classic MCP"),
            ("codemode", CODEMODE, "Code Mode"),
        ):
            ps, los, his, xs = [], [], [], []
            for i, task_id in enumerate(CORE_TASKS):
                cell = main[
                    (main["model_id"] == model_id)
                    & (main["mode"] == mode)
                    & (main["task_id"] == task_id)
                ]
                p, lo, hi = wilson(int(cell["success"].sum()), len(cell))
                ps.append(p * 100)
                los.append((p - lo) * 100)
                his.append((hi - p) * 100)
                xs.append(i + (-1 if mode == "classic" else 1) * width / 2)
            if mode == "classic":
                probe = trials[
                    (trials["model_id"] == model_id) & (trials["task_id"] == PAGINATED_PROBE_TASK)
                ]
                p, lo, hi = wilson(int(probe["success"].sum()), len(probe))
                ps.append(p * 100)
                los.append((p - lo) * 100)
                his.append((hi - p) * 100)
                xs.append(len(CORE_TASKS) - width / 2)
            ax.bar(
                xs, ps, width, color=color, edgecolor="black", linewidth=0.5, label=lab, zorder=2
            )
            ax.errorbar(
                xs,
                ps,
                yerr=[los, his],
                fmt="none",
                ecolor="black",
                elinewidth=0.6,
                capsize=1.8,
                zorder=3,
            )
        ax.axvline(len(CORE_TASKS) - 0.5, color="black", linewidth=0.5, linestyle=":")
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 108)
        ax.set_title(model_id, loc="left", fontweight="bold")
        _style_axes(ax)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(
        loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, handlelength=1.2, columnspacing=1.2
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out_path)


def fig_latency(trials: pd.DataFrame, out_path: Path) -> None:
    """4. Median end-to-end latency per trial, classic vs Code Mode,
    one subplot per model (main 12-tool series, T1-T6 only)."""
    save(
        _grouped_median_iqr_figure(trials, "latency_total_s", "Median latency per trial, s (IQR)"),
        out_path,
    )


def fig_tool_count_effect(trials: pd.DataFrame, out_path: Path) -> None:
    """5. Effect of tool count (12 vs 60) on median input tokens, T3/T5
    only, one subplot per model, bars grouped by mode (lighter = 12
    tools)."""
    sub_all = trials[trials["task_id"].isin(TOOL_BLOAT_TASKS)]
    models = sorted(sub_all["model_id"].unique())
    combos = [(t, m) for t in TOOL_BLOAT_TASKS for m in ("classic", "codemode")]
    labels = [f"{t}\n{m}" for t, m in combos]
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 3.2))
    axes = axes if hasattr(axes, "__len__") else [axes]
    width = 0.36
    for ax, model_id in zip(axes, models, strict=True):
        for offset, tool_count, alpha in ((-1, 12, 0.55), (1, 60, 1.0)):
            for i, (task_id, mode) in enumerate(combos):
                cell = sub_all[
                    (sub_all["model_id"] == model_id)
                    & (sub_all["mode"] == mode)
                    & (sub_all["task_id"] == task_id)
                    & (sub_all["tool_count"] == tool_count)
                ]
                med, lo, hi = _median_iqr(cell["prompt_tokens"])
                color = CLASSIC if mode == "classic" else CODEMODE
                xi = i + offset * width / 2
                ax.bar(
                    xi,
                    med,
                    width,
                    color=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=2,
                )
                ax.errorbar(
                    [xi],
                    [med],
                    yerr=[[lo], [hi]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.6,
                    capsize=1.8,
                    zorder=3,
                )
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Median input tokens (IQR)")
        ax.set_title(model_id, loc="left", fontweight="bold")
        _style_axes(ax)
    handles = [
        Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.55, ec="black", lw=0.5),
        Rectangle((0, 0), 1, 1, fc="#777777", ec="black", lw=0.5),
    ]
    fig.legend(
        handles, ["12 tools", "60 tools"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out_path)


def fig_failure_breakdown(trials: pd.DataFrame, out_path: Path) -> None:
    """6. Failure-type breakdown (core 6-task design, both tool-count
    variants; T4p is excluded here and reported separately, REPORT.md
    §4.5), one subplot per model, stacked by failure type, split by
    mode."""
    core = trials[trials["task_id"].isin(CORE_TASKS)]
    failures = core[~core["success"]]
    models = sorted(core["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 3.0))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, model_id in zip(axes, models, strict=True):
        sub = failures[failures["model_id"] == model_id]
        pivot = sub.pivot_table(
            index="mode", columns="failure_type", values="trial_id", aggfunc="count", fill_value=0
        )
        modes = list(pivot.index)
        bottom = [0.0] * len(modes)
        for failure_type, color in FAILURE_TYPE_COLORS.items():
            if failure_type not in pivot.columns:
                continue
            vals = list(pivot[failure_type])
            ax.bar(
                modes, vals, bottom=bottom, color=color, edgecolor="black", linewidth=0.5, zorder=2
            )
            bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
        ax.set_ylabel("Failed trials (count)")
        ax.set_title(model_id, loc="left", fontweight="bold")
        _style_axes(ax)
    handles = [
        Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.5) for c in FAILURE_TYPE_COLORS.values()
    ]
    fig.legend(
        handles,
        list(FAILURE_TYPE_COLORS.keys()),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.2),
        handlelength=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    save(fig, out_path)


# --- supplementary figures (not part of CLAUDE.md §6, but reused by
# scripts/paper_figures.py via this module to avoid duplication) ------


def _request_size_from_detail(detail: object) -> int | None:
    """Extract the rejected request's token count from a vLLM/openai
    context-length error message. `detail` comes back as a float NaN
    (not None) for missing values once it has round-tripped through a
    pandas DataFrame/CSV, so this checks the type explicitly rather
    than relying on `detail or ""` (NaN is truthy)."""
    if not isinstance(detail, str):
        return None
    match = re.search(r"Input length \((\d+)\)", detail)
    return int(match.group(1)) if match else None


def fig_context_limit_scatter(trials: pd.DataFrame, out_path: Path) -> None:
    """8. Per-request size on T4 (classic, unpaginated tool), T4p
    (classic, paginated tool) and T4 (Code Mode), against the
    32,768-token context limit -- the concrete picture behind H7 and
    the T4p probe (REPORT.md §4.5)."""
    rng = np.random.default_rng(0)
    models = sorted(trials["model_id"].unique())
    markers = {models[0]: "o", models[1]: "s"} if len(models) >= 2 else {models[0]: "o"}
    cats = [
        ("T4", "classic", "T4, classic\n(unpaginated tool)"),
        (PAGINATED_PROBE_TASK, "classic", "T4p, classic\n(paginated tool)"),
        ("T4", "codemode", "T4, Code Mode"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    offsets = np.linspace(-0.12, 0.12, len(models)) if len(models) > 1 else [0.0]
    for i, (task_id, mode, _label) in enumerate(cats):
        for model_id, off in zip(models, offsets, strict=True):
            s = trials[
                (trials["model_id"] == model_id)
                & (trials["task_id"] == task_id)
                & (trials["mode"] == mode)
                & (trials["tool_count"] == 12)
            ]
            color = CODEMODE if mode == "codemode" else "#444444"
            if mode == "codemode":
                ys = (s["prompt_tokens"] / s["num_requests"]).to_numpy()
                facecolor = "white"
            else:
                overflow = s[s["failure_type"] == "context_overflow"]
                ys = np.array([_request_size_from_detail(d) or 32769 for d in overflow["detail"]])
                facecolor = color
            xs = i + off + rng.uniform(-0.05, 0.05, len(ys))
            ax.scatter(
                xs,
                ys / 1e3,
                marker=markers[model_id],
                s=16,
                facecolors=facecolor,
                edgecolors=color,
                linewidths=0.8,
                zorder=3,
            )
    ax.axhline(32.768, color="#D55E00", linestyle="--", linewidth=0.9)
    ax.text(
        len(cats) - 0.55,
        33.6,
        "context limit (32,768)",
        color="#D55E00",
        ha="right",
        va="bottom",
        fontsize=8.5,
    )
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[2] for c in cats])
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylabel("Tokens per request (x1,000)")
    ax.set_ylim(0, 56)
    _style_axes(ax)
    leg = [
        Line2D(
            [],
            [],
            marker=markers[m],
            ls="",
            mfc="#444444",
            mec="#444444",
            ms=4.5,
            label=f"{m}: rejected request (classic)",
        )
        for m in models
    ] + [
        Line2D(
            [],
            [],
            marker=markers[m],
            ls="",
            mfc="white",
            mec=CODEMODE,
            ms=4.5,
            label=f"{m}: mean tokens/request (Code Mode)",
        )
        for m in models
    ]
    ax.legend(
        handles=leg,
        loc="upper left",
        bbox_to_anchor=(0, -0.22),
        ncol=1,
        handletextpad=0.3,
        fontsize=7.5,
    )
    save(fig, out_path)


def fig_setup_diagram(out_path: Path) -> None:
    """7. Harness architecture: task prompt/grader, classic vs Code
    Mode inside the agent harness, the MCP server and reference
    database. A static diagram, no trial data."""

    def box(
        ax: Axes,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        sub: str | None = None,
        fc: str = "white",
    ) -> None:
        ax.add_patch(Rectangle((x, y), w, h, fill=True, fc=fc, ec="black", lw=0.8))
        if sub:
            ax.text(
                x + w / 2,
                y + h * 0.66,
                title,
                ha="center",
                va="center",
                fontsize=8.6,
                fontweight="bold",
            )
            ax.text(
                x + w / 2,
                y + h * 0.30,
                sub,
                ha="center",
                va="center",
                fontsize=7.6,
                linespacing=1.25,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=8.6,
                fontweight="bold",
            )

    def arrow(
        ax: Axes,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        both: bool = False,
        txt: str | None = None,
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "<|-|>" if both else "-|>",
                "lw": 0.8,
                "color": "black",
                "mutation_scale": 7,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        if txt:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 1.4,
                txt,
                fontsize=7.3,
                ha="center",
                va="center",
                style="italic",
            )

    fig, ax = plt.subplots(figsize=(10.0, 4.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47)
    ax.axis("off")
    box(ax, 1, 25, 15, 9, "Task prompt", "T1-T6, identical\nin both modes")
    box(ax, 1, 6, 15, 9, "Grader", "SQL ground truth,\nDB state for T4")
    box(
        ax,
        31,
        37.5,
        34,
        9.3,
        "Model server (vLLM)",
        "gpt-oss-20b or Qwen3.5-9B\n1x L40S-24Q vGPU, 32K-token context",
    )
    ax.add_patch(Rectangle((21, 1.5), 54, 32, fill=False, ec="black", lw=0.8, ls=(0, (4, 2))))
    ax.text(22.2, 31.6, "Agent harness", fontsize=8.6, fontweight="bold", va="center")
    box(
        ax,
        24,
        19.5,
        48,
        9,
        "(a) Classic mode",
        "12 tool schemas in the prompt - one tool call per turn\n"
        "full tool results appended to the conversation (<=15 turns)",
        fc="#f2f2f2",
    )
    box(ax, 24, 4, 48, 13, "", None, fc="#e8f1f8")
    ax.text(48, 14.8, "(b) Code Mode", ha="center", va="center", fontsize=8.6, fontweight="bold")
    box(ax, 25.5, 5.3, 13, 7, "execute_python", "single tool,\ntyped tools.py")
    box(ax, 42, 5.3, 13, 7, "Sandbox", "no network,\n120s, stdout <=8k chars")
    box(ax, 57.5, 5.3, 13.5, 7, "JSON-RPC bridge", "tool calls from\nthe script")
    arrow(ax, 38.5, 8.8, 42, 8.8)
    arrow(ax, 55, 8.8, 57.5, 8.8)
    box(ax, 81, 22, 18, 9, "MCP server", "12 tools (or 60)")
    box(ax, 81, 6, 18, 9, "SQLite CRM", "6 tables, seed 42,\nreset per trial")
    arrow(ax, 16, 29.5, 21, 29.5)
    arrow(ax, 21, 10.5, 16, 10.5, txt="answer")
    arrow(ax, 48, 33.5, 48, 37.5, both=True, txt="requests / responses (usage = tokens)")
    arrow(ax, 72, 24, 81, 26.5, both=True)
    arrow(ax, 70.5, 9.5, 81, 24, both=True)
    arrow(ax, 90, 22, 90, 15, both=True)
    save(fig, out_path)


def write_all(trials: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Generate the 6 figures required by CLAUDE.md §6 into `out_dir`.
    The 2 supplementary figures (setup diagram, context-limit scatter)
    are generated separately by `scripts/paper_figures.py`, which calls
    `fig_setup_diagram`/`fig_context_limit_scatter` directly."""
    outputs = {
        "input_tokens_by_task": out_dir / "01_input_tokens_by_task.png",
        "gpu_time_per_trial": out_dir / "02_gpu_time_per_trial.png",
        "success_rate_by_task": out_dir / "03_success_rate_by_task.png",
        "latency_by_task": out_dir / "04_latency_by_task.png",
        "tool_count_effect": out_dir / "05_tool_count_effect.png",
        "failure_breakdown": out_dir / "06_failure_breakdown.png",
    }
    fig_input_tokens_by_task(trials, outputs["input_tokens_by_task"])
    fig_gpu_time_per_trial(trials, outputs["gpu_time_per_trial"])
    fig_success_rate_with_ci(trials, outputs["success_rate_by_task"])
    fig_latency(trials, outputs["latency_by_task"])
    fig_tool_count_effect(trials, outputs["tool_count_effect"])
    fig_failure_breakdown(trials, outputs["failure_breakdown"])
    return outputs
