"""Regenerate every figure directly into results/figures/: the 6
required by CLAUDE.md §6 (via `cmbench.report.plots.write_all`, the
same function `cmbench report` calls) plus 2 supplementary figures
specific to this project's T4/T4p story (the harness architecture
diagram and the T4/T4p context-limit scatter). All colors, statistics
(Wilson CI, median+IQR) and drawing helpers live in
`cmbench.report.plots` -- this script only calls them, so there is one
figure style in the repository, not two.

Usage: uv run python scripts/paper_figures.py
"""

from __future__ import annotations

import pathlib

from cmbench.report import plots
from cmbench.report.aggregate import load_trials

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures"


def main() -> None:
    trials = load_trials(ROOT / "results" / "raw")
    outputs = plots.write_all(trials, OUT_DIR)
    plots.fig_setup_diagram(OUT_DIR / "07_setup_diagram.png")
    plots.fig_context_limit_scatter(trials, OUT_DIR / "08_context_limit.png")
    for name, path in outputs.items():
        print("ok", name, "->", path)
    print("ok setup_diagram ->", OUT_DIR / "07_setup_diagram.png")
    print("ok context_limit ->", OUT_DIR / "08_context_limit.png")


if __name__ == "__main__":
    main()
