# code-mode-bench

> **Status: work in progress.** This README is a placeholder filled in as the
> benchmark is built; see `PROGRESS.md` for the current state. The full
> professional README (architecture diagram, results, citation, etc.) is
> written at the end of the project, per `../CLAUDE.md` §8.

Benchmark comparing **Code Mode** (LLM writes Python against a typed API,
executed in a sandbox) against **classic MCP tool-calling** (LLM calls MCP
tools one at a time), on 4 open-weight LLMs served locally with vLLM on a
single 24 GB GPU.

See `../PLAN.md` for the full experimental design.
