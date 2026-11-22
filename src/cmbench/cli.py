"""Command-line interface for cmbench.

Example: cmbench run --model gpt-oss-20b --mode codemode --task T3 --reps 10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import typer
import yaml

from cmbench.data.seed import generate_database
from cmbench.providers.openai_compat import OpenAICompatProvider
from cmbench.runner import (
    ResultsWriter,
    TrialSpec,
    load_completed_trial_ids,
    run_all_trials_for_model,
    run_trial,
)
from cmbench.tasks.loader import load_tasks

app = typer.Typer(help="cmbench: Code Mode vs classic MCP tool-calling benchmark")

DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")
DEFAULT_TEMPLATE_DB = Path("./data/crm.db")
DEFAULT_WORKING_DIR = Path("./data/working")
DEFAULT_RESULTS_DIR = Path("./results")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))


def _model_config(models_config: Path, model_id: str) -> dict[str, Any]:
    data = _load_yaml(models_config)
    for model in data["models"]:
        if model["id"] == model_id:
            return cast("dict[str, Any]", model)
    raise typer.BadParameter(f"model {model_id!r} not found in {models_config}")


def _build_provider(model_id: str, models_config: Path, base_url: str) -> OpenAICompatProvider:
    model = _model_config(models_config, model_id)
    return OpenAICompatProvider(base_url=base_url, model=model["served_model_name"])


@app.command()
def data(
    out: Path = typer.Option(DEFAULT_TEMPLATE_DB, help="output SQLite file"),
    seed: int = typer.Option(42, help="random seed"),
) -> None:
    """Generate the deterministic CRM reference database."""
    _setup_logging()
    generate_database(out, seed=seed)
    typer.echo(f"wrote {out}")


@app.command()
def run(
    model: str = typer.Option(..., help="model id from configs/models.yaml"),
    mode: str = typer.Option(..., help="'classic' or 'codemode'"),
    task: str = typer.Option(..., help="task id, e.g. T3"),
    reps: int = typer.Option(1, help="number of repetitions"),
    tool_count: int = typer.Option(12, help="12 (normal) or 60 (noise-tool variant)"),
    base_url: str = typer.Option("http://localhost:8000/v1", help="vLLM OpenAI-compatible URL"),
    models_config: Path = typer.Option(DEFAULT_MODELS_CONFIG),
    experiment_config: Path = typer.Option(DEFAULT_EXPERIMENT_CONFIG),
    template_db: Path = typer.Option(DEFAULT_TEMPLATE_DB),
    working_dir: Path = typer.Option(DEFAULT_WORKING_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
) -> None:
    """Run N repetitions of one (model, mode, task) cell. Used for smoke
    tests and ad-hoc debugging; the full series uses `bench`."""
    _setup_logging()
    provider = _build_provider(model, models_config, base_url)
    experiment = _load_yaml(experiment_config)
    tasks = load_tasks()
    writer = ResultsWriter(results_dir / "raw")
    working_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(reps):
        spec = TrialSpec(model, mode, task, tool_count, rep)
        record, transcript = run_trial(
            spec,
            provider,
            tasks,
            template_db,
            working_dir,
            max_turns=experiment["limits"]["max_turns"],
            max_code_exec_seconds=experiment["limits"]["max_code_exec_seconds"],
            gpu_hourly_usd=experiment.get("gpu_hourly_usd", 0.50),
        )
        writer.append(record, transcript)
        status = "OK" if record.success else f"FAIL ({record.failure_type})"
        typer.echo(
            f"{spec.trial_id}: {status} | tokens={record.total_tokens} "
            f"requests={record.num_requests} latency={record.latency_total_s:.1f}s"
        )


@app.command()
def smoke(
    model: str = typer.Option(..., help="model id from configs/models.yaml"),
    base_url: str = typer.Option("http://localhost:8000/v1"),
    models_config: Path = typer.Option(DEFAULT_MODELS_CONFIG),
    experiment_config: Path = typer.Option(DEFAULT_EXPERIMENT_CONFIG),
    template_db: Path = typer.Option(DEFAULT_TEMPLATE_DB),
    working_dir: Path = typer.Option(DEFAULT_WORKING_DIR),
) -> None:
    """1 trial of T1 in each mode -- a quick check that a freshly-served
    model actually works before running the full series."""
    _setup_logging()
    provider = _build_provider(model, models_config, base_url)
    experiment = _load_yaml(experiment_config)
    tasks = load_tasks()
    working_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for mode in ("classic", "codemode"):
        spec = TrialSpec(model, mode, "T1", 12, 0)
        record, _transcript = run_trial(
            spec,
            provider,
            tasks,
            template_db,
            working_dir,
            max_turns=experiment["limits"]["max_turns"],
            max_code_exec_seconds=experiment["limits"]["max_code_exec_seconds"],
        )
        status = "OK" if record.success else f"FAIL ({record.failure_type}: {record.detail})"
        typer.echo(f"[smoke] {model} {mode} T1: {status}")
        if not record.valid:
            typer.echo(f"[smoke] WARNING: trial marked invalid ({record.detail})")
        if not record.success or not record.valid:
            all_ok = False

    if not all_ok:
        raise typer.Exit(code=1)


@app.command()
def bench(
    model: str = typer.Option(..., help="model id from configs/models.yaml"),
    base_url: str = typer.Option("http://localhost:8000/v1"),
    models_config: Path = typer.Option(DEFAULT_MODELS_CONFIG),
    experiment_config: Path = typer.Option(DEFAULT_EXPERIMENT_CONFIG),
    template_db: Path = typer.Option(DEFAULT_TEMPLATE_DB),
    working_dir: Path = typer.Option(DEFAULT_WORKING_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
) -> None:
    """Run every planned trial for one model (resumable)."""
    _setup_logging()
    provider = _build_provider(model, models_config, base_url)
    experiment = _load_yaml(experiment_config)
    writer = ResultsWriter(results_dir / "raw")
    working_dir.mkdir(parents=True, exist_ok=True)

    run_all_trials_for_model(model, provider, experiment, template_db, working_dir, writer)


@app.command(name="serve-cmd")
def serve_cmd(
    model: str = typer.Option(..., help="model id from configs/models.yaml"),
    port: int = typer.Option(8000),
    models_config: Path = typer.Option(DEFAULT_MODELS_CONFIG),
) -> None:
    """Print the `vllm serve ...` command for one model, built from
    configs/models.yaml, so scripts/serve_vllm.sh never has to parse YAML
    in bash. Prints one shell-escaped token per line's worth of args,
    space-joined on stdout (no trailing newline inside tokens)."""
    cfg = _model_config(models_config, model)
    args = [
        "vllm",
        "serve",
        cfg["repo_id"],
        "--served-model-name",
        cfg["served_model_name"],
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(cfg["gpu_memory_utilization"]),
        "--max-model-len",
        str(cfg["max_model_len"]),
        "--no-enable-prefix-caching",
        # The harness only ever sends one request at a time (never
        # concurrent trials), so a small max-num-seqs avoids vLLM
        # over-committing VRAM to a KV-cache warm-up batch it will
        # never actually need (default 256 caused a CUDA OOM at
        # startup on this 24GB GPU).
        "--max-num-seqs",
        "16",
        # CUDA graph capture still OOM'd with max-num-seqs=16 (21.2 of
        # 23.76 GiB used, 28 MiB short). We don't need the throughput
        # CUDA graphs buy for single-sequence trials, so skip them.
        "--enforce-eager",
    ]
    if cfg.get("tokenizer"):
        args += ["--tokenizer", cfg["tokenizer"]]
    if cfg.get("trust_remote_code"):
        args.append("--trust-remote-code")
    if cfg.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
    if cfg.get("tool_call_parser"):
        args += ["--tool-call-parser", cfg["tool_call_parser"]]
    if cfg.get("reasoning_parser"):
        args += ["--reasoning-parser", cfg["reasoning_parser"]]
    args += [str(a) for a in cfg.get("vllm_extra_args", [])]
    typer.echo(" ".join(args))


@app.command(name="list-models")
def list_models(models_config: Path = typer.Option(DEFAULT_MODELS_CONFIG)) -> None:
    """Print each model id, one per line, in the order defined by its
    `order` field in configs/models.yaml. Used by scripts/run_all.sh."""
    data = _load_yaml(models_config)
    for model in sorted(data["models"], key=lambda m: m["order"]):
        typer.echo(model["id"])


@app.command()
def status(
    model: str = typer.Option(..., help="model id"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
) -> None:
    """Print how many trials have completed for a model."""
    writer = ResultsWriter(results_dir / "raw")
    done = load_completed_trial_ids(writer.jsonl_path(model))
    typer.echo(f"{model}: {len(done)} trials completed")


if __name__ == "__main__":
    app()
