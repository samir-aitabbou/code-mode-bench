"""Orchestrates benchmark trials: builds the planned trial list (shuffled,
fixed seed), resumes from `results/raw/*.jsonl` by trial id, runs each
trial against a fresh working database copy, grades it, and writes one
JSON line (+ a separate trace file) per trial.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cmbench.data.seed import reset_database
from cmbench.mcp_server.adapter import mcp_tools_to_specs
from cmbench.mcp_server.exceptions_compat import ToolExecutionError
from cmbench.mcp_server.server import build_server
from cmbench.metrics import gpu_cost_usd
from cmbench.modes.classic import run_classic_trial
from cmbench.modes.codemode import run_codemode_trial
from cmbench.providers.base import ChatMessage, Provider, UsageMissingError
from cmbench.tasks.graders import grade_task
from cmbench.tasks.loader import Task, load_tasks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a CRM assistant for a company's support and sales team. "
    "Answer the user's request precisely using the tools available to you. "
    "Only state facts you obtained through a tool call; never guess or make "
    "up data."
)

TRIAL_WALL_CLOCK_LIMIT_S = 600.0
FAILURE_TYPES = frozenset(
    {
        "wrong_answer",
        "tool_error",
        "code_error",
        "timeout",
        "max_turns",
        "context_overflow",
        "format_error",
    }
)
_CONTEXT_OVERFLOW_MARKERS = ("maximum context length", "context length", "too long for model")


@dataclass(frozen=True)
class TrialSpec:
    model_id: str
    mode: str  # "classic" | "codemode"
    task_id: str
    tool_count: int  # 12 or 60
    rep_index: int

    @property
    def trial_id(self) -> str:
        return (
            f"{self.model_id}__{self.mode}__{self.task_id}"
            f"__tools{self.tool_count}__rep{self.rep_index}"
        )


@dataclass
class TrialRecord:
    trial_id: str
    model_id: str
    mode: str
    task_id: str
    tool_count: int
    rep_index: int
    timestamp: str
    valid: bool
    success: bool
    failure_type: str | None
    num_requests: int
    num_tool_calls: int
    num_code_executions: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_total_s: float
    throughput_tokens_per_s: float
    gpu_seconds: float
    cost_usd_estimate: float
    final_answer_text: str | None
    expected: Any
    detail: str


def build_trial_specs(
    model_id: str,
    *,
    tasks: list[str],
    modes: list[str],
    main_repetitions: int,
    tool_bloat_tasks: list[str],
    tool_bloat_repetitions: int,
    seed: int,
) -> list[TrialSpec]:
    """Build every planned trial for one model: the main 12-tool series
    plus the 60-tool variant, then shuffle with a seed derived from the
    model id so the order differs per model but is reproducible."""
    specs: list[TrialSpec] = []
    for task_id in tasks:
        for mode in modes:
            for rep in range(main_repetitions):
                specs.append(TrialSpec(model_id, mode, task_id, 12, rep))
    for task_id in tool_bloat_tasks:
        for mode in modes:
            for rep in range(tool_bloat_repetitions):
                specs.append(TrialSpec(model_id, mode, task_id, 60, rep))

    rng = random.Random(f"{seed}-{model_id}")
    rng.shuffle(specs)
    return specs


def load_completed_trial_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    ids: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.add(json.loads(line)["trial_id"])
    return ids


def _unwrap_structured_content(structured_content: Any) -> Any:
    """MCP wraps a non-object tool return (list/str/int/...) as
    `{"result": value}`, since structured content must be a JSON object.
    None of this project's 12 real tools has a field literally named
    "result", so this heuristic is safe here."""
    if isinstance(structured_content, dict) and set(structured_content.keys()) == {"result"}:
        return structured_content["result"]
    return structured_content


def make_tool_executor(server: Any) -> Any:
    """Build a synchronous (tool_name, arguments) -> result callable that
    calls the given in-process MCP server, for use by both agent modes."""

    def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
        try:
            result = asyncio.run(server.call_tool(name, arguments))
        except ToolExecutionError as exc:
            # In-process call_tool() raises rather than returning an
            # is_error result (unlike a real stdio/HTTP MCP transport);
            # unwrap to the tool's own exception message.
            cause = exc.__cause__
            raise RuntimeError(str(cause) if cause else str(exc)) from exc
        if result.is_error:
            text = result.content[0].text if result.content else "tool error"
            raise RuntimeError(text)
        return _unwrap_structured_content(result.structured_content)

    return execute_tool


def _is_context_overflow(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CONTEXT_OVERFLOW_MARKERS)


def run_trial(
    spec: TrialSpec,
    provider: Provider,
    tasks: dict[str, Task],
    template_db_path: Path,
    working_db_dir: Path,
    *,
    max_turns: int = 15,
    max_code_exec_seconds: float = 120.0,
    gpu_hourly_usd: float = 0.50,
) -> tuple[TrialRecord, list[ChatMessage]]:
    """Run one trial end to end and return (record, transcript)."""
    working_db = working_db_dir / f"{spec.trial_id}.db"
    reset_database(template_db_path, working_db)

    server = build_server(working_db, with_noise_tools=spec.tool_count == 60)
    mcp_tools = asyncio.run(server.list_tools())
    tool_specs = mcp_tools_to_specs(mcp_tools)
    execute_tool = make_tool_executor(server)

    task = tasks[spec.task_id]
    task_prompt = task.build_prompt()

    def _invoke() -> Any:
        if spec.mode == "classic":
            return run_classic_trial(
                provider, SYSTEM_PROMPT, task_prompt, tool_specs, execute_tool, max_turns=max_turns
            )
        return run_codemode_trial(
            provider,
            SYSTEM_PROMPT,
            task_prompt,
            tool_specs,
            execute_tool,
            max_turns=max_turns,
            max_code_exec_seconds=max_code_exec_seconds,
        )

    timestamp = datetime.now(UTC).isoformat()
    start = time.monotonic()
    # Not a `with` block on purpose: ThreadPoolExecutor.__exit__ calls
    # shutdown(wait=True), which would block until the abandoned worker
    # thread finishes even after future.result() already raised
    # TimeoutError -- defeating the wall-clock limit entirely (observed:
    # a trial ran 100+ minutes past its 600s budget because of this).
    # shutdown(wait=False) lets this function return promptly; the
    # orphaned thread (and its still-running vLLM request) is abandoned.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_invoke)
        outcome = future.result(timeout=TRIAL_WALL_CLOCK_LIMIT_S)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        wall = time.monotonic() - start
        return (
            TrialRecord(
                trial_id=spec.trial_id,
                model_id=spec.model_id,
                mode=spec.mode,
                task_id=spec.task_id,
                tool_count=spec.tool_count,
                rep_index=spec.rep_index,
                timestamp=timestamp,
                valid=True,
                success=False,
                failure_type="timeout",
                num_requests=0,
                num_tool_calls=0,
                num_code_executions=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_total_s=wall,
                throughput_tokens_per_s=0.0,
                gpu_seconds=wall,
                cost_usd_estimate=gpu_cost_usd(wall, gpu_hourly_usd=gpu_hourly_usd),
                final_answer_text=None,
                expected=None,
                detail=f"trial exceeded the {TRIAL_WALL_CLOCK_LIMIT_S}s wall-clock limit",
            ),
            [],
        )
    except UsageMissingError as exc:
        pool.shutdown(wait=False)
        return (
            TrialRecord(
                trial_id=spec.trial_id,
                model_id=spec.model_id,
                mode=spec.mode,
                task_id=spec.task_id,
                tool_count=spec.tool_count,
                rep_index=spec.rep_index,
                timestamp=timestamp,
                valid=False,
                success=False,
                failure_type=None,
                num_requests=0,
                num_tool_calls=0,
                num_code_executions=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_total_s=0.0,
                throughput_tokens_per_s=0.0,
                gpu_seconds=0.0,
                cost_usd_estimate=0.0,
                final_answer_text=None,
                expected=None,
                detail=str(exc),
            ),
            [],
        )
    except Exception as exc:  # noqa: BLE001 -- classified below, never silently dropped
        pool.shutdown(wait=False)
        wall = time.monotonic() - start
        failure_type = "context_overflow" if _is_context_overflow(exc) else "tool_error"
        return (
            TrialRecord(
                trial_id=spec.trial_id,
                model_id=spec.model_id,
                mode=spec.mode,
                task_id=spec.task_id,
                tool_count=spec.tool_count,
                rep_index=spec.rep_index,
                timestamp=timestamp,
                valid=True,
                success=False,
                failure_type=failure_type,
                num_requests=0,
                num_tool_calls=0,
                num_code_executions=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_total_s=wall,
                throughput_tokens_per_s=0.0,
                gpu_seconds=wall,
                cost_usd_estimate=gpu_cost_usd(wall, gpu_hourly_usd=gpu_hourly_usd),
                final_answer_text=None,
                expected=None,
                detail=str(exc),
            ),
            [],
        )
    pool.shutdown(wait=False)

    latency_total = sum(outcome.per_request_latencies)
    total_tokens = outcome.total_usage.total_tokens
    throughput = (outcome.total_usage.completion_tokens / latency_total) if latency_total else 0.0
    num_code_executions = getattr(outcome, "num_code_executions", 0)
    had_error = outcome.had_error

    final_failure_type: str | None
    if outcome.stopped_reason == "max_turns":
        success = False
        if had_error:
            final_failure_type = "code_error" if spec.mode == "codemode" else "tool_error"
        else:
            final_failure_type = "max_turns"
        expected = None
        detail = ""
    else:
        grade = grade_task(spec.task_id, template_db_path, working_db, outcome.final_answer_text)
        success = grade.correct
        final_failure_type = grade.failure_type
        expected = grade.expected
        detail = grade.detail

    record = TrialRecord(
        trial_id=spec.trial_id,
        model_id=spec.model_id,
        mode=spec.mode,
        task_id=spec.task_id,
        tool_count=spec.tool_count,
        rep_index=spec.rep_index,
        timestamp=timestamp,
        valid=True,
        success=success,
        failure_type=final_failure_type,
        num_requests=outcome.num_requests,
        num_tool_calls=outcome.num_tool_calls,
        num_code_executions=num_code_executions,
        prompt_tokens=outcome.total_usage.prompt_tokens,
        completion_tokens=outcome.total_usage.completion_tokens,
        total_tokens=total_tokens,
        latency_total_s=latency_total,
        throughput_tokens_per_s=throughput,
        gpu_seconds=latency_total,
        cost_usd_estimate=gpu_cost_usd(latency_total, gpu_hourly_usd=gpu_hourly_usd),
        final_answer_text=outcome.final_answer_text,
        expected=expected,
        detail=detail,
    )
    return record, outcome.transcript


@dataclass
class ResultsWriter:
    """Appends trial records/traces, one file per model so a partial run
    is easy to resume (CLAUDE.md §0.8)."""

    raw_dir: Path
    traces_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.traces_dir = self.raw_dir / "traces"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def jsonl_path(self, model_id: str) -> Path:
        return self.raw_dir / f"{model_id}.jsonl"

    def append(self, record: TrialRecord, transcript: list[ChatMessage]) -> None:
        with self.jsonl_path(record.model_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        trace_path = self.traces_dir / f"{record.trial_id}.jsonl"
        with trace_path.open("w", encoding="utf-8") as f:
            for message in transcript:
                f.write(json.dumps(asdict(message)) + "\n")


def run_all_trials_for_model(
    model_id: str,
    provider: Provider,
    experiment_config: dict[str, Any],
    template_db_path: Path,
    working_db_dir: Path,
    writer: ResultsWriter,
    *,
    tasks_path: Path | None = None,
) -> None:
    """Run every planned (and not-yet-completed) trial for one model,
    writing each result as it finishes."""
    tasks = load_tasks(tasks_path) if tasks_path else load_tasks()
    seed = experiment_config["seed"]
    main = experiment_config["main_series"]
    bloat = experiment_config["tool_bloat_variant"]
    specs = build_trial_specs(
        model_id,
        tasks=main["tasks"],
        modes=experiment_config["modes"],
        main_repetitions=main["repetitions"],
        tool_bloat_tasks=bloat["tasks"],
        tool_bloat_repetitions=bloat["repetitions"],
        seed=seed,
    )
    done = load_completed_trial_ids(writer.jsonl_path(model_id))
    gpu_hourly_usd = experiment_config.get("gpu_hourly_usd", 0.50)
    max_turns = experiment_config["limits"]["max_turns"]
    max_code_exec_seconds = experiment_config["limits"]["max_code_exec_seconds"]

    for spec in specs:
        if spec.trial_id in done:
            logger.info("skipping already-completed trial %s", spec.trial_id)
            continue
        logger.info("running trial %s", spec.trial_id)
        record, transcript = run_trial(
            spec,
            provider,
            tasks,
            template_db_path,
            working_db_dir,
            max_turns=max_turns,
            max_code_exec_seconds=max_code_exec_seconds,
            gpu_hourly_usd=gpu_hourly_usd,
        )
        writer.append(record, transcript)
