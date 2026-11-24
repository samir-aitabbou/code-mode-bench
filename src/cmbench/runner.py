"""Orchestrates benchmark trials: builds the planned trial list (shuffled,
fixed seed), resumes from `results/raw/*.jsonl` by trial id, runs each
trial against a fresh working database copy, grades it, and writes one
JSON line (+ a separate trace file) per trial.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
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
from cmbench.modes.common import TrialAborted
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

# Default when not overridden by configs/experiment.yaml's
# limits.max_trial_seconds (see run_all_trials_for_model).
TRIAL_WALL_CLOCK_LIMIT_S = 900.0
FAILURE_TYPES = frozenset(
    {
        "wrong_answer",
        "tool_error",
        "code_error",
        "provider_error",
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
    # A snapshot taken right after the trial finishes, so a shared VM
    # (another user's own GPU job, seen in the wild 2026-11-23) can be
    # told apart from this harness's own load when reading results:
    # `gpu_compute_procs` > 1 means something other than our own vLLM
    # server was also on the GPU during (or just after) this trial.
    # None when nvidia-smi isn't available (e.g. the fake-provider CI
    # tests) -- never fabricated.
    gpu_util_pct: int | None = None
    gpu_mem_used_mib: int | None = None
    gpu_mem_free_mib: int | None = None
    gpu_compute_procs: int | None = None


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


def _gpu_snapshot() -> dict[str, int] | None:
    """Best-effort `nvidia-smi` snapshot (utilization/memory/compute-process
    count). Returns None on any failure (no GPU, no nvidia-smi, parse
    error, ...) -- never raises, so this can't break a trial."""
    try:
        usage = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        util_pct, mem_used, mem_free = (int(x.strip()) for x in usage.split(","))
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        num_procs = len([line for line in procs.splitlines() if line.strip()])
    except Exception:  # noqa: BLE001 -- a snapshot is a nice-to-have, never a trial blocker
        return None
    return {
        "gpu_util_pct": util_pct,
        "gpu_mem_used_mib": mem_used,
        "gpu_mem_free_mib": mem_free,
        "gpu_compute_procs": num_procs,
    }


def _attach_gpu_snapshot(record: TrialRecord) -> TrialRecord:
    snapshot = _gpu_snapshot()
    if snapshot is None:
        return record
    return replace(
        record,
        gpu_util_pct=snapshot["gpu_util_pct"],
        gpu_mem_used_mib=snapshot["gpu_mem_used_mib"],
        gpu_mem_free_mib=snapshot["gpu_mem_free_mib"],
        gpu_compute_procs=snapshot["gpu_compute_procs"],
    )


def _is_context_overflow(exc: BaseException) -> bool:
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
    trial_wall_clock_limit_s: float = TRIAL_WALL_CLOCK_LIMIT_S,
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

    cancel_event = threading.Event()

    def _invoke() -> Any:
        if spec.mode == "classic":
            return run_classic_trial(
                provider,
                SYSTEM_PROMPT,
                task_prompt,
                tool_specs,
                execute_tool,
                max_turns=max_turns,
                cancel_event=cancel_event,
            )
        return run_codemode_trial(
            provider,
            SYSTEM_PROMPT,
            task_prompt,
            tool_specs,
            execute_tool,
            max_turns=max_turns,
            max_code_exec_seconds=max_code_exec_seconds,
            cancel_event=cancel_event,
        )

    timestamp = datetime.now(UTC).isoformat()
    start = time.monotonic()
    # A manual daemon thread + Queue, not ThreadPoolExecutor:
    # 1. ThreadPoolExecutor.__exit__/shutdown(wait=True) blocks until an
    #    abandoned worker finishes, even after future.result() already
    #    raised TimeoutError -- defeating the wall-clock limit entirely
    #    (observed: a trial ran 100+ minutes past its 600s budget).
    #    shutdown(wait=False) made this *function* return promptly, but
    #    the abandoned thread (and its request against vLLM) kept
    #    running regardless, for up to `max_turns` more turns -- CPython
    #    joins non-daemon threads at interpreter exit, and in the
    #    meantime it keeps competing for vLLM's KV cache with whatever
    #    trial runs next (observed: "Running: 2 reqs", usage climbing
    #    past 86% within a minute).
    # 2. `daemon=True` means this process is never blocked waiting for an
    #    abandoned thread to finish.
    # 3. `cancel_event` (checked by classic.py/codemode.py at the top of
    #    each turn) stops the abandoned thread from starting a new turn,
    #    bounding its extra runtime to roughly one more in-flight request
    #    rather than up to `max_turns` more.
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result_queue.put(("ok", _invoke()))
        except BaseException as exc:  # noqa: BLE001 -- forwarded to the main thread, not swallowed
            result_queue.put(("error", exc))

    threading.Thread(target=_run, daemon=True).start()
    try:
        kind, payload = result_queue.get(timeout=trial_wall_clock_limit_s)
    except queue.Empty:
        cancel_event.set()
        wall = time.monotonic() - start
        return (
            _attach_gpu_snapshot(
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
                    detail=f"trial exceeded the {trial_wall_clock_limit_s}s wall-clock limit",
                )
            ),
            [],
        )
    if kind == "error":
        exc = payload
        wall = time.monotonic() - start
        if isinstance(exc, TrialAborted):
            # provider.chat() raised mid-trial (classic.py/codemode.py
            # always wrap this as TrialAborted -- see modes/common.py).
            # Unlike the old behaviour (every counter zeroed, trace
            # always empty), use the partial outcome so failed trials
            # stay reviewable: this is what CLAUDE.md's consistency
            # check and the report's failure analysis both need traces
            # for.
            # Always raised as `raise TrialAborted(outcome) from exc`
            # (see modes/common.py), so __cause__ is never actually None.
            assert exc.__cause__ is not None
            cause: BaseException = exc.__cause__
            partial = exc.outcome
            if isinstance(cause, UsageMissingError):
                return (
                    _attach_gpu_snapshot(
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
                            num_requests=partial.num_requests,
                            num_tool_calls=partial.num_tool_calls,
                            num_code_executions=getattr(partial, "num_code_executions", 0),
                            prompt_tokens=partial.total_usage.prompt_tokens,
                            completion_tokens=partial.total_usage.completion_tokens,
                            total_tokens=partial.total_usage.total_tokens,
                            latency_total_s=sum(partial.per_request_latencies),
                            throughput_tokens_per_s=0.0,
                            gpu_seconds=wall,
                            cost_usd_estimate=gpu_cost_usd(wall, gpu_hourly_usd=gpu_hourly_usd),
                            final_answer_text=None,
                            expected=None,
                            detail=str(cause),
                        )
                    ),
                    partial.transcript,
                )
            # "context_overflow" is distinguished from a generic
            # provider-side failure (a malformed tool call the parser
            # rejected, a client-side request timeout, a 5xx from
            # vLLM, ...). Neither is a genuine MCP tool-execution error
            # (that case is "tool_error"/"code_error" via had_error +
            # max_turns below) -- mislabeling them that way would hide a
            # real source of failures behind a misleading name.
            failure_type = "context_overflow" if _is_context_overflow(cause) else "provider_error"
            return (
                _attach_gpu_snapshot(
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
                        num_requests=partial.num_requests,
                        num_tool_calls=partial.num_tool_calls,
                        num_code_executions=getattr(partial, "num_code_executions", 0),
                        prompt_tokens=partial.total_usage.prompt_tokens,
                        completion_tokens=partial.total_usage.completion_tokens,
                        total_tokens=partial.total_usage.total_tokens,
                        latency_total_s=sum(partial.per_request_latencies),
                        throughput_tokens_per_s=0.0,
                        gpu_seconds=wall,
                        cost_usd_estimate=gpu_cost_usd(wall, gpu_hourly_usd=gpu_hourly_usd),
                        final_answer_text=None,
                        expected=None,
                        detail=str(cause),
                    )
                ),
                partial.transcript,
            )
        # Unexpected (not from provider.chat(), which is always wrapped
        # as TrialAborted); classified generically, never silently
        # dropped.
        failure_type = "context_overflow" if _is_context_overflow(exc) else "provider_error"
        return (
            _attach_gpu_snapshot(
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
                )
            ),
            [],
        )
    outcome = payload

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
    return _attach_gpu_snapshot(record), outcome.transcript


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
    trial_wall_clock_limit_s = experiment_config["limits"].get(
        "max_trial_seconds", TRIAL_WALL_CLOCK_LIMIT_S
    )

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
            trial_wall_clock_limit_s=trial_wall_clock_limit_s,
        )
        writer.append(record, transcript)
