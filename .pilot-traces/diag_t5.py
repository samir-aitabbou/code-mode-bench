"""One-off diagnostic (not part of the harness): run classic-mode T5 for
gpt-oss-20b directly, with NO 600s trial wall-clock cap, to see whether
it naturally reaches max_turns or just makes slow progress on one huge
tool result. Prints turn-by-turn timing and tool-result sizes."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from cmbench.data.seed import reset_database  # noqa: E402
from cmbench.mcp_server.adapter import mcp_tools_to_specs  # noqa: E402
from cmbench.mcp_server.server import build_server  # noqa: E402
from cmbench.providers.base import ChatMessage  # noqa: E402
from cmbench.providers.openai_compat import OpenAICompatProvider  # noqa: E402
from cmbench.runner import SYSTEM_PROMPT, make_tool_executor  # noqa: E402
from cmbench.tasks.loader import load_tasks  # noqa: E402

template_db = Path("data/crm.db")
working_dir = Path("data/working")
working_dir.mkdir(parents=True, exist_ok=True)
working_db = working_dir / "diag_t5_classic.db"
reset_database(template_db, working_db)

server = build_server(working_db, with_noise_tools=False)
mcp_tools = asyncio.run(server.list_tools())
tool_specs = mcp_tools_to_specs(mcp_tools)
execute_tool = make_tool_executor(server)

tasks = load_tasks()
task_prompt = tasks["T5"].build_prompt()

provider = OpenAICompatProvider(base_url="http://localhost:8000/v1", model="gpt-oss-20b")

messages: list[ChatMessage] = [
    ChatMessage(role="system", content=SYSTEM_PROMPT),
    ChatMessage(role="user", content=task_prompt),
]

MAX_TURNS = 15
for turn in range(MAX_TURNS):
    t0 = time.monotonic()
    try:
        response = provider.chat(messages, tools=tool_specs, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"turn {turn}: provider.chat raised after {time.monotonic() - t0:.1f}s: {exc!r}")
        break
    dt = time.monotonic() - t0
    messages.append(response.message)
    tc = response.message.tool_calls or []
    print(
        f"turn {turn}: {dt:.1f}s, prompt_tokens={response.usage.prompt_tokens}, "
        f"completion_tokens={response.usage.completion_tokens}, "
        f"tool_calls={[c.name + str(c.arguments) for c in tc]}, "
        f"content_preview={(response.message.content or '')[:150]!r}"
    )
    if tc:
        for c in tc:
            try:
                result = execute_tool(c.name, c.arguments)
                result_text = str(result)
            except Exception as exc:  # noqa: BLE001
                result_text = f"ERROR: {exc}"
            print(f"  -> tool result len={len(result_text)} preview={result_text[:200]!r}")
            messages.append(
                ChatMessage(role="tool", content=result_text, tool_call_id=c.id, name=c.name)
            )
        continue
    content = response.message.content or ""
    if "FINAL_ANSWER:" in content:
        print(f"turn {turn}: FINAL_ANSWER reached: {content}")
        break
    messages.append(
        ChatMessage(
            role="user",
            content="Reminder: when you have the final answer, reply with exactly "
            "'FINAL_ANSWER: <json>' and nothing else -- no tool call, no extra text.",
        )
    )
else:
    print("hit max_turns without a final answer")
