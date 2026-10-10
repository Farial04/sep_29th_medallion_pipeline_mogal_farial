import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") not in {"thinking", "reasoning", "redacted_thinking"}
                and isinstance(block.get("text"), str)
            ):
                text_parts.append(block["text"])
        return "\n".join(text_parts)
    return "" if content is None else str(content)


class AgentTrace:
    def __init__(self, agent_name: str, run_id: str) -> None:
        self.agent_name = agent_name
        self.run_id = run_id
        self._started_at = time.time()
        self.trace: dict[str, Any] = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": None,
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": None,
            "status": "running",
        }

    def set_input(self, **kwargs: Any) -> "AgentTrace":
        self.trace["input"] = kwargs
        return self

    def set_plan(self, plan_str: str) -> "AgentTrace":
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs: Any) -> "AgentTrace":
        self.trace["output"] = kwargs
        return self

    def extract_from_messages(self, messages: list[Any]) -> "AgentTrace":
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        for message in messages:
            if isinstance(message, HumanMessage):
                self.trace["reasoning_steps"].append(
                    {"type": "task_input", "content": _message_text(message.content)}
                )
            elif isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None) or []
                self.trace["tool_calls"].extend(tool_calls)
                content = _message_text(message.content)
                if content.strip():
                    if self.trace["plan"] is None:
                        self.trace["plan"] = content
                    self.trace["reasoning_steps"].append(
                        {"type": "assistant_message", "content": content}
                    )
            elif isinstance(message, ToolMessage):
                self.trace["reasoning_steps"].append(
                    {
                        "type": "tool_result",
                        "tool": getattr(message, "name", None),
                        "tool_call_id": getattr(message, "tool_call_id", None),
                        "content": _message_text(message.content),
                    }
                )

        return self

    def complete(self, status: str = "success") -> None:
        self.trace["duration_seconds"] = time.time() - self._started_at
        self.trace["status"] = status
        safe_agent_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.agent_name)
        trace_path = TRACES_DIR / f"trace_{safe_agent_name}_{self.run_id[:8]}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as trace_file:
            json.dump(self.trace, trace_file, ensure_ascii=False, indent=2, default=str)
            trace_file.write("\n")

        print(
            f"Trace saved: {trace_path} | agent={self.agent_name} | "
            f"run_id={self.run_id} | status={status} | "
            f"duration={self.trace['duration_seconds']:.3f}s"
        )

    def fail(self, error: Any) -> None:
        self.trace["error"] = str(error)
        self.complete(status="failed")
