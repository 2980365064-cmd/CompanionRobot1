"""Console monitor — Agent 运作过程与回复（过滤无关日志）."""

from __future__ import annotations

import logging
import sys
import time

_AGENT_LOGGER = "agent"
_QUIET_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "httpx",
    "httpcore",
    "chromadb",
    "openai",
    "app.memory.semantic",
    "app.rag",
    "app.llm",
    "app.memory.router",
    "app.memory.extractor",
    "app.ws_handler",
)


def _clip(text: str, n: int = 72) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


class AgentMonitor:
    """Structured console output for chat turns."""

    def __init__(self) -> None:
        self._log = logging.getLogger(_AGENT_LOGGER)
        self._configured = False

    def configure(self) -> None:
        if self._configured:
            return
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._log.handlers.clear()
        self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)
        self._log.propagate = False

        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        # app.* 默认静默，仅 agent 与显式 warning 输出
        logging.getLogger("app").setLevel(logging.WARNING)
        self._log.setLevel(logging.INFO)

        self._configured = True

    def _line(self, msg: str) -> None:
        self._log.info(msg)

    def banner(self, title: str) -> None:
        self._line(f"\n--- {title} " + "-" * max(0, 40 - len(title)))

    def startup(self, msg: str) -> None:
        self._line(f"[启动] {msg}")

    def warn(self, msg: str) -> None:
        self._line(f"[警告] {msg}")

    def chat_user(self, device_id: str, message: str) -> None:
        self.banner("对话")
        dev = device_id if device_id != "default" else ""
        prefix = f"[{dev}] " if dev else ""
        self._line(f">> 用户 {prefix}{message}")

    def chat_memory(self, memory: dict, query: str) -> None:
        l1 = len(memory.get("working") or [])
        l2 = len(memory.get("episodic") or [])
        l3_n = len(memory.get("semantic") or [])
        intent = memory.get("l3_intent", False)
        triggered = memory.get("l3_triggered", False)

        l3_status = "未检索"
        if triggered:
            l3_status = f"命中 {l3_n} 条" if l3_n else "已检索·无匹配"
        elif intent:
            l3_status = "轻量检索"
        elif l3_n:
            l3_status = f"轻量 {l3_n} 条"

        intent_tag = "是" if intent else "否"
        self._line(f"  记忆  L1={l1}轮  L2={l2}条  L3意图={intent_tag}  L3={l3_status}")

        for i, hit in enumerate((memory.get("semantic") or [])[:2], 1):
            self._line(f"       [{i}] {_clip(hit, 64)}")

        if triggered and l3_n == 0:
            self._line(f"       (查询: {_clip(query, 40)})")

    def chat_reply(self, reply: str, elapsed_ms: float) -> None:
        self._line(f"<< 回复 {reply}")
        self._line(f"   耗时 {elapsed_ms:.0f}ms")

    def event(self, msg: str) -> None:
        """后台事件（压缩、入库等），仅关键节点."""
        self._line(f"   * {msg}")

    def finish_turn(self, memory: dict, query: str, reply: str, t0: float) -> None:
        self.chat_memory(memory, query)
        self.chat_reply(reply, (time.perf_counter() - t0) * 1000)


agent_monitor = AgentMonitor()
