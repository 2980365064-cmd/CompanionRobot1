"""控制台监控 —— 结构化输出每轮对话的记忆命中与后台事件。

阶段 3.0 — "记忆测试驾驶舱"
支持四模式：
  silent  只显示启动和错误
  normal  显示每轮核心链路（box-drawing 风格）
  debug   显示 MemoryPackV2、写入裁决、prompt 摘要
  trace   显示底层召回和候选项

设计原则：
  - 静音所有第三方库日志噪音
  - 仅通过 "agent" logger 通道输出 INFO 级别
  - box-drawing 风格统一终端布局
  - 所有时序由 agent.py 通过 set_timing() 注入，monitor 仅负责展示
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any

from app.config import settings

# agent 监控专用 logger 名称
_AGENT_LOGGER = "agent"

# 需要静音的第三方/框架 logger
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

# 工程术语检测列表（用于 prompt_summary）
_ENGINEERING_TERMS = [
    "L0", "L1", "L2", "L3",
    "向量", "检索", "命中", "命中率",
    "记忆库", "数据库中", "关联网络", "strength",
    "FTS", "embedding", "未匹配", "无相关记录",
]


def _clip(text: str, n: int = 72) -> str:
    """单行截断：合并空白后截取前 n 字符。"""
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_score(score: float | None) -> str:
    """向量相似度格式化。"""
    if score is None:
        return "recent"
    return f"{score:.3f}"


def _count_engineering_terms(text: str) -> int:
    """统计 prompt 文本中出现的工程术语数量。"""
    count = 0
    for term in _ENGINEERING_TERMS:
        if term in text:
            count += 1
    return count


class _TurnTimer:
    """计时上下文管理器 —— 由 AgentMonitor.turn_timer() 返回。"""

    def __init__(self, monitor: "AgentMonitor", name: str) -> None:
        self._monitor = monitor
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_TurnTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        self._monitor.set_timing(self._name, elapsed)


class AgentMonitor:
    """结构化的控制台输出器 —— 阶段 3.0 升级版。

    新增方法：
      start_turn()    — box-drawing 轮次头
      identity()      — 身份信息行
      memory_pack_v2() — MemoryPackV2 详细展开（debug）
      prompt_summary() — prompt 安全摘要
      consolidation() — Consolidator 结果展示
      end_turn()      — 回复+耗时+box-drawing 尾
      set_timing()    — 注入阶段耗时
      turn_timer()    — 计时上下文管理器

    旧方法保留兼容：
      chat_user(), chat_memory(), chat_reply(), finish_turn()
    """

    def __init__(self) -> None:
        self._log = logging.getLogger(_AGENT_LOGGER)
        self._configured = False
        self._turn_counter = 0
        self._timings: dict[str, float] = {}
        # SSE 日志缓冲（后台管理页面实时日志）
        self._log_buffer: deque[str] = deque(maxlen=2000)

    # ── 初始化 ──────────────────────────────────────────────────────

    def configure(self, *, force: bool = False) -> None:
        """初始化日志通道（静音第三方，仅 agent 输出）。"""
        if self._configured and not force:
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

        logging.getLogger("app").setLevel(logging.WARNING)
        self._log.setLevel(logging.INFO)

        self._configured = True

    def ensure_ready(self) -> None:
        """确保 agent logger handler 存在（uvicorn 可能冲掉配置）。"""
        if not self._log.handlers or not self._configured:
            self.configure(force=True)

    def _line(self, msg: str) -> None:
        """写一行到 agent 日志通道。"""
        self.ensure_ready()
        self._log.info(msg)
        # 同时写入 SSE 日志缓冲
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_buffer.append(f"[{ts}] {msg}")

    def _mode_is(self, *levels: str) -> bool:
        """检查当前日志模式是否匹配任一给定级别。"""
        current = str(settings.console_log_mode)
        return current in levels

    # ── 公用方法（所有模式可用） ────────────────────────────────────

    def banner(self, title: str) -> None:
        """分节标题线（兼容旧代码）。"""
        if self._mode_is("silent"):
            return
        self._line(f"\n--- {title} " + "-" * max(0, 40 - len(title)))

    def startup(self, msg: str) -> None:
        """服务启动信息（silent 模式也显示）。"""
        self._line(f"[启动] {msg}")

    def warn(self, msg: str) -> None:
        """非致命警告（silent 模式也显示）。"""
        self._line(f"[警告] {msg}")

    def event(self, msg: str) -> None:
        """后台异步事件。"""
        if self._mode_is("silent"):
            return
        self._line(f"   * {msg}")

    # ── 轮次启动（阶段 3.0 核心） ──────────────────────────────────

    def start_turn(self, device_id: str, message: str, session_id: str = "") -> None:
        """打印 box-drawing 轮次头和用户消息。

        在 agent.py 的 handle_chat/stream 入口调用。
        """
        if self._mode_is("silent"):
            return

        self._turn_counter += 1
        now_str = datetime.now().strftime("%H:%M:%S")
        width = settings.console_log_width
        dev = device_id if device_id != "default" else ""

        session_suffix = ""
        if self._mode_is("debug", "trace") and session_id:
            session_suffix = f" · session={session_id[:12]}"

        self._line(f"╭─ Turn #{self._turn_counter} · {now_str} · {dev}{session_suffix}")
        self._line(f"│ 用户: {_clip(message, width - 10)}")

    def set_timing(self, name: str, ms: float) -> None:
        """注入阶段耗时。"""
        self._timings[name] = ms

    def turn_timer(self, name: str) -> _TurnTimer:
        """返回一个计时上下文管理器。

        用法：
            with agent_monitor.turn_timer("llm"):
                reply = await llm_call(...)
        """
        return _TurnTimer(self, name)

    # ── 身份展示 ──────────────────────────────────────────────────

    def identity(
        self,
        person_profile: dict | None,
        memory: dict,
        interlocutor_mode: str,
    ) -> None:
        """打印当前对话对象的身份信息。"""
        if self._mode_is("silent"):
            return

        if self._mode_is("debug", "trace"):
            # 多行详细
            if person_profile:
                from app.memory.profile import (
                    normalize_profile,
                    profile_display_name,
                    profile_nicknames,
                )
                p = normalize_profile(person_profile)
                nick = profile_display_name(p) or (
                    profile_nicknames(p)[0] if profile_nicknames(p) else "?"
                )
                confirmed = str(p.get("confirmed", False)).lower()
                guest = str(memory.get("guest_mode", False)).lower()
                self._line(f"│ 身份:")
                self._line(f"│   mode={interlocutor_mode} · person={nick} · confirmed={confirmed} · guest={guest}")
            else:
                self._line(f"│ 身份:")
                self._line(f"│   mode={interlocutor_mode} · 未绑定画像")
        else:
            # 单行紧凑
            if person_profile:
                from app.memory.profile import (
                    normalize_profile,
                    profile_display_name,
                    profile_nicknames,
                )
                p = normalize_profile(person_profile)
                nick = profile_display_name(p) or (
                    profile_nicknames(p)[0] if profile_nicknames(p) else "?"
                )
                confirmed_str = "已确认" if p.get("confirmed") else "待确认"
                pid = str(p.get("person_id") or "")[:12]
                self._line(
                    f"│ 身份: {nick} · {interlocutor_mode} · {confirmed_str} · id={pid}"
                )
            elif memory.get("guest_mode"):
                mem_pid = str(memory.get("person_id") or "—")[:16]
                self._line(f"│ 身份: 访客 · tmp={mem_pid}…")
            else:
                self._line(f"│ 身份: {interlocutor_mode}")

    # ── MemoryPack 摘要 ────────────────────────────────────────────

    def memory_pack_v2(self, memory_pack: Any) -> None:
        """打印 MemoryPackV2 详细内容（debug/trace 模式）。"""
        if not self._mode_is("debug", "trace"):
            return

        # 尝试获取 MemoryPackV2
        pack_v2 = None
        if memory_pack is not None:
            try:
                pack_v2 = getattr(memory_pack, "_v2", None) or (
                    memory_pack.to_v2() if hasattr(memory_pack, "to_v2") else None
                )
            except Exception:
                pack_v2 = None

        if pack_v2 is None:
            self._line(f"│ MemoryPack: (unavailable)")
            return

        import app.memory.schema as _schema

        # 月份查询标记
        raw = getattr(pack_v2, "_raw", {}) or {}
        mk = str(raw.get("month_key", "") or "")
        if mk:
            self._line(f"│ MemoryPackV2:")
            self._line(f"│   month_key={mk}")
        else:
            self._line(f"│ MemoryPackV2:")

        # L3 命中摘要（source / category / score）
        l3_matches = (raw.get("matches") or {}).get("l3") or []
        if l3_matches:
            l3_lines: list[str] = []
            for m in l3_matches[:6]:
                src = str(m.get("source", "—"))[:24]
                cat = str(m.get("category", "—"))[:16]
                scr = f"{m.get('score', 0):.3f}" if m.get("score") is not None else "recent"
                l3_lines.append(f"{src}|{cat}|{scr}")
            self._line(f"│   l3_hits={' · '.join(l3_lines)}")

        # 关系状态块
        rel = pack_v2.relationship
        mood_str = f"mood={rel.recent_mood or '—'}"
        att_str = f"attitude={rel.recent_attitude or '—'}"
        temp_str = f"temp={rel.relationship_temperature:.2f}"
        self._line(f"│   relationship: {mood_str} · {att_str} · {temp_str}")

        # 当前状态
        topic_str = pack_v2.current_topic or "—"
        self._line(f"│   current: topic={topic_str}")

        # MemoryItem 统计
        try:
            items = pack_v2.items_for_prompt() or []
        except Exception:
            items = []
        kind_counts: dict[str, int] = {}
        for item in items:
            k = item.kind.value if hasattr(item.kind, "value") else str(item.kind)
            kind_counts[k] = kind_counts.get(k, 0) + 1

        kind_summary = " · ".join(f"{k}={v}" for k, v in sorted(kind_counts.items()))
        self._line(f"│   items={len(items)} · {kind_summary}" if kind_summary else f"│   items={len(items)}")

        # 前 5 条项目
        for item in items[:5]:
            text = _clip(item.humanized_text, 80) or _clip(item.content, 80)
            kind_val = item.kind.value if hasattr(item.kind, "value") else str(item.kind)
            conf = f"{item.confidence:.2f}"
            self._line(f"│   [{kind_val}/{conf}] {text}")

    def memory_pack_summary(self, memory: dict, memory_pack: Any = None) -> None:
        """打印单行记忆包摘要（normal 模式）。"""
        if self._mode_is("silent", "debug", "trace"):
            return

        if memory.get("guest_mode"):
            self._line(f"│ 记忆: 访客模式 · 仅 L1 上下文")
            return

        # 尝试从 MemoryPackV2 获取摘要
        pack_v2 = None
        if memory_pack is not None:
            try:
                pack_v2 = getattr(memory_pack, "_v2", None) or (
                    memory_pack.to_v2() if hasattr(memory_pack, "to_v2") else None
                )
            except Exception:
                pack_v2 = None

        if pack_v2 and settings.console_log_memory_detail:
            try:
                items = pack_v2.items_for_prompt() or []
            except Exception:
                items = []
            kind_counts: dict[str, int] = {}
            for item in items:
                k = item.kind.value if hasattr(item.kind, "value") else str(item.kind)
                kind_counts[k] = kind_counts.get(k, 0) + 1
            summary_parts = [f"记忆项={len(items)}"]
            for k, v in sorted(kind_counts.items()):
                summary_parts.append(f"{k}={v}")
            self._line(f"│ 记忆包: {' · '.join(summary_parts)}")
            return

        # 降级到旧 dict 格式
        matches = memory.get("matches") or {}
        l2_n = len(matches.get("l2") or [])
        l3_n = len(matches.get("l3") or [])
        related_n = len(matches.get("related") or [])
        miss = "是" if memory.get("memory_miss") else "否"
        self._line(f"│ 记忆: L2={l2_n} L3={l3_n} 联想={related_n} 未命中={miss}")

    # ── Prompt 安全摘要 ────────────────────────────────────────────

    def prompt_summary(self, messages: list[dict]) -> None:
        """打印 prompt 安全摘要。
        debug/trace 模式或 console_log_prompt_preview=True 时显示。
        """
        if self._mode_is("silent"):
            return
        show_preview = settings.console_log_prompt_preview
        if self._mode_is("normal") and not show_preview:
            return

        if not messages:
            return

        system_content = messages[0].get("content", "") if messages else ""
        chars = len(system_content)
        eng_count = _count_engineering_terms(system_content)
        sections = system_content.count("## ")

        # 检测是否为 V2 人类化格式（简单的启发式）
        has_human_sections = "## 你和她的关系状态" in system_content
        has_old_terms = eng_count > 0
        fallback = "no" if (has_human_sections and not has_old_terms) else "yes" if has_old_terms else "no"

        if self._mode_is("debug", "trace"):
            self._line(f"│ Prompt:")
            self._line(f"│   sections={sections} · chars={chars} · engineering_terms={eng_count} · fallback={fallback}")
            if show_preview:
                preview = system_content[:200].replace("\n", "\n│   ")
                self._line(f"│   preview: {preview}…")
        else:
            self._line(f"│ Prompt: {sections}段 {chars}字 工程词={eng_count}")

    # ── Consolidator 结果展示 ──────────────────────────────────────

    def consolidation(self, result: Any) -> None:
        """打印 Consolidator 处理详细结果（debug/trace 模式）。

        必须在 end_turn 之前调用（因为输出在 box 内部）。
        """
        if not self._mode_is("debug", "trace"):
            return
        if not result:
            return

        self._line(f"│ Consolidator:")
        if hasattr(result, "classification"):
            self._line(f"│   classification={result.classification.reason}")

        if hasattr(result, "quality_decision") and result.quality_decision:
            self._line(f"│   quality={result.quality_decision}")

        if hasattr(result, "contacts_updated") and result.contacts_updated:
            self._line(f"│   contact=hit {result.contacts_updated} contacts")
        elif getattr(result, "classification", None) and result.classification.is_third_party:
            self._line(f"│   contact=query")

        ol_created = len(result.open_loops_created) if hasattr(result, "open_loops_created") else 0
        ol_resolved = len(result.open_loops_resolved) if hasattr(result, "open_loops_resolved") else 0
        self._line(f"│   open_loop=new {ol_created} resolved {ol_resolved}")

        if result.open_loops_created:
            self._line(f"│   open_loop_titles={' '.join(result.open_loops_created)}")
        if result.open_loops_resolved:
            self._line(f"│   open_loop_resolved={' '.join(result.open_loops_resolved)}")

        temp_before = result.relationship_before.get("relationship_temperature", "?") if hasattr(result, "relationship_before") else "?"
        temp_after = result.relationship_after.get("relationship_temperature", "?") if hasattr(result, "relationship_after") else "?"
        self._line(f"│   relationship=temp {temp_before}→{temp_after}")

        if hasattr(result, "corrections_applied") and result.corrections_applied:
            s = result.corrections_applied
            self._line(f"│   correction=del_mem {s.get('deleted_facts',0)} del_chunk {s.get('deleted_chunks',0)}")

        if hasattr(result, "emotional_events_detected") and result.emotional_events_detected:
            self._line(f"│   emotional_events={' '.join(result.emotional_events_detected)}")

    # ── 轮次收尾 ──────────────────────────────────────────────────

    def end_turn(self, reply: str, t0: float, *, consolidation_result: Any = None) -> None:
        """打印回复、耗时和 box-drawing 结尾。

        Args:
            reply:                LLM 回复文本
            t0:                   轮次开始时间戳（time.perf_counter）
            consolidation_result: 可选 ConsolidationResult，normal 模式显示紧凑后台行
        """
        if self._mode_is("silent"):
            return

        elapsed = (time.perf_counter() - t0) * 1000
        width = settings.console_log_width

        if self._mode_is("debug", "trace"):
            self._line(f"│ 回复:")
            self._line(f"│   {_clip(reply, width - 4)}")
            if self._timings and settings.console_log_timing:
                timing_items = [
                    f"{k}={v:.0f}ms"
                    for k, v in self._timings.items()
                    if v is not None
                ]
                timing_str = " · ".join(timing_items)
                self._line(f"│ Timings:")
                self._line(f"│   {timing_str} total={elapsed:.0f}ms")
            elif settings.console_log_timing:
                self._line(f"│ Timings: total={elapsed:.0f}ms")
            self._line("╰─ " + "─" * max(0, width - 4))
        else:
            # normal 模式
            self._line(f"│ 回复: {_clip(reply, width - 14)}")
            if settings.console_log_timing:
                timing_parts: list[str] = []
                if self._timings:
                    timing_parts.extend(
                        f"{k}={v:.0f}ms"
                        for k, v in self._timings.items()
                        if v is not None and k in ("recall", "llm", "prompt")
                    )
                timing_parts.append(f"total={elapsed:.0f}ms")
                self._line(f"│ 耗时: {' '.join(timing_parts)}")

            # 紧凑后台行（Consolidator 摘要）
            if consolidation_result:
                bg_events = self._compact_background_events(consolidation_result)
                if bg_events:
                    self._line(f"│ 后台: {' · '.join(bg_events)}")

            self._line("╰─ " + "─" * max(0, width - 4))

    @staticmethod
    def _compact_background_events(result: Any) -> list[str]:
        """从 ConsolidationResult 中提取紧凑后台事件摘要（normal 模式用）。"""
        events: list[str] = []
        if not result:
            return events

        if hasattr(result, "open_loops_created") and result.open_loops_created:
            events.append(f"待跟进新增 · {' '.join(result.open_loops_created[:2])}")
        if hasattr(result, "open_loops_resolved") and result.open_loops_resolved:
            events.append(f"待跟进完成 · {' '.join(result.open_loops_resolved)}")
        if hasattr(result, "contacts_updated") and result.contacts_updated:
            events.append(f"第三方画像 · {result.contacts_updated}")
        if hasattr(result, "emotional_events_detected") and result.emotional_events_detected:
            events.append(f"情感事件 · {' '.join(result.emotional_events_detected)}")
        if hasattr(result, "corrections_applied") and result.corrections_applied:
            s = result.corrections_applied
            events.append(f"记忆修正 · 删{s.get('deleted_facts',0)} 增{s.get('added_facts',0)}")
        return events

    # ── 旧方法兼容（仍可使用，但新代码推荐用上面的新方法） ──────────

    def chat_user(self, device_id: str, message: str) -> None:
        """旧版轮次入口（保持兼容）。"""
        if self._mode_is("silent"):
            return
        self.banner("对话")
        dev = device_id if device_id != "default" else ""
        prefix = f"[{dev}] " if dev else ""
        self._line(f">> 用户 {prefix}{message}")

    def _print_match_layer(
        self,
        layer: str,
        items: list[dict],
        *,
        empty_msg: str,
        fmt_item,
    ) -> None:
        """打印单层记忆命中列表（兼容旧代码）。"""
        if self._mode_is("silent"):
            return
        if not items:
            self._line(f"  {layer}  {empty_msg}")
            return
        self._line(f"  {layer}  命中 {len(items)} 条")
        for i, item in enumerate(items, 1):
            self._line(f"    [{i}] {fmt_item(item)}")

    def chat_memory(
        self,
        memory: dict,
        query: str,
        *,
        person_profile: dict | None = None,
        promotion_eval=None,
    ) -> None:
        """旧版记忆摘要（保持兼容）。"""
        if self._mode_is("silent"):
            return

        if person_profile:
            from app.memory.profile import (
                normalize_profile,
                profile_display_name,
                profile_nicknames,
            )
            p = normalize_profile(person_profile)
            nick = profile_display_name(p) or (
                profile_nicknames(p)[0] if profile_nicknames(p) else "?"
            )
            rel = getattr(p, "relationship", None) or p.get("relationship", "—")
            pid = str(p.get("person_id") or "")[:12]
            confirmed = "已确认" if p.get("confirmed") else "待确认"
            self._line(f"  对象  {nick}（{rel}）  {confirmed} id={pid}…")
        elif memory.get("guest_mode"):
            mem_pid = memory.get("person_id") or "—"
            self._line(f"  对象  访客（仅 L1） tmp={str(mem_pid)[:16]}…")
        else:
            mem_pid = memory.get("person_id")
            if mem_pid:
                self._line(f"  对象  已绑定 person_id={str(mem_pid)[:12]}…")
            else:
                self._line("  对象  未绑定")

        matches = memory.get("matches") or {}
        l0_n = len(memory.get("l0") or [])
        l2_hit = memory.get("l2_hit", False)
        l3_hit = memory.get("l3_hit", memory.get("facts_hit", False))
        memory_miss = memory.get("memory_miss", False)
        l1_ctx = len(memory.get("working") or [])

        if memory.get("guest_mode"):
            self._line(f"  记忆  访客模式 · 仅 L1={l1_ctx}条")
            return

        miss_tag = "须承认不清楚" if memory_miss else "—"
        self._line(
            f"  记忆  L0={l0_n}条(全量必载)"
            f"  L1上下文={l1_ctx}条(仅注入不检索)"
            f"  L2={'命中' if l2_hit else '未命中'}"
            f"  L3={'命中' if l3_hit else '未命中'}"
            f"  联想={len(matches.get('related') or [])}条"
            f"  未命中={miss_tag}"
        )
        self._print_match_layer(
            "L2",
            matches.get("l2") or [],
            empty_msg="无相关摘要",
            fmt_item=lambda m: f"sim={_fmt_score(m.get('score'))} {_clip(m.get('text', ''), 96)}",
        )
        l3_items = matches.get("l3") or []
        self._print_match_layer(
            "L3",
            l3_items,
            empty_msg="无相关 Facts/语料",
            fmt_item=lambda m: (
                f"sim={_fmt_score(m.get('score'))} "
                f"[{m.get('category') or m.get('source') or 'memory'}] "
                f"{_clip(m.get('text', ''), 88)}"
            ),
        )
        related_items = matches.get("related") or []
        if related_items:
            self._print_match_layer(
                "联想",
                related_items,
                empty_msg="—",
                fmt_item=lambda m: (
                    f"{m.get('relation_type', 'related')}·{m.get('strength', 0.5)} "
                    f"{_clip(m.get('text', ''), 88)}"
                ),
            )
        if memory_miss:
            self._line(f"  ⚠ L2/L3 均无相关内容，查询: {_clip(query, 48)}")

    def chat_reply(self, reply: str, elapsed_ms: float) -> None:
        """旧版回复+耗时（保持兼容）。"""
        if self._mode_is("silent"):
            return
        self._line(f"<< 回复 {reply}")
        self._line(f"   耗时 {elapsed_ms:.0f}ms")

    def finish_turn(
        self,
        memory: dict,
        query: str,
        reply: str,
        t0: float,
        *,
        person_profile: dict | None = None,
        promotion_eval=None,
    ) -> None:
        """旧版轮次收尾（保持兼容）。"""
        self.chat_memory(
            memory, query, person_profile=person_profile, promotion_eval=promotion_eval
        )
        self.chat_reply(reply, (time.perf_counter() - t0) * 1000)


# 全局单例
agent_monitor = AgentMonitor()


# ── SSE 日志 API ──────────────────────────────────────────────

def sse_get_history(last_index: int = 0) -> tuple[list[str], int]:
    """获取历史日志（从 last_index 位置开始），返回 (lines, new_index)。"""
    buf = list(agent_monitor._log_buffer)
    total = len(buf)
    if last_index >= total:
        return [], total
    return buf[last_index:], total
