"""控制台监控 —— 结构化输出每轮对话的记忆命中与后台事件。

本模块的角色：
  陪伴机器人的"黑匣子"仪表盘。在终端运行时，每轮对话都会结构化打印：
  - 用户输入
  - 当前对话对象（实名用户/访客/画像信息）
  - 各记忆层命中情况（L0/L1/L2/L3/联想网络）
  - 助手回复与耗时
  - 后台异步事件（L1 压缩、L0 入库、记忆修正、语料同步等）

设计原则：
  - 静音所有第三方库（uvicorn、httpx、chromadb、openai 等）的日志噪音
  - 仅通过 "agent" logger 通道输出 INFO 级别信息
  - 每行格式统一，便于 grep 和人工阅读
  - chat_memory() 的字段与 memory_router.recall() 返回值一一对应，便于调试召回链路

与 app/log_config.py 的关系：
  log_config.py 控制 Uvicorn 启动时的日志级别（WARNING），
  本模块在 AgentMonitor.configure() 中进一步静音所有应用库日志，
  两者配合实现"控制台只看 agent 通道"的效果。
"""

from __future__ import annotations

import logging
import sys
import time

# agent 监控专用 logger 名称，与配置中静音的其他 logger 区分
_AGENT_LOGGER = "agent"

# 需要静音的第三方/框架 logger 列表
# 这些库在 DEBUG/INFO 级别会输出大量无用信息，统一设为 WARNING
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
    """单行截断：合并空白后截取前 n 字符，用于控制台预览记忆片段。

    记忆文本通常较长，控制台一行显示不下，截断到 72 字加省略号，
    既保留关键信息又不撑破终端布局。
    """
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_score(score: float | None) -> str:
    """向量相似度格式化。

    None 表示通过时间倒序 fallback 注入（如 L2 recent 模式），输出 "recent"；
    有值则格式化为 3 位小数（如 0.856）。
    """
    if score is None:
        return "recent"
    return f"{score:.3f}"


class AgentMonitor:
    """单轮对话的结构化控制台输出器。

    核心方法：
      configure()    初始化日志通道（静音第三方，仅 agent 输出）
      banner()       分节标题（如 "--- 对话 -----------"）
      startup()      服务启动信息
      chat_user()    用户输入（每轮开始）
      chat_memory()  记忆命中摘要（L0/L1/L2/L3/联想网络）
      chat_reply()   助手回复与耗时
      finish_turn()  轮次结束（先记忆再回复）
      event()        后台异步事件（L1 压缩等）
      warn()         非致命警告
    """

    def __init__(self) -> None:
        self._log = logging.getLogger(_AGENT_LOGGER)
        self._configured = False

    def configure(self, *, force: bool = False) -> None:
        """初始化日志配置：静音所有第三方库，仅 agent 通道输出 INFO。

        只执行一次（通过 _configured 标志），多次调用安全。
        传 force=True 可重建 handler（uvicorn --log-level 可能冲掉配置）。

        静音策略（自底向上）：
          1. root logger 设为 WARNING —— 所有未显式配置的 logger 默认静音
          2. agent logger 的 handlers 清空重新绑定 —— 避免重复输出
          3. 所有 _QUIET_LOGGERS 中的第三方/框架 logger 设为 WARNING
          4. app 包下的 logger 也设为 WARNING —— agent 除外（保持 INFO）
             agent logger 通过 propagate=False 独立输出，不进入 root 管道

        最终效果：控制台 stdout 只输出 "agent" 通道的 INFO 消息，
        所有其他日志（uvicorn、httpx、chromadb、openai 等）全部静音。
        """
        if self._configured and not force:
            return
        # Step 1: 清空 root logger 的所有 handler 并设为 WARNING
        # 这样所有 logger 默认不会再输出 INFO/DEBUG 到控制台
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

        # Step 2: 确保输出流使用 UTF-8 编码（Windows 终端可能默认 gbk）
        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        # Step 3: 为 agent logger 创建独立的 handler，格式为纯消息（无时间戳/logger 名）
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._log.handlers.clear()  # 清空已有 handler，避免多次 configure 导致重复
        self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)
        self._log.propagate = False  # 关键：不向 root 传播，因为 root 已是 WARNING

        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        logging.getLogger("app").setLevel(logging.WARNING)
        self._log.setLevel(logging.INFO)

        self._configured = True

    def ensure_ready(self) -> None:
        """Ensure agent logger still has a handler (uvicorn may reset logging)."""
        if not self._log.handlers or not self._configured:
            self.configure(force=True)

    def _line(self, msg: str) -> None:
        """写一行到 agent 日志通道（内部辅助方法）。"""
        self.ensure_ready()
        self._log.info(msg)

    def banner(self, title: str) -> None:
        """打印分节标题线，如 "--- 对话 ------------------------------" """
        self._line(f"\n--- {title} " + "-" * max(0, 40 - len(title)))

    def startup(self, msg: str) -> None:
        """服务启动信息（lifespan 阶段输出）。

        包括：LLM/向量就绪状态、监听地址、语料同步进度等。
        """
        self._line(f"[启动] {msg}")

    def warn(self, msg: str) -> None:
        """非致命警告（入库失败、配置缺失、API 错误等）。

        这些不会导致服务崩溃，但需要开发者关注。
        """
        self._line(f"[警告] {msg}")

    def chat_user(self, device_id: str, message: str) -> None:
        """轮次开始：打印用户输入消息。

        参数:
            device_id: 设备标识（default 时不显示）
            message:   用户发送的文本
        """
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
        """打印单层记忆命中列表的通用方法。

        参数:
            layer:     层名（"L2" / "L3" / "联想"）
            items:     命中条目列表
            empty_msg: 无命中时的提示文案
            fmt_item:  单条格式函数，接收 dict 返回 str
        """
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
        """打印本轮记忆召回摘要（对话对象 + 各层命中详情）。

        参数:
            memory:         memory_router.recall() 的返回值，包含 L0/L1/L2/L3/匹配信息
            query:          用户本轮输入（用于未命中时展示查询内容）
            person_profile: 当前对话对象的画像（含 draft 转正状态）
            promotion_eval: 可选，临时画像转正评估结果（由 profile_promotion 模块生成）

        输出结构：
          1. 对话对象：实名用户显示昵称/关系/是否确认，访客显示 tmp_* ID
          2. 记忆摘要：L0 条数、L1 上下文条数、L2/L3 命中状态、联想条数、未命中标记
          3. L2 命中明细：相似度 + 文本片段
          4. L3 命中明细：相似度 + 来源 + 文本片段
          5. 联想记忆明细：关系类型 + 强度 + 文本片段
          6. 未命中警告（仅当 L2 和 L3 均无相关内容时）
        """
        if person_profile:
            # 实名用户：展示昵称、关系、确认状态
            # 引用的 profile 工具函数负责从画像字典中提取可读的展示信息
            from app.memory.profile import (
                normalize_profile,
                profile_display_name,
                profile_nicknames,
                profile_relationship,
            )

            p = normalize_profile(person_profile)
            nick = profile_display_name(p) or (
                profile_nicknames(p)[0] if profile_nicknames(p) else "?"
            )
            rel = profile_relationship(p) or "—"
            pid = str(p.get("person_id") or "")[:12]
            confirmed = "已确认" if p.get("confirmed") else "待确认"
            self._line(
                f"  对象  {nick}（{rel}）"
                f"  {confirmed} id={pid}…"
            )
        elif memory.get("guest_mode"):
            # 访客模式：未实名，不检索任何长期记忆
            mem_pid = memory.get("person_id") or "—"
            self._line(
                f"  对象  访客（仅 L1） tmp={str(mem_pid)[:16]}…"
                "  未实名不检索 L0/L2/L3/画像"
            )
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
            # 访客模式快速返回：仅展示 L1 条数，其余层均不检索
            self._line(
                f"  记忆  访客模式 · 仅 L1={l1_ctx}条"
                "  L0/L2/L3/画像均不检索"
            )
            return

        # 已实名模式：逐层展示命中状态
        # L0 和 L1 是全量注入的（不走检索），所以算条数而非"命中/未命中"
        # L2/L3 走向量检索，用命中/未命中标记
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
        """打印助手回复与耗时。

        参数:
            reply:      生成的回复文本
            elapsed_ms: 本轮处理耗时（毫秒）
        """
        self._line(f"<< 回复 {reply}")
        self._line(f"   耗时 {elapsed_ms:.0f}ms")

    def event(self, msg: str) -> None:
        """后台异步事件日志。

        用于记录不阻塞对话响应的后台操作，如：
          - L1→L2 压缩完成
          - L0 核心事实入库
          - 记忆修正（用户纠错、冲突合并）
          - 语料同步进度
          - 空闲会话清理
        """
        self._line(f"   * {msg}")

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
        """轮次结束：依次打印记忆命中摘要和回复与耗时。

        参数:
            memory:        记忆召回结果
            query:         用户输入
            reply:         LLM 生成的回复
            t0:            time.perf_counter() 记录的轮次开始时间戳
            person_profile: 对话对象画像
            promotion_eval: 临时画像转正评估结果

        输出顺序：先记忆摘要（chat_memory），再回复+耗时（chat_reply）。
        这样设计是因为控制台阅读时记忆命中是"排查过程"，
        回复是"结果"，从过程到结果自上而下自然。
        """
        # 先打印记忆召回详情（对话对象 + 各层命中）
        self.chat_memory(
            memory, query, person_profile=person_profile, promotion_eval=promotion_eval
        )
        # 再打印回复文本与耗时（elapsed = 当前时间 - t0）
        self.chat_reply(reply, (time.perf_counter() - t0) * 1000)


# 全局单例：整个应用共享同一个 AgentMonitor 实例
# 在 main.py 的 lifespan 中调用 agent_monitor.configure() 初始化日志通道
# 之后所有模块（agent.py、main.py、ws_handler.py 等）通过 import 使用同一实例
# 为什么是全局单例而非依赖注入：
#   monitor 是横切关注点（cross-cutting），几乎所有模块都可能调用，
#   通过 import 比通过构造函数传递更简洁，不会污染每个模块的接口
agent_monitor = AgentMonitor()
