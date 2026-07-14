"""MemoryPackV2 到 prompt 上下文的适配层。"""

from __future__ import annotations

from app.memory.schema import MemoryPackV2


def build_prompt_context(pack: MemoryPackV2) -> dict:
    """生成 MemoryPackV2 的边界上下文。

    召回入口统一使用 MemoryPackV2。这里不再把存储来源字段（core/recent/long_term/matches）
    重新暴露给 prompt，只保留当前会话、诊断和 pack 本体。
    """
    diag = pack.diagnostics
    return {
        "history": list(pack.history),
        "memory_pack": pack,
        "diagnostics": dict(diag),
        "person_id": diag.get("person_id"),
        "guest_mode": pack.guest_mode,
        "identity_hint": diag.get("identity_hint", ""),
        "interlocutor_mode": diag.get("interlocutor_mode", "girlfriend"),
        "memory_miss": bool(diag.get("memory_miss", 0) > 0),
    }
