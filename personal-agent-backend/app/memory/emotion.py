"""
情感轨迹追踪 —— 从 近期记忆 摘要中提取情感快照，构建跨会话的情感趋势感知。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  每次 近期记忆 摘要生成时，LLM 会附带一个 emotion JSON（包含 mood/情绪标签、
  intensity/强度、trigger/诱因、attitude/用户对话态度）。
  本模块读取最近 N 次会话的情感快照，构建一条"情感轨迹"，
  注入 system prompt 让 Agent 感知用户的情绪走向。

关键能力：
  - 情感极性判定：低落/焦虑/难过/生气/烦躁/疲惫/害怕/伤心 → 负面
                    开心/兴奋/期待 → 正面
                    平静/轻松 → 中性
  - 趋势预警：连续 2+ 次负面 → 提示"近期情绪持续偏负面，多倾听共情"
  - 单次预警：上次负面 → 提示"本轮留意用户状态，勿轻浮调侃"

数据来源：memory_items 中 kind=episode 记录的情感字段（JSON 格式）。
============================================================================
"""

from __future__ import annotations

import json

from app.session import store


def emotion_trajectory(device_id: str, person_id: str, last_n: int = 5) -> list[dict]:
    """读取最近 N 次会话的情感快照，按时间倒序排列。

    从近期记忆记录中提取有 emotion 字段的记录，
    解析 mood/intensity/trigger/attitude 四个维度，附上会话日期。

    数据获取策略：取 last_n * 2 条 近期记忆 记录作为备选（因为部分记录可能没有
    emotion 字段，或 emotion JSON 格式不合法），从中筛选出有效情感记录，
    最终返回最多 last_n 条。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        last_n:    最多取最近 N 条情感记录（默认 5）

    Returns:
        情感快照列表，按时间倒序（最近的在前面），每条含：
        - mood:      情绪标签（如 "开心"/"低落"/"焦虑"/"平静"）
        - intensity: 情绪强度（0.0 ~ 1.0，1.0 为最强烈）
        - trigger:   情绪诱因文本（什么事件触发了该情绪）
        - attitude:  用户对话态度标签（如 "倾诉"/"求助"/"吐槽"）
        - ts:        会话日期（YYYY-MM-DD 格式）
        无数据时返回空列表。
    """
    pid = str(person_id or "").strip()
    if not pid or not device_id:
        return []
    # 多取一些备选：部分 近期记忆 记录可能没有 emotion 字段，
    # 或者 emotion JSON 解析失败，所以需要比目标数更大的候选池
    rows = store.list_active_recent_memory(device_id, pid, limit=last_n * 2)
    out: list[dict] = []
    for row in rows:
        raw = row.get("emotion") or ""
        if not raw:
            continue
        try:
            # emotion 字段存储为 JSON 字符串，在 近期记忆 写入时由 LLM 生成
            emo = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            # LLM 输出偶尔可能不是合法 JSON，跳过该条记录
            continue
        # 至少需要有 mood 字段才算有效情感记录（仅有 intensity 无意义）
        if not isinstance(emo, dict) or not emo.get("mood"):
            continue
        emo["ts"] = str(row.get("created_at", ""))[:10]  # 截取日期部分，附加到快照中
        out.append(emo)
        if len(out) >= last_n:
            break
    return out


def format_emotion_prompt(device_id: str, person_id: str) -> str:
    """生成情感感知提示块，用于注入 system prompt。

    包含两部分内容：
      1. 最近几次对话的情感轨迹列表（时间、情绪、触发、态度）
      2. 基于情感极性的分级预警提示（负面情绪累积时提醒 Agent 共情倾听）

    预警分级策略（基于最近 3 次情感快照的负面次数）：
      - neg_count >= 2：高优先级——提示 Agent 多倾听共情，禁止说教
      - neg_count == 1：温和提醒——提示 Agent 留意用户状态，勿轻浮
      - neg_count == 0：不追加任何预警，Agent 正常行事

    Args:
        device_id: 设备标识
        person_id: 用户 ID

    Returns:
        Markdown 格式的情感轨迹提示块；无数据时返回空字符串（不会注入 prompt）。
    """
    trajectory = emotion_trajectory(device_id, person_id)
    if not trajectory:
        return ""

    lines = ["## 情绪轨迹（最近几次对话）"]

    # 情感极性映射表：将 LLM 输出的情绪标签映射到三分类。
    # neg = 负面（需要 Agent 特别关注），neu = 中性，pos = 正面
    mood_polarity = {
        "低落": "neg", "焦虑": "neg", "难过": "neg", "生气": "neg",
        "烦躁": "neg", "疲惫": "neg", "害怕": "neg", "伤心": "neg",
        "平静": "neu", "轻松": "pos",
        "开心": "pos", "兴奋": "pos", "期待": "pos",
    }
    # 取最近 3 次情感快照，统计负面情绪次数——这是预警判断的核心依据
    recent_moods = [e.get("mood", "") for e in trajectory[:3]]
    neg_count = sum(1 for m in recent_moods if mood_polarity.get(m) == "neg")

    # 逐条格式化情感轨迹，格式为：- 日期: 情绪(强度) — 诱因 [态度]
    for e in trajectory:
        mood = e.get("mood", "")
        intensity = e.get("intensity", 0)
        trigger = e.get("trigger", "")
        attitude = e.get("attitude", "")
        ts = e.get("ts", "")
        parts = [f"- {ts}: {mood}({intensity})"]
        if trigger:
            parts.append(f" — {trigger}")
        if attitude:
            parts.append(f" [{attitude}]")
        lines.append("".join(parts))

    # 情感预警：根据负面次数给出分级提示，引导 Agent 调整对话策略
    if neg_count >= 2:
        # 连续多轮负面情绪：用户可能处于低谷期，Agent 应侧重情感支持
        lines.append("【注意】近期情绪持续偏负面，多倾听共情，避免强行正能量或说教。")
    elif neg_count == 1:
        # 仅上一轮负面：可能是暂时性事件，Agent 保持敏感即可
        lines.append("【注意】上次对话情绪偏低，本轮留意用户状态，勿轻浮调侃。")

    return "\n".join(lines)
