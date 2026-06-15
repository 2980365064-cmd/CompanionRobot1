"""
人物画像（Person Profile）—— 每个陪伴对象一份 JSON 文档的归档级数据。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Profile 是"人物履历层"——存储性格、关键经历、情绪规律等深度画像数据。
  与 L0（核心事实）分工明确，不重复存储：

  对比：
    L0（核心事实层）     → 身份、禁忌、核心关系、纪念日、强偏好（每轮必载）
    Profile（人物履历层） → 性格/沟通特征、重要经历、情绪规律（低频深度谈心时启用）

画像生命周期：
  draft（临时） → 积累实质内容 → 确认转正（confirmed=True）
  转正时触发 sync_l0_from_profile 将身份/关系同步到 L0

设计简化（相比老版 person_profile.py + pipeline/promotion.py）：
  - 取消了多版本号（v1/v2/v3/draft），confirmed 标志替代 provisional/v1
  - 取消了复杂的晋升规则评估，有关系+实质内容即确认
  - 取消了批量存档 LLM 扫描，改为手动或定时触发
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from app.session import store

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(name: str) -> str:
    """规范化名字：去除所有空白字符，用于名字匹配比较。"""
    return re.sub(r"\s+", "", name.strip())


# ══════════════════════════════════════════════════════════════════════════════
# 画像 CRUD —— 创建、读取、规范化
# ══════════════════════════════════════════════════════════════════════════════

def empty_profile(
    nickname: str,
    *,
    person_id: str = "",
    relationship: str = "",
    aliases: list[str] | None = None,
) -> dict:
    """创建一个空的画像骨架（所有字段初始为默认值）。

    新用户注册时调用，生成一份结构完整但内容为空的画像。
    confirmed 初始为 False，需要后续积累实质内容后通过
    try_promote_provisional_profile 确认转正。

    Args:
        nickname:     用户昵称/显示名
        person_id:    用户唯一 ID（不提供则自动生成 UUID）
        relationship: 与 Agent（叶鹏祥）的关系（如"女朋友""闺蜜"等）
        aliases:      用户别名列表（用于多名字匹配）

    Returns:
        包含所有必要字段的画像字典骨架。
    """
    uid = str(person_id or uuid4()).strip()
    now = _utc_now()
    return {
        "person_id": uid,           # 用户唯一标识
        "display_name": nickname.strip() or "未知",  # 显示名
        "aliases": list(aliases or []),              # 别名列表
        "relationship": relationship.strip(),         # 与叶鹏祥的关系
        "personality": [],           # 性格/沟通特征列表
        "experiences": [],           # 关键经历列表
        "emotional_habit": [],       # 情绪规律列表
        "confirmed": False,          # 是否已确认转正
        "created_at": now,           # 创建时间
        "updated_at": now,           # 最后更新时间
    }


def normalize_profile(profile: dict) -> dict:
    """规范化画像字典：确保包含所有预期字段，兼容旧版数据结构迁移。

    核心职责：
    1. 统一 person_id 字段名（兼容老版 user_id）
    2. 从老版嵌套结构（extend_custom/personality_feature/memory_data）提取字段
    3. 清理废弃字段（provisional/version/user_id 等）
    4. 为缺失字段填充默认值

    Args:
        profile: 原始画像字典（可能是旧版格式）

    Returns:
        规范化后的画像字典（不含任何旧版废弃字段）。
    """
    p = deepcopy(profile)
    uid = str(p.get("person_id") or p.get("user_id") or uuid4())
    p["person_id"] = uid

    # 迁移 display_name：多个可能来源按优先级尝试
    if not p.get("display_name"):
        p["display_name"] = str(
            p.get("claimed_name")
            or p.get("display_name")
            or (p.get("basic_info") or {}).get("nickname")
            or "未知"
        ).strip()

    # 迁移 relationship：从老版 extend_custom.relationship_to_me 提取
    if not p.get("relationship"):
        ext = p.get("extend_custom") or {}
        p["relationship"] = str(
            p.get("relationship_to_me") or ext.get("relationship_to_me") or ""
        ).strip()

    # 迁移 aliases：从老版 extend_custom.aliases 提取
    if not p.get("aliases"):
        ext = p.get("extend_custom") or {}
        p["aliases"] = list(ext.get("aliases") or p.get("aliases") or [])

    # 迁移 personality：从老版 personality_feature 提取
    if isinstance(p.get("personality_feature"), list) and not p.get("personality"):
        p["personality"] = [str(x).strip() for x in p["personality_feature"] if str(x).strip()]
    if isinstance(p.get("personality_feature"), dict):
        items: list[str] = []
        pf = p["personality_feature"]
        # 从两个子字段合并：personality_label（性格标签）+ frustration_trigger（挫败触发点）
        for key in ("personality_label", "frustration_trigger"):
            for x in pf.get(key) or []:
                s = str(x).strip()
                if s:
                    items.append(s)
        if not p.get("personality"):
            p["personality"] = items

    # 迁移 experiences 和 emotional_habit：从老版 memory_data 提取
    md = p.get("memory_data") or {}
    if isinstance(md, dict):
        if md.get("important_experience") and not p.get("experiences"):
            p["experiences"] = [
                str(x.get("content", x) if isinstance(x, dict) else x).strip()
                for x in md["important_experience"]
                if str(x.get("content", x) if isinstance(x, dict) else x).strip()
            ]
        if md.get("emotional_habit") and not p.get("emotional_habit"):
            p["emotional_habit"] = [str(x).strip() for x in md["emotional_habit"] if str(x).strip()]

    # 为缺失字段填充默认值
    p.setdefault("personality", [])
    p.setdefault("experiences", [])
    p.setdefault("emotional_habit", [])
    p.setdefault("aliases", [])
    # confirmed 默认值：非 provisional 且非 draft 版本 → 视为已确认
    p.setdefault("confirmed", not p.get("provisional") and p.get("version") != "draft")
    p.setdefault("created_at", p.get("update_time") or _utc_now())
    p.setdefault("updated_at", p.get("update_time") or p.get("created_at") or _utc_now())

    # 删除所有旧版废弃字段，保持画像结构干净
    for drop in (
        "user_id", "basic_info", "preference", "behavior_stat",
        "personality_feature", "memory_data", "extend_custom",
        "relationship_to_me", "claimed_name", "known_in_memory",
        "provisional", "version", "update_time", "verified",
    ):
        p.pop(drop, None)

    return p


def profile_display_name(profile: dict | None) -> str:
    """从画像中提取显示名。无画像或无名时返回"未知"。"""
    if not profile:
        return "未知"
    p = normalize_profile(profile)
    return str(p.get("display_name") or "未知").strip() or "未知"


def profile_relationship(profile: dict) -> str:
    """从画像中提取与叶鹏祥的关系描述（如"女朋友"）。"""
    return str(normalize_profile(profile).get("relationship", "")).strip()


def profile_nicknames(profile: dict) -> list[str]:
    """提取画像中所有可能的称呼名（显示名 + 所有别名），用于多名字匹配。"""
    p = normalize_profile(profile)
    names: list[str] = []
    dn = profile_display_name(p)
    if dn and dn != "未知":
        names.append(dn)
    for a in p.get("aliases") or []:
        s = str(a).strip()
        if s:
            names.append(s)
    return names


def profile_matches_name(profile: dict, name: str) -> bool:
    """检查画像的显示名或别名是否与给定名字匹配。

    比较时忽略空白字符差异（通过 _normalize_name 处理）。
    """
    key = _normalize_name(name)
    if not key:
        return False
    for n in profile_nicknames(profile):
        if _normalize_name(n) == key:
            return True
    return False


def find_profile_by_name(device_id: str, name: str) -> dict | None:
    """在指定设备的所有画像中按名字查找匹配的画像。

    用于 identity 模块做名字→画像的模糊匹配。
    """
    for row in store.list_person_profiles(device_id):
        p = normalize_profile(row["profile"])
        if profile_matches_name(p, name):
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 画像确认转正（Promotion）—— 临时 draft 画像 → 正式画像
# ══════════════════════════════════════════════════════════════════════════════

def has_profile_substance(profile: dict) -> bool:
    """判断画像是否有实质内容（关系/性格/经历/情绪至少有一项有值）。

    画像确认的必要条件：没有实质内容的 draft 不应转正。
    """
    p = normalize_profile(profile)
    return bool(
        p.get("relationship")
        or p.get("personality")
        or p.get("experiences")
        or p.get("emotional_habit")
    )


def try_promote_provisional_profile(
    device_id: str, person_id: str, profile: dict | None, *, memory: dict | None = None,
) -> tuple[dict | None, str]:
    """尝试将临时画像转正为正式画像。

    转正条件：画像有实质内容（has_profile_substance 为 True）。
    转正后：
      1. confirmed 设为 True
      2. 触发 sync_l0_from_profile 将身份/关系同步到 L0
      3. 生成确认日志

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        profile:   画像字典（可能为 None）
        memory:    当前召回的记忆数据（预留，当前未直接使用）

    Returns:
        (profile, reason) —— profile 为转正后的画像（或 None），
        reason 为转正原因描述（如"画像确认 · 刘远慧"）。
    """
    if not profile:
        return None, ""
    p = normalize_profile(profile)
    if p.get("confirmed"):
        return p, "already confirmed"

    from app.memory.l0 import sync_l0_from_profile

    if has_profile_substance(p):
        # 确认转正：标记 confirmed，更新时间戳，写入存储
        p["confirmed"] = True
        p["updated_at"] = _utc_now()
        store.save_person_profile(device_id, p)
        # 转正后同步身份/关系到 L0 核心事实
        sync_l0_from_profile(device_id, person_id, p)
        nick = profile_display_name(p)
        logger.info("profile confirmed: %s", nick)
        return p, f"画像确认 · {nick}"

    return p, "no substance yet"


# ══════════════════════════════════════════════════════════════════════════════
# System prompt 注入块 —— 访客模式提示 / 人物履历归档
# ══════════════════════════════════════════════════════════════════════════════

def format_provisional_person_block(profile: dict | None) -> str:
    """生成访客模式下的提示块：告知 LLM 当前对话对象未确认身份。

    访客模式下，LLM 必须以"不认识"的态度交流，禁止假装老熟人寒暄。
    身份和关系信息以 L0 为准（访客没有 L0，所以是空的）。

    Args:
        profile: 临时画像（可能为 None）

    Returns:
        访客模式提示文本块。
    """
    if not profile:
        return (
            "## 当前对话对象\n"
            "（尚未确认对方是谁；身份与关系以 L0 为准，勿擅自假定）"
        )
    name = str(profile.get("claimed_name") or profile_display_name(profile) or "对方")
    return f"""## 当前对话对象（仅本轮自称，记忆库无此人记录）

用户自称「{name}」；L2/L3/L0 中**此前均无此人**完整记录。
- **禁止**假装认识、老熟人寒暄
- 身份/关系/喜恶以 L0 为准；无则可口语追问"""


def has_profile_archive_content(profile: dict | None) -> bool:
    """判断画像是否包含归档级别内容（性格/经历/情绪规律）。

    仅在画像已确认且至少有一项归档字段非空时返回 True。
    归档内容用于深度谈心场景，不同于 L0 的日常核心事实。
    """
    if not profile:
        return False
    p = normalize_profile(profile)
    if not p.get("confirmed"):
        return False
    return bool(p.get("personality") or p.get("experiences") or p.get("emotional_habit"))


def _format_profile_archive_entry(profile: dict) -> str:
    """将单份画像格式化为履历归档条目。

    格式示例：
    - 【刘远慧】
    - 性格/沟通：外向、直率
    - 关键经历：曾在杭州实习
    - 情绪规律：晚上容易焦虑

    Args:
        profile: 已确认的画像字典

    Returns:
        适合注入 system prompt 的归档文本块；无内容时返回空字符串。
    """
    p = normalize_profile(profile)
    if not has_profile_archive_content(p):
        return ""
    name = profile_display_name(p)
    lines: list[str] = [f"【{name}】"]
    if p.get("personality"):
        lines.append(f"性格/沟通：{'、'.join(str(x) for x in p['personality'][:8])}")
    if p.get("experiences"):
        lines.append("关键经历：" + "；".join(str(x) for x in p["experiences"][:6]))
    if p.get("emotional_habit"):
        lines.append("情绪规律：" + "；".join(str(x) for x in p["emotional_habit"][:6]))
    return "\n".join(f"- {ln}" if not ln.startswith("-") else ln for ln in lines)


def format_profile_archive_query_block(
    user_message: str, *, active_profile: dict | None = None,
) -> str:
    """当用户询问人生经历/性格等深度问题时，注入画像履历归档块。

    触发条件：user_message 包含"我的经历/成长经历/人生故事"等关键词。
    仅深度谈心场景启用，日常对话不注入（日常以 L0 为准）。

    Args:
        user_message:   用户当前消息
        active_profile: 当前活跃用户的画像（可选）

    Returns:
        画像履历归档 prompt 块；不需要或无可注入内容时返回空字符串。
    """
    from app.memory.guard import needs_profile_archive

    if not needs_profile_archive(user_message):
        return ""

    rows = store.list_all_person_profiles()
    profiles = [normalize_profile(r["profile"]) for r in rows]
    # 只取有归档内容且已确认的画像
    profiles = [p for p in profiles if has_profile_archive_content(p) and p.get("confirmed")]
    if not profiles:
        return ""

    from app.memory.guard import extract_asked_person_name, extract_self_name

    msg = user_message.strip()
    # 尝试从用户消息中提取目标人物名，做画像精准匹配
    focus = extract_self_name(msg) or extract_asked_person_name(msg)
    if focus:
        profiles = [p for p in profiles if profile_matches_name(p, focus)] or profiles

    # 确保活跃画像排在最前面
    if active_profile and has_profile_archive_content(active_profile):
        aid = str(active_profile.get("person_id") or "")
        if aid and not any(str(p.get("person_id")) == aid for p in profiles):
            profiles.insert(0, normalize_profile(active_profile))

    parts = [_format_profile_archive_entry(p) for p in profiles[:5]]
    body = "\n".join(x for x in parts if x)
    if not body:
        return ""
    return f"""## 人物履历归档（低频；仅深度谈心/聊自身经历时启用；基础事实以 L0 为准）

{body}

- 日常身份/喜恶/禁忌/纪念日 → 只用 L0，勿与归档混用
- 只许使用上述归档内容，禁止编造成长经历"""


# ══════════════════════════════════════════════════════════════════════════════
# 周期性画像更新 —— LLM 驱动的按需更新
# ══════════════════════════════════════════════════════════════════════════════

# LLM 提示词：根据近期的 L2 摘要和 L3 长期记忆，更新画像的性格/经历/情绪规律
# 仅新增不删除，避免丢失已确认内容
_PROFILE_UPDATE_PROMPT = """你是人物履历归档器。根据记忆材料更新性格/经历/情绪规律（仅新增，不删除已有）。

对象：{name}
当前归档：
{current}

材料 — L2 摘要：
{l2_text}

材料 — L3 长期记忆：
{l3_text}

只输出 JSON：
{{"need_update": true/false, "reason": "一句话",
  "patch": {{"personality": [], "experiences": [], "emotional_habit": []}}}}"""


def update_profile(device_id: str, person_id: str) -> dict | None:
    """对单一用户的画像做增量更新：用 LLM 分析近期 L2+L3 材料，提取新的归档条目。

    更新策略：仅新增（append only），不删除已有内容。LLM 判断是否需要更新，
    返回 need_update=true 时执行 patch 合并。

    Args:
        device_id: 设备标识
        person_id: 用户 ID

    Returns:
        更新后的画像字典；无更新需求时返回 None。
    """
    from app.llm import chat_completion_small

    raw = store.get_person_profile(person_id)
    if not raw:
        return None
    profile = normalize_profile(raw)
    # 只更新已确认的画像
    if not profile.get("confirmed"):
        return None

    # 取最近 3 天内的 L2 摘要和 L3 长期记忆作为分析材料
    since = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    l2 = store.list_episodic_since(person_id, since, limit=15)
    l3_rows = store.l3_list_person_memory(person_id, limit=30)
    l3_texts = [str(r.get("text", "")).strip() for r in l3_rows if str(r.get("text", "")).strip()][:20]

    if not l2 and not l3_texts:
        return None

    name = profile_display_name(profile)
    current = {
        "personality": profile.get("personality") or [],
        "experiences": profile.get("experiences") or [],
        "emotional_habit": profile.get("emotional_habit") or [],
    }
    prompt = _PROFILE_UPDATE_PROMPT.format(
        name=name,
        current=json.dumps(current, ensure_ascii=False),
        l2_text=json.dumps([str(r.get("summary", "")) for r in l2 if r.get("summary")], ensure_ascii=False),
        l3_text=json.dumps(l3_texts, ensure_ascii=False),
    )

    raw_out = chat_completion_small([{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", raw_out, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not data.get("need_update"):
        return None
    patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
    if not patch:
        return None

    # 合并 patch：仅新增不存在的条目，保留已有内容
    for key in ("personality", "experiences", "emotional_habit"):
        if isinstance(patch.get(key), list):
            existing = set(str(x) for x in profile.get(key) or [])
            for item in patch[key]:
                s = str(item).strip()
                if s and s not in existing:
                    profile[key].append(s)

    profile["updated_at"] = _utc_now()
    store.save_person_profile(device_id, profile)
    logger.info("profile updated: %s reason=%s", name, str(data.get("reason", ""))[:60])
    return profile


def update_all_profiles() -> dict[str, int]:
    """对所有已确认的画像执行周期性更新。

    Returns:
        {"updated": N, "total": M} —— N 为本次成功更新的画像数，M 为总画像数。
    """
    rows = store.list_all_person_profiles()
    updated = 0
    for row in rows:
        pid = str(row.get("person_id") or "")
        device_id = str(row.get("device_id") or "")
        profile = row.get("profile") or {}
        if not pid or not normalize_profile(profile).get("confirmed"):
            continue
        if update_profile(device_id, pid):
            updated += 1
    return {"updated": updated, "total": len(rows)}
