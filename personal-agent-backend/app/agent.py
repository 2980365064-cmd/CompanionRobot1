"""单轮对话编排 —— 身份识别 → 记忆召回 → Prompt 组装 → LLM 生成 → 异步入库。

本模块是陪伴机器人的"对话引擎"，编排每轮对话的完整处理流水线。

============================
记忆分层架构（由浅到深）
============================

  L1 working   本会话滑动窗口（最后 N 轮对话，访客仅有此层）
  L0 core      高置信核心事实（身份/禁忌/关键人物/里程碑/偏好）
               每轮全量注入 prompt，无需检索
  L2 episodic  近 7 天会话摘要，通过向量语义检索注入 prompt
  L3 corpus    长期记忆库（聊天记忆 + 语料知识），统一向量检索
               collection='memory' 为用户记忆，'corpus' 为知识库
  Profile      人物性格/履历归档（非向量库，固定注入 Profile Card）

============================
单轮主路径（handle_chat）
============================

  1. get_or_create_session         获取或创建当前会话
  2. resolve_person_before_memory  身份门控：解析「名字 XXX ID xxx」
                                   或维持 tmp_* 访客身份
                                   必须在 recall 之前执行，否则访客会误查 L2/L3
  3. memory_router.recall          按 person_id 召回各层记忆
                                   访客模式跳过 L0/L2/L3，仅保留 L1
  4. load_profile_card             加载机器人人格描述
  5. build_messages                拼装 system prompt（含反幻觉规则、
                                   多层记忆块、主动话题规则、输出要求）
  6. chat_completion_async         单次调用 DeepSeek，同时生成主回复和
                                   可选的主动话题（||| 分隔）
  7. _parse_reply                  解析 ||| 分隔符，校验主动话题合法性
  8. _append_turn                  用户/主回复/主动话题写入 L1
  9. _post_process（后台异步）     核心记忆/长期记忆捕获、记忆修正、
                                   显式「记住」语料入库
                                   不阻塞本轮回复返回

============================
关键设计决策
============================

  - 身份门控放在 recall 之前：访客不允许访问任何长期记忆，
    这是隐私保护的核心
  - 单次 LLM 调用生成回复+话题：用 ||| 分隔符在同一个推理上下文内
    同时输出主回复和主动话题，从根源杜绝模型将自身输出误判为用户输入
  - _post_process 使用 asyncio.create_task 异步执行：
    用户不需要等待后台写入完成才看到回复
  - 主动话题经三道纯规则防线校验后才发送：必须含"你"、问号结尾、
    长度≤20字、禁止自问自答开头
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone

from app.config import settings
from app.llm import chat_completion_async, chat_completion_small_async, chat_completion_stream_async
from app.memory.extractor import consolidate_session
from app.memory.guard import (
    ANTI_HALLUCINATION_RULES,
    extract_self_name,
    memory_l3_texts,
    memory_miss_level,
    name_in_memory_text,
    should_force_active_topic,
    should_suppress_active_topic,
    user_message_hints,
    validate_active_topic,
)
from app.memory.self_state import format_self_state_prompt
from app.memory.l0 import (
    extract_l0_from_session_summary,
    format_l0_block,
)
from app.memory.l1 import working_memory
from app.memory.contacts import format_contacts_prompt_block
from app.memory.identity import (
    is_guest_bind_failure_hint,
    is_verified_person_id,
)
from app.memory.interlocutor import (
    MODE_GIRLFRIEND,
    MODE_VISITOR,
    enforce_mode_switch_reply,
    resolve_interlocutor_before_memory,
)
from app.memory.emotion import format_emotion_prompt
from app.memory.profile import (
    format_provisional_person_block,
    format_profile_archive_query_block,
    profile_display_name,
)
from app.memory.orchestrator import orchestrator
from app.memory.router import memory_router  # 保留旧 router 以备降级
from app.persona.card import load_profile_card
from app.monitor import agent_monitor
from app.session import store

logger = logging.getLogger(__name__)


# ============================
# 时间感知辅助函数
# ============================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _relative_time(iso_str: str | None, now: datetime | None = None) -> str:
    """将 ISO 时间戳转成中文相对时间描述，用于 prompt 中标记记忆时效。

    返回示例："3分钟前"、"2小时前"、"昨天"、"3天前"、"上周"、"2周前"、"较早"

    为什么不用绝对时间：
      绝对时间（如 2026-06-07 14:30）需要 LLM 自行计算时间差，
      大部分 LLM 不擅长这个，容易出错。相对时间直接给出结论，
      LLM 可以准确地感知"这是最近的事"还是"很久以前的事"。
    """
    if not iso_str:
        return ""
    try:
        ts = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or _now_utc()
    delta = now - ts
    seconds = delta.total_seconds()
    if seconds < 0:
        return "刚刚"
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时前"
    if seconds < 172800:
        return "昨天"
    if seconds < 604800:
        return f"{int(seconds // 86400)}天前"
    if seconds < 1209600:
        return "上周"
    if seconds < 2592000:
        return f"{int(seconds // 604800)}周前"
    return "较早"


def _abs_time(iso_str: str | None) -> str:
    """将 ISO 时间戳转成绝对时间字符串，用于记忆块时效标注。

    返回示例："2026年6月7日 14:30"
    时区为 UTC，与 system prompt 中当前时间头保持一致。
    """
    if not iso_str:
        return ""
    try:
        ts = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return f"{ts.year}年{ts.month}月{ts.day}日 {ts.strftime('%H:%M')}"


# ============================
# Prompt 组装辅助函数
# ============================

def _l3_covers_name(name: str, memory: dict) -> bool:
    """检查 L3 长期记忆中是否出现了某个人的名字。

    用途：当用户自称某个名字时，检查 L3 中是否有该名字的相关记忆。
    如果有，说明这个身份之前出现过，可以按记忆内容自然回应；
    如果没有，说明这可能是首次见面或冒充，需要谨慎处理。

    参数:
        name:   要检查的人名
        memory: memory_router.recall() 的返回值
    返回:
        True 如果 L3 匹配的文本中包含该名字
    """
    if not name:
        return False
    texts = memory_l3_texts(memory)
    for s in texts:
        if name_in_memory_text(name, str(s)):
            return True
    return False


def _safe_memory_prompt_fallback() -> str:
    """安全 fallback：当 MemoryPackV2.format_prompt_block() 异常时使用。

    绝对不包含 L0/L2/L3/向量/检索/命中/记忆库 等工程词，
    确保 LLM 在任何意外情况下也不会看到旧式工程术语结构。
    """
    return (
        "## 你该记得的相关事\n"
        "- 这轮记忆整理有点不完整，只按已确认事实自然回复，不要编。\n"
        "- 不确定的地方要坦诚说不太记得，可以让对方提醒你。\n"
    )


def _format_l3_block(memory: dict) -> str:
    """将长程记忆命中结果格式化为 prompt 中的相关回忆块。

    参数:
        memory: memory_router.recall() 的返回值

    返回:
        str: 格式化的回忆块文本
    """
    if memory.get("memory_miss"):
        return (
            "（你不记得相关的事 → 别编，诚实说不太清楚就好）"
        )
    items = (memory.get("matches") or {}).get("l3") or []
    if not items:
        return "（没有找到相关的回忆；不太记得的话诚实说就好）"
    lines: list[str] = []
    for m in items:
        rel = _abs_time(m.get("created_at", ""))
        time_label = f"（{rel}的事）" if rel else ""
        lines.append(f"- {time_label} {m.get('text', '')}")
    body = "\n".join(lines)
    return body + "\n（只能说以上提到的内容，没提到的别瞎编）"


def _format_related_block(memory: dict) -> str:
    """将关联记忆结果格式化为 prompt 中的关联回忆块。

    参数:
        memory: memory_router.recall() 的返回值

    返回:
        str: 格式化的关联回忆块，无命中时返回空字符串
    """
    related = (memory.get("matches") or {}).get("related") or []
    if not related:
        return ""
    lines = [
        f"- {r.get('text', '')}"
        for r in related[:6]
        if r.get("text")
    ]
    if not lines:
        return ""
    return "## 顺带想起的\n" + "\n".join(lines) + "\n"


_GF_SCENE_EMOTION = re.compile(
    r"想你了|撒娇|委屈|生气|不开心|难过|哭了|哄|抱抱"
)
_GF_SCENE_TEASE = re.compile(
    r"哼|嘿嘿|嘻嘻|哦\?|哦？|又来了|咋滴|笑死|满意没"
)
_GF_SCENE_PHILOSOPHY = re.compile(
    r"人生|意义|活着|未来|永远|距离|陪伴|时间|迷茫|孤独|爱是什么"
)
_GF_SCENE_CASUAL = re.compile(
    r"吃了|睡了|干嘛|在哪|忙吗|到家|上班|上课|实习"
)


# 中文括号旁白/动作描写，语音场景必须去除（代码层兜底，prompt 已禁止但 LLM 偶尔违反）
_PAREN_STAGE_RE = re.compile(r"（[^）]*）")


def _strip_stage_directions(text: str) -> str:
    """去除中文括号旁白/动作描写，防止语音朗读时读出括号内容。"""
    return _PAREN_STAGE_RE.sub("", text).strip()


def _gf_scene_hint(message: str) -> str:
    """女友模式本轮风格路由（persona 路由表的代码层提示，避免三种风格混用）。"""
    msg = (message or "").strip()
    if not msg:
        return ""
    if _GF_SCENE_PHILOSOPHY.search(msg):
        return "- 【本轮路由】谈感受/人生 → 可走心，哲理最多一句，说完回到她"
    if _GF_SCENE_EMOTION.search(msg):
        return "- 【本轮路由】她要情绪/撒娇 → 先深情再轻逗，哲理不要"
    if _GF_SCENE_TEASE.search(msg):
        return "- 【本轮路由】斗嘴场景 → 俏皮甜损为主，别强行哲理"
    if _GF_SCENE_CASUAL.search(msg) or len(msg) <= 12:
        return "- 【本轮路由】日常闲聊 → 斗嘴+温情，别每轮哲理"
    return ""


# ============================
# Prompt 组装主函数
# ============================

def build_messages(
    profile: str,
    memory: dict,
    user_message: str,
    *,
    device_id: str,
    person_profile: dict | None = None,
    memory_pack: object = None,
) -> list[dict]:
    """核心函数：拼装完整的 LLM messages 列表（system + history + 当前用户消息）。

    参数:
        profile:        机器人人格描述文本（persona.md 内容）
        memory:         记忆召回结果，含各层匹配数据和身份提示
        user_message:   用户本轮输入的文本
        device_id:      设备标识（用于特定逻辑如女友语气判断）
        person_profile: 当前对话对象的画像（可为 None，访客模式）
        memory_pack:    可选的 MemoryPack 实例（提供后使用人类化 prompt 格式，
                        替代工程分层术语）

    返回:
        list[dict]: OpenAI 格式的消息列表，可直接传给 chat_completion

    组装逻辑分两大分支：

    --- 分支一：访客模式 ---
      仅使用本会话窗口，不访问长期记忆。
      system prompt 包含：
        - 机器人人格（profile）
        - 反幻觉规则
        - 身份提示（若检测到自我介绍）
        - 访客模式警告（禁止假装有记忆）
        - 本会话缓存
        - 每轮顺带问一句"怎么称呼你呀"以推动实名（轮换问法，禁止复读）

    --- 分支二：已实名模式 ---
      注入用户稳定记忆 + 近期会话摘要 + 相关回忆。
      system prompt 包含：
        - 机器人人格
        - 反幻觉规则
        - 关系状态 + 当前场景（由 MemoryPack 格式化）
        - 身份提示 + 人物块（provisional 或 archive）
        - 情绪提示 + 自我状态
        - 相关回忆（长期记忆命中）
        - 用户消息提示（检测到特定意图时注入额外指令）
        - 输出要求（叶鹏祥口吻、字数限制、女友语气等）
    """
    now = _now_utc()
    # 格式化近期情景摘要（最近几天的对话压缩），带时间戳标注时效
    l2_matches_raw = (memory.get("matches") or {}).get("l2") or []
    if l2_matches_raw:
        episodic_block = "\n".join(
            f"- [{_abs_time(m.get('created_at', ''))}] {m.get('text', '')}"
            for m in l2_matches_raw
        ) or "（最近暂时没有特别的事需要想起）"
    else:
        episodic_block = "（最近暂时没有特别的事需要想起）"

    # 提取用户声称的名字（如果本轮有自我介绍）
    claimed_name = ""
    if person_profile:
        claimed_name = str(
            person_profile.get("claimed_name")
            or profile_display_name(person_profile)
            or ""
        ).strip()
    # 检查 L3 中是否有该名字的相关记忆
    l3_knows_claimed = bool(claimed_name) and _l3_covers_name(claimed_name, memory)

    # 构建人物块：描述当前对话对象的信息
    person_block = ""
    if person_profile and (
        person_profile.get("provisional") or person_profile.get("known_in_memory") is False
    ):
        # 临时画像：对象身份尚未完全确认
        if person_profile.get("provisional") and l3_knows_claimed:
            # L3 中有此人相关记忆，可以按记忆回应但仍然谨慎
            person_block = (
                f"## 当前对话对象\n"
                f"用户自称「{claimed_name}」；你的长期记忆中有与此人相关的记录，"
                f"可按已有记忆自然回应，没提到的不编。"
            )
        else:
            person_block = format_provisional_person_block(person_profile)
    else:
        # 正式画像：查询 archive query block
        archive_block = format_profile_archive_query_block(
            user_message, active_profile=person_profile
        )
        if archive_block:
            person_block = archive_block

    # L0 核心记忆块：全量注入，不需要检索
    # L0 块必须放在 system prompt 最前面（优先级最高），确保 LLM 优先遵循
    # 这些是不可变的高置信度事实（身份、禁忌、关键人物等）
    person_id = (person_profile or {}).get("person_id") if person_profile else memory.get("person_id")
    l0_block = format_l0_block(str(person_id or memory.get("person_id") or ""))
    l0_prefix = f"{l0_block}\n\n" if l0_block else ""
    person_section = f"\n{person_block}\n" if person_block else ""
    contacts_block = ""
    if person_id and not memory.get("guest_mode"):
        contacts_block = format_contacts_prompt_block(
            device_id,
            str(person_id),
            user_message,
            owner_profile=person_profile,
        )
        if contacts_block:
            contacts_block = f"\n{contacts_block}\n"

    # 身份提示（来自 interlocutor / identity 模块）
    identity_hint = str(memory.get("identity_hint") or "").strip()
    identity_section = f"\n{identity_hint}\n" if identity_hint else ""
    interlocutor_mode = str(memory.get("interlocutor_mode") or MODE_GIRLFRIEND)

    # ═══════════════════════════════════════
    # 分支一：访客对话角色（口令「访客模式」）
    # ═══════════════════════════════════════
    if interlocutor_mode == MODE_VISITOR:
        working_block = "\n".join(
            f"- [{_relative_time(m.get('created_at', ''), now)}] {m['role']}: {m['content'][:120]}"
            for m in (memory.get("working") or [])[-6:]
        ) or "（本会话尚无上下文）"

        l3_block = _format_l3_block(memory)
        related_block = _format_related_block(memory)
        episodic_block = "（最近暂时没有特别的事需要想起）"
        l2_matches_raw = (memory.get("matches") or {}).get("l2") or []
        if l2_matches_raw:
            episodic_block = "\n".join(
                f"- [{_abs_time(m.get('created_at', ''))}] {m.get('text', '')}"
                for m in l2_matches_raw
            ) or episodic_block

        system = f"""{profile}

## 当前时间
现在是 {now.strftime('%Y年%m月%d日 %H:%M')}（UTC）。

{ANTI_HALLUCINATION_RULES}
{identity_section}
## 访客模式（对外人/朋友 · 非女友）
- 当前是跟**访客/朋友**聊天，不是跟女友刘远慧私密聊
- **禁止**情侣亲密、撒娇、性癖、私密话题；persona 里「只能跟刘远慧聊」的内容一律不提
- **禁止**称呼对方为大炮/秋雨/乖乖等女友专属称呼
- 叶鹏祥第一人称，友好自然，像跟朋友微信闲聊
- 可参考下方记忆，但不要照搬对女友的语气

## 本会话上下文
{working_block}

## 最近聊过的
{episodic_block}

## 相关回忆
{l3_block}
{related_block}

## 本轮输出要求
- 按 profile 身份：对朋友/访客**大大咧咧**、随性短句；禁止情侣亲密与女友专属称呼
- 口语 5～{settings.max_reply_chars} 字；禁止括号动作/客服腔
- 禁止每轮追问名字+ID或实名绑定

{_ACTIVE_TOPIC_RULES}
"""
        messages: list[dict] = [{"role": "system", "content": system}]
        for m in memory["working"][-6:]:
            if m["role"] == "user":
                messages.append({"role": "user", "content": f"[对方] {m['content']}"})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": f"[对方] {user_message}"})
        return messages

    # ═══════════════════════════════════════
    # 分支二：旧版访客实名（兼容遗留 session，极少触发）
    # ═══════════════════════════════════════
    if memory.get("guest_mode"):
        working_block = "\n".join(
            f"- [{_relative_time(m.get('created_at', ''), now)}] {m['role']}: {m['content'][:120]}"
            for m in (memory.get("working") or [])[-6:]  # 只取最近 6 条
        ) or "（本会话尚无上下文）"

        # 访客身份引导
        # 1. pending 待确认 → 口语追问确定/不是
        # 2. 绑定失败 → 口语说明没绑上，请重发名字+ID
        # 3. 其余访客轮次 → 每轮必须问对方是谁（轮换问法）
        has_pending = "待确认" in identity_hint
        has_bind_failure = is_guest_bind_failure_hint(identity_hint)
        if has_pending:
            identity_followup = "- 先接话，后半段用自然口语确认这个新身份（每次说法不一样），用户确认/否认后按规则处理"
        elif has_bind_failure:
            identity_followup = "- 按绑定失败提示口语说明问题，请对方重新发名字+ID，禁止客服通知腔"
        else:
            identity_followup = "- 先正常接话，后半段顺口问对方是谁（名字+ID），说法每轮不同，禁止复读"

        system = f"""{profile}

## 当前时间
现在是 {now.strftime('%Y年%m月%d日 %H:%M')}（UTC），请据此判断记忆时效，严禁把很久前的事当刚发生的。

{ANTI_HALLUCINATION_RULES}
{identity_section}
## 访客模式（未实名）
- 你不记得这个人，也没有关于 ta 的记忆——不要假装认识
- 仅依据下方的会话内容自然闲聊，不要编造对方的身份、关系或经历

## 本会话内容
{working_block}

## 本轮输出要求
- 叶鹏祥第一人称，口语短句像微信语音转文字
{identity_followup}
- 禁止括号动作/客服腔/[对方]前缀；5～{settings.max_reply_chars}字

{_ACTIVE_TOPIC_RULES}
"""
        messages: list[dict] = [{"role": "system", "content": system}]
        for m in memory["working"][-6:]:
            if m["role"] == "user":
                messages.append({"role": "user", "content": f"[对方] {m['content']}"})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": f"[对方] {user_message}"})
        return messages

    # ═══════════════════════════════════════
    # 分支二：已实名模式（全量记忆注入）
    # ═══════════════════════════════════════

    l3_block = _format_l3_block(memory)

    # 临时画像 + L3 不认识此人：额外警告勿混淆人名
    if person_profile and (
        person_profile.get("provisional") or person_profile.get("known_in_memory") is False
    ) and not l3_knows_claimed:
        l3_block += (
            "\n（注意：该对象仅本轮自称，跟你记忆中的其他人名勿混为一谈）"
        )

    # 记忆未命中提示：按级别注入不同强度的提示
    # 不再一刀切禁止——让 LLM 带着 persona 自然回应
    miss_lv = memory_miss_level(memory)
    memory_miss_block = ""
    if miss_lv == 2:
        memory_miss_block = (
            "\n## 记忆提示\n"
            "你翻了一下记忆，对这件事确实没什么印象。"
            "诚实说不太记得就好，用你自己的口吻——可以追问对方补充，但别编。\n"
        )
    elif miss_lv == 1:
        memory_miss_block = (
            "\n## 记忆提示\n"
            "你隐约记得一些相关的，但不太确定。"
            "可以提一下你记得的部分，不确定的地方坦诚说。\n"
        )

    # 联想记忆块
    related_block = _format_related_block(memory)

    # 用户消息额外提示：检测到特定意图时注入指令
    # 例如"你还记得XXX吗"会触发提示 LLM 检查 L2/L3 是否有相关内容
    hints = user_message_hints(
        user_message,
        memory=memory,
        person_profile=person_profile,
        device_id=device_id,
    )
    hints_block = f"\n{hints}\n" if hints else ""

    scene_hint = ""
    if interlocutor_mode == MODE_GIRLFRIEND:
        hint_line = _gf_scene_hint(user_message)
        if hint_line:
            scene_hint = f"{hint_line}\n"

    # 情绪提示块：基于近几轮对话判断当前对话对象的情绪状态
    emotion_block = ""
    if person_id and not memory.get("guest_mode"):
        emotion_block = format_emotion_prompt(device_id, str(person_id))

    # 自我状态：机器人的当前活动/心情，让机器人能自然分享自己的生活
    self_state_block = ""
    if interlocutor_mode == MODE_GIRLFRIEND:
        self_state_block = format_self_state_prompt()

    # 拼装最终的 system prompt
    # 区块顺序经过精心设计：L0（最高优先级）→ 人格 → 反幻觉规则 →
    # 身份信息 → 情绪提示 → 联想记忆 → L2 摘要 → L3 长期记忆 →
    # 未命中警告 → 消息提示 → 输出要求
    # 顺序很重要：L0 在最前，输出要求（格式指令）在最后

    # ── 人类化提示块（Phase 4 / 阶段 3.0 安全加固）───────────────────
    if memory_pack is not None:
        try:
            main_memory_block = memory_pack.format_prompt_block()
        except Exception as exc:
            logger.warning("MemoryPack format_prompt_block failed: %s", exc)
            agent_monitor.warn(f"MemoryPack prompt fallback: {exc}")
            main_memory_block = _safe_memory_prompt_fallback()
    else:
        main_memory_block = _safe_memory_prompt_fallback()

    system = f"""{profile}

## 当前时间
现在是 {now.strftime('%Y年%m月%d日 %H:%M')}（UTC），请据此判断记忆时效，严禁把很久前的事当刚发生的。

{ANTI_HALLUCINATION_RULES}
{identity_section}{person_section}{contacts_block}{emotion_block}{self_state_block}
{main_memory_block}
{hints_block}
## 本轮输出要求（必须执行）
- 按 profile 与场景路由；第一人称口语，像微信语音转文字
{scene_hint}- 禁止括号动作/客服腔；「你妈」绝对禁止
- 不超过 {settings.max_reply_chars} 字；禁止助手腔、markdown、分点列表
- 禁止输出 [对方] 前缀——那是角色标记，不是对话内容

{_ACTIVE_TOPIC_RULES}
"""
    messages: list[dict] = [{"role": "system", "content": system}]
    for m in memory["working"]:
        if m["role"] == "user":
            messages.append({"role": "user", "content": f"[对方] {m['content']}"})
        else:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"[对方] {user_message}"})
    return messages


# ============================
# 主动话题规则（注入 system prompt 输出要求）
# ============================

_ACTIVE_TOPIC_RULES = """## 主动话题规则
输出格式：主回复写完后，如果判断适合开启新话题，在末尾追加 |||话题内容
不生成也完全可以——只输出主回复即可，不需要加 |||

**跳过（不要加 |||）：**
- 对方在倾诉负面情绪或提问
- 对方不想聊 / 要结束对话
- 你实在想不到合适话题

**生成（加 |||）：**
- 对方只回了嗯/哦/好，明显等你带话题
- 你的主回复太短（≤5字），话要断了
- 其余场景自行判断，可以主动聊聊对方相关的事

**话题要求：**
- 必须向对方提问，带"你"字，以？结尾
- 从记忆里找对方感兴趣的话题，或顺着当前聊的自然延伸
- 5～20字，口语，像朋友随口一问
- 禁止：是的/对呀/没错/我也觉得/你呢/你咋样 开头
- 禁止反问，禁止模仿对方语气"""


# ============================
# 回复解析：拆分主回复和主动话题
# ============================

def _parse_reply(raw: str) -> tuple[str, str | None]:
    """解析 LLM 原始输出，按 ||| 分隔符拆分为主回复和主动话题。

    参数:
        raw: LLM 原始输出文本

    返回:
        (main_reply, active_topic): 主回复文本和可选的主动话题（无则为 None）

    解析规则：
      - 用 ||| 分隔，最多取第一段和最后一段之间的内容
      - 只用第一个 ||| 来分割（防止回复内容中出现多个 |||）
      - 主回复和话题两端去空白
      - 话题为空字符串时返回 None
    """
    if "|||" not in raw:
        return raw, None
    parts = raw.split("|||", 1)
    main = parts[0].strip()
    topic = parts[1].strip() if len(parts) > 1 else ""
    if not topic or len(topic) < 2:
        return main, None
    return main, topic


# ============================
# L1 工作记忆写入
# ============================

def _append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """将本轮 user/assistant 消息追加写入 L1（messages 表）。

    L1 是当前会话的完整对话历史，存储在 SQLite messages 表中。
    每轮结束后同步写入（因为内容很少，没必要异步），
    后续轮次通过 get_recent_messages 读取形成上下文窗口。
    """
    working_memory.append(session_id, "user", user_msg)
    working_memory.append(session_id, "assistant", assistant_msg)


# ============================
# 单轮对话主函数
# ============================

async def handle_chat(device_id: str, session_id: str, message: str) -> tuple[str, str, str | None]:
    """处理一轮用户消息，返回 (回复文本, session_id, 主动话题文本或None)。

    参数:
        device_id:  设备标识
        session_id: 当前会话 ID（空字符串表示创建新会话）
        message:    用户输入文本

    返回:
        (reply_text, session_id, follow_up_text)：助手回复、更新后的会话 ID、
        以及可选的主动话题文本（模型自主判断 + 三道防线校验）

    处理步骤：
      Step 1: 获取或创建 session
      Step 2: 身份门控（resolve_person_before_memory）
      Step 3: 并行加载人格卡片和记忆召回
      Step 4: 拼装 prompt（build_messages，含主动话题规则）
      Step 5: 单次 LLM 调用，同时生成主回复 + 可选的主动话题（||| 分隔）
      Step 6: 解析 |||，三道防线校验主动话题
      Step 7: 截断回复至 max_reply_chars（代码层双保险）
      Step 8: 写入 L1（同步）
      Step 9: 控制台输出（agent_monitor）
      Step 10: 异步入库（asyncio.create_task）
    """
    t0 = time.perf_counter()

    # Step 1: 获取或创建会话
    session_id = await asyncio.to_thread(
        store.get_or_create_session, device_id, session_id or None
    )
    _t1 = time.perf_counter()
    agent_monitor.set_timing("session", (_t1 - t0) * 1000)

    # 阶段 3.0：box-drawing 轮次头
    agent_monitor.start_turn(device_id, message, session_id)

    # Step 2: 对话角色（默认女友，口令切换访客/女友）
    ictx = await asyncio.to_thread(
        resolve_interlocutor_before_memory,
        device_id,
        session_id,
        message,
    )
    person_id = ictx.person_id
    person_profile = ictx.person_profile
    identity_hint = ictx.hint
    identity_event = ictx.monitor_event
    _t2 = time.perf_counter()
    agent_monitor.set_timing("identity", (_t2 - _t1) * 1000)

    # Step 3: 并行加载人格卡片和记忆召回
    # 通过 orchestrator 获取 MemoryPack，兼容旧 dict 格式
    memory_pack, profile = await asyncio.gather(
        asyncio.to_thread(
            orchestrator.recall,
            device_id,
            session_id,
            message,
            person_id=person_id,
        ),
        asyncio.to_thread(load_profile_card, ictx.interlocutor_mode, message),
    )
    memory = memory_pack.to_legacy_dict()
    memory["identity_hint"] = identity_hint
    # recall() 已按 person_id 设置 guest_mode；勿用 interlocutor 的 False 覆盖（女友模式≠已实名）
    memory["interlocutor_mode"] = ictx.interlocutor_mode
    memory["person_id"] = person_id
    _t3 = time.perf_counter()
    agent_monitor.set_timing("recall", (_t3 - _t2) * 1000)

    # 身份展示
    agent_monitor.identity(person_profile, memory, ictx.interlocutor_mode)

    if identity_event:
        agent_monitor.event(identity_event)
    elif identity_hint:
        agent_monitor.event(identity_hint[:48])

    # MemoryPack 摘要
    agent_monitor.memory_pack_v2(memory_pack)
    agent_monitor.memory_pack_summary(memory, memory_pack)

    # Step 4: 拼装 prompt
    # 记忆未命中不再是硬阻断——LLM 带着 persona 自然回应，
    # build_messages 会根据 memory_miss_level 注入不同强度的提示
    messages = build_messages(
        profile, memory, message, device_id=device_id, person_profile=person_profile,
        memory_pack=memory_pack,
    )
    _t4 = time.perf_counter()
    agent_monitor.set_timing("prompt", (_t4 - _t3) * 1000)
    agent_monitor.prompt_summary(messages)

    temp = settings.chat_temperature
    if user_message_hints(message, memory=memory, person_profile=person_profile, device_id=device_id):
        temp = min(temp, 0.72)

    # Step 5: 单次 LLM 调用，同时生成主回复和可选的主动话题
    reply_raw = await chat_completion_async(messages, temperature=temp)
    reply_raw = reply_raw.strip()
    _t5 = time.perf_counter()
    agent_monitor.set_timing("llm", (_t5 - _t4) * 1000)

    # Step 6: 解析 ||| 分隔符，拆分主回复和主动话题
    reply, active_topic = _parse_reply(reply_raw)

    # Step 6.5: 去除括号旁白（代码层兜底，语音场景必须）
    reply = _strip_stage_directions(reply)

    # Step 7: 代码层校验 + 截断
    # 根据用户消息和主回复内容判断是否应该屏蔽主动话题
    if active_topic and should_suppress_active_topic(message):
        active_topic = None
    # 三道纯规则防线校验
    if active_topic:
        active_topic = validate_active_topic(active_topic)
    # 截断主回复
    if len(reply) > settings.max_reply_chars:
        reply = reply[: settings.max_reply_chars]
    reply = enforce_mode_switch_reply(reply, ictx.mode_switch_ack)

    # Step 8: 写入 L1
    await asyncio.to_thread(_append_turn, session_id, message, reply)
    if active_topic:
        await asyncio.to_thread(
            working_memory.append, session_id, "assistant", active_topic
        )

    # Step 9: 控制台输出（阶段 3.0）
    # end_turn 输出回复、耗时和 box-drawing 结尾
    # 后台事件（Consolidator 结果）由 _post_process 中的 agent_monitor.event() 展示
    agent_monitor.end_turn(reply, t0)

    # Step 10: 后台异步入库
    asyncio.create_task(
        _post_process(device_id, session_id, message, reply, memory, person_id)
    )
    return reply, session_id, active_topic


async def handle_chat_stream(device_id: str, session_id: str, message: str):
    """流式版 handle_chat：逐步 yield (event_type, data) 元组。

    与 handle_chat 逻辑相同，但 LLM 调用使用流式模式，
    每收到一个 token 即 yield 给调用方，实现逐字输出。

    yield 的事件类型:
        ("token", str)    - 增量文本 token
        ("done", None)    - 生成完成
        ("error", str)    - 错误信息

    用法:
        async for event, data in handle_chat_stream(...):
            if event == "token":
                send_to_client(data)
            elif event == "done":
                # data = (reply, session_id, active_topic)
                finalize(data)
    """
    t0 = time.perf_counter()

    session_id = await asyncio.to_thread(
        store.get_or_create_session, device_id, session_id or None
    )
    _t1 = time.perf_counter()
    agent_monitor.set_timing("session", (_t1 - t0) * 1000)

    agent_monitor.start_turn(device_id, message, session_id)

    ictx = await asyncio.to_thread(
        resolve_interlocutor_before_memory, device_id, session_id, message,
    )
    person_id = ictx.person_id
    person_profile = ictx.person_profile
    identity_hint = ictx.hint
    identity_event = ictx.monitor_event
    _t2 = time.perf_counter()
    agent_monitor.set_timing("identity", (_t2 - _t1) * 1000)

    memory_pack, profile = await asyncio.gather(
        asyncio.to_thread(
            orchestrator.recall, device_id, session_id, message, person_id=person_id,
        ),
        asyncio.to_thread(load_profile_card, ictx.interlocutor_mode, message),
    )
    memory = memory_pack.to_legacy_dict()
    memory["identity_hint"] = identity_hint
    memory["interlocutor_mode"] = ictx.interlocutor_mode
    memory["person_id"] = person_id
    _t3 = time.perf_counter()
    agent_monitor.set_timing("recall", (_t3 - _t2) * 1000)

    agent_monitor.identity(person_profile, memory, ictx.interlocutor_mode)

    if identity_event:
        agent_monitor.event(identity_event)
    elif identity_hint:
        agent_monitor.event(identity_hint[:48])

    agent_monitor.memory_pack_v2(memory_pack)
    agent_monitor.memory_pack_summary(memory, memory_pack)

    messages = build_messages(
        profile, memory, message, device_id=device_id, person_profile=person_profile,
        memory_pack=memory_pack,
    )
    _t4 = time.perf_counter()
    agent_monitor.set_timing("prompt", (_t4 - _t3) * 1000)
    agent_monitor.prompt_summary(messages)

    temp = settings.chat_temperature
    if user_message_hints(message, memory=memory, person_profile=person_profile, device_id=device_id):
        temp = min(temp, 0.72)

    # 流式生成：收集完整回复文本用于后续处理
    reply_parts: list[str] = []
    try:
        async for token in chat_completion_stream_async(messages, temperature=temp):
            if token.startswith("\n[ERROR]"):
                reply_parts.append(token)
                yield ("token", token)
                break
            reply_parts.append(token)
            yield ("token", token)
    except Exception as exc:
        err_msg = f"调用 DeepSeek 失败：{str(exc)[:120]}"
        reply_parts.append(err_msg)
        yield ("token", err_msg)
    _t5 = time.perf_counter()
    agent_monitor.set_timing("llm", (_t5 - _t4) * 1000)

    reply_raw = "".join(reply_parts).strip()
    reply, active_topic = _parse_reply(reply_raw)

    # 去除括号旁白（代码层兜底，语音场景必须）
    reply = _strip_stage_directions(reply)

    if active_topic and should_suppress_active_topic(message):
        active_topic = None
    if active_topic:
        active_topic = validate_active_topic(active_topic)
    if len(reply) > settings.max_reply_chars:
        reply = reply[: settings.max_reply_chars]
    reply = enforce_mode_switch_reply(reply, ictx.mode_switch_ack)

    await asyncio.to_thread(_append_turn, session_id, message, reply)
    if active_topic:
        await asyncio.to_thread(
            working_memory.append, session_id, "assistant", active_topic
        )

    agent_monitor.end_turn(reply, t0)

    asyncio.create_task(
        _post_process(device_id, session_id, message, reply, memory, person_id)
    )
    yield ("done", (reply, session_id, active_topic))


# ============================
# 沉默破冰：从 L1/L2/L3 真实记忆生成话题
# ============================

# 兜底话题库：无记忆命中时随机选一条
_SAFE_TOPICS = [
    "今天有没有什么开心的事呀？",
    "你现在在干嘛呢？",
    "有没有什么想跟我分享的？",
    "最近有没有看什么好看的剧？",
    "今天天气怎么样呀？",
]

# 负面情绪标签集合——用户难过/低落时不主动发话题
_NEGATIVE_MOODS = {"难过", "伤心", "低落", "焦虑", "生气", "烦躁", "疲惫", "害怕"}


async def generate_memory_topic(device_id: str, person_id: str, *, session_id: str = "") -> str | None:
    """从记忆中生成真实话题，用于用户沉默后主动破冰。

    检索优先级：
    1. 用当前对话最后 3 轮内容作为检索 query（非空查询）
    2. 优先 L2 近期会话摘要（向量检索）
    3. L2 无命中 → L3 长期记忆 hybrid 检索（仅查最近 3 个月）
    4. 全部无命中 → 兜底安全话题

    情感适配：
    - 用户难过/低落 → 不主动发话题，安静等待
    - 开心/轻松/中性 → 正常生成

    话题生成结合当前对话上下文 + 检索到的 Top1 记忆。
    """
    import random as _random
    from app.memory.identity import is_verified_person_id, parse_identity_credentials
    from app.memory.guard import is_noise_memory_for_l3
    from app.memory.l3 import semantic_memory
    from app.memory.emotion import emotion_trajectory
    from app.monitor import agent_monitor

    if not is_verified_person_id(person_id):
        return None

    # Step 0: 情感门控 —— 用户难过时不发话题
    emotion_traj = emotion_trajectory(device_id, person_id, last_n=2)
    if emotion_traj:
        last_mood = str(emotion_traj[0].get("mood", "")).strip()
        if last_mood in _NEGATIVE_MOODS:
            agent_monitor.event(f"沉默话题跳过: 情感负面 mood={last_mood}")
            return None

    # Step 1: 取 L1 最后 3 轮（6 条消息）作为检索 query 和上下文
    # 过滤掉身份凭证消息（如"刘远慧 123"），避免沉默话题将其当成人名追问
    l1 = working_memory.get_recent(session_id) if session_id else []
    l1 = (l1 or [])[-6:]  # 最后 3 轮（user + assistant = 6 条）

    def _is_identity_msg(m: dict) -> bool:
        """判断是否为身份凭证消息，这类消息不应出现在沉默话题上下文中。"""
        text = str(m.get("content") or "").strip()
        if len(text) > 48:  # 长消息几乎不会是身份凭证
            return False
        return parse_identity_credentials(text) is not None

    l1_user_msgs = [
        str(m["content"]).strip()
        for m in l1
        if m.get("role") == "user" and not _is_identity_msg(m)
    ]
    context_str = "\n".join(
        f"{'女友' if m.get('role') == 'user' else '你'}: {m.get('content', '')}"
        for m in l1
        if not _is_identity_msg(m)
    ) or "（无最近对话）"

    # 用最后 3 轮用户消息拼接检索 query
    search_query = " ".join(l1_user_msgs[-3:]).strip()[:300]
    if not search_query:
        # 兜底：用 L2 最新摘要主题
        for ep in (store.list_episodic_active(device_id, person_id, limit=1) or []):
            search_query = str(ep.get("summary", "")).strip()[:200]
            break
    if not search_query:
        search_query = "最近聊天"

    agent_monitor.event(
        f"沉默话题-上下文: L1轮数={len(l1)//2} query_len={len(search_query)}"
    )

    # Step 2: 优先 L2 向量检索（近期会话摘要）
    memory_text: str = ""
    try:
        from app.memory.l2 import episodic_memory as _l2
        l2_matches = _l2.recall_scored(device_id, person_id, search_query, 3)
        if l2_matches:
            top = l2_matches[0]
            memory_text = str(top.get("text", "")).strip()
            agent_monitor.event(f"沉默话题-L2命中: score={top.get('score')}")
    except Exception:
        pass

    # Step 3: L2 无命中 → L3 hybrid 检索（仅最近 3 个月）
    if not memory_text:
        try:
            l3_results = semantic_memory.recall_l3_scored(
                device_id, person_id, search_query, 5,
                persona_person_id=str(getattr(settings, "persona_fact_person_id", "") or "").strip(),
            )
            # 过滤噪音 + 只保留最近 3 个月的记忆
            cutoff = datetime.now(timezone.utc)
            from datetime import timedelta as _td
            three_months_ago = cutoff - _td(days=90)
            for r in l3_results:
                if is_noise_memory_for_l3(str(r.get("text", ""))):
                    continue
                created = r.get("created_at", "")
                if created:
                    try:
                        ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < three_months_ago:
                            continue
                    except (ValueError, AttributeError):
                        pass
                memory_text = str(r.get("text", "")).strip()
                if memory_text:
                    break
            agent_monitor.event(
                f"沉默话题-L3检索: hit={'Y' if memory_text else 'N'} query_len={len(search_query)}"
            )
        except Exception:
            pass

    # Step 4: 无记忆 → 兜底安全话题（随机选一条）
    if not memory_text:
        topic = _random.choice(_SAFE_TOPICS)
        agent_monitor.event(f"沉默话题-兜底: topic=「{topic}」")
        return topic

    # Step 5: 用上下文 + Top1 记忆生成话题
    prompt = f"""你是男朋友「叶鹏祥」。现在女友沉默了一会儿没说话，你要结合正在聊的内容和记忆中相似的事，自然地问她一个问题。

当前正在聊的（最后几轮对话）：
{context_str[:400]}

从记忆中检索到的最相关内容：
{memory_text[:300]}

要求：
- 从记忆中和当前聊天中找关联，自然地带出话题
- 10字以内口语问句，以？结尾
- 像突然想起来随口一问，不要"我记得..."开头
- 你是男朋友，禁止说女友说过的原话
- 只输出问句本身"""

    try:
        result = await chat_completion_small_async(
            [{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=40,
        )
        topic = result.strip().strip("\"'""''\"")
        if len(topic) < 3 or len(topic) > 25:
            agent_monitor.event(f"沉默话题丢弃: 长度={len(topic)}")
            return None
        if "?" not in topic and "？" not in topic:
            agent_monitor.event(f"沉默话题丢弃: 非问句")
            return None
        agent_monitor.event(f"沉默话题生成成功: topic=「{topic}」")
        return topic
    except Exception:
        agent_monitor.warn("沉默话题LLM调用异常")
        return None


# ============================
# 后台异步入库
# ============================

async def _post_process(
    device_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    memory: dict,
    person_id: str | None,
) -> None:
    """后台处理：统一裁决本轮记忆沉淀。

    委托 MemoryConsolidator 完成统一分类与路由，避免 6 个写入操作
    各自独立裁决。所有操作 fire-and-forget，失败仅记录日志。

    写入路径（按优先级）：
      1. 纠错 → 修正旧记忆
      2. 压缩 L1→L2 → 持久化情景摘要
      3. 事实捕获 → 核心记忆 + 长期记忆
      4. 第三方人物 → Contact Profile
    """
    # ---- 统一裁决：委托 MemoryConsolidator ----
    # 所有写入决策集中在 consolidator，避免各模块各自判断
    try:
        from app.memory.consolidator import consolidator

        # 将 profile 附着在 memory 上供 contacts 使用
        if person_id and "contacts" not in memory:
            try:
                memory["_person_profile"] = store.get_person_profile(person_id or "")
            except Exception:
                pass

        result = await consolidator.process_turn_async(
            device_id, session_id,
            user_msg, assistant_msg,
            memory, person_id,
        )

        # 控制台输出写入摘要
        if result.skipped:
            return

        events: list[str] = []
        if result.did_compress_l1:
            events.append("L1→L2 会话已压缩入库")
        if result.l0_saved_count:
            events.append(f"L0 入库 · {result.l0_saved_count} 条")
        if result.corrections_applied:
            s = result.corrections_applied
            events.append(
                f"记忆修正 · 删事实{s.get('deleted_facts', 0)} "
                f"删块{s.get('deleted_chunks', 0)} "
                f"改块{s.get('patched_chunks', 0)} "
                f"新增{s.get('added_facts', 0)} "
                f"删L0{s.get('deleted_l0', 0)}"
            )
        if result.contacts_updated:
            events.append(f"第三方画像更新 · {result.contacts_updated} 个")
        if result.open_loops_created:
            events.append(f"待跟进新增 · {' '.join(result.open_loops_created[:2])}")
        if result.open_loops_resolved:
            events.append(f"待跟进完成 · {' '.join(result.open_loops_resolved)}")
        if result.emotional_events_detected:
            events.append(f"情感事件 · {' '.join(result.emotional_events_detected)}")
        if result.errors:
            for err in result.errors[:2]:
                agent_monitor.warn(err)

        for ev in events:
            agent_monitor.event(ev)
    except Exception as exc:
        logger.warning("Consolidator process failed: %s", exc)


# ============================
# 会话结束处理
# ============================

async def handle_session_end(
    device_id: str, session_id: str, *, create_new: bool = True
) -> str | None:
    """会话结束处理：将 L1 压缩为 L2 摘要（实名用户）或丢弃（访客）。

    参数:
        device_id:  设备标识
        session_id: 要结束的会话 ID
        create_new: 是否同时创建新会话（默认 True，供 WebSocket 续聊用）

    返回:
        str | None: 新创建的 session_id（create_new=True 时），否则 None

    处理逻辑：
      - 实名用户：consolidate_session 读取全部 L1 消息 → LLM 压缩为 L2 摘要
        → 异步提取 L0 核心事实
      - 访客用户：直接 finalize_session（清空 L1，不写入 L2/L3/画像）
    """
    from app.memory.guard import prune_self_intro_noise
    from app.session import store

    if session_id:
        pid = store.get_session_active_person_id(session_id) or ""
        if is_verified_person_id(pid):
            # 实名用户：L1 → L2 压缩
            l2_summary = await asyncio.to_thread(consolidate_session, device_id, session_id)
            # 后台异步从 L2 摘要中提取 L0 核心事实（不阻塞会话关闭）
            if l2_summary:
                asyncio.create_task(_background_l0_extraction(device_id, pid, l2_summary))
            if create_new:
                # 清理 L1 中的自我介绍噪音（如每轮重复的"我是XXX"）
                await asyncio.to_thread(prune_self_intro_noise, device_id)
                agent_monitor.event("会话结束 · L1 已清空，摘要已写入 L2（要点进 L3/画像）")
        else:
            # 访客：直接丢弃 L1
            await asyncio.to_thread(store.finalize_session, session_id)
            if create_new:
                agent_monitor.event("访客会话结束 · L1 已丢弃，未写入 L2/L3/画像/L0")

    if create_new:
        return await asyncio.to_thread(store.get_or_create_session, device_id, None)
    return None


async def _background_l0_extraction(device_id: str, person_id: str, l2_summary: str) -> None:
    """Fire-and-forget：从会话结束时的 L2 摘要中提取 L0 核心事实。

    使用轻量 LLM（chat_completion_small）分析 L2 摘要，
    识别其中的高置信度事实（身份、偏好、里程碑等）并写入 L0。

    为什么从 L2 摘要提取而不是 L1 原始消息：
      L2 摘要已经是压缩后的结构化内容，噪音少，提取准确率更高，
      且 token 消耗也更少。
    """
    try:
        saved = await asyncio.to_thread(
            extract_l0_from_session_summary, device_id, person_id, l2_summary
        )
        if saved:
            agent_monitor.event(f"L0 提取 · 会话结束 · {len(saved)} 条")
    except Exception as exc:
        logger.warning("L0 session-end extraction failed: %s", exc)
