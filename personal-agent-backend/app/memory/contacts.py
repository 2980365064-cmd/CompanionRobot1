"""
第三方人物画像 —— 聊天中提到的亲友/熟人，与对话对象（owner）区分存储。

识别策略（避免乱建 / 避免漏建）：
  - 高置信（立刻建档/更新）：明确关系（「刘远航是我姐」）、明确事实（「刘远航在杭州实习」）
  - 中置信（立刻建档）：用户主动问「你认识 XXX 吗」且本轮有描述
  - 低置信（累计）：仅点名提及 → 先记 mention_count；达到阈值或出现事实后再建档

画像 scope：同一 owner_person_id（如女友 123）下的第三方互不混淆。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.memory.guard import (
    extract_asked_person_name,
    extract_self_name,
    is_asking_about_third_party,
    is_valid_person_name,
)
from app.memory.profile import (
    empty_profile,
    normalize_profile,
    profile_display_name,
    profile_matches_name,
    profile_nicknames,
    profile_relationship,
)
from app.session import store

logger = logging.getLogger(__name__)

PROFILE_ROLE_OWNER = "owner"
PROFILE_ROLE_CONTACT = "contact"

# 单独出现时不建档的泛称（无具体人名）
_GENERIC_NAMES = frozenset({
    "有人", "别人", "大家", "他们", "她们", "群里", "快递", "外卖", "客服",
    "老板", "同事", "同学", "朋友", "老师", "医生", "司机", "阿姨", "叔叔",
    "哥哥", "姐姐", "弟弟", "妹妹", "爸爸", "妈妈", "爷爷", "奶奶",
    "今天", "明天", "昨天", "那个", "这个", "啥人", "哪位",
})

# 高置信：第三方关系
_REL_TO_USER = re.compile(
    r"([一-鿿·A-Za-z0-9_]{2,10})\s*是我(?:的)?"
    r"(女朋友|女友|老婆|男朋友|男友|老公|"
    r"爸爸|妈妈|父亲|母亲|爷爷|奶奶|外公|外婆|"
    r"哥哥|弟弟|姐姐|妹妹|闺蜜|兄弟|同事|同学|室友|朋友|老板|导员|老师)"
)
_REL_USER_HAS = re.compile(
    r"我(?:的)?(姐姐|哥哥|弟弟|妹妹|闺蜜|兄弟|同事|同学|室友|老板|导员|老师|朋友)"
    r"([一-鿿·A-Za-z0-9_]{2,10})"
)
# 高置信：关于某人的事实
_FACT_ABOUT = re.compile(
    r"([一-鿿·A-Za-z0-9_]{2,10})(?:在|去|到|来自)([一-鿿·A-Za-z0-9_\u4e00-\u9fff]{2,12})"
    r"(?:工作|上班|读书|上学|实习|住|留学)?"
)
_FACT_IS = re.compile(
    r"([一-鿿·A-Za-z0-9_]{2,10})\s*(?:是|叫|名叫|名字(?:叫|是)?)\s*"
    r"([一-鿿·A-Za-z0-9_\u4e00-\u9fff]{2,16})"
)
# 中置信：聊起某人
_TALK_ABOUT = re.compile(
    r"(?:聊到|说起|谈到|关于|提到|见过|认识了?)"
    r"([一-鿿·A-Za-z0-9_]{2,10})"
)

_CONTACT_UPDATE_PROMPT = """你是第三方人物档案整理器。根据用户本轮原话，提取**关于第三方人物**的可验证信息。

对话对象（主人）是：{owner_name}
当前第三方档案：
{current}

用户说：{user_msg}

只输出 JSON：
{{"need_update": true/false, "reason": "一句话",
  "patch": {{"relationship": "", "aliases": [], "personality": [], "experiences": [], "notes": []}}}}

规则：
- 只更新 patch 中非空字段；relationship 为「此人与对话对象的关系」（如姐姐、同事）
- notes 放短事实（地点、职业、近况）；experiences 放较完整经历
- 用户随口一提、无法核实 → need_update=false
- 禁止编造用户未说的内容"""


@dataclass
class ContactSignal:
    name: str
    confidence: str  # high | medium | low
    relationship: str = ""
    note: str = ""
    source: str = ""


def _clean_person_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip()).rstrip("了啦哈呢啊嘛吧")


def _normalize_name(name: str) -> str:
    return _clean_person_name(name)


def _contact_person_id(owner_person_id: str, name: str) -> str:
    key = _normalize_name(name).lower()
    h = hashlib.md5(f"{owner_person_id}:{key}".encode()).hexdigest()[:12]
    return f"ct_{h}"


def _is_contact_profile(profile: dict) -> bool:
    p = normalize_profile(profile)
    return str(p.get("profile_role") or PROFILE_ROLE_OWNER) == PROFILE_ROLE_CONTACT


def _owner_names(owner_profile: dict | None, owner_person_id: str) -> set[str]:
    names: set[str] = set()
    if owner_profile:
        for n in profile_nicknames(owner_profile):
            names.add(_normalize_name(n))
    owner = store.get_person_profile(owner_person_id)
    if owner:
        for n in profile_nicknames(owner):
            names.add(_normalize_name(n))
    display = str(settings.default_owner_display_name or "").strip()
    if display:
        names.add(_normalize_name(display))
    names.add(_normalize_name("叶鹏祥"))
    return {n for n in names if n}


def should_skip_contact_name(name: str, *, exclude: set[str]) -> bool:
    n = _normalize_name(name)
    if not n or not is_valid_person_name(n):
        return True
    if n in exclude or n in _GENERIC_NAMES:
        return True
    if len(n) > 10:
        return True
    return False


def find_contact_profile(
    device_id: str, owner_person_id: str, name: str,
) -> dict | None:
    key = _normalize_name(name)
    if not key:
        return None
    for row in store.list_person_profiles(device_id):
        p = normalize_profile(row["profile"])
        if not _is_contact_profile(p):
            continue
        if str(p.get("owner_person_id") or "") != str(owner_person_id):
            continue
        if profile_matches_name(p, name):
            return p
    return None


def list_contacts_for_owner(device_id: str, owner_person_id: str) -> list[dict]:
    out: list[dict] = []
    for row in store.list_person_profiles(device_id):
        p = normalize_profile(row["profile"])
        if not _is_contact_profile(p):
            continue
        if str(p.get("owner_person_id") or "") != str(owner_person_id):
            continue
        out.append(p)
    return out


def _empty_contact(
    device_id: str,
    owner_person_id: str,
    name: str,
    *,
    relationship: str = "",
) -> dict:
    pid = _contact_person_id(owner_person_id, name)
    p = empty_profile(name, person_id=pid, relationship=relationship)
    p["profile_role"] = PROFILE_ROLE_CONTACT
    p["owner_person_id"] = str(owner_person_id)
    p["notes"] = []
    p["mention_count"] = 0
    p["last_mentioned_at"] = p["created_at"]
    return p


def _append_unique(target: list, items: list[str]) -> None:
    seen = {str(x).strip() for x in target}
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            target.append(s)
            seen.add(s)


def _apply_patch(profile: dict, patch: dict) -> bool:
    changed = False
    if isinstance(patch.get("relationship"), str):
        rel = patch["relationship"].strip()
        if rel and rel != profile.get("relationship"):
            profile["relationship"] = rel
            changed = True
    for key in ("aliases", "personality", "experiences", "notes"):
        if isinstance(patch.get(key), list):
            before = len(profile.get(key) or [])
            _append_unique(profile.setdefault(key, []), patch[key])
            if len(profile[key]) > before:
                changed = True
    return changed


def _detect_signals(user_msg: str, *, exclude_names: set[str]) -> list[ContactSignal]:
    msg = (user_msg or "").strip()
    if not msg:
        return []
    signals: list[ContactSignal] = []
    seen: set[str] = set()

    def add(sig: ContactSignal) -> None:
        sig.name = _clean_person_name(sig.name)
        key = _normalize_name(sig.name)
        if should_skip_contact_name(sig.name, exclude=exclude_names):
            return
        if key in seen:
            return
        seen.add(key)
        signals.append(sig)

    for m in _REL_TO_USER.finditer(msg):
        add(ContactSignal(
            m.group(1).strip(), "high",
            relationship=m.group(2).strip(), source="rel_to_user",
        ))
    for m in _REL_USER_HAS.finditer(msg):
        add(ContactSignal(
            m.group(2).strip(), "high",
            relationship=m.group(1).strip(), source="rel_user_has",
        ))
    for m in _FACT_ABOUT.finditer(msg):
        note = f"{m.group(1)}与{m.group(2)}相关：{m.group(0).strip()}"
        add(ContactSignal(m.group(1).strip(), "high", note=note, source="fact_place"))
    for m in _FACT_IS.finditer(msg):
        add(ContactSignal(
            m.group(1).strip(), "medium",
            note=f"{m.group(1)}：{m.group(2).strip()}", source="fact_is",
        ))
    if is_asking_about_third_party(msg):
        asked = extract_asked_person_name(msg)
        if asked:
            add(ContactSignal(asked, "medium", source="ask_about"))
    for m in _TALK_ABOUT.finditer(msg):
        add(ContactSignal(m.group(1).strip(), "low", source="talk_about"))

    return signals


def upsert_contact_from_signal(
    device_id: str,
    owner_person_id: str,
    sig: ContactSignal,
) -> tuple[dict | None, str]:
    """根据信号创建或更新第三方画像。返回 (profile, event_msg)。"""
    from datetime import datetime, timezone

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    existing = find_contact_profile(device_id, owner_person_id, sig.name)
    mention_count = int((existing or {}).get("mention_count") or 0) + 1
    threshold = max(1, int(getattr(settings, "contact_min_casual_mentions", 2)))

    if existing:
        profile = normalize_profile(existing)
    elif sig.confidence == "high":
        profile = _empty_contact(device_id, owner_person_id, sig.name, relationship=sig.relationship)
    elif sig.confidence == "medium":
        profile = _empty_contact(device_id, owner_person_id, sig.name, relationship=sig.relationship)
    elif sig.confidence == "low":
        profile = _empty_contact(device_id, owner_person_id, sig.name)
    else:
        return None, ""

    profile["mention_count"] = mention_count
    profile["last_mentioned_at"] = _now()

    if sig.relationship and not profile.get("relationship"):
        profile["relationship"] = sig.relationship
    if sig.note:
        _append_unique(profile.setdefault("notes", []), [sig.note])

    if sig.confidence == "high" or (sig.relationship or sig.note):
        profile["confirmed"] = True
    elif mention_count >= threshold and (
        profile.get("relationship") or profile.get("notes") or profile.get("experiences")
    ):
        profile["confirmed"] = True
    else:
        profile["confirmed"] = bool(profile.get("confirmed"))

    profile["updated_at"] = _now()
    store.save_person_profile(device_id, profile)
    action = "更新" if existing else "新建"
    nick = profile_display_name(profile)
    if not profile.get("confirmed") and sig.confidence == "low":
        return profile, f"第三方提及 · {nick}（{mention_count}/{threshold}）"
    logger.info("contact %s: %s owner=%s", action, nick, owner_person_id[:8])
    return profile, f"第三方画像{action} · {nick}"


def _llm_patch_contact(
    device_id: str,
    owner_person_id: str,
    owner_profile: dict | None,
    contact: dict,
    user_msg: str,
) -> tuple[dict | None, str]:
    from app.llm import chat_completion_small

    if len((user_msg or "").strip()) < 8:
        return None, ""
    owner_name = profile_display_name(owner_profile) if owner_profile else owner_person_id
    current = {
        "display_name": profile_display_name(contact),
        "relationship": profile_relationship(contact),
        "aliases": contact.get("aliases") or [],
        "personality": contact.get("personality") or [],
        "experiences": contact.get("experiences") or [],
        "notes": contact.get("notes") or [],
    }
    prompt = _CONTACT_UPDATE_PROMPT.format(
        owner_name=owner_name,
        current=json.dumps(current, ensure_ascii=False),
        user_msg=user_msg.strip()[:400],
    )
    raw = chat_completion_small([{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None, ""
    if not data.get("need_update"):
        return None, ""
    patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
    if not _apply_patch(contact, patch):
        return None, ""
    if contact.get("relationship") or contact.get("notes") or contact.get("experiences"):
        contact["confirmed"] = True
    from datetime import datetime, timezone

    contact["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.save_person_profile(device_id, contact)
    return contact, f"第三方画像补充 · {profile_display_name(contact)}"


def process_third_party_from_turn(
    device_id: str,
    owner_person_id: str,
    user_msg: str,
    assistant_msg: str,
    *,
    owner_profile: dict | None = None,
) -> list[str]:
    """处理一轮对话中的第三方人物信号，返回 monitor 事件文案列表。"""
    pid = str(owner_person_id or "").strip()
    if not pid or pid.startswith("tmp_"):
        return []

    exclude = _owner_names(owner_profile, pid)
    if extract_self_name(user_msg or ""):
        who = extract_self_name(user_msg or "")
        if who:
            exclude.add(_normalize_name(who))

    events: list[str] = []
    signals = _detect_signals(user_msg, exclude_names=exclude)
    touched: dict[str, dict] = {}

    for sig in signals:
        prof, ev = upsert_contact_from_signal(device_id, pid, sig)
        if ev:
            events.append(ev)
        if prof:
            touched[_normalize_name(sig.name)] = prof

    # 对已确认联系人，用 LLM 从本轮用户话里增量补充（仅 high/medium 信号或已确认）
    for sig in signals:
        if sig.confidence == "low" and not sig.note:
            prof = touched.get(_normalize_name(sig.name)) or find_contact_profile(
                device_id, pid, sig.name,
            )
            if prof and not prof.get("confirmed"):
                continue
        key = _normalize_name(sig.name)
        prof = touched.get(key) or find_contact_profile(device_id, pid, sig.name)
        if not prof or not prof.get("confirmed"):
            continue
        updated, ev = _llm_patch_contact(device_id, pid, owner_profile, prof, user_msg)
        if ev:
            events.append(ev)
        if updated:
            touched[key] = updated

    del assistant_msg  # 预留：未来可从助手复述中反查
    return events


def contacts_mentioned_in_message(
    user_msg: str,
    contacts: list[dict],
    *,
    exclude_names: set[str],
) -> list[dict]:
    msg = user_msg or ""
    hits: list[dict] = []
    for p in contacts:
        if not p.get("confirmed"):
            continue
        for n in profile_nicknames(p):
            if should_skip_contact_name(n, exclude=exclude_names):
                continue
            if n and n in msg:
                hits.append(p)
                break
    return hits[:5]


def format_contacts_prompt_block(
    device_id: str,
    owner_person_id: str,
    user_message: str,
    *,
    owner_profile: dict | None = None,
) -> str:
    """为本轮可能相关的第三方注入 prompt 块。"""
    pid = str(owner_person_id or "").strip()
    if not pid or pid.startswith("tmp_"):
        return ""
    contacts = list_contacts_for_owner(device_id, pid)
    if not contacts:
        return ""
    exclude = _owner_names(owner_profile, pid)
    relevant = contacts_mentioned_in_message(user_message, contacts, exclude_names=exclude)
    if not relevant:
        confirmed = [p for p in contacts if p.get("confirmed")]
        relevant = confirmed[:2]
    if not relevant:
        return ""

    lines = [
        "## 第三方人物（对话对象提到的人 · 只许用下列档案，禁止编造）",
        f"（以下人物与当前对话对象 {profile_display_name(owner_profile) if owner_profile else pid} 相关，不是对话对象本人）",
    ]
    for p in relevant[:4]:
        name = profile_display_name(p)
        rel = profile_relationship(p)
        rel_line = f"与对话对象关系：{rel}" if rel else "关系：未明"
        parts = [f"【{name}】{rel_line}"]
        if p.get("notes"):
            parts.append("备注：" + "；".join(str(x) for x in p["notes"][:4]))
        if p.get("personality"):
            parts.append("性格：" + "、".join(str(x) for x in p["personality"][:4]))
        if p.get("experiences"):
            parts.append("经历：" + "；".join(str(x) for x in p["experiences"][:3]))
        lines.append("- " + " | ".join(parts))
    lines.append("- 档案未写明的细节 → 说不记得或请对方补充，禁止编造")
    return "\n".join(lines) + "\n"
