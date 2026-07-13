"""
第三方人物画像 —— 聊天中提到的亲友/熟人，与对话对象（owner）区分存储。

识别策略（宁可漏建，不可误建）：
  - 首次建档：只接受明确、具名的关系陈述（「刘远航是我姐」）
  - 已有档案：可由明确的具名事实补充内容
  - 单纯提问、随口提及、模糊断言：不创建候选、不写画像

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
from app.memory.person_resolver import PersonResolution, resolve_person
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
ROBOT_OWNER_PERSON_ID = "robot:sparkbot"
ROBOT_OWNER_DISPLAY_NAME = "叶鹏祥 / SparkBot"

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
    return re.sub(r"\s+", "", (name or "").strip()).rstrip("了啦哈呢啊嘛吧吗？?！!。")


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
    rows = (
        store.list_all_person_profiles()
        if str(owner_person_id or "").strip() == ROBOT_OWNER_PERSON_ID
        else store.list_person_profiles(device_id)
    )
    for row in rows:
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


_FIRST_CONTACT_SOURCES = frozenset({"rel_to_user", "rel_user_has"})
_CONTACT_UPDATE_SOURCES = _FIRST_CONTACT_SOURCES | frozenset({"fact_place", "fact_is"})


def has_contact_admission_signal(user_msg: str) -> bool:
    """本轮是否包含值得进入第三方画像管线的明确陈述。

    这里只做轻量语法门控；真正的 owner 排除与是否允许首次建档由
    ``process_third_party_from_turn`` 再次校验。询问某人或随口提到某人
    从不构成写入理由。
    """
    return any(
        signal.source in _CONTACT_UPDATE_SOURCES
        for signal in _detect_signals(user_msg, exclude_names=set())
    )


def upsert_contact_from_signal(
    device_id: str,
    owner_person_id: str,
    sig: ContactSignal,
    *,
    resolution: PersonResolution | None = None,
) -> tuple[dict | None, str]:
    """根据信号创建或更新第三方画像。返回 (profile, event_msg)。

    未知人物只有明确关系陈述可以首次建档。提问和随口提及不写入；
    明确事实仅用于补充既有档案，避免把普通句子误识别为新人物。
    """
    from datetime import datetime, timezone

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── 长期记忆 已知但不结构化 → 跳过 ──
    if resolution and resolution.source == "memory_items":
        return None, ""

    # 问句与随口提及没有持久化价值；此处再次守住直接调用本函数的路径。
    if sig.source not in _CONTACT_UPDATE_SOURCES:
        return None, ""

    existing = find_contact_profile(device_id, owner_person_id, sig.name)
    mention_count = int((existing or {}).get("mention_count") or 0) + 1

    # ── Wiki 已知 → 从 wiki 元数据创建/补全为 confirmed contact ──
    if resolution and resolution.source == "wiki" and not existing:
        profile = _empty_contact(device_id, owner_person_id, sig.name,
                                 relationship=resolution.wiki_meta.get("relationship", "") if resolution.wiki_meta else "")
        profile["source"] = "wiki"
        profile["source_path"] = resolution.wiki_source_path
        profile["wiki_synced_at"] = _now()
        profile["confirmed"] = True
        if resolution.wiki_meta and resolution.wiki_meta.get("aliases"):
            profile["aliases"] = list(resolution.wiki_meta["aliases"])
        if resolution.wiki_body_facts:
            profile["notes"] = list(resolution.wiki_body_facts)
        existing = None  # 不进入 "更新" 分支
    elif existing:
        profile = normalize_profile(existing)
    elif sig.source in _FIRST_CONTACT_SOURCES:
        profile = _empty_contact(device_id, owner_person_id, sig.name, relationship=sig.relationship)
    else:
        return None, ""

    profile["mention_count"] = mention_count
    profile["last_mentioned_at"] = _now()

    if sig.relationship and not profile.get("relationship"):
        profile["relationship"] = sig.relationship
    if sig.note:
        _append_unique(profile.setdefault("notes", []), [sig.note])

    if sig.source in _FIRST_CONTACT_SOURCES or (sig.relationship or sig.note):
        profile["confirmed"] = True
    else:
        profile["confirmed"] = bool(profile.get("confirmed"))

    profile["updated_at"] = _now()
    store.save_person_profile(device_id, profile)

    action = "更新" if existing else "新建"
    nick = profile_display_name(profile)

    # ── 生成差异化 monitor 文案 ──
    if resolution:
        if resolution.source == "wiki":
            if existing:
                logger.info("contact wiki-sync update: %s owner=%s", nick, owner_person_id[:8])
                return profile, f"Wiki人物同步(更新) · {nick}"
            logger.info("contact wiki-sync new: %s owner=%s", nick, owner_person_id[:8])
            return profile, f"Wiki人物同步 · {nick}"
        if resolution.source == "contact":
            logger.info("contact mention: %s owner=%s", nick, owner_person_id[:8])
            return profile, f"第三方画像命中 · {nick}"

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
    """处理一轮对话中的第三方人物信号，返回 monitor 事件文案列表。

    首次建档仅限明确关系；已确认联系人可接受明确事实补充。任何问句
    或随口提及都不会留下持久化候选，避免污染 ``person_profiles``。
    """
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

    # ── 先统一解析每个人名的认知来源 ──
    resolutions: dict[str, PersonResolution] = {}
    for sig in signals:
        key = _normalize_name(sig.name)
        if key in resolutions:
            continue
        if should_skip_contact_name(sig.name, exclude=exclude):
            continue
        resolutions[key] = resolve_person(sig.name, device_id, pid)

    for sig in signals:
        key = _normalize_name(sig.name)
        resolution = resolutions.get(key)

        # 长期记忆 已知但不结构化 → 跳过，不创建 contact
        if resolution and resolution.source == "memory_items":
            continue

        prof, ev = upsert_contact_from_signal(
            device_id, pid, sig,
            resolution=resolution,
        )
        if ev:
            events.append(ev)
        if prof:
            touched[key] = prof

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
    robot_contacts = list_contacts_for_owner(device_id, ROBOT_OWNER_PERSON_ID)
    known_ids = {str(p.get("person_id") or "") for p in contacts}
    contacts.extend(
        p for p in robot_contacts
        if str(p.get("person_id") or "") not in known_ids
    )
    exclude = _owner_names(owner_profile, pid)
    relevant = contacts_mentioned_in_message(user_message, contacts, exclude_names=exclude)
    if not relevant:
        confirmed = [p for p in contacts if p.get("confirmed")]
        relevant = confirmed[:2]

    # 额外检查：对话中提到的 Wiki 已知人物（无 contact 时也能注入）
    wiki_fallback: list[PersonResolution] = []
    asked = extract_asked_person_name(user_message or "")
    if asked or not relevant:
        # 先获取已有 contact 的所有称呼，避免重复注入
        mentioned_contact_names = set()
        for p in confirmed_contacts_for_owner(device_id, pid):
            for n in profile_nicknames(p):
                mentioned_contact_names.add(_normalize_name(n))

        # 如果用户在问某个人，且此人 Wiki 已知但不在 contact 中
        if asked:
            res = resolve_person(asked, device_id, pid)
            if res.source == "wiki" and _normalize_name(asked) not in mentioned_contact_names:
                wiki_fallback.append(res)

    if not relevant and not wiki_fallback:
        return ""

    lines = [
        "## 第三方人物（对话对象提到的人 · 只许用下列档案，禁止编造）",
        f"（以下人物与当前对话对象 {profile_display_name(owner_profile) if owner_profile else pid} 或智能体本人相关，不是对话对象本人）",
    ]
    for p in relevant[:4]:
        name = profile_display_name(p)
        rel = profile_relationship(p)
        owner_id = str(p.get("owner_person_id") or "")
        owner_label = "智能体本人" if owner_id == ROBOT_OWNER_PERSON_ID else "对话对象"
        rel_line = f"与{owner_label}关系：{rel}" if rel else f"与{owner_label}关系：未明"
        parts = [f"【{name}】{rel_line}"]
        if p.get("notes"):
            parts.append("备注：" + "；".join(str(x) for x in p["notes"][:4]))
        if p.get("personality"):
            parts.append("性格：" + "、".join(str(x) for x in p["personality"][:4]))
        if p.get("experiences"):
            parts.append("经历：" + "；".join(str(x) for x in p["experiences"][:3]))
        lines.append("- " + " | ".join(parts))
    # Wiki 已知但无 contact 的人物
    for res in wiki_fallback[:2]:
        wiki = res.wiki_meta or {}
        rel = wiki.get("relationship", "")
        rel_line = f"与对话对象关系：{rel}" if rel else "关系：未明"
        name = res.display_name
        parts = [f"【{name}】{rel_line}（Wiki 记录）"]
        if res.wiki_body_facts:
            parts.append("备注：" + "；".join(res.wiki_body_facts[:4]))
        lines.append("- " + " | ".join(parts))
    lines.append("- 档案未写明的细节 → 说不记得或请对方补充，禁止编造")
    return "\n".join(lines) + "\n"


def confirmed_contacts_for_owner(device_id: str, owner_person_id: str) -> list[dict]:
    """获取当前 owner 下所有已确认的第三方联系人。"""
    out: list[dict] = []
    for row in store.list_person_profiles(device_id):
        p = normalize_profile(row["profile"])
        if not _is_contact_profile(p):
            continue
        if str(p.get("owner_person_id") or "") != str(owner_person_id):
            continue
        if p.get("confirmed"):
            out.append(p)
    return out
