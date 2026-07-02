"""对话角色（Interlocutor）—— 默认女友模式，口令切换访客/女友。

切换口令（用户消息中包含即可）：
  - 「访客模式」→ visitor，回复须含「访客模式开启」
  - 「女友模式」→ girlfriend，回复须含「女友模式开启」
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import settings
from app.memory.identity import memory_scoped_to_person
from app.memory.profile import find_profile_by_name, normalize_profile
from app.session import store

logger = logging.getLogger(__name__)

MODE_GIRLFRIEND = "girlfriend"
MODE_VISITOR = "visitor"

MODE_ACK = {
    MODE_VISITOR: "访客模式开启",
    MODE_GIRLFRIEND: "女友模式开启",
}

_VISITOR_SWITCH = re.compile(r"访客模式")
_GIRLFRIEND_SWITCH = re.compile(r"女友模式")


@dataclass
class InterlocutorTurnResult:
    person_id: str
    person_profile: dict | None = None
    hint: str = ""
    guest_mode: bool = False
    monitor_event: str = ""
    interlocutor_mode: str = MODE_GIRLFRIEND
    mode_switched: bool = False
    mode_switch_ack: str = ""


def get_default_owner_person_id(device_id: str) -> str:
    """解析设备默认女友 person_id：配置优先，否则按显示名查画像。"""
    configured = str(settings.default_owner_person_id or "").strip()
    if configured and store.get_person_profile(configured):
        return configured
    if configured:
        return configured
    name = str(settings.default_owner_display_name or "").strip()
    if name:
        prof = find_profile_by_name(device_id, name)
        if prof:
            return str(prof.get("person_id") or "").strip()
    return ""


def ensure_session_defaults(device_id: str, session_id: str) -> None:
    """新会话或旧会话缺省值：默认女友模式 + 绑定 owner person_id。"""
    mode = store.get_session_interlocutor_mode(session_id)
    if not mode:
        store.set_session_interlocutor_mode(session_id, MODE_GIRLFRIEND)
        mode = MODE_GIRLFRIEND

    owner_id = get_default_owner_person_id(device_id)
    active = store.get_session_active_person_id(session_id) or ""

    if mode == MODE_GIRLFRIEND and owner_id:
        if not active or str(active).startswith("tmp_"):
            store.set_session_active_person(session_id, owner_id)
    store.clear_session_identity_pending(session_id)


def _detect_mode_switch(message: str) -> str | None:
    msg = (message or "").strip()
    if not msg:
        return None
    if _GIRLFRIEND_SWITCH.search(msg):
        return MODE_GIRLFRIEND
    if _VISITOR_SWITCH.search(msg):
        return MODE_VISITOR
    return None


def _mode_switch_hint(mode: str) -> str:
    ack = MODE_ACK[mode]
    if mode == MODE_VISITOR:
        return (
            f"【模式切换 · 必须执行】用户开启了访客模式。"
            f"你本轮回复正文**必须完整包含**「{ack}」这六个字，"
            f"然后再用对朋友/访客的口吻简短接话；禁止情侣亲密语气与女友专属称呼。"
        )
    return (
        f"【模式切换 · 必须执行】用户开启了女友模式。"
        f"你本轮回复正文**必须完整包含**「{ack}」这六个字，"
        f"然后再用对女友刘远慧的深情、俏皮、会斗嘴的口吻接话（可偶尔一句短哲理）。"
    )


def enforce_mode_switch_reply(reply: str, ack_phrase: str) -> str:
    """代码层保证模式切换确认语出现在回复中。"""
    text = (reply or "").strip()
    ack = (ack_phrase or "").strip()
    if not ack:
        return text
    if ack in text:
        return text
    if not text:
        return ack
    return f"{ack}。{text}"


def resolve_interlocutor_before_memory(
    device_id: str, session_id: str, message: str,
) -> InterlocutorTurnResult:
    """每轮对话的角色解析入口（替代旧版访客实名流程）。"""
    ensure_session_defaults(device_id, session_id)

    switch = _detect_mode_switch(message)
    if switch:
        store.set_session_interlocutor_mode(session_id, switch)
        if switch == MODE_GIRLFRIEND:
            owner_id = get_default_owner_person_id(device_id)
            if owner_id:
                store.set_session_active_person(session_id, owner_id)
        store.clear_session_identity_pending(session_id)
        ack = MODE_ACK[switch]
        label = "访客模式" if switch == MODE_VISITOR else "女友模式"
        return InterlocutorTurnResult(
            person_id=_active_person_id(device_id, session_id),
            person_profile=_load_active_profile(device_id, session_id),
            hint=_mode_switch_hint(switch),
            guest_mode=False,
            monitor_event=f"模式切换 · {label}",
            interlocutor_mode=switch,
            mode_switched=True,
            mode_switch_ack=ack,
        )

    mode = store.get_session_interlocutor_mode(session_id) or MODE_GIRLFRIEND
    person_id = _active_person_id(device_id, session_id)
    profile = _load_active_profile(device_id, session_id)
    scoped = memory_scoped_to_person(person_id)

    if mode == MODE_VISITOR:
        return InterlocutorTurnResult(
            person_id=person_id,
            person_profile=profile,
            hint=(
                "【对话角色 · 访客/朋友】叶鹏祥第一人称，大大咧咧、随性自然，"
                "像跟兄弟/熟人微信闲聊；禁止情侣亲密语气与女友专属称呼（大炮/秋雨/乖乖等）。"
            ),
            guest_mode=not scoped,
            interlocutor_mode=MODE_VISITOR,
        )

    return InterlocutorTurnResult(
        person_id=person_id,
        person_profile=profile,
        hint=(
            "【对话角色 · 女友刘远慧】深情、走心，会逗她损她一下（损友+情人）；"
            "按 persona 场景路由选风格，勿混斗嘴/走心/哲理。"
        ),
        guest_mode=not scoped,
        interlocutor_mode=MODE_GIRLFRIEND,
    )


def _active_person_id(device_id: str, session_id: str) -> str:
    pid = store.get_session_active_person_id(session_id) or ""
    if pid and not str(pid).startswith("tmp_"):
        return pid
    owner_id = get_default_owner_person_id(device_id)
    if owner_id:
        store.set_session_active_person(session_id, owner_id)
        return owner_id
    return pid or owner_id


def _load_active_profile(device_id: str, session_id: str) -> dict | None:
    pid = _active_person_id(device_id, session_id)
    if not pid:
        return None
    prof = store.get_person_profile(pid)
    if prof:
        return normalize_profile(prof)
    name = str(settings.default_owner_display_name or "").strip()
    if name:
        return find_profile_by_name(device_id, name)
    return None


def is_mode_switch_message(message: str) -> bool:
    """是否包含模式切换口令（用于 WS 沉默监听等）。"""
    return _detect_mode_switch(message) is not None
