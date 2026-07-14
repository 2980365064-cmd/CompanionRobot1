"""访客角色切换不能继承已实名对象的记忆或画像。"""

from unittest.mock import patch

from app.memory.interlocutor import (
    MODE_GIRLFRIEND,
    MODE_VISITOR,
    _active_person_id,
    _load_active_profile,
    resolve_interlocutor_before_memory,
    ensure_session_defaults,
)
from app.memory.identity import IdentityTurnResult, ensure_guest_person_id


def test_visitor_switch_replaces_verified_identity_with_temporary_identity():
    with patch("app.memory.interlocutor.store") as store, \
         patch("app.memory.interlocutor.get_default_owner_person_id", return_value="123"), \
         patch("app.memory.interlocutor.new_temp_person_id", return_value="tmp_visitor"):
        store.get_session_interlocutor_mode.return_value = MODE_GIRLFRIEND
        store.get_session_active_person_id.return_value = "123"

        result = resolve_interlocutor_before_memory("dev", "session", "访客模式")

    assert result.interlocutor_mode == MODE_VISITOR
    assert result.guest_mode is True
    assert result.person_id == "tmp_visitor"
    assert result.person_profile is None
    store.set_session_active_person.assert_called_with("session", "tmp_visitor")


def test_temporary_visitor_identity_is_not_rebound_to_default_owner():
    with patch("app.memory.interlocutor.store") as store, \
         patch("app.memory.interlocutor.get_default_owner_person_id", return_value="123"):
        store.get_session_active_person_id.return_value = "tmp_visitor"

        assert _active_person_id("dev", "session") == "tmp_visitor"
        assert _load_active_profile("dev", "session") is None

    store.set_session_active_person.assert_not_called()


def test_guest_memory_pack_does_not_add_state_register_items():
    from app.memory.orchestrator import MemoryOrchestrator

    with patch("app.memory.orchestrator.relationship_manager") as relationship, \
         patch("app.memory.orchestrator.open_loop_manager") as open_loops, \
         patch("app.memory.orchestrator.emotional_extractor") as emotional_events:
        pack = MemoryOrchestrator()._build_v2_pack(
            {
                "history": [],
                "items": [],
                "diagnostics": {"retrieval_plan": {"needs_memory": False}},
                "guest_mode": True,
                "memory_miss": False,
            },
            [],
            "tmp_visitor",
            MODE_VISITOR,
            "",
            "你好",
        )

    assert pack.items == []
    relationship.load.assert_not_called()
    open_loops.list_relevant.assert_not_called()
    emotional_events.extract_all_from_recent_memory.assert_not_called()


def test_identity_can_replace_default_owner_with_temporary_guest():
    with patch("app.memory.identity.store") as store, \
         patch("app.memory.identity.new_temp_person_id", return_value="tmp_pending"):
        store.get_session_active_person_id.return_value = "123"

        assert ensure_guest_person_id("session", replace_verified=True) == "tmp_pending"

    store.set_session_active_person.assert_called_once_with("session", "tmp_pending")


def test_interlocutor_uses_identity_state_machine_for_credentials():
    identity = IdentityTurnResult(
        person_id="tmp_pending",
        guest_mode=True,
        hint="请确认身份",
        monitor_event="身份待确认",
    )
    with patch("app.memory.interlocutor.store") as store, \
         patch("app.memory.interlocutor.get_default_owner_person_id", return_value="123"), \
         patch("app.memory.interlocutor.resolve_identity_turn", return_value=identity) as resolve:
        store.get_session_interlocutor_mode.return_value = MODE_GIRLFRIEND
        store.get_session_active_person_id.return_value = "123"

        result = resolve_interlocutor_before_memory("dev", "session", "名字小明 ID xm001")

    resolve.assert_called_once_with("dev", "session", "名字小明 ID xm001")
    assert result.person_id == "tmp_pending"
    assert result.guest_mode is True
    assert result.hint == "请确认身份"


def test_session_defaults_preserve_pending_identity_registration():
    with patch("app.memory.interlocutor.store") as store, \
         patch("app.memory.interlocutor.get_default_owner_person_id", return_value="123"):
        store.get_session_interlocutor_mode.return_value = MODE_GIRLFRIEND
        store.get_session_active_person_id.return_value = "tmp_pending"

        ensure_session_defaults("dev", "session")

    store.clear_session_identity_pending.assert_not_called()
