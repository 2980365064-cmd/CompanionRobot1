"""第三方联系人画像的准入边界。"""

from unittest.mock import patch

from app.memory.person_resolver import PersonResolution


class _Store:
    def __init__(self):
        self.profiles: dict[str, dict] = {}
        self.saved: list[dict] = []

    def get_person_profile(self, person_id: str):
        return self.profiles.get(person_id)

    def list_person_profiles(self, device_id: str):
        del device_id
        return [
            {"profile": profile}
            for profile in self.profiles.values()
        ]

    def save_person_profile(self, device_id: str, profile: dict):
        del device_id
        self.saved.append(profile)
        self.profiles[profile["person_id"]] = profile


def _unknown(name: str, *_args) -> PersonResolution:
    return PersonResolution(known=False, source="unknown", name=name)


def test_question_about_unknown_person_never_creates_contact_profile():
    from app.memory.contacts import process_third_party_from_turn

    fake_store = _Store()
    with patch("app.memory.contacts.store", fake_store), \
         patch("app.memory.contacts.resolve_person", side_effect=_unknown):
        events = process_third_party_from_turn(
            "device", "owner-1", "你认识小王吗？", "不确定。",
        )

    assert events == []
    assert fake_store.saved == []


def test_casual_mention_of_unknown_person_never_creates_contact_profile():
    from app.memory.contacts import process_third_party_from_turn

    fake_store = _Store()
    with patch("app.memory.contacts.store", fake_store), \
         patch("app.memory.contacts.resolve_person", side_effect=_unknown):
        events = process_third_party_from_turn(
            "device", "owner-1", "刚才说起小王了。", "嗯。",
        )

    assert events == []
    assert fake_store.saved == []


def test_explicit_named_relationship_creates_confirmed_contact_profile():
    from app.memory.contacts import process_third_party_from_turn

    fake_store = _Store()
    with patch("app.memory.contacts.store", fake_store), \
         patch("app.memory.contacts.resolve_person", side_effect=_unknown):
        events = process_third_party_from_turn(
            "device", "owner-1", "小王是我同事。", "知道了。",
        )

    assert len(events) == 1
    assert len(fake_store.saved) == 1
    assert fake_store.saved[0]["display_name"] == "小王"
    assert fake_store.saved[0]["relationship"] == "同事"
    assert fake_store.saved[0]["confirmed"] is True


def test_consolidator_only_routes_explicit_relationship_to_contact_pipeline():
    from app.memory.consolidator import classify_turn

    assert not classify_turn("你认识小王吗？", "", {}).is_third_party
    assert not classify_turn("刚才说起小王了。", "", {}).is_third_party
    assert classify_turn("小王是我同事。", "", {}).is_third_party
