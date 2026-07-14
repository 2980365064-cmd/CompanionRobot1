import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch


class AdminOpsModuleTests(unittest.TestCase):
    def test_config_masks_sensitive_values(self):
        from app.admin.config import mask_secret

        self.assertEqual(mask_secret(""), "")
        self.assertEqual(mask_secret("sk-1234567890"), "sk-1******890")

    def test_env_patch_preserves_comments_and_updates_values(self):
        from app.admin.config import apply_env_patch

        original = "# hello\nLLM_MODEL=deepseek-chat\nAPI_TOKEN=old\n"
        updated = apply_env_patch(original, {"LLM_MODEL": "deepseek-reasoner", "MAX_REPLY_CHARS": 90})
        self.assertIn("# hello", updated)
        self.assertIn("LLM_MODEL=deepseek-reasoner", updated)
        self.assertIn("API_TOKEN=old", updated)
        self.assertIn("MAX_REPLY_CHARS=90", updated)

    def test_file_guard_rejects_path_traversal(self):
        from app.admin.files import AdminFileStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "persona"
            allowed.mkdir()
            store = AdminFileStore({"persona": allowed})
            with self.assertRaises(ValueError):
                store.resolve("persona/../../secret.txt")

    def test_self_cognition_store_writes_persona_and_profile_card(self):
        from app.admin.self_cognition import AgentSelfCognitionStore

        with tempfile.TemporaryDirectory() as tmp:
            persona = Path(tmp) / "persona.md"
            profile_card = Path(tmp) / "profile_card.md"
            persona.write_text("old persona\n", "utf-8")
            profile_card.write_text("old card\n", "utf-8")
            cleared = []

            store = AgentSelfCognitionStore(
                persona_path=persona,
                profile_card_path=profile_card,
                clear_cache=lambda: cleared.append(True),
            )

            saved = store.save(
                persona_text="我是叶鹏祥，喜欢克制表达。",
                profile_card_text="Profile Card: 我是叶鹏祥。",
            )

            self.assertEqual(persona.read_text("utf-8"), "我是叶鹏祥，喜欢克制表达。\n")
            self.assertEqual(profile_card.read_text("utf-8"), "Profile Card: 我是叶鹏祥。\n")
            self.assertEqual(saved["persona"]["content"], "我是叶鹏祥，喜欢克制表达。")
            self.assertEqual(cleared, [True])

    def test_create_person_admin_rejects_agent_name_but_allows_creator_label(self):
        from app.admin.persons import create_person_admin

        saved = []

        class FakeStore:
            def get_person_profile(self, person_id):
                return None

            def save_person_profile(self, device_id, profile):
                saved.append((device_id, profile))

        with patch("app.admin.persons.store", FakeStore()):
            with self.assertRaisesRegex(ValueError, "reserved for agent itself"):
                create_person_admin("creator_yepengxiang", "叶鹏祥", "admin")
            result = create_person_admin("creator_yepengxiang", "创造者", "admin")

        self.assertTrue(result["created"])
        self.assertEqual(result["display_name"], "创造者")
        self.assertEqual(saved[0][1]["profile_role"], "owner")

    def test_create_contact_admin_allows_robot_itself_as_owner(self):
        from app.admin.contacts import ROBOT_OWNER_PERSON_ID, create_contact_admin

        saved = []

        class FakeStore:
            def __init__(self):
                self.profiles = {}

            def get_person_profile(self, person_id):
                return self.profiles.get(person_id)

            def get_person_device_id(self, person_id):
                return ""

            def save_person_profile(self, device_id, profile):
                saved.append((device_id, profile))
                self.profiles[profile["person_id"]] = profile

        with patch("app.admin.contacts.store", FakeStore()):
            result = create_contact_admin(
                owner_person_id=ROBOT_OWNER_PERSON_ID,
                display_name="唐凯",
                relationship="初中同学",
                notes=["叶鹏祥的初中同学"],
            )

        self.assertEqual(result["owner_person_id"], ROBOT_OWNER_PERSON_ID)
        self.assertIn("叶鹏祥", result["owner_display_name"])
        self.assertEqual(saved[0][0], "admin")

    def test_robot_owned_contact_is_injected_for_any_owner_prompt(self):
        from app.memory import contacts

        robot_contact = {
            "person_id": "ct_tangkai",
            "display_name": "唐凯",
            "relationship": "初中同学",
            "aliases": [],
            "personality": [],
            "experiences": [],
            "notes": ["叶鹏祥的初中同学"],
            "profile_role": "contact",
            "owner_person_id": contacts.ROBOT_OWNER_PERSON_ID,
            "confirmed": True,
        }

        class FakeStore:
            def list_person_profiles(self, device_id):
                return []

            def list_all_person_profiles(self):
                return [{"profile": robot_contact}]

            def get_person_profile(self, person_id):
                return None

        with patch("app.memory.contacts.store", FakeStore()):
            block = contacts.format_contacts_prompt_block(
                "esp32",
                "creator_yepengxiang",
                "唐凯是谁",
            )

        self.assertIn("唐凯", block)
        self.assertIn("与智能体本人关系：初中同学", block)

    def test_person_detail_uses_admin_contacts_module(self):
        from app.admin.person_detail import build_person_detail

        class FakeStore:
            def list_memory_items_detailed(self, **kwargs):
                return []

        with patch("app.admin.person_detail.store", FakeStore()), \
             patch("app.admin.person_detail.list_persons_admin", return_value=[
                 {"person_id": "p1", "display_name": "用户"}
             ]), \
             patch("app.admin.person_detail.get_profile_admin", return_value={"display_name": "用户"}), \
             patch("app.admin.person_detail.list_core_memory_admin", return_value=[]), \
             patch("app.admin.contacts.list_contacts_admin", return_value=[
                 {"person_id": "ct1", "display_name": "唐凯", "relationship": "同学", "confirmed": True}
             ]), \
             patch("app.admin.contacts.build_graph_admin", return_value={"nodes": [], "links": []}):
            detail = build_person_detail("p1")

        self.assertEqual(detail["contacts"][0]["display_name"], "唐凯")

    def test_recent_memory_admin_crud_rewrites_unified_items(self):
        from app.admin.memory import (
            create_recent_memory_admin,
            delete_recent_memory_admin,
            list_recent_memory_admin,
            update_recent_memory_admin,
        )

        class FakeStore:
            def __init__(self):
                self.items = {}
                self.seq = 0

            def get_person_profile(self, person_id):
                return {"person_id": person_id, "display_name": "用户"} if person_id == "p1" else None

            def get_person_device_id(self, person_id):
                return "dev1"

            def write_memory_item(self, **kwargs):
                self.seq += 1
                iid = f"m{self.seq}"
                self.items[iid] = {"id": iid, **kwargs, "created_at": "now", "updated_at": "now"}
                return iid

            def get_memory_item(self, item_id):
                return self.items.get(str(item_id))

            def list_memory_items(self, person_id, *, limit=50, kinds=None, visibility=None):
                rows = [x for x in self.items.values() if x["person_id"] == person_id]
                if kinds:
                    rows = [x for x in rows if x["kind"] in kinds]
                return rows[:limit]

            def delete_memory_item(self, item_id):
                return self.items.pop(str(item_id), None) is not None

        fake = FakeStore()
        with patch("app.admin.memory.store", fake):
            created = create_recent_memory_admin("p1", content="今天一起讨论了机器人外壳", kind="episode")
            self.assertEqual(created["content"], "今天一起讨论了机器人外壳")
            self.assertEqual(len(list_recent_memory_admin("p1")), 1)
            updated = update_recent_memory_admin("p1", created["id"], content="今天一起讨论了机器人外壳和后台", kind="milestone")
            self.assertEqual(updated["kind"], "milestone")
            self.assertNotIn(created["id"], fake.items)
            deleted = delete_recent_memory_admin("p1", updated["id"])

        self.assertTrue(deleted["deleted"])

    def test_long_term_memory_admin_crud_rewrites_unified_items(self):
        from app.admin.memory import (
            create_long_term_memory_admin,
            delete_long_term_memory_admin,
            list_long_term_memory_admin,
            update_long_term_memory_admin,
        )

        class FakeStore:
            def __init__(self):
                self.items = {}
                self.seq = 0

            def get_person_profile(self, person_id):
                return {"person_id": person_id, "display_name": "用户"} if person_id == "p1" else None

            def get_person_device_id(self, person_id):
                return "dev1"

            def write_memory_item(self, **kwargs):
                self.seq += 1
                iid = f"lt{self.seq}"
                self.items[iid] = {"id": iid, **kwargs, "created_at": "now", "updated_at": "now"}
                return iid

            def get_memory_item(self, item_id):
                return self.items.get(str(item_id))

            def search_memory_items(self, person_id, *, kinds=None, visibility=None, limit=20, **kwargs):
                rows = [x for x in self.items.values() if x["person_id"] == person_id]
                if kinds:
                    rows = [x for x in rows if x["kind"] in kinds]
                if visibility:
                    rows = [x for x in rows if x["visibility"] == visibility]
                return rows[:limit]

            def list_memory_items(self, person_id, *, limit=50, kinds=None, visibility=None):
                return self.search_memory_items(
                    person_id,
                    kinds=kinds,
                    visibility=visibility,
                    limit=limit,
                )

            def delete_memory_item(self, item_id):
                return self.items.pop(str(item_id), None) is not None

        fake = FakeStore()
        with patch("app.admin.memory.store", fake):
            created = create_long_term_memory_admin("p1", content="用户喜欢清爽的后台界面", kind="fact")
            self.assertEqual(len(list_long_term_memory_admin("p1")), 1)
            updated = update_long_term_memory_admin("p1", created["id"], content="用户喜欢清爽克制的后台界面", kind="wiki")
            self.assertEqual(updated["kind"], "wiki")
            self.assertNotIn(created["id"], fake.items)
            deleted = delete_long_term_memory_admin("p1", updated["id"])

        self.assertTrue(deleted["deleted"])

    def test_task_manager_rejects_duplicate_running_task(self):
        from app.admin.tasks import AdminTaskManager

        manager = AdminTaskManager()

        async def noop():
            return {"ok": True}

        manager._running_names.add("same")
        with self.assertRaises(ValueError):
            manager.start("same", "same", noop)

    def test_detect_restart_mode_uses_local_safe_without_systemd_runtime(self):
        from app.admin import ops as admin_ops

        mode = admin_ops.detect_restart_mode(
            systemd_runtime=Path("/definitely/not/systemd"),
            systemctl_path="",
            service_file=Path("/tmp/sparkbot-agent.service"),
        )

        self.assertEqual(mode["mode"], "local_dev_safe")
        self.assertFalse(mode["can_restart"])

    def test_request_restart_schedules_fixed_systemd_command(self):
        from app.admin import ops as admin_ops

        calls = []

        async def fake_runner(command, delay_sec):
            calls.append((command, delay_sec))

        mode = {
            "mode": "systemd",
            "can_restart": True,
            "unit": "sparkbot-agent.service",
            "command": ["systemctl", "restart", "sparkbot-agent.service"],
            "reason": "systemd available",
        }

        async def run_case():
            return await admin_ops.request_restart(
                restart_runner=fake_runner,
                restart_mode=mode,
                delay_sec=0,
            )

        result = asyncio.run(run_case())

        self.assertTrue(result["accepted"])
        self.assertEqual(result["mode"], "systemd")
        self.assertEqual(result["command"], ["systemctl", "restart", "sparkbot-agent.service"])
        self.assertEqual(calls, [(["systemctl", "restart", "sparkbot-agent.service"], 0)])

    def test_request_start_schedules_fixed_systemd_command(self):
        from app.admin import ops as admin_ops

        calls = []

        async def fake_runner(command, delay_sec):
            calls.append((command, delay_sec))

        mode = {
            "mode": "systemd",
            "can_restart": True,
            "unit": "sparkbot-agent.service",
            "command": ["systemctl", "restart", "sparkbot-agent.service"],
            "reason": "systemd available",
        }

        async def run_case():
            return await admin_ops.request_service_action(
                "start",
                action_runner=fake_runner,
                restart_mode=mode,
                delay_sec=0,
            )

        result = asyncio.run(run_case())

        self.assertTrue(result["accepted"])
        self.assertEqual(result["action"], "start")
        self.assertEqual(result["command"], ["systemctl", "start", "sparkbot-agent.service"])
        self.assertEqual(calls, [(["systemctl", "start", "sparkbot-agent.service"], 0)])

    def test_request_stop_schedules_fixed_systemd_command(self):
        from app.admin import ops as admin_ops

        calls = []

        async def fake_runner(command, delay_sec):
            calls.append((command, delay_sec))

        mode = {
            "mode": "systemd",
            "can_restart": True,
            "unit": "sparkbot-agent.service",
            "command": ["systemctl", "restart", "sparkbot-agent.service"],
            "reason": "systemd available",
        }

        async def run_case():
            return await admin_ops.request_service_action(
                "stop",
                action_runner=fake_runner,
                restart_mode=mode,
                delay_sec=0,
            )

        result = asyncio.run(run_case())

        self.assertTrue(result["accepted"])
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["command"], ["systemctl", "stop", "sparkbot-agent.service"])
        self.assertEqual(calls, [(["systemctl", "stop", "sparkbot-agent.service"], 0)])

    def test_request_service_action_rejects_unknown_action(self):
        from app.admin import ops as admin_ops

        async def run_case():
            return await admin_ops.request_service_action("remove", delay_sec=0)

        with self.assertRaises(ValueError):
            asyncio.run(run_case())

    def test_request_restart_does_not_kill_local_dev_process(self):
        from app.admin import ops as admin_ops

        calls = []
        mode = {
            "mode": "local_dev_safe",
            "can_restart": False,
            "unit": "sparkbot-agent.service",
            "command": [],
            "reason": "not systemd",
        }

        async def fake_runner(command, delay_sec):
            calls.append((command, delay_sec))

        async def run_case():
            return await admin_ops.request_restart(
                restart_runner=fake_runner,
                restart_mode=mode,
                delay_sec=0,
            )

        result = asyncio.run(run_case())

        self.assertFalse(result["accepted"])
        self.assertEqual(result["mode"], "local_dev_safe")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
