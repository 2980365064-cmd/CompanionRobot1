"""后台身份管理 API —— 列出/修改/删除已实名用户的 person_id 和 display_name。

本模块的角色：
  陪伴机器人支持多用户（通过 person_id 区分），每个用户的记忆、画像、核心事实 核心事实
  等数据全部按 person_id 隔离。本模块提供后台管理员对已实名用户进行增删改查的能力，
  区别于自动身份识别（app/memory/identity.py）。

关键业务规则：
  - 访客 tmp_* 开头的临时 ID 不允许通过后台管理操作（这些是自动分配的）
  - persona_global 系统 ID 不允许删除（那是语料导入的全局知识）
  - 修改 person_id 会级联更新所有记忆表（核心事实/近期记忆/长期记忆/Facts/画像）
  - 修改 display_name 仅影响画像中的显示名，不改 person_id
"""

from __future__ import annotations

from app.memory.guard import is_valid_person_name
from app.memory.identity import is_temp_person_id, validate_person_id
from app.memory.profile import normalize_profile, profile_display_name
from app.session import store

RESERVED_AGENT_DISPLAY_NAMES = {"叶鹏祥", "叶鹏祥大侠"}
CREATOR_DISPLAY_NAME = "创造者"


def _is_valid_admin_display_name(name: str) -> bool:
    return is_valid_person_name(name)


def _validate_admin_display_name(name: str) -> None:
    if name in RESERVED_AGENT_DISPLAY_NAMES:
        raise ValueError(
            f"display_name reserved for agent itself; use {CREATOR_DISPLAY_NAME} for creator debugging"
        )
    if not _is_valid_admin_display_name(name):
        raise ValueError("invalid display_name")


def list_persons_admin() -> list[dict]:
    """列出所有已实名的用户（过滤掉访客 tmp_* 开头的临时 ID）。

    返回:
        list[dict]: 每个元素包含 person_id, display_name, device_id, updated_at

    业务说明：
      陪伴机器人在用户首次连接时会自动分配 tmp_* 访客 ID，当他们通过
      "名字 XXX ID xxx" 格式完成实名后才会创建正式画像。本函数只返回后者。
    """
    rows = store.list_all_person_profiles()
    out: list[dict] = []
    for row in rows:
        pid = str(row["person_id"])
        # 跳过访客临时 ID，只展示已实名的用户
        if is_temp_person_id(pid):
            continue
        profile = row["profile"]
        if str(profile.get("profile_role") or "owner") == "contact":
            continue
        out.append(
            {
                "person_id": pid,
                "display_name": profile_display_name(profile),
                "device_id": row["device_id"],
                "updated_at": row["updated_at"],
            }
        )
    return out


def update_person_admin(
    person_id: str,
    *,
    new_person_id: str | None = None,
    display_name: str | None = None,
) -> dict:
    """修改已实名用户的 person_id 和/或 display_name。

    参数:
        person_id:     当前用户 ID（必须已存在于画像表中）
        new_person_id: 新的用户 ID（可选），若提供且与当前不同则执行级联重命名
        display_name:  新的显示名（可选），用于对话中称呼该用户

    返回:
        dict: 包含更新后的 person_id, display_name, device_id, updated_at
              若执行了重命名则额外包含 renamed_from 和 migrate_stats

    级联影响：
      修改 person_id 会通过 store.rename_person_id 在同一事务中更新以下所有表：
        - person_profiles（画像）
        - memory_items（统一记忆库）
        - sessions（活跃会话绑定的 person_id）
      这样可以保证 person_id 改名后所有历史记忆不丢失。
    """
    current_id = str(person_id or "").strip()
    if not current_id:
        raise ValueError("person_id required")
    # 不允许后台修改访客临时 ID，这些是自动生命周期管理的
    if is_temp_person_id(current_id):
        raise ValueError("cannot edit guest tmp_* id")

    profile = store.get_person_profile(current_id)
    if not profile:
        raise ValueError("person not found")

    renamed_from: str | None = None
    migrate_stats: dict[str, int] | None = None
    target_id = current_id

    # 处理 person_id 重命名
    if new_person_id is not None:
        new_id = str(new_person_id).strip()
        if new_id != current_id:
            if not validate_person_id(new_id):
                raise ValueError("invalid person_id format")
            if is_temp_person_id(new_id):
                raise ValueError("person_id cannot use tmp_ prefix")
            # 执行级联重命名：一个事务内更新所有包含 person_id 的表
            migrate_stats = store.rename_person_id(current_id, new_id)
            renamed_from = current_id
            target_id = new_id
            # 重新加载新 ID 的画像数据
            profile = store.get_person_profile(new_id) or profile
            profile["person_id"] = new_id
            profile["user_id"] = new_id

    # 处理 display_name 修改
    if display_name is not None:
        name = str(display_name).strip()
        _validate_admin_display_name(name)
        profile["display_name"] = name

    # 只有实际有变更时才写库
    if renamed_from or display_name is not None:
        device_id = store.get_person_device_id(target_id) or "default"
        store.save_person_profile(device_id, normalize_profile(profile))

    refreshed = store.get_person_profile(target_id) or profile
    result = {
        "person_id": target_id,
        "display_name": profile_display_name(refreshed),
        "device_id": store.get_person_device_id(target_id),
        "updated_at": refreshed.get("update_time") or refreshed.get("updated_at"),
    }
    if renamed_from:
        result["renamed_from"] = renamed_from
        result["migrate_stats"] = migrate_stats or {}
    return result


def delete_person_admin(person_id: str) -> dict:
    """删除已实名用户及其全部关联记忆数据。

    参数:
        person_id: 要删除的用户 ID

    返回:
        dict: {"deleted": True, "person_id": ..., "display_name": ..., "delete_stats": {...}}
              delete_stats 包含各表删除的记录数

    注意事项：
      - 不允许删除访客 tmp_* 临时 ID
      - 不允许删除 persona_global 系统级语料 ID
      - 删除操作不可逆，会清除该用户的所有 核心事实/近期记忆/长期记忆/Facts/画像/关联图数据
    """
    pid = str(person_id or "").strip()
    if not pid:
        raise ValueError("person_id required")
    if is_temp_person_id(pid):
        raise ValueError("cannot delete guest tmp_* id")

    from app.config import settings

    # 保护系统语料 ID：persona_global 存储的是通用知识（如陪伴技巧、语气风格），
    # 不属于任何一个真实用户，删除会导致语料库残缺
    persona_pid = str(getattr(settings, "persona_fact_person_id", "") or "").strip()
    if persona_pid and pid == persona_pid:
        raise ValueError("cannot delete persona system id")

    profile = store.get_person_profile(pid)
    if not profile:
        raise ValueError("person not found")

    display_name = profile_display_name(profile)
    # store.delete_person_id 会清理所有关联表的数据
    delete_stats = store.delete_person_id(pid)
    return {
        "deleted": True,
        "person_id": pid,
        "display_name": display_name,
        "delete_stats": delete_stats,
    }


def create_person_admin(person_id: str, display_name: str, device_id: str = "") -> dict:
    """后台新建已实名用户。

    参数:
        person_id:    用户 ID（不能是 tmp_ 开头）
        display_name: 显示名
        device_id:    设备 ID（可选，默认为 "admin"）

    返回:
        dict: 包含新建用户的 person_id, display_name, device_id
    """
    pid = str(person_id or "").strip()
    if not pid:
        raise ValueError("person_id required")
    if is_temp_person_id(pid):
        raise ValueError("person_id cannot use tmp_ prefix")
    if not validate_person_id(pid):
        raise ValueError("invalid person_id format")
    name = str(display_name or "").strip()
    _validate_admin_display_name(name)
    if store.get_person_profile(pid):
        raise ValueError("person already exists")

    from datetime import datetime, timezone

    dev = str(device_id or "").strip() or "admin"
    profile = {
        "person_id": pid,
        "user_id": pid,
        "display_name": name,
        "aliases": [],
        "relationship": "",
        "personality": [],
        "experiences": [],
        "emotional_habit": [],
        "profile_role": "owner",
        "confirmed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    store.save_person_profile(dev, profile)
    return {
        "person_id": pid,
        "display_name": name,
        "device_id": dev,
        "created": True,
    }
