"""
身份状态机 —— person_id 是记忆的主键，名字是显示标签。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Identity 是"记忆的门禁系统"——决定了用户是否有权访问各层记忆。

状态机三态流转：

  ┌─────────┐   解析名字+ID     ┌──────────┐   用户"是/确认"    ┌──────────┐
  │  访客    │ ──────────────→  │  待确认   │ ────────────────→ │  已实名   │
  │  tmp_*   │   register pending│ pending  │   register_verified│ verified  │
  └─────────┘                    └──────────┘                    └──────────┘
       │        用户"不是/取消"        │
       └──────────────────────────────┘

  guest (tmp_*)    → 工作上下文 only；每轮口语引导实名；不调用 embedding
  pending          → 已解析名字+ID，等待用户发送"是/对"确认
  verified (非tmp) → 核心事实/近期记忆/长期记忆/Profile 全部可用

身份注册规则：
  - ID 由用户自定义（2-64 位字母数字 + _ -），系统不会自动生成正式 ID
  - 相同名字 + 不同 ID = 不同人；仅靠名字不能绑定任何人
  - 新 ID + 新名字 → 需要用户确认才能入库
  - 已有 ID → 名字匹配则直接绑定，不匹配则提示校验失败

访客引导策略：
  - 先正常接话、回答用户问题
  - 再用口语顺带请对方发"名字 XXX ID xxx"
  - 避免客服腔/公告腔；像微信朋友顺嘴问一句
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from uuid import uuid4

from app.config import settings
from app.memory.guard import is_valid_person_name
from app.memory.profile import (
    empty_profile,
    find_profile_by_name,
    normalize_profile,
    profile_display_name,
)
from app.session import store

logger = logging.getLogger(__name__)

# 访客 ID 前缀：所有未实名用户的 person_id 以此开头
TEMP_PREFIX = "tmp_"

# ── 身份解析正则表达式 ────────────────────────────────────────────────────
# 这些正则是身份状态机的词法分析层，负责从自然语言消息中提取结构化凭证。

# 仅 ID（格式："id: xxx" 或 "编号：xxx"）
# 匹配独立的 ID 声明，不依赖名字配对。用于场景：用户此前说过名字，
# 本轮只补充了 ID。
_ID_CRED = re.compile(
    r"(?:^|[\s，。；;])"
    r"(?:id|ID|编号)[:：\s]*"
    r"([a-zA-Z0-9_-]{2,64})"
    r"(?:[\s，。；;]|$)",
)

# 名字 + ID（格式："名字XXX ID: xxx"）
# 中文名字/英文名 + 自定 ID 的配对格式，是最常见的新用户注册消息。
_NAME_ID_PAIR = re.compile(
    r"(?:名字|姓名|我叫|我是)[:：\s]*"
    r"([一-鿿·A-Za-z0-9_]{2,12})"
    r"[\s，。；;]+"
    r"(?:id|ID|编号)[:：\s]*"
    r"([a-zA-Z0-9_-]{2,64})",
    re.I,
)

# ID + 名字（格式："ID: xxx 姓名：XXX"）
# 与 _NAME_ID_PAIR 顺序相反，兼容用户先说 ID 后说名字的习惯。
_ID_NAME_PAIR = re.compile(
    r"(?:id|ID|编号)[:：\s]*"
    r"([a-zA-Z0-9_-]{2,64})"
    r"[\s，。；;]+"
    r"(?:名字|姓名)[:：\s]*"
    r"([一-鿿·A-Za-z0-9_]{2,12})",
    re.I,
)

# 确认/否认信号 —— 仅匹配纯信号词，不匹配含凭证或长句的消息。
# 正则要求消息完全由信号词+可选标点组成，确保"是的，名字XX ID yy"不被误判为确认。
_CONFIRM = re.compile(r"^(?:是|对|没错|确认|确定|是的|嗯是|好的|好|可以|行|嗯嗯|嗯|ok|OK|Ok)[。！？…~\s]*$")
_DENY = re.compile(r"^(?:不是|不对|否|取消|算了)[。！？…~\s]*$")


@dataclass
class IdentityTurnResult:
    """身份解析的结果数据类。

    Fields:
        person_id:           当前活跃的用户 ID（可能是 tmp_ 的访客 ID）
        person_profile:      用户画像（已实名时为完整画像，否则为 None）
        verified:            是否已实名验证通过
        guest_mode:          是否处于访客模式
        hint:                本轮注入的提示文本（如"身份验证通过"或错误提示）
        monitor_event:       监控事件标签（用于日志/统计）
        pending_registration: 待确认的注册信息（{name, person_id}，仅在 pending 状态有值）
    """
    person_id: str
    person_profile: dict | None = None
    verified: bool = False
    guest_mode: bool = True
    hint: str = ""
    monitor_event: str = ""
    pending_registration: dict | None = None


def is_temp_person_id(person_id: str | None) -> bool:
    """判断是否为访客临时 ID（以 tmp_ 前缀开头）。"""
    return str(person_id or "").strip().startswith(TEMP_PREFIX)


def is_verified_person_id(person_id: str | None) -> bool:
    """判断是否为已实名用户（非空且非 tmp_ 前缀）。"""
    pid = str(person_id or "").strip()
    return bool(pid) and not is_temp_person_id(pid)


def validate_person_id(raw: str) -> bool:
    """校验 person_id 格式：2-64 位字母数字 + _ -，且不是 tmp_ 前缀。

    用户自定义 ID 的格式约束，用于防止注入无效 ID。
    """
    pid = str(raw or "").strip()
    if len(pid) < 2 or len(pid) > 64:
        return False
    if pid.startswith(TEMP_PREFIX):
        return False
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]+", pid))


def _normalize_name(name: str) -> str:
    """规范化名字：去除所有空白字符（用于模糊匹配）。

    为什么要去空白：中文名字中间可能插入空格（"张 三"），
    存储时的格式不保证与用户输入一致，去空白后做子串包含匹配更鲁棒。
    """
    return re.sub(r"\s+", "", (name or "").strip())


def names_match(stored: str, claimed: str) -> bool:
    """宽松名字匹配：完全相等、子串包含均视为匹配。

    用于身份验证时比较用户声明的名字和存储的名字。
    为什么用子串匹配而非严格相等：用户可能只发"慧"而非"刘远慧"全名，
    或存储时多/少空格，子串包含比严格匹配更宽容、体验更好。
    """
    a = _normalize_name(stored)
    b = _normalize_name(claimed)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def new_temp_person_id() -> str:
    """生成新的访客临时 ID：tmp_ + 12 位十六进制 UUID 片段。"""
    return f"{TEMP_PREFIX}{uuid4().hex[:12]}"


def _parse_bare_name_id(msg: str) -> tuple[str, str] | None:
    """解析裸「姓名 ID」格式（机器人口语引导后用户最常见的回复）。

    示例：「刘远慧 123」「名字 刘远慧 123」「刘远慧123」
    仅在短消息上匹配，避免从长句里误抽两个词当凭证。
    """
    text = (msg or "").strip()
    if not text or len(text) > 48:
        return None
    text = re.sub(r"^(?:名字|姓名)[:：\s]*", "", text, flags=re.I).strip()
    if not text:
        return None

    parts = [p for p in re.split(r"[\s，。；;]+", text) if p]
    if len(parts) == 2:
        name, pid = parts[0].strip(), parts[1].strip()
        if is_valid_person_name(name) and validate_person_id(pid):
            return name, pid

    glued = re.fullmatch(r"([一-鿿·]{2,12})([a-zA-Z0-9_-]{2,64})", text)
    if glued:
        name, pid = glued.group(1).strip(), glued.group(2).strip()
        if is_valid_person_name(name) and validate_person_id(pid):
            return name, pid
    return None


def parse_identity_credentials(message: str) -> tuple[str, str] | None:
    """从用户消息中解析身份凭证（名字 + ID）。

    支持四种格式：
      1. "名字XXX ID: xxx"
      2. "ID: xxx 姓名：XXX"
      3. 仅 ID（"id: xxx"）——此时从上下文尝试提取名字
      4. 裸「姓名 ID」（如「刘远慧 123」，与口语引导一致）

    Args:
        message: 用户消息文本

    Returns:
        (name, person_id) 元组；无法解析时返回 None。
    """
    msg = (message or "").strip()
    if not msg:
        return None

    # 尝试格式 1 和 2：名字+ID 成对出现
    for pat in (_NAME_ID_PAIR, _ID_NAME_PAIR):
        m = pat.search(msg)
        if m:
            if pat is _NAME_ID_PAIR:
                name, pid = m.group(1).strip(), m.group(2).strip()
            else:
                pid, name = m.group(1).strip(), m.group(2).strip()
            if is_valid_person_name(name) and validate_person_id(pid):
                return name, pid

    bare = _parse_bare_name_id(msg)
    if bare:
        return bare

    # 尝试格式 3：仅 ID，从上下文中提取名字
    id_m = _ID_CRED.search(msg)
    if not id_m:
        return None
    pid = id_m.group(1).strip()
    if not validate_person_id(pid):
        return None

    # 从消息中除去 ID 部分，尝试在剩余文本中寻找有效名字
    name_part = _ID_CRED.sub(" ", msg)
    name_part = re.sub(r"(?:id|ID|编号)[:：\s]*[a-zA-Z0-9_-]+", " ", name_part, flags=re.I)
    name_part = re.sub(
        r"(?:名字|姓名|我叫|我是)[:：\s]*", " ", name_part, flags=re.I
    ).strip(" ，。；;")
    for token in re.split(r"[\s，。；;]+", name_part):
        t = token.strip()
        if is_valid_person_name(t):
            return t, pid
    return None


def register_verified_person(device_id: str, person_id: str, display_name: str) -> dict:
    """注册新的已实名用户：创建画像并标记 confirmed。

    仅在用户首次注册时调用。ID 必须不存在于已有画像中。

    Args:
        device_id:    设备标识
        person_id:    用户自定义的唯一 ID
        display_name: 用户显示名

    Returns:
        规范化后的完整画像字典。

    Raises:
        ValueError: person_id 无效 / display_name 无效 / person_id 已存在
    """
    pid = str(person_id).strip()
    name = str(display_name).strip()
    if not validate_person_id(pid):
        raise ValueError("invalid person_id")
    if not is_valid_person_name(name):
        raise ValueError("invalid display_name")
    if store.get_person_profile(pid):
        raise ValueError("person_id already exists")
    profile = empty_profile(name, person_id=pid)
    profile["confirmed"] = True
    store.save_person_profile(device_id, profile)
    logger.info("registered person id=%s name=%s", pid, name)
    return normalize_profile(profile)


def _load_pending(session_id: str) -> dict | None:
    """从会话存储中加载待确认的注册信息。

    待确认状态存储在 session 表的 identity_pending 字段（JSON 字符串），
    包含 name 和 person_id 两个字段。只有同时存在两个字段才视为有效 pending。
    返回 None 表示无待确认或数据损坏。
    """
    raw = store.get_session_identity_pending(session_id)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("person_id") and data.get("name"):
        return data
    return None


def _save_pending(session_id: str, pending: dict | None) -> None:
    """保存/清除待确认的注册信息到会话存储。

    pending=None 时清除（如用户取消或确认后），否则将 dict 序列化为 JSON 存储。
    """
    if not pending:
        store.set_session_identity_pending(session_id, None)
        return
    store.set_session_identity_pending(session_id, json.dumps(pending, ensure_ascii=False))


def _bind_verified(
    device_id: str,
    session_id: str,
    person_id: str,
    profile: dict,
    *,
    hint: str = "",
) -> IdentityTurnResult:
    """将已验证的用户绑定到当前会话。

    执行三个副作用操作（不可逆）：
      1. 设置会话的活跃 person_id（该 session 后续所有记忆操作绑定此 ID）
      2. 重置访客轮数计数器（用户已实名，不再做口语引导）
      3. 清除 pending 注册状态（确认/绑定后 pending 失效）

    返回值 guest_mode=False 表示当前会话从访客模式切换为已实名模式，
    agent 模块据此启用 核心事实/近期记忆/长期记忆 检索。
    """
    store.set_session_active_person(session_id, person_id)
    store.reset_guest_turn_count(session_id)
    _save_pending(session_id, None)
    return IdentityTurnResult(
        person_id=person_id,
        person_profile=normalize_profile(profile),
        verified=True,
        guest_mode=False,
        hint=hint,
        monitor_event=f"实名绑定 · {profile_display_name(profile)} id={person_id[:12]}",
    )


def ensure_guest_person_id(session_id: str, *, replace_verified: bool = False) -> str:
    """确保会话有一个访客 ID。

    ``replace_verified`` 仅用于用户主动开始身份登记或切换访客角色时，
    让会话脱离此前的已实名对象，避免错误继承其记忆。
    """
    active = store.get_session_active_person_id(session_id)
    if active and is_temp_person_id(active):
        return active
    if active and is_verified_person_id(active) and not replace_verified:
        return active
    tmp = new_temp_person_id()
    store.set_session_active_person(session_id, tmp)
    return tmp


def resolve_identity_turn(
    device_id: str, session_id: str, message: str,
) -> IdentityTurnResult:
    """每轮对话的身份解析入口 —— 状态机主逻辑。

    根据当前会话状态和用户消息，执行对应的身份转换：

    状态机分支：
      1. 有 pending + 确认信号（"是/对"）→ 注册入库，绑定 verified
      2. 有 pending + 否认信号（"不是/取消"）→ 取消注册，回退 guest
      3. 有 pending + 无新凭证 → 保持 pending，提示确认
      4. 无 pending + 消息含身份凭证 → 查库：
         a. ID 存在 + 名字匹配 → 直接绑定 verified
         b. ID 存在 + 名字不匹配 → 错误提示
         c. 名字匹配已有画像但 ID 不同 → 错误提示
         d. 新 ID + 新名字 → 进入 pending 待确认
      5. 无 pending + 无身份凭证 + 已实名 → 保持 verified
      6. 无 pending + 无身份凭证 + 访客 → 增加访客轮数计数

    Args:
        device_id:  设备标识
        session_id: 会话标识
        message:    用户当前消息

    Returns:
        IdentityTurnResult 包含本轮后的身份状态和提示信息。
    """
    msg = (message or "").strip()
    active = store.get_session_active_person_id(session_id) or ""
    pending = _load_pending(session_id)

    # 状态机入口：pending 是"是否处于待确认"的核心标记，active 是"当前是谁"。
    # 以下 6 个分支按优先级排列，高优先级的先匹配（如确认/否认优先于凭证解析）。
    # 顺序不能乱：确认/否认必须在凭证解析之前检查，否则"是的"会被当做名字尝试解析。

    # 分支 1：待确认 + 用户确认 → 注册入库
    if pending and _CONFIRM.match(msg):
        try:
            profile = register_verified_person(
                device_id, str(pending["person_id"]), str(pending["name"]),
            )
            return _bind_verified(
                device_id, session_id, profile["person_id"], profile,
                hint=(
                    f"【身份入库成功 · 用口语回应】"
                    f"用户确认了身份：{pending['name']}（ID:{pending['person_id']}）。"
                    f"你自然回复一句，像「好嘞{ pending['name'] }，记住了～」"
                    f"之后正常接话。这是已实名用户，你有权限访问 核心事实/近期记忆/长期记忆/画像了。"
                ),
            )
        except ValueError as exc:
            _save_pending(session_id, None)
            return IdentityTurnResult(
                person_id=ensure_guest_person_id(session_id, replace_verified=True),
                guest_mode=True,
                hint=_hint_bind_failure_register(str(exc)),
            )

    # 分支 2：待确认 + 用户否认 → 取消注册
    if pending and _DENY.match(msg):
        _save_pending(session_id, None)
        return IdentityTurnResult(
            person_id=ensure_guest_person_id(session_id, replace_verified=True),
            guest_mode=True,
            hint=(
                "【身份已取消】用户说不是，取消了刚才的注册。"
                "自然接一句，后半段仍可顺口问名字+ID。仍按访客模式（仅 工作上下文）。"
            ),
        )

    # 分支 3：待确认 + 用户没做确认/否认 → 继续用口语追问
    if pending and not parse_identity_credentials(msg):
        return IdentityTurnResult(
            person_id=ensure_guest_person_id(session_id, replace_verified=True),
            guest_mode=True,
            pending_registration=pending,
            hint=f"""【待确认新身份 · 你必须在本轮回复中口语确认】
用户刚才自称「{pending['name']}」ID「{pending['person_id']}」——库里还没有。
你**必须先正常接话**回复用户的问题，然后**后半段用口语追问**，像：
「对了，{pending['name']}是新角色呀？确定了哈？」或「{pending['name']}对吧，确定了我记下来」
注意：不要用【】格式；不要客服腔；像朋友随口确认。
对方回复「确定」就入库；回复「不是」就取消。""",
        )

    # 分支 4：消息含身份凭证 → 查库验证
    creds = parse_identity_credentials(msg)
    if creds:
        name, pid = creds
        existing = store.get_person_profile(pid)
        if existing:
            stored_name = profile_display_name(existing)
            # 4a：ID 存在且名字匹配 → 直接绑定
            if names_match(stored_name, name):
                return _bind_verified(
                    device_id, session_id, pid, existing,
                    hint=f"【身份验证通过】{name} / {pid}，已加载个人记忆。",
                )
            # 4b：ID 对但名字错 → 拒绝
            return IdentityTurnResult(
                person_id=ensure_guest_person_id(session_id, replace_verified=True),
                guest_mode=True,
                hint=_hint_bind_failure_id_name_mismatch(pid, stored_name, name),
                monitor_event=f"身份校验失败 · ID对名错 id={pid[:12]}",
            )
        # 4c：用名字反查 → 名字匹配已有画像但 ID 不同
        by_name = find_profile_by_name(device_id, name)
        if by_name:
            stored_pid = str(by_name.get("person_id") or "").strip()
            return IdentityTurnResult(
                person_id=ensure_guest_person_id(session_id, replace_verified=True),
                guest_mode=True,
                hint=_hint_bind_failure_name_id_mismatch(name, stored_pid),
                monitor_event=f"身份校验失败 · 名对ID错 id={stored_pid[:12]}",
            )
        # 4d：新 ID + 新名字 → 进入待确认
        pending_new = {"name": name, "person_id": pid}
        _save_pending(session_id, pending_new)
        return IdentityTurnResult(
            person_id=ensure_guest_person_id(session_id, replace_verified=True),
            guest_mode=True,
            pending_registration=pending_new,
            hint=f"""【新身份待确认 · 你必须在本轮回复中口语确认】
用户第一次自称「{name}」ID「{pid}」——库里没有这个人。
你**必须先正常接话**，然后用自然口语确认对方身份。
**不要**用「请确认」「回复确认」这种客服话术，不要用固定句式，每次说法不一样。
对方回复「确定/是/对」就入库，回复「不是」就取消。""",
            monitor_event=f"新身份待确认 · {name} id={pid[:12]}",
        )

    # 分支 5：无凭证 + 已实名 → 保持已验证状态
    if is_verified_person_id(active):
        prof = store.get_person_profile(active)
        return IdentityTurnResult(
            person_id=active,
            person_profile=normalize_profile(prof) if prof else None,
            verified=True,
            guest_mode=False,
        )

    # 分支 6：无凭证 + 访客 → 每轮口语引导实名（轮换问法，避免复读）
    guest_pid = ensure_guest_person_id(session_id)
    turn = store.increment_guest_turn_count(session_id)
    incomplete = _incomplete_identity_hint(msg)
    if incomplete:
        return IdentityTurnResult(
            person_id=guest_pid,
            guest_mode=True,
            hint=incomplete,
            monitor_event=f"身份未绑上 · 格式不全 turn={turn}",
        )
    return IdentityTurnResult(
        person_id=guest_pid,
        guest_mode=True,
        hint=guest_identity_nudge_for_turn(turn),
        monitor_event="访客 工作上下文" if turn == 1 else "",
    )


def resolve_person_before_memory(
    device_id: str, session_id: str, message: str,
) -> tuple[str | None, dict | None, str, bool, str]:
    """为对话入口返回身份解析元组。

    Returns:
        (person_id, person_profile, hint, guest_mode, monitor_event)
    """
    result = resolve_identity_turn(device_id, session_id, message)
    return (
        result.person_id, result.person_profile, result.hint,
        result.guest_mode, result.monitor_event or "",
    )


def memory_scoped_to_person(person_id: str | None) -> bool:
    """判断记忆检索是否应限定到特定用户（= 非访客）。

    这是 router 模块的入口门控：为 True 时启用 核心事实/近期记忆/长期记忆 检索，
    为 False 时仅返回 工作上下文（当前会话窗口），节省 embedding API 费用。
    """
    return is_verified_person_id(person_id)


# ── 访客口语引导提示词 ─────────────────────────────────────────────────────

_GUEST_NUDGE_ANGLES = (
    "用「你谁啊」那种随口语气，后半段问名字+ID",
    "用「还没认识你」起头，轻松要名字和ID",
    "半开玩笑催对方自报家门，别太正经",
    "像在问微信好友备注：怎么称呼、ID多少",
    "根据对方刚聊的话题自然转折去问身份",
    "简短起头（诶/话说/那个），别又用「对了」",
    "像发语音顺口问一句，2秒能读完",
    "对方情绪一般时：先接话再轻飘飘问一句",
    "俏皮一点：「你还没报ID呢」这类",
    "直接了当：「名字+ID扔我一下，ID随便起」",
)

_GREETING_ONLY = frozenset({
    "你好", "您好", "嗨", "哈喽", "在吗", "在么", "嗯", "哦", "啊", "哈",
})

_BIND_FAILURE_MARKERS = ("绑定失败", "不匹配", "入库失败", "身份未绑上")


def guest_identity_nudge_for_turn(turn: int) -> str:
    """按轮次轮换问法角度，降低复读感。"""
    angle = _GUEST_NUDGE_ANGLES[(max(1, turn) - 1) % len(_GUEST_NUDGE_ANGLES)]
    return f"""## 顺带问对方名字+ID（本条回复必须执行）
对方是访客，系统还没绑定身份，你暂时记不住TA。先正常接话，后半段**必须**问对方是谁。
本轮风格：{angle}。让对方发「名字+ID」即可，ID随便起，如「刘远慧 123」。
**禁止复读**之前任何一轮的问法；禁止「请您」「实名」「绑定」「登记」等客服词。
每轮都要问，像微信朋友顺口一句。"""


def _hint_bind_failure_id_name_mismatch(pid: str, stored_name: str, claimed_name: str) -> str:
    return f"""【绑定失败 · 名字和ID对不上 · 你必须口语告知用户重新输入】
系统没绑上。ID「{pid}」在库里是「{stored_name}」，不是你说的「{claimed_name}」。
先正常接话，口语说清楚哪里不对，请对方重新发「名字+ID」，例如「{stored_name} {pid}」。
禁止客服腔；每次换说法；不要假装已经记住了对方。"""


def _hint_bind_failure_name_id_mismatch(name: str, stored_pid: str) -> str:
    return f"""【绑定失败 · 名字和ID对不上 · 你必须口语告知用户重新输入】
系统没绑上。名字「{name}」在库里的ID是「{stored_pid}」，跟你这次发的不一致。
先正常接话，口语请对方按正确ID重发，如「{name} {stored_pid}」。
禁止客服腔；每次换说法；不要假装已经记住了对方。"""


def _hint_bind_failure_register(reason: str) -> str:
    return f"""【绑定失败 · 入库失败 · 你必须口语告知用户重新输入】
系统没绑上（{reason}）。先正常接话，口语请对方重新发「名字+ID」，ID用2位以上字母数字。
禁止客服腔；不要假装已经记住了对方。"""


def _incomplete_identity_hint(msg: str) -> str | None:
    """用户似乎想绑定但格式不全 → 口语提示补全。"""
    text = (msg or "").strip()
    if not text or len(text) > 32 or parse_identity_credentials(text):
        return None

    if re.fullmatch(r"[一-鿿·A-Za-z0-9_]{2,12}", text):
        if text in _GREETING_ONLY:
            return None
        if is_valid_person_name(text):
            return f"""【绑定失败 · 只有名字 · 你必须口语告知用户重新输入】
用户发了「{text}」但没绑上——还缺ID。先接话，口语说名字收到了，再发个ID就行，
比如「{text} 123」这种，ID随便起。禁止客服腔；不要假装已经绑定了。"""

    if _ID_CRED.search(text):
        name_part = _ID_CRED.sub(" ", text)
        name_part = re.sub(
            r"(?:id|ID|编号)[:：\s]*[a-zA-Z0-9_-]+", " ", name_part, flags=re.I
        ).strip(" ，。；;")
        has_name = any(
            is_valid_person_name(t.strip())
            for t in re.split(r"[\s，。；;]+", name_part)
            if t.strip()
        )
        if not has_name:
            return """【绑定失败 · 只有ID · 你必须口语告知用户重新输入】
用户发了ID但没绑上——还缺名字。先接话，口语请对方把名字也带上，
比如「刘远慧 123」这种格式。禁止客服腔；不要假装已经绑定了。"""

    if re.search(r"(?:名字|姓名|我叫|我是)", text, re.I) and not re.search(
        r"(?:id|ID|编号)", text, re.I
    ):
        return """【绑定失败 · 缺ID · 你必须口语告知用户重新输入】
用户说了名字但没绑上——还缺ID。先接话，口语请对方补个ID，随便起，如「名字 123」。
禁止客服腔；不要假装已经绑定了。"""
    return None


def is_guest_bind_failure_hint(identity_hint: str) -> bool:
    """绑定/格式失败类 hint，agent 需口语告知用户重新输入。"""
    hint = (identity_hint or "").strip()
    return bool(hint) and any(m in hint for m in _BIND_FAILURE_MARKERS)


def guest_needs_casual_identity_nudge(identity_hint: str) -> bool:
    """访客模式下是否需要顺带问名字+ID（待确认/已实名成功除外）。"""
    hint = (identity_hint or "").strip()
    if not hint:
        return False
    skip = ("待确认", "身份已确认", "身份验证通过", "身份入库成功")
    return not any(m in hint for m in skip)
