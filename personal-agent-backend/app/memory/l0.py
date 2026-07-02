"""
L0 核心事实记忆 —— 记忆系统的最顶层，优先级最高。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  L0 是"铁律层"——每轮对话无条件加载并注入 system prompt，优先级高于 L2/L3。
  L0 仅存储用户明确陈述的核心事实（身份、禁忌、核心关系、纪念日、强偏好），
  不包含临时计划、普通经历、次要偏好、寒暄或推断内容。

铁律：宁可漏存，不可错存（better to miss than to store wrong）。

L0 数据的三种来源路径：
  1. capture_l0_from_user_message —— 用户发言时立即正则提取（同步、实时）
  2. extract_l0_from_session_summary —— 会话结束时 LLM 从 L2 摘要提取（异步、不阻塞）
  3. sync_l0_from_profile —— 画像确认转正时，从 profile 同步身份/关系事实

L0 五大类别（按优先级排序）：
  1. identity    —— 身份与关系事实
  2. taboo       —— 绝对禁忌与雷区（生成回复前必须先检查）
  3. key_people  —— 核心亲人与重要他人（最多5人）
  4. milestone   —— 重大人生节点与固定纪念日
  5. preference  —— 顶级偏好与生活习惯
============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from app.config import settings
from app.llm import chat_completion_small
from app.memory.guard import extract_self_name, is_valid_person_name
from app.session import store

logger = logging.getLogger(__name__)

# ── L0 内存缓存 ────────────────────────────────────────────────────────────
# L0 数据低频变化（仅在用户声明新事实、记忆修正时更新），
# 每轮对话都从 DB 读取是浪费。缓存 TTL = 60 秒。
_L0_CACHE_TTL = 60.0
_l0_cache: dict[str, tuple[float, list[dict]]] = {}


def _bust_l0_cache(person_id: str) -> None:
    """使指定用户的 L0 缓存失效（写入新事实后调用）。"""
    _l0_cache.pop(str(person_id or "").strip(), None)


def list_l0_cached(person_id: str) -> list[dict]:
    """带 TTL 缓存的 L0 全量加载。"""
    pid = str(person_id or "").strip()
    if not pid:
        return []
    now = time.monotonic()
    entry = _l0_cache.get(pid)
    if entry and now - entry[0] < _L0_CACHE_TTL:
        return entry[1]
    rows = store.l0_list(pid)
    _l0_cache[pid] = (now, rows)
    return rows

# ── L0 五大类别键名与标签 ────────────────────────────────────────────────────
# 每个类别有固定的键名和中文标签，用于存储和 system prompt 注入

L0_IDENTITY = "identity"       # 身份与关系事实（最高优先级）
L0_TABOO = "taboo"             # 绝对禁忌与雷区
L0_KEY_PEOPLE = "key_people"   # 核心亲人与重要他人
L0_MILESTONE = "milestone"     # 重大人生节点与固定纪念日
L0_PREFERENCE = "preference"   # 顶级偏好与生活习惯

# 类别处理顺序（保证 system prompt 注入时身份/禁忌排最前面）
L0_CATEGORY_ORDER = (
    L0_IDENTITY, L0_TABOO, L0_KEY_PEOPLE, L0_MILESTONE, L0_PREFERENCE,
)

# 类别对应的中文标签（注入 system prompt 时作为小节标题）
L0_CATEGORY_LABELS: dict[str, str] = {
    L0_IDENTITY: "1. 身份与关系事实（最高优先级）",
    L0_TABOO: "2. 绝对禁忌与雷区（生成回复前必须先检查）",
    L0_KEY_PEOPLE: "3. 核心亲人与重要他人",
    L0_MILESTONE: "4. 重大人生节点与固定纪念日",
    L0_PREFERENCE: "5. 顶级偏好与生活习惯",
}

# ── 数据来源标签 ────────────────────────────────────────────────────────────
# 用于 is_l0_blocked 做不同来源的差异化门控：
#   user_declared  → 用户直接陈述，门控最严
#   profile_sync   → 画像转正同步，允许信任度最高的身份/禁忌类别通过
#   remember_intent → 用户明确说"记住"，中等信任度

_SOURCE_USER = "user_declared"
_SOURCE_PROFILE = "profile_sync"
_SOURCE_REMEMBER = "remember_intent"

# 核心亲人最多存 5 人（防止溢出和提示词过长）
_MAX_KEY_PEOPLE = 5

# ══════════════════════════════════════════════════════════════════════════════
# 正则表达式库 —— 从用户消息中捕获和过滤事实
# ══════════════════════════════════════════════════════════════════════════════

# ── 捕获用正则（正向匹配，从消息中提取事实）──────────────────────────────

# 禁忌声明：匹配"不要提/别聊/禁止"等表述
_TABOO_FORBID = re.compile(
    r"(?:不要|别|勿|禁止|不准)(?:提|聊|说|讲|问|提起来|再提|跟我聊|跟我说|再聊|再讲|再提)"
    r"|(?:别|不要)跟?我(?:聊|说|讲|提)"
)
# 提取禁忌对象："不要提XXX"中的XXX
_TABOO_TARGET = re.compile(
    r"(?:不要提|别提|别(?:跟我)?(?:聊|说|讲|问)|不要问(?:我的)?|别问(?:我的)?|"
    r"不要(?:再)?(?:提|聊|说|讲))"
    r"(.+?)(?:[，。！？\s]|$|了|啊|呀|嘛|呢|吧)"
)
# "永远记住"类声明（表达强烈记忆意图）
_REMEMBER_FOREVER = re.compile(
    r"(?:永远|一直|务必|千万)?(?:记住|记得|别忘了|别忘)(?:住)?(.+?)(?:[，。！？\s]|$|啊|呀|嘛|呢|吧)"
)
# "我一直/从来/绝对"类强烈偏好
_ALWAYS_PREF = re.compile(
    r"我(?:一直|永远|从来|绝对|压根)(?:都|不|没)?(.+?)(?:[，。！？\s]|$|啊|呀|嘛|呢|吧)"
)
# 用户与 Agent（叶鹏祥）的关系定位："你做我女朋友"等
_REL_TO_AGENT = re.compile(
    r"你(?:是|做)我(?:的)?(女朋友|女友|老婆|男朋友|男友|老公|闺蜜|兄弟|家人|陪伴者|对象)"
    r"|我(?:是|做)你(?:的)?(女朋友|女友|老婆|男朋友|男友|老公|闺蜜|兄弟|家人|对象)"
)
# 第三方关系："XXX是我女朋友/妈妈"等
_REL_THIRD = re.compile(
    r"([一-鿿·]{2,8})\s*是我\s*(女朋友|女友|老婆|男朋友|男友|老公|"
    r"爸爸|妈妈|爷爷|奶奶|哥哥|弟弟|姐姐|妹妹|闺蜜|兄弟|宠物|猫|狗|最好的朋友)"
)
# 真实姓名
_REAL_NAME = re.compile(r"(?:真名|全名|大名)(?:叫|是)([一-鿿·]{2,10})")
# 年龄
_AGE = re.compile(r"(?:我|今年)(?:已经)?(\d{1,2})\s*岁")
# 性别
_GENDER = re.compile(r"我是(男|女)(?:的|生)?(?:[^，。！？\s]{0,4})?")
# 常住城市
_CITY = re.compile(
    r"(?:住在|常住|来自)([一-鿿]{2,8}(?:市|省|区|县)?)(?:[，。！？\s]|$|呢|啊)"
)
# 生日/出生日期
_BIRTHDAY = re.compile(
    r"(?:生日|出生(?:日期)?)(?:是|在)?(\d{1,2}\s*月\s*\d{1,2}\s*日?|[一-鿿\d]{4,12})"
)
# 亲属关系词（用于 key_people 类别判断）
_KINSHIP = re.compile(
    r"(?:我)?(?:的)?(爸爸|妈妈|父亲|母亲|爷爷|奶奶|外公|外婆|哥哥|弟弟|姐姐|妹妹|"
    r"男友|女友|男朋友|女朋友|老公|老婆|宠物|猫|狗|闺蜜|兄弟|最好的朋友)"
)

# ── 过滤用正则（负向匹配，拦截不应进入 L0 的内容）───────────────────────

# 临时性内容：当天/近期计划类，不属于永久事实
_TEMPORAL = re.compile(
    r"今天|明天|后天|大后天|昨天|前天|上周|下周|下月|上月|这周|这次|下次|"
    r"眼前|最近要|打算|准备|即将|临时|要去|将要|过两天|过几天|待会|一会儿|"
    r"考试|出差|开会|看电影|看剧|逛街|超市|饭局|聚餐|加班|请假"
)
# 一次性经历："昨天去了XXX"等过去事件，不应固化为长期事实
_PAST_STORY = re.compile(
    r"(?:昨天|前天|上次|曾经|以前|小时候|那年|有一回|记得那次|之前)"
    r".{0,12}(?:去|逛|玩|看|吃|喝|旅游|到过|去过|经历了)"
    r"|(?:去|逛|玩|看|吃|喝|旅游|到过|去过).{0,8}(?:超市|商场|北京|上海|电影|剧)"
)
# 次要/模糊偏好："觉得还不错"等不够强烈的偏好表达
_MINOR_PREF = re.compile(
    r"觉得.{0,8}(?:不错|还行|还可以|挺好|蛮好|一般|蛮)"
    r"|(?:偶尔|有时|有时候|可能|大概|好像|挺|蛮)(?:会|想|要|觉得)?"
    r"|还不错|挺好看|还行吧|还可以吧|蛮喜欢的"
)
# 纯寒暄/噪音短句
_CASUAL_NOISE = re.compile(
    r"^(?:在吗|在不在|你好|您好|嗨|哈喽|hello|hi|嗯+|[哈呵]+|哦+|[啊呀]+|"
    r"早安|晚安|吃了吗|干嘛呢|忙吗|谢谢|多谢|好的|好哒|ok|OK|666)[。！？…~\s]*$"
    r"|^(?:没事|算了|随便|哈哈哈+|笑死)[。！？…~\s]*$"
)
# 推断性标记词："可能/好像"类不确定表述，不应作为事实存入
_INFERRED_MARKERS = re.compile(
    r"可能|大概|好像|似乎|应该|估计|听说|据说|看起来|感觉像|像是|猜"
    r"|用户可能|推断|推测|或许|也许"
)
# L2/L3 叙述性套话：LLM 生成的摘要格式模板，不是用户事实
_NARRATIVE_ROLLUP = re.compile(
    r"对话中|本轮|摘要|L2|L3|语料|用户自称「|仅用户说明|勿自动推断|"
    r"会话|压缩|叙述|里程碑叙述"
)
# 强偏好信号词：用于 preference 类别的正向确认
_CORE_PREF_STRONG = re.compile(
    r"绝对|永远|一直|从来|从不|压根|只(?:喝|吃|用|要)|唯独|必须|每天|每晚|"
    r"每周|过敏|不能吃|不吃|不喝|最讨厌|最喜欢|只喜欢|固定习惯"
)
# 固定日期信号词：用于 milestone 类别的正向确认
_FIXED_DATE = re.compile(
    r"生日|纪念日|周年|每年|固定|每逢|腊月|正月|\d{1,2}\s*月\s*\d{1,2}\s*日?"
)
# 沟通偏好："别发长语音/别啰嗦"等
_COMM_PREF = re.compile(
    r"(?:喜欢|希望|要)(?:直接|简短|别发长语音|不要长语音|别啰嗦|简单点)"
    r"|(?:不要|别)(?:发长语音|啰嗦|说教|讲大道理|跟我讲大道理)"
)


def _content_hash(text: str) -> str:
    """对文本内容做 SHA256 哈希，取前 16 位十六进制作为去重键。"""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _normalize_content(text: str) -> str:
    """规范化文本：去除首尾空白，多空格合并为单个空格。"""
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_temporary(text: str) -> bool:
    """判断文本是否包含临时性/计划性内容（今天/明天/打算等）。"""
    return bool(_TEMPORAL.search(text))


def is_l0_blocked(
    content: str, category: str, *, source: str = _SOURCE_USER,
) -> bool:
    """L0 门控函数：阻止不应进入 L0 的内容。

    L0 铁律的核心实现 —— 宁可漏存，不可错存。
    对每条候选内容按类别和来源做多道过滤检查。

    Args:
        content:  待存入的事实文本
        category: 目标 L0 类别（identity/taboo/key_people/milestone/preference）
        source:   数据来源（user_declared/profile_sync/remember_intent）

    Returns:
        True 表示应阻止（不存入），False 表示可以通过。

    过滤逻辑（按检查顺序）：
      1. 长度 < 2 字 ── 直接拒绝
      2. 纯寒暄噪音 ── 拒绝
      3. 含推断标记词（可能/好像）── 拒绝（除非 profile_sync 且为身份/禁忌类）
      4. 含 L2/L3 叙述套话 ── 拒绝
      5. profile_sync 来源 → 身份/禁忌/带日期的里程碑 → 可信，放行
      6. 临时性内容检查
      7. 一次性经历检查
      8. 各类别专项检查（如 preference 验证强偏好信号等）
    """
    text = _normalize_content(content)
    if len(text) < 2:
        return True

    cat = str(category or "").strip()

    # 噪音快速过滤
    if _CASUAL_NOISE.match(text):
        return True
    if _INFERRED_MARKERS.search(text):
        return True
    if _NARRATIVE_ROLLUP.search(text):
        return True

    # profile_sync 来源的信任策略：身份和禁忌类无条件信任
    # 因为 profile_sync 来自经过确认的画像，数据质量远高于用户实时消息
    if source == _SOURCE_PROFILE and cat in (L0_IDENTITY, L0_TABOO):
        return False
    if source == _SOURCE_PROFILE and cat == L0_MILESTONE and _FIXED_DATE.search(text):
        return False

    # 临时计划/一次性经历 → 不进入永久记忆
    if _is_temporary(text):
        return True
    if _PAST_STORY.search(text):
        return True

    # 各类别专项门控
    # preference 类别：非 profile_sync 来源必须有强偏好信号词；次要偏好拒绝
    if cat == L0_PREFERENCE:
        if _MINOR_PREF.search(text):
            return True
        if source != _SOURCE_PROFILE and not _CORE_PREF_STRONG.search(text):
            return True

    # milestone 类别：必须有固定日期信号，否则拒绝（profile_sync 来源例外）
    if cat == L0_MILESTONE:
        if not _FIXED_DATE.search(text) and source != _SOURCE_PROFILE:
            return True

    # key_people 类别：必须有亲属关系词或纪念日类信号
    if cat == L0_KEY_PEOPLE:
        if not _KINSHIP.search(text) and not re.search(
            r"生日|纪念日|永远记住|一定要记得", text
        ):
            return True
        if _MINOR_PREF.search(text) or _PAST_STORY.search(text):
            return True

    # 身份和禁忌类的额外长度限制（太长的可能是叙述而非事实）
    if cat in (L0_IDENTITY, L0_TABOO):
        if len(text) > 120 and cat == L0_IDENTITY:
            return True

    return False


def upsert_l0(
    person_id: str, category: str, content: str,
    *, device_id: str = "", source: str = _SOURCE_USER, confidence: float = 1.0,
) -> bool:
    """插入或更新一条 L0 记录。

    写入前经过 is_l0_blocked 门控 + 类别校验 + key_people 数量上限检查。
    如果对同一用户的同一类别下已存在相同内容，则更新元数据而非重复插入。

    Args:
        person_id:  用户唯一标识
        category:   L0 类别键名（必须是 L0_CATEGORY_LABELS 中的键）
        content:    事实内容文本
        device_id:  设备标识
        source:     数据来源标签
        confidence: 置信度 0.0~1.0

    Returns:
        True 表示已写入存储，False 表示被门控拒绝或参数无效。
    """
    pid = str(person_id or "").strip()
    cat = str(category or "").strip()
    body = _normalize_content(content)
    if not pid or cat not in L0_CATEGORY_LABELS or len(body) < 2:
        return False

    # 门控检查：被拦截则打日志并返回 False
    if is_l0_blocked(body, cat, source=source):
        logger.debug("L0 rejected [%s/%s]: %s", cat, source, body[:96])
        return False

    # key_people 数量上限保护：最多存 5 人
    # 如果已达上限且内容不重复，静默拒绝（不再新增）
    if cat == L0_KEY_PEOPLE and store.l0_count(pid, L0_KEY_PEOPLE) >= _MAX_KEY_PEOPLE:
        rows = store.l0_list(pid, category=L0_KEY_PEOPLE)
        if store.l0_find_by_content(pid, body, L0_KEY_PEOPLE):
            pass  # 内容已存在 → 允许更新
        else:
            return False  # 新内容 → 拒绝，防止超出上限

    store.l0_upsert(pid, cat, body, device_id=device_id, source=source, confidence=confidence)
    _bust_l0_cache(pid)
    return True


def list_l0(person_id: str) -> list[dict]:
    """获取指定用户的所有 L0 记录列表（按存储顺序）。"""
    pid = str(person_id or "").strip()
    if not pid:
        return []
    return store.l0_list(pid)


def format_l0_block(person_id: str | None) -> str:
    """将所有 L0 记忆格式化为 system prompt 注入块。

    这是 L0 注入 system prompt 的入口 —— 每轮对话调用。
    按 L0_CATEGORY_ORDER 排序输出，去重后注入。

    Args:
        person_id: 用户 ID（访客模式 None 或临时 ID 返回空字符串）

    Returns:
        Markdown 格式的 L0 记忆块；无数据时返回空字符串。
        访客模式下返回空字符串（L0 仅服务已实名用户）。
    """
    pid = str(person_id or "").strip()
    if not pid:
        return ""

    rows = list_l0_cached(pid)
    if not rows:
        return ""

    # 按类别分组并去重（同类别相同文本只保留一份）
    by_cat: dict[str, list[str]] = {c: [] for c in L0_CATEGORY_ORDER}
    seen: set[str] = set()
    for row in rows:
        cat = str(row.get("category", ""))
        text = str(row.get("content", "")).strip()
        if not text or cat not in by_cat:
            continue
        key = f"{cat}:{text}"
        if key in seen:
            continue
        seen.add(key)
        by_cat[cat].append(text)

    # 构建 Markdown 格式的 system prompt 块
    # 先写说明头，告知 LLM 这些是核心事实、需要优先遵守
    sections: list[str] = [
        "## L0 核心记忆（每轮必载，优先级高于 L2/L3；生成回复前须先核对禁忌）",
        "（L0 仅含用户明确陈述的核心事实；不含临时计划、普通经历、次要偏好、寒暄或推断内容）",
    ]
    for cat in L0_CATEGORY_ORDER:
        items = by_cat.get(cat) or []
        if not items:
            continue
        sections.append(f"### {L0_CATEGORY_LABELS[cat]}")
        for item in items:
            sections.append(f"- {item}")

    if len(sections) <= 1:
        return ""
    return "\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
# 路径 1：即时捕获 —— 从用户消息中正则提取 L0 事实（同步、实时）
# ══════════════════════════════════════════════════════════════════════════════

def capture_l0_from_user_message(
    device_id: str, person_id: str, user_msg: str,
) -> list[str]:
    """从用户消息中即时提取并存储 L0 事实（无推断，仅正则匹配）。

    这是 L0 最实时的数据入口。每次用户发消息时调用，
    用一系列正则表达式扫描用户原话，提取身份/关系/禁忌/生日等事实。

    设计原则：
    - 仅做正则匹配，不做 LLM 推断（避免幻觉风险）
    - 每条提取文本经过 is_l0_blocked 门控后才写入
    - 返回已保存的事实列表供调用方记录

    Args:
        device_id: 设备标识
        person_id: 已确认的用户 ID
        user_msg:  用户原始消息文本

    Returns:
        已成功保存到 L0 的事实文本列表。
    """
    msg = user_msg.strip()
    pid = str(person_id or "").strip()
    if not msg or not pid:
        return []

    saved: list[str] = []

    def _save(cat: str, text: str, *, source: str = _SOURCE_USER) -> None:
        """内部辅助：尝试写入 L0，成功则加入 saved 列表。"""
        if upsert_l0(pid, cat, text, device_id=device_id, source=source):
            saved.append(text)

    # 1. 提取自称/常用名
    who = extract_self_name(msg)
    if who:
        _save(L0_IDENTITY, f"用户自称/常用名：{who}")

    # 2. 提取真实姓名
    m = _REAL_NAME.search(msg)
    if m and is_valid_person_name(m.group(1)):
        _save(L0_IDENTITY, f"用户真实姓名：{m.group(1)}")

    # 3. 提取年龄
    m = _AGE.search(msg)
    if m:
        _save(L0_IDENTITY, f"用户年龄：{m.group(1)}岁")

    # 4. 提取性别
    m = _GENDER.search(msg)
    if m:
        _save(L0_IDENTITY, f"用户性别：{m.group(1)}")

    # 5. 提取常住城市
    m = _CITY.search(msg)
    if m:
        _save(L0_IDENTITY, f"用户常住城市：{m.group(1)}")

    # 6. 提取生日
    m = _BIRTHDAY.search(msg)
    if m:
        _save(L0_MILESTONE, f"用户生日：{m.group(1)}")

    # 7. 提取与 Agent（叶鹏祥）的关系定位
    m = _REL_TO_AGENT.search(msg)
    if m:
        rel = next((g for g in m.groups() if g), None)
        if rel:
            _save(L0_IDENTITY, f"用户与叶鹏祥的关系定位：{rel}")

    # 8. 提取第三方关系："XXX是我女朋友"等
    m = _REL_THIRD.search(msg)
    if m:
        _save(L0_KEY_PEOPLE, f"{m.group(1)}是用户的{m.group(2)}")

    # 9. 提取禁忌声明
    # 先检测是否包含禁忌类动词，再尝试提取具体禁忌对象
    if _TABOO_FORBID.search(msg):
        m = _TABOO_TARGET.search(msg)
        if m:
            target = m.group(1).strip("「」\"' ")
            if len(target) >= 2:
                _save(L0_TABOO, f"绝对禁止提及/触碰：{target}")
        elif len(msg) <= 80:
            # 无法提取具体对象时，整体消息作为禁忌
            _save(L0_TABOO, f"用户声明禁忌：{msg}")

    # 10. 提取"永远记住"类声明
    m = _REMEMBER_FOREVER.search(msg)
    if m:
        topic = m.group(1).strip("「」\"' ")
        if len(topic) >= 2 and not is_l0_blocked(topic, L0_KEY_PEOPLE, source=_SOURCE_REMEMBER):
            # 根据内容特征归类：含亲属词 → key_people，含纪念日词 → milestone
            if _KINSHIP.search(topic) or _KINSHIP.search(msg):
                _save(L0_KEY_PEOPLE, topic, source=_SOURCE_REMEMBER)
            elif re.search(r"生日|纪念日|周年|每年", topic):
                _save(L0_MILESTONE, topic, source=_SOURCE_REMEMBER)

    # 11. 提取"我一直/从来不"类强偏好
    m = _ALWAYS_PREF.search(msg)
    if m:
        pref = m.group(0).strip()
        if len(pref) >= 4 and _CORE_PREF_STRONG.search(pref) and not is_l0_blocked(pref, L0_PREFERENCE):
            _save(L0_PREFERENCE, pref)

    # 12. 提取沟通偏好（如"别发长语音"）
    if _COMM_PREF.search(msg) and len(msg) <= 60:
        if not is_l0_blocked(msg.strip(), L0_PREFERENCE):
            _save(L0_PREFERENCE, msg.strip())

    # 13. 提取过敏/饮食禁忌
    allergy = re.search(
        r"我(?:对|)([一-鿿]{1,6})(?:过敏|不能吃)|"
        r"我(?:绝对|从来|永远|一直)?不(?:吃|喝)([一-鿿]{1,6})",
        msg,
    )
    if allergy:
        line = f"饮食禁忌/偏好：{allergy.group(0).strip()}"
        if not is_l0_blocked(line, L0_PREFERENCE):
            _save(L0_PREFERENCE, line)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# 路径 2：会话结束提取 —— LLM 从 L2 摘要中提取 L0 级事实（异步、不阻塞）
# ══════════════════════════════════════════════════════════════════════════════

# LLM 提取提示词：要求 LLM 从 L2 会话摘要中筛选应当永久记住的核心事实
# 严格限定四种类别，明确禁止提取临时计划/普通经历/次要偏好等
_L0_EXTRACTION_PROMPT = """你是记忆过滤器。从下方 L2 会话摘要中筛选**应当永久记住的核心事实**。

只提取以下 4 类（缺一则不提取）：
1. identity — 身份/关系：姓名、年龄、性别、自称关系、与叶鹏祥的关系
2. taboo — 禁忌：明确说"不要提/别聊/别问"的话题
3. milestone — 固定纪念日：具体日期（生日、纪念日）
4. preference — 顶级偏好：带"绝对/永远/从不/每天/只/过敏/不吃"的强偏好

绝对不要提取：
- 临时计划（明天看电影、下周考试）
- 普通经历（昨天逛街、去过北京）
- 次要偏好（觉得还不错、偶尔）
- 闲聊（在吗、哈哈哈、早安）
- 推断猜测（可能、好像、似乎）

L2 摘要：
{summary}

只输出 JSON，无则 found 为空数组：
{{"found": [{{"category": "identity|taboo|milestone|preference", "content": "..."}}]}}"""


def extract_l0_from_session_summary(
    device_id: str, person_id: str, l2_summary: str,
) -> list[str]:
    """会话结束时：用小模型从 L2 摘要提取 L0 级事实（异步调用，不阻塞会话关闭）。

    这是路径 2 —— 与路径 1 的实时正则捕获互补。
    LLM 能理解语境和隐含信息，可以捕获正则无法匹配的复杂事实际述。
    但 LLM 结果仍需经过 is_l0_blocked 门控二次过滤。

    Args:
        device_id:  设备标识
        person_id:  用户 ID
        l2_summary: 由 consolidate_session 生成的 L2 摘要文本

    Returns:
        已成功保存到 L0 的事实文本列表。
    """
    pid = str(person_id or "").strip()
    if not pid or not device_id or len(l2_summary) < 20:
        return []

    prompt = _L0_EXTRACTION_PROMPT.format(summary=l2_summary)
    raw = chat_completion_small([{"role": "user", "content": prompt}])

    # 安全解析 LLM JSON 输出（容错：即使 LLM 输出不是纯 JSON）
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        return []

    found = data.get("found") if isinstance(data.get("found"), list) else []
    if not found:
        return []

    # 逐条通过门控后写入 L0
    saved: list[str] = []
    for item in found:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        cat = str(item.get("category", "")).strip()
        if not content or cat not in L0_CATEGORY_LABELS:
            continue
        if is_l0_blocked(content, cat, source=_SOURCE_USER):
            continue
        if upsert_l0(pid, cat, content, device_id=device_id, source=_SOURCE_USER):
            saved.append(content)
            logger.info("L0 extracted from session-end: [%s] %s", cat, content[:80])

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# 路径 3：画像转正同步 —— 画像确认后将身份/关系事实推入 L0
# ══════════════════════════════════════════════════════════════════════════════

def push_identity_on_profile_promotion(
    device_id: str, person_id: str, profile: dict,
) -> list[str]:
    """画像确认转正时：将身份/关系事实从 profile 同步到 L0。

    仅对已确认（confirmed=True）且非 draft/provisional 的画像执行。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        profile:   画像字典（会被 normalize_profile 规范化）

    Returns:
        已成功保存到 L0 的身份事实文本列表。
    """
    from app.memory.profile import normalize_profile, profile_display_name, profile_relationship

    p = normalize_profile(profile)
    # 只有已确认（confirmed=True）的正式画像才同步 L0
    # normalize_profile 会移除旧字段（provisional/version），统一用 confirmed 表示
    if not p.get("confirmed"):
        return []

    pid = str(person_id or p.get("person_id") or "").strip()
    if not pid:
        return []

    saved: list[str] = []
    nick = profile_display_name(p)
    rel = profile_relationship(p)
    parts: list[str] = []
    if nick and nick != "未知":
        parts.append(f"常用名/昵称：{nick}")
    if rel:
        parts.append(f"与叶鹏祥的关系：{rel}")
    if parts:
        line = "；".join(parts)
        if upsert_l0(pid, L0_IDENTITY, line, device_id=device_id, source=_SOURCE_PROFILE):
            saved.append(line)
    return saved


def push_substantive_facts_to_l0(device_id: str, person_id: str) -> list[str]:
    """画像确认时：将高置信度 L3 事实推入 L0。

    从 L3 语义记忆中检索该用户的记忆条目，仅将置信度 >= 0.85 且
    包含强信号词的事实推入对应的 L0 类别。

    Args:
        device_id: 设备标识
        person_id: 用户 ID

    Returns:
        已成功保存到 L0 的事实文本列表。
    """
    from app.memory.guard import is_noise_memory_for_l3

    pid = str(person_id or "").strip()
    if not pid:
        return []
    saved: list[str] = []
    # 最多取 30 条 L3 记忆进行评估
    for row in store.l3_list_person_memory(pid, device_id=device_id, limit=30):
        fact = str(row.get("text", "")).strip()
        if not fact or is_noise_memory_for_l3(fact):
            continue
        cat = str(row.get("category", "general")).lower()
        conf = float(row.get("confidence") or 0.0)
        # 仅高置信度事实才推入 L0（L0 铁律：宁可漏存，不可错存）
        if conf < 0.85:
            continue
        target = ""
        # 按类别和内容特征映射到 L0 类别
        if cat in ("relationship", "person", "correction"):
            target = L0_IDENTITY
        elif cat == "preference" and _CORE_PREF_STRONG.search(fact):
            target = L0_PREFERENCE
        elif cat in ("milestone", "event") and _FIXED_DATE.search(fact):
            target = L0_MILESTONE
        elif _KINSHIP.search(fact):
            target = L0_KEY_PEOPLE
        if target and upsert_l0(pid, target, fact, device_id=device_id, source=_SOURCE_USER, confidence=conf):
            saved.append(fact)
    return saved


def sync_l0_from_profile(device_id: str, person_id: str, profile: dict) -> list[str]:
    """画像转正时的一站式 L0 同步：身份 + 高置信度 L3 事实。

    组合调用 push_identity_on_profile_promotion 和 push_substantive_facts_to_l0，
    在画像确认时一次性完成 L0 同步。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        profile:   画像字典

    Returns:
        所有已保存的事实文本列表（身份类 + L3 高置信度事实类）。
    """
    identity = push_identity_on_profile_promotion(device_id, person_id, profile)
    facts = push_substantive_facts_to_l0(device_id, person_id)
    return identity + facts
