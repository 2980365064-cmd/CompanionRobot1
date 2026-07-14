"""
核心事实（Core Facts）—— 高置信、不可遗忘的底层事实（自包含模块）。

============================================================================
业务含义：
  - 身份信息：名字、年龄、关系定位
  - 绝对禁忌：称呼雷区、不能提的话题
  - 核心人物：重要亲友关系
  - 人生节点：纪念日、里程碑
  - 顶级偏好：强偏好（"不吃辣"、"喜欢喝奶茶"）

这些事实每轮对话无条件注入 system prompt。

铁律：宁可漏存，不可错存（better to miss than to store wrong）。

数据的三种来源路径：
  1. capture_core_fact_from_message —— 用户发言时立即正则提取（同步、实时）
  2. extract_core_facts_from_summary —— 会话结束时 LLM 从近期摘要提取（异步、不阻塞）
  3. sync_core_facts_from_profile —— 画像确认转正时，从 profile 同步身份/关系事实

五大类别（按优先级排序）：
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

# ── TTL 缓存 ────────────────────────────────────────────────────────────
# 核心事实低频变化（仅在用户声明新事实、记忆修正时更新），
# 每轮对话都从 DB 读取是浪费。缓存 TTL = 60 秒。
_CORE_FACT_CACHE_TTL = 60.0
_core_fact_cache: dict[str, tuple[float, list[dict]]] = {}


def invalidate_core_fact_cache(person_id: str) -> None:
    """使指定用户的核心事实缓存失效（写入后调用）。"""
    invalidate_core_fact_cache(person_id)


# ── 五大类别键名与标签 ────────────────────────────────────────────────────

CORE_FACT_IDENTITY = "identity"       # 身份与关系事实（最高优先级）
CORE_FACT_TABOO = "taboo"             # 绝对禁忌与雷区
CORE_FACT_KEY_PEOPLE = "key_people"   # 核心亲人与重要他人
CORE_FACT_MILESTONE = "milestone"     # 重大人生节点与固定纪念日
CORE_FACT_PREFERENCE = "preference"   # 顶级偏好与生活习惯


# 类别处理顺序（保证 system prompt 注入时身份/禁忌排最前面）
CORE_FACT_CATEGORY_ORDER = (
    CORE_FACT_IDENTITY, CORE_FACT_TABOO, CORE_FACT_KEY_PEOPLE, CORE_FACT_MILESTONE, CORE_FACT_PREFERENCE,
)


# 类别对应的中文标签（注入 system prompt 时作为小节标题）
CORE_FACT_CATEGORY_LABELS: dict[str, str] = {
    CORE_FACT_IDENTITY: "1. 身份与关系事实（最高优先级）",
    CORE_FACT_TABOO: "2. 绝对禁忌与雷区（生成回复前必须先检查）",
    CORE_FACT_KEY_PEOPLE: "3. 核心亲人与重要他人",
    CORE_FACT_MILESTONE: "4. 重大人生节点与固定纪念日",
    CORE_FACT_PREFERENCE: "5. 顶级偏好与生活习惯",
}

_SOURCE_USER = "user_declared"
_SOURCE_REMEMBER = "remember_intent"

# 核心亲人最多存 5 人（防止溢出和提示词过长）
_MAX_KEY_PEOPLE = 5

# ══════════════════════════════════════════════════════════════════════════════
# 正则表达式库 —— 从用户消息中捕获和过滤事实
# ══════════════════════════════════════════════════════════════════════════════

# ── 捕获用正则（正向匹配，从消息中提取事实）──────────────────────────────
_TABOO_FORBID = re.compile(
    r"(?:不要|别|勿|禁止|不准)(?:提|聊|说|讲|问|提起来|再提|跟我聊|跟我说|再聊|再讲|再提)"
    r"|(?:别|不要)跟?我(?:聊|说|讲|提)"
)
_TABOO_TARGET = re.compile(
    r"(?:不要提|别提|别(?:跟我)?(?:聊|说|讲|问)|不要问(?:我的)?|别问(?:我的)?|"
    r"不要(?:再)?(?:提|聊|说|讲))"
    r"(.+?)(?:[，。！？\s]|$|了|啊|呀|嘛|呢|吧)"
)
_REMEMBER_FOREVER = re.compile(
    r"(?:永远|一直|务必|千万)?(?:记住|记得|别忘了|别忘)(?:住)?(.+?)(?:[，。！？\s]|$|啊|呀|嘛|呢|吧)"
)
_ALWAYS_PREF = re.compile(
    r"我(?:一直|永远|从来|绝对|压根)(?:都|不|没)?(.+?)(?:[，。！？\s]|$|啊|呀|嘛|呢|吧)"
)
_REL_TO_AGENT = re.compile(
    r"你(?:是|做)我(?:的)?(女朋友|女友|老婆|男朋友|男友|老公|闺蜜|兄弟|家人|陪伴者|对象)"
    r"|我(?:是|做)你(?:的)?(女朋友|女友|老婆|男朋友|男友|老公|闺蜜|兄弟|家人|对象)"
)
_REL_THIRD = re.compile(
    r"([一-鿿·]{2,8})\s*是我\s*(女朋友|女友|老婆|男朋友|男友|老公|"
    r"爸爸|妈妈|爷爷|奶奶|哥哥|弟弟|姐姐|妹妹|闺蜜|兄弟|宠物|猫|狗|最好的朋友)"
)
_REAL_NAME = re.compile(r"(?:真名|全名|大名)(?:叫|是)([一-鿿·]{2,10})")
_AGE = re.compile(r"(?:我|今年)(?:已经)?(\d{1,2})\s*岁")
_GENDER = re.compile(r"我是(男|女)(?:的|生)?(?:[^，。！？\s]{0,4})?")
_CITY = re.compile(
    r"(?:住在|常住|来自)([一-鿿]{2,8}(?:市|省|区|县)?)(?:[，。！？\s]|$|呢|啊)"
)
_BIRTHDAY = re.compile(
    r"(?:生日|出生(?:日期)?)(?:是|在)?(\d{1,2}\s*月\s*\d{1,2}\s*日?|[一-鿿\d]{4,12})"
)
_KINSHIP = re.compile(
    r"(?:我)?(?:的)?(爸爸|妈妈|父亲|母亲|爷爷|奶奶|外公|外婆|哥哥|弟弟|姐姐|妹妹|"
    r"男友|女友|男朋友|女朋友|老公|老婆|宠物|猫|狗|闺蜜|兄弟|最好的朋友)"
)

# ── 过滤用正则（负向匹配，拦截不应进入核心事实的内容）────────────────────

_TEMPORAL = re.compile(
    r"今天|明天|后天|大后天|昨天|前天|上周|下周|下月|上月|这周|这次|下次|"
    r"眼前|最近要|打算|准备|即将|临时|要去|将要|过两天|过几天|待会|一会儿|"
    r"考试|出差|开会|看电影|看剧|逛街|超市|饭局|聚餐|加班|请假"
)
_PAST_STORY = re.compile(
    r"(?:昨天|前天|上次|曾经|以前|小时候|那年|有一回|记得那次|之前)"
    r".{0,12}(?:去|逛|玩|看|吃|喝|旅游|到过|去过|经历了)"
    r"|(?:去|逛|玩|看|吃|喝|旅游|到过|去过).{0,8}(?:超市|商场|北京|上海|电影|剧)"
)
_MINOR_PREF = re.compile(
    r"觉得.{0,8}(?:不错|还行|还可以|挺好|蛮好|一般|蛮)"
    r"|(?:偶尔|有时|有时候|可能|大概|好像|挺|蛮)(?:会|想|要|觉得)?"
    r"|还不错|挺好看|还行吧|还可以吧|蛮喜欢的"
)
_CASUAL_NOISE = re.compile(
    r"^(?:在吗|在不在|你好|您好|嗨|哈喽|hello|hi|嗯+|[哈呵]+|哦+|[啊呀]+|"
    r"早安|晚安|吃了吗|干嘛呢|忙吗|谢谢|多谢|好的|好哒|ok|OK|666)[。！？…~\s]*$"
    r"|^(?:没事|算了|随便|哈哈哈+|笑死)[。！？…~\s]*$"
)
_INFERRED_MARKERS = re.compile(
    r"可能|大概|好像|似乎|应该|估计|听说|据说|看起来|感觉像|像是|猜"
    r"|用户可能|推断|推测|或许|也许"
)
_NARRATIVE_ROLLUP = re.compile(
    r"对话中|本轮|摘要|语料|用户自称「|仅用户说明|勿自动推断|"
    r"会话|压缩|叙述|里程碑叙述"
)
_CORE_PREF_STRONG = re.compile(
    r"绝对|永远|一直|从来|从不|压根|只(?:喝|吃|用|要)|唯独|必须|每天|每晚|"
    r"每周|过敏|不能吃|不吃|不喝|最讨厌|最喜欢|只喜欢|固定习惯"
)
_FIXED_DATE = re.compile(
    r"生日|纪念日|周年|每年|固定|每逢|腊月|正月|\d{1,2}\s*月\s*\d{1,2}\s*日?"
)
_COMM_PREF = re.compile(
    r"(?:喜欢|希望|要)(?:直接|简短|别发长语音|不要长语音|别啰嗦|简单点)"
    r"|(?:不要|别)(?:发长语音|啰嗦|说教|讲大道理|跟我讲大道理)"
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _normalize_content(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_temporary(text: str) -> bool:
    return bool(_TEMPORAL.search(text))


def is_core_fact_blocked(
    content: str, category: str, *, source: str = _SOURCE_USER,
) -> bool:
    """核心事实门控函数：阻止不应进入核心事实的内容。

    对每条候选内容按类别和来源做多道过滤检查。
    过滤逻辑（按检查顺序）：
      1. 长度 < 2 字 —— 直接拒绝
      2. 纯寒暄噪音 —— 拒绝
      3. 含推断标记词（可能/好像）—— 拒绝
      4. 含叙述套话 —— 拒绝
      5. 身份/禁忌类 —— 无条件信任
      6. 临时性内容检查
      7. 一次性经历检查
      8. 各类别专项检查
    """
    text = _normalize_content(content)
    if len(text) < 2:
        return True

    cat = str(category or "").strip()

    if _CASUAL_NOISE.match(text):
        return True
    if _INFERRED_MARKERS.search(text):
        return True
    if _NARRATIVE_ROLLUP.search(text):
        return True

    if source == _SOURCE_USER and cat in (CORE_FACT_IDENTITY, CORE_FACT_TABOO):
        return False
    if source == _SOURCE_USER and cat == CORE_FACT_MILESTONE and _FIXED_DATE.search(text):
        return False

    if _is_temporary(text):
        return True
    if _PAST_STORY.search(text):
        return True

    if cat == CORE_FACT_PREFERENCE:
        if _MINOR_PREF.search(text):
            return True
        if not _CORE_PREF_STRONG.search(text):
            return True

    if cat == CORE_FACT_MILESTONE:
        if not _FIXED_DATE.search(text):
            return True

    if cat == CORE_FACT_KEY_PEOPLE:
        if not _KINSHIP.search(text) and not re.search(
            r"生日|纪念日|永远记住|一定要记得", text
        ):
            return True
        if _MINOR_PREF.search(text) or _PAST_STORY.search(text):
            return True

    if cat in (CORE_FACT_IDENTITY, CORE_FACT_TABOO):
        if len(text) > 120 and cat == CORE_FACT_IDENTITY:
            return True

    return False


def write_core_fact(
    person_id: str, category: str, content: str,
    *, device_id: str = "", source: str = _SOURCE_USER, confidence: float = 1.0,
) -> bool:
    """插入或更新一条核心事实。

    写入前经过 is_core_fact_blocked 门控 + 类别校验 + key_people 数量上限检查。

    Args:
        person_id:  用户唯一标识
        category:   类别键名
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
    if not pid or cat not in CORE_FACT_CATEGORY_LABELS or len(body) < 2:
        return False

    if is_core_fact_blocked(body, cat, source=source):
        logger.debug("Core fact rejected [%s/%s]: %s", cat, source, body[:96])
        return False

    if cat == CORE_FACT_KEY_PEOPLE:
        existing = store.search_memory_items(pid, visibility="always", kinds=[cat], limit=_MAX_KEY_PEOPLE + 1)
        if len(existing) >= _MAX_KEY_PEOPLE:
            found = store.search_memory_items(pid, visibility="always", kinds=[cat], query=body, limit=1)
            if not found:
                return False

    store.write_memory_item(
        person_id=pid,
        device_id=device_id,
        kind=cat,
        source=source,
        visibility="always",
        content=body,
        confidence=confidence,
    )
    invalidate_core_fact_cache(pid)
    return True


def list_core_facts(person_id: str) -> list[dict]:
    """获取指定用户的所有核心事实列表。"""
    pid = str(person_id or "").strip()
    if not pid:
        return []
    return store.list_core_facts(pid)


def load_core_facts(person_id: str) -> list[dict]:
    """加载指定用户的核心事实（带 TTL 缓存）。"""
    pid = str(person_id or "").strip()
    if not pid:
        return []
    now = time.monotonic()
    entry = _core_fact_cache.get(pid)
    if entry and now - entry[0] < _CORE_FACT_CACHE_TTL:
        return entry[1]
    rows = list_core_facts(pid)
    _core_fact_cache[pid] = (now, rows)
    return rows


def format_core_facts_block(person_id: str | None) -> str:
    """将所有核心事实格式化为 system prompt 注入块。

    按 CORE_FACT_CATEGORY_ORDER 排序输出，去重后注入。
    访客模式下返回空字符串。

    Returns:
        Markdown 格式的核心事实块；无数据时返回空字符串。
    """
    pid = str(person_id or "").strip()
    if not pid:
        return ""

    rows = load_core_facts(pid)
    if not rows:
        return ""

    by_cat: dict[str, list[str]] = {c: [] for c in CORE_FACT_CATEGORY_ORDER}
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

    sections: list[str] = [
        "## 核心事实（每轮必载；生成回复前须先核对禁忌）",
        "（仅含用户明确陈述的核心事实；不含临时计划、普通经历、次要偏好、寒暄或推断内容）",
    ]
    for cat in CORE_FACT_CATEGORY_ORDER:
        items = by_cat.get(cat) or []
        if not items:
            continue
        sections.append(f"### {CORE_FACT_CATEGORY_LABELS[cat]}")
        for item in items:
            sections.append(f"- {item}")

    if len(sections) <= 1:
        return ""
    return "\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
# 路径 1：即时捕获 —— 从用户消息中正则提取核心事实（同步、实时）
# ══════════════════════════════════════════════════════════════════════════════

def capture_core_fact_from_message(
    device_id: str, person_id: str, user_msg: str,
) -> list[str]:
    """从用户消息中即时提取并存储核心事实（无推断，仅正则匹配）。

    用一系列正则表达式扫描用户原话，提取身份/关系/禁忌/生日等事实。
    仅做正则匹配，不做 LLM 推断（避免幻觉风险）。

    Args:
        device_id: 设备标识
        person_id: 已确认的用户 ID
        user_msg:  用户原始消息文本

    Returns:
        已成功保存的事实文本列表。
    """
    msg = user_msg.strip()
    pid = str(person_id or "").strip()
    if not msg or not pid:
        return []

    saved: list[str] = []

    def _save(cat: str, text: str, *, source: str = _SOURCE_USER) -> None:
        if write_core_fact(pid, cat, text, device_id=device_id, source=source):
            saved.append(text)

    # 1. 自称/常用名
    who = extract_self_name(msg)
    if who:
        _save(CORE_FACT_IDENTITY, f"用户自称/常用名：{who}")

    # 2. 真实姓名
    m = _REAL_NAME.search(msg)
    if m and is_valid_person_name(m.group(1)):
        _save(CORE_FACT_IDENTITY, f"用户真实姓名：{m.group(1)}")

    # 3. 年龄
    m = _AGE.search(msg)
    if m:
        _save(CORE_FACT_IDENTITY, f"用户年龄：{m.group(1)}岁")

    # 4. 性别
    m = _GENDER.search(msg)
    if m:
        _save(CORE_FACT_IDENTITY, f"用户性别：{m.group(1)}")

    # 5. 常住城市
    m = _CITY.search(msg)
    if m:
        _save(CORE_FACT_IDENTITY, f"用户常住城市：{m.group(1)}")

    # 6. 生日
    m = _BIRTHDAY.search(msg)
    if m:
        _save(CORE_FACT_MILESTONE, f"用户生日：{m.group(1)}")

    # 7. 与 Agent（叶鹏祥）的关系定位
    m = _REL_TO_AGENT.search(msg)
    if m:
        rel = next((g for g in m.groups() if g), None)
        if rel:
            _save(CORE_FACT_IDENTITY, f"用户与叶鹏祥的关系定位：{rel}")

    # 8. 第三方关系
    m = _REL_THIRD.search(msg)
    if m:
        _save(CORE_FACT_KEY_PEOPLE, f"{m.group(1)}是用户的{m.group(2)}")

    # 9. 禁忌声明
    if _TABOO_FORBID.search(msg):
        m = _TABOO_TARGET.search(msg)
        if m:
            target = m.group(1).strip("「」\"' ")
            if len(target) >= 2:
                _save(CORE_FACT_TABOO, f"绝对禁止提及/触碰：{target}")
        elif len(msg) <= 80:
            _save(CORE_FACT_TABOO, f"用户声明禁忌：{msg}")

    # 10. "永远记住"类声明
    m = _REMEMBER_FOREVER.search(msg)
    if m:
        topic = m.group(1).strip("「」\"' ")
        if len(topic) >= 2 and not is_core_fact_blocked(topic, CORE_FACT_KEY_PEOPLE, source=_SOURCE_REMEMBER):
            if _KINSHIP.search(topic) or _KINSHIP.search(msg):
                _save(CORE_FACT_KEY_PEOPLE, topic, source=_SOURCE_REMEMBER)
            elif re.search(r"生日|纪念日|周年|每年", topic):
                _save(CORE_FACT_MILESTONE, topic, source=_SOURCE_REMEMBER)

    # 11. "我一直/从来不"类强偏好
    m = _ALWAYS_PREF.search(msg)
    if m:
        pref = m.group(0).strip()
        if len(pref) >= 4 and _CORE_PREF_STRONG.search(pref) and not is_core_fact_blocked(pref, CORE_FACT_PREFERENCE):
            _save(CORE_FACT_PREFERENCE, pref)

    # 12. 沟通偏好
    if _COMM_PREF.search(msg) and len(msg) <= 60:
        if not is_core_fact_blocked(msg.strip(), CORE_FACT_PREFERENCE):
            _save(CORE_FACT_PREFERENCE, msg.strip())

    # 13. 过敏/饮食禁忌
    allergy = re.search(
        r"我(?:对|)([一-鿿]{1,6})(?:过敏|不能吃)|"
        r"我(?:绝对|从来|永远|一直)?不(?:吃|喝)([一-鿿]{1,6})",
        msg,
    )
    if allergy:
        line = f"饮食禁忌/偏好：{allergy.group(0).strip()}"
        if not is_core_fact_blocked(line, CORE_FACT_PREFERENCE):
            _save(CORE_FACT_PREFERENCE, line)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# 路径 2：会话结束提取 —— LLM 从会话摘要中提取核心事实（异步、不阻塞）
# ══════════════════════════════════════════════════════════════════════════════

_CORE_FACT_EXTRACTION_PROMPT= """你是记忆过滤器。从下方会话摘要中筛选**应当永久记住的核心事实**。

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

会话摘要：
{summary}

只输出 JSON，无则 found 为空数组：
{{"found": [{{"category": "identity|taboo|milestone|preference", "content": "..."}}]}}"""


def extract_core_facts_from_summary(
    device_id: str, person_id: str, session_summary: str,
) -> list[str]:
    """会话结束时：用小模型从会话摘要提取核心事实（异步调用，不阻塞会话关闭）。

    Args:
        device_id:      设备标识
        person_id:      用户 ID
        session_summary: 会话摘要文本

    Returns:
        已成功保存的事实文本列表。
    """
    pid = str(person_id or "").strip()
    if not pid or not device_id or len(session_summary) < 20:
        return []

    prompt = _CORE_FACT_EXTRACTION_PROMPT.format(summary=session_summary)
    raw = chat_completion_small([{"role": "user", "content": prompt}])

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

    saved: list[str] = []
    for item in found:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        cat = str(item.get("category", "")).strip()
        if not content or cat not in CORE_FACT_CATEGORY_LABELS:
            continue
        if is_core_fact_blocked(content, cat, source=_SOURCE_USER):
            continue
        if write_core_fact(pid, cat, content, device_id=device_id, source=_SOURCE_USER):
            saved.append(content)
            logger.info("Core fact extracted from session-end: [%s] %s", cat, content[:80])

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# 路径 3：画像转正同步 —— 画像确认后将身份/关系事实推入核心事实
# ══════════════════════════════════════════════════════════════════════════════

def push_identity_on_profile_promotion(
    device_id: str, person_id: str, profile: dict,
) -> list[str]:
    """画像确认转正时：将身份/关系事实从 profile 同步到核心事实。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        profile:   画像字典

    Returns:
        已成功保存的身份事实文本列表。
    """
    from app.memory.profile import normalize_profile, profile_display_name, profile_relationship

    p = normalize_profile(profile)
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
        if write_core_fact(pid, CORE_FACT_IDENTITY, line, device_id=device_id, source=_SOURCE_USER):
            saved.append(line)
    return saved


def push_substantive_facts_to_core_facts(device_id: str, person_id: str) -> list[str]:
    """画像确认时：将高置信度长期记忆事实推入核心事实。

    仅将置信度 >= 0.85 且包含强信号词的事实推入对应的类别。

    Args:
        device_id: 设备标识
        person_id: 用户 ID

    Returns:
        已成功保存的事实文本列表。
    """
    from app.memory.guard import is_noise_memory

    pid = str(person_id or "").strip()
    if not pid:
        return []
    saved: list[str] = []
    for row in store.list_person_long_term_memory(pid, device_id=device_id, limit=30):
        fact = str(row.get("text", "")).strip()
        if not fact or is_noise_memory(fact):
            continue
        cat = str(row.get("category", "general")).lower()
        conf = float(row.get("confidence") or 0.0)
        if conf < 0.85:
            continue
        target = ""
        if cat in ("relationship", "person", "correction"):
            target = CORE_FACT_IDENTITY
        elif cat == "preference" and _CORE_PREF_STRONG.search(fact):
            target = CORE_FACT_PREFERENCE
        elif cat in ("milestone", "event") and _FIXED_DATE.search(fact):
            target = CORE_FACT_MILESTONE
        elif _KINSHIP.search(fact):
            target = CORE_FACT_KEY_PEOPLE
        if target and write_core_fact(pid, target, fact, device_id=device_id, source=_SOURCE_USER, confidence=conf):
            saved.append(fact)
    return saved


def sync_core_facts_from_profile(device_id: str, person_id: str, profile: dict) -> list[str]:
    """画像转正时的一站式同步：身份 + 高置信度事实。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        profile:   画像字典

    Returns:
        所有已保存的事实文本列表。
    """
    identity = push_identity_on_profile_promotion(device_id, person_id, profile)
    facts = push_substantive_facts_to_core_facts(device_id, person_id)
    return identity + facts
