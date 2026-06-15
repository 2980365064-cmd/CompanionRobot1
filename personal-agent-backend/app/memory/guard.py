"""
对话守卫 —— 召回门控 + 反幻觉 + 用户事实捕获。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Guard 是"安全阀和护城河"——承担两个关键职责：

职责一：召回门控（原 intent.py 已并入前半段）
  决定当前轮是否需要触发 L2/L3 记忆检索。
  寒暄短句跳过检索（省 API 调用），回忆/追问类必须检索。
  核心函数：query_needs_memory_answer / has_recall_intent / is_casual_smalltalk

职责二：反幻觉守卫（原 guard.py 后半段）
  防止 Agent 产生"幻觉记忆"——即把刚说的当成早就知道的、
  把自称噪声当老熟人、编造记忆中没有的信息。
  核心函数：user_message_hints（生成本轮反幻觉提示注入 system prompt）

守卫的核心原则：
  1. 记忆库里没有的信息 → 必须说不知道/不清楚，可追问补充
  2. 本轮首次出现的自称/经历/关系 → 不能回复成老熟人
  3. 不确定 → 宁可短句追问，不编造
  4. 人名 ≠ 身份：L3 中某人名出现不代表当前说话人就是那个人

情感业务语义：
  - 访客模式：用户未实名时，仅 L1 可用，不能假装认识
  - 画像确认：名字/关系已在画像中确认的，才是"认识的人"
  - 女友口吻：刘远慧/刘大炮确认关系 → 启用调侃亲密口吻
============================================================================
"""

from __future__ import annotations

import re

from app.session import store


# ══════════════════════════════════════════════════════════════════════════════
# 第一道防线：召回意图门控 —— 决定是否触发 L2/L3 向量检索
# ══════════════════════════════════════════════════════════════════════════════

# 强回忆信号 —— 这些词出现时，用户明显在调用长期记忆
# 匹配到 → 必须触发 L2/L3 检索
_RECALL_STRONG = [
    r"还记得",
    r"记不记得",
    r"想起",
    r"想起来",
    r"记起",
    r"记不起来",
    r"以前(是不是|说过|提过)",
    r"之前.{0,12}(说|聊|提|去|见|吃|在|玩|有)",
    r"那时候",
    r"上次.{0,12}(说|聊|提|去|见|吃|在|玩)",
    r"你说过",
    r"很久以前",
    r"小时候",
    r"\d+年前",
    r"\d+年\d*月",          # 具体年月：2023年6月
    r"\d+年",                # 具体年份
    r"去年|前年|当初",
    r"那天",
    r"那次",
    r"回想",
    r"回忆",
    r"忘了",
    r"忘记",
    r"不记得",
]

# 追问/施压信号 —— 用户对 Agent 的记忆能力表示怀疑
# 匹配到 → 触发更深层的检索，同时标记 memory_miss
_RECALL_PRESS = [
    r"再想想",
    r"你再想想",
    r"你肯定记得",
    r"真的不记得",
    r"怎么可能忘",
    r"别装(傻|了)?",
    r"好好想想",
    r"提示一下",
    r"想不起来吗",
    r"忘了\s*[?？]?$",
    r"真的(想起来了|记起来了|记得|想起来了吗)",
    r"那你说说",
    r"具体(说说|呢|点)",
    r"还有呢",
    r"继续",
]

# 个人信息类关键词 —— 涉及用户的身份、关系、经历
# 匹配到 → 需要记忆检索参与回答
_PERSONAL = [
    r"女朋友|男友|老婆|老公|远慧|刘远慧|刘大炮|秋雨|刘远航",
    r"你谁|我是谁|你是谁|叫什么|输错|打错|姐姐|弟弟|亲戚",
    r"我俩|咱们|我们一起|咱俩",
    r"见面|约会|去哪|吃过|玩过",
    r"实习|杭州|南溪|爱琴海",
    r"喜欢什么|爱吃|忌口|喝什么",
    r"周末.*(干嘛|做什么|干啥)",
]

# 事实类短查询 —— 明显需要从记忆中匹配答案的问题
_FACT_QUERY = re.compile(
    r"(叫什么|叫什么名字|忌口|喜欢喝|猫叫|生日|老家)"
)

# 询问认识某人 —— "你认识XXX吗"
_KNOWS_PERSON = re.compile(
    r"(?:你)?认(?:识|得)(?:[一-鿿·]{2,10})|[一-鿿·]{2,10}是谁"
)

# 问句结尾检测（？/吗/嘛/么）
_QUESTION_MARK = re.compile(r"[?？吗嘛么]$")

# 纯寒暄短句（跳过检索，直接 LLM 自由回答即可）
_CASUAL = re.compile(
    r"^(在吗|在不在|嗯嗯?|哦哦?|好哒?|行|收到|嗨|hello|hi|哈喽)[\?？!！。…]*$",
    re.I,
)

# 扩展寒暄（更多问候/关心短句模式）
_CASUAL_EXTRA = [
    r"^(你好|您好|你好啊|您好啊|嗨喽|喂)[\?？!！。…\s]*$",
    r"^(咋了|怎么了|干嘛呢|干啥呢|在么|在嘛)[\?？!！。…\s]*$",
    r"^(早上好|晚上好|下午好|早安|晚安)[\?？!！。…\s]*$",
]


def is_casual_smalltalk(query: str) -> bool:
    """判断是否为寒暄/短句闲聊，这类消息跳过 L2/L3 向量检索以节省 embedding 调用。

    规则：长度 ≤ 28 字符 + 匹配寒暄模式。
    超过 28 字符的消息即使包含"你好"也不跳过（可能是实际内容）。

    Args:
        query: 用户消息文本

    Returns:
        True 表示跳过检索，False 表示正常检索。
    """
    q = query.strip()
    if not q or len(q) > 28:
        return False
    if _CASUAL.match(q):
        return True
    for pat in _CASUAL_EXTRA:
        if re.match(pat, q, re.I):
            return True
    return False


def needs_profile_archive(query: str) -> bool:
    """判断是否为深度谈心/人生经历类问题，需注入 Profile 人物履历归档块。

    这种问题不是日常闲聊，而是用户希望 Agent 了解自己的性格、经历、
    人生故事，因此需要启用低频的画像归档（日常以 L0 核心事实优先）。

    Args:
        query: 用户消息文本

    Returns:
        True 表示需要注入人物履历归档。
    """
    q = query.strip()
    if not q or is_casual_smalltalk(q):
        return False
    if re.search(
        r"我的经历|成长经历|人生经历|聊聊我的过去|我的过去|我的故事|"
        r"你懂我吗|了解我吗|我是什么样的人|我这个人|我的性格|"
        r"聊聊我自己|说说我的|我的履历|人生低谷|童年|幼年",
        q,
    ):
        return True
    # 短消息 + 特定人生关键词 → 也可能是深度话题
    if re.search(r"小时候|那年|曾经.*低谷|失恋|和奶奶", q) and len(q) <= 80:
        return True
    return False


def needs_l3_recall(query: str) -> bool:
    """兼容别名：判断是否需要 L3 长期记忆检索（等同 has_recall_intent）。"""
    return has_recall_intent(query)


def has_recall_intent(query: str) -> bool:
    """判断是否有明显的长期回忆意图（如"还记得/小时候/去年"等信号词）。"""
    q = query.strip()
    if not q:
        return False
    for pat in _RECALL_STRONG + _RECALL_PRESS:
        if re.search(pat, q):
            return True
    return False


def is_identity_question(query: str) -> bool:
    """判断是否为身份类问题（"我是谁/你记得我吗/我叫什么"）。

    这类问题有特殊性：
    - 优先依赖 L0 核心记忆（身份事实）和 L2 近期摘要
    - 不应使用 L3 中的自称噪声（可能包含不同人的名字）
    - 无明确答案时应坦诚说"不记得"，而非罗列矛盾名字
    """
    q = query.strip()
    if not q:
        return False
    return bool(
        re.search(
            r"你知道我是谁|还记得我是谁|我是谁[吗嘛]?|你记得我吗|我叫什么|我的名字|你知道我叫什么",
            q,
        )
    )


def memory_l3_texts(memory: dict | None) -> list[str]:
    """从召回结果中提取 L3 文本列表（兼容旧 semantic 字段名）。"""
    mem = memory or {}
    return list(mem.get("l3") or mem.get("semantic") or [])


def memory_l3_hit(memory: dict | None) -> bool:
    """判断本轮召回中 L3 是否命中（兼容旧 facts_hit 字段）。"""
    mem = memory or {}
    return bool(mem.get("l3_hit") or mem.get("facts_hit"))


def query_needs_memory_answer(query: str) -> bool:
    """综合判断本轮是否需要记忆支撑回答。

    这是召回门控的核心判断函数——决定是否调用 L3 检索、是否标记 memory_miss。

    决策逻辑（按优先级）：
      1. 空消息 → 不需要
      2. 寒暄短句 → 不需要（跳过 embedding 调用）
      3. 身份类问题 → 需要
      4. 含回忆信号词 → 需要
      5. 询问认识某人 → 需要
      6. 含个人信息关键词 → 需要
      7. 短事实类查询 → 需要
      8. 短问句 → 需要
      9. 含疑问词 + 短消息 → 需要

    Args:
        query: 用户消息文本

    Returns:
        True 表示需要记忆检索，False 表示 LLM 自由回答即可。
    """
    q = query.strip()
    if not q:
        return False
    if is_casual_smalltalk(q):
        return False
    if is_identity_question(q):
        return True
    if has_recall_intent(q):
        return True
    if _KNOWS_PERSON.search(q):
        return True
    for pat in _PERSONAL:
        if re.search(pat, q):
            return True
    if _FACT_QUERY.search(q) and len(q) <= 48:
        return True
    if _QUESTION_MARK.search(q) and len(q) <= 40:
        return True
    if re.search(r"(什么|谁|哪|几|咋|怎么|为啥|为什么)", q) and len(q) <= 60:
        return True
    return False


# ── 记忆未命中：硬回复（不经主 LLM 编造）────────────────────────────────────

_MONTH_IN_QUERY = re.compile(
    r"(\d{4})年(?:([一二三四五六七八九十]{1,3})|(\d{1,2}))月"
)
_CN_MONTH = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}
_QUERY_STOP_TERMS = frozenset({
    "什么", "怎么", "为什么", "为啥", "有没有", "记得", "还记得", "之间",
    "发生", "事情", "我们", "你们", "知道", "告诉", "一下", "可以", "是不是",
    "当时", "那次", "怎样", "如何", "哪里", "哪儿", "什么时候", "多久",
})

_UNKNOWN_GF = (
    "这事儿我真没印象，你跟我讲讲呗？",
    "记忆库里没翻到这段，你说说是哪件？",
    "这个我不太记得了，你再提示我一下？",
)
_UNKNOWN_VISITOR = (
    "这个我不太清楚，你说说看？",
    "我这边没这段记录，你再讲讲？",
)


def _parse_query_month_label(query: str) -> tuple[str, int] | None:
    m = _MONTH_IN_QUERY.search(query.strip())
    if not m:
        return None
    year = m.group(1)
    mo_token = (m.group(2) or m.group(3) or "").strip()
    if mo_token.isdigit():
        mo = int(mo_token)
    else:
        mo = _CN_MONTH.get(mo_token, 0)
    if 1 <= mo <= 12:
        return year, mo
    return None


def _extract_month_section(text: str, mk: str) -> str:
    """只取 ## YYYY-MM 标题下的正文，避免其它月份内容误判为命中。"""
    heading = f"## {mk}"
    idx = text.find(heading)
    if idx < 0:
        return ""
    rest = text[idx + len(heading):]
    nxt = re.search(r"\n## \d{4}-\d{2}\b", rest)
    if nxt:
        rest = rest[: nxt.start()]
    return rest.strip()


def memory_evidence_supports_query(user_msg: str, memory: dict) -> bool:
    """检索结果是否足以支撑回答（防止弱相关 chunk 诱发编造）。"""
    if is_identity_question(user_msg):
        if memory.get("l0"):
            return True
        return bool(memory.get("l2_hit") or memory.get("l3_hit"))

    l3_items = (memory.get("matches") or {}).get("l3") or []
    l2_items = [
        m for m in ((memory.get("matches") or {}).get("l2") or [])
        if m.get("score") is not None
    ]

    month = _parse_query_month_label(user_msg)
    if month:
        year, mo = month
        mk = f"{year}-{mo:02d}"
        sections: list[str] = []
        for m in l3_items:
            sec = _extract_month_section(str(m.get("text", "")), mk)
            if sec:
                sections.append(sec)
        if not sections:
            return False
        combined = "\n".join(sections)
        if re.search(r"发生了什么|什么事|怎么回事|怎么样|咋样|如何|还记得|有没有", user_msg):
            return True
        terms = set(re.findall(r"[\u4e00-\u9fff]{2,4}", user_msg.strip())) - _QUERY_STOP_TERMS
        terms -= {
            f"{year}年", f"{mo}月", "月份", "发生", "什么", "之间", "我们", "怎么", "回事",
            "那年", "当时", "那时候", "还记得", "记得",
        }
        terms = {t for t in terms if not re.fullmatch(r"[一二三四五六七八九十]+月", t)}
        if not terms:
            return True
        return any(t in combined for t in terms)

    blobs: list[str] = [str(m.get("text", "")) for m in l3_items]
    blobs.extend(str(m.get("text", "")) for m in l2_items)
    if not blobs:
        return False

    terms = set(re.findall(r"[\u4e00-\u9fff]{2,}", user_msg.strip())) - _QUERY_STOP_TERMS
    if not terms:
        return True
    for blob in blobs:
        if any(t in blob for t in terms):
            return True
    return False


def memory_requires_unknown_reply(user_msg: str, memory: dict) -> bool:
    """是否应直接回复「不知道/请补充」，禁止走创意生成。"""
    if memory.get("guest_mode"):
        return False
    if not query_needs_memory_answer(user_msg):
        return False
    if memory.get("memory_miss"):
        return True
    return not memory_evidence_supports_query(user_msg, memory)


def build_unknown_reply(
    user_msg: str,
    *,
    interlocutor_mode: str = "girlfriend",
    person_profile: dict | None = None,
) -> str:
    """生成口语化「不记得/请补充」回复（模板，不调用主 LLM）。"""
    import random

    del person_profile
    msg = user_msg.strip()
    asked = extract_asked_person_name(msg)
    if asked:
        asked = asked.rstrip("吗嘛呢啊呀")
        if interlocutor_mode == "visitor":
            return f"关于{asked}的我真不太熟，记忆里没有，你跟我说说？"
        return f"关于{asked}的事儿我记忆库里真没有，你知道的话跟我说说？"

    month = _parse_query_month_label(msg)
    if month:
        year, mo = month
        if interlocutor_mode == "visitor":
            return f"{year}年{mo}月的事儿我这边没记录，你跟我讲讲？"
        return f"{year}年{mo}月那段我翻翻记忆也没有，你跟我回忆一下？"

    pool = _UNKNOWN_VISITOR if interlocutor_mode == "visitor" else _UNKNOWN_GF
    return random.choice(pool)


# ══════════════════════════════════════════════════════════════════════════════
# 第二道防线：反幻觉与事实捕获 —— 防止 Agent 编造记忆中没有的信息
# ══════════════════════════════════════════════════════════════════════════════

# ── 反幻觉核心正则表达式 ──────────────────────────────────────────────────

# 用户纠错信号："打错了/说错了/搞错了"
_CORRECTION = re.compile(r"输错了|打错了|说错了|搞错了|打错字|写错了|刚才.*错")

# 从"我是XXX"中提取自称名
_SELF_NAME = re.compile(
    r"(?:^|[，。！？\s])我是([一-鿿·]{2,10})(?=[，。！？\s吗嘛呢啊]|$)"
)

# 询问第三方："你认识XXX吗/XXX是谁"
_ASK_ABOUT_PERSON = re.compile(
    r"(?:你)?认识([一-鿿·]{2,10})[吗嘛]?|"
    r"知道([一-鿿·]{2,10})[吗嘛]?|"
    r"([一-鿿·]{2,10})是谁|"
    r"认(?:识|得)([一-鿿·]{2,10})"
)

# 无效自称名模式：语气词、指代词等
_INVALID_NAME_RE = re.compile(
    r"^(谁呀|谁啊|谁嘛|什么|啥|哪位|哪个|你吗|你呢|嗯|哦|啊|呀|吗|呢|的|了|我|你|他|她)$"
)

# 用户改名/澄清："我不是XXX，其实是YYY"
_RENAME = re.compile(r"(?:不是|非).{0,12}?是([一-鿿·]{2,10})|其实是([一-鿿·]{2,10})")

# 提取中文姓名 token（2-4 个汉字）
_NAME_TOKEN = re.compile(r"[一-鿿]{2,4}")

# 自称噪声模板：LLM 生成的"本轮用户自称"等套话
_SELF_INTRO_FACT = re.compile(r"对话中用户自称|用户自称「|本轮用户自称")

# 已确认人物标记：L3 语料中出现了这些词才算"认识"
_KNOWN_PERSON_MARKERS = re.compile(
    r"女朋友|女友|老婆|男朋友|男友|是叶鹏祥的|叶鹏祥的|认识|在一起|同学|同事"
)


# ── 反幻觉铁律（注入 system prompt 的最高优先级规则）────────────────────

ANTI_HALLUCINATION_RULES = """## 事实铁律（最高优先级，覆盖口吻与调侃）

**只允许**使用以下来源中的具体事实：L0 核心记忆（身份/喜恶/禁忌/纪念日，日常优先）、L2 近期摘要、L3 长期记忆、已入库事实；人物履历归档仅在深度谈心时启用。
- **角色铁律**：消息中 [对方] 开头的是用户说的，[你] 开头的是你（叶鹏祥）说过的话。**绝对禁止**把 [你] 开头的消息当成用户消息来回复——那是你自己说的话，不是用户说的。只回应最新一条 [对方] 消息。
- L0 禁忌类条目：**生成回复前必须先检查**，任何情况下不得触碰。
- **禁止**把用户**本条刚说的**内容当成「早就知道」：仅本轮首次出现的自称/经历/关系，不能回复成老熟人（禁止：认识你、好久不见、我记得你、你终于来了、咱上次…）。
- 记忆库**没有**的人名、关系、经历、地点、约定、喜好 → 必须口语表示**不清楚或没印象**，并可口语追问一句补记忆；**禁止编造**。
- 用户问具体事件/日期/人物而 L2/L3 **未给出可直接回答的条目** → 只能说不记得/没查到/你跟我讲讲，**禁止**用 persona 口吻「圆故事」或凭常识瞎编。
- **禁止**根据姓氏相近、谐音、名字像就推断关系（例：刘远航≠刘远慧的姐姐）。
- 用户纠正打错字/改名：只认**最新说法**；勿补全记忆未写明的「你姐」「你们家」等。
- L3 语料里出现的人名**不等于**当前发消息的人就是那人，除非记忆明确对应。
- 不确定时宁可短句追问，也不要「帮你圆故事」。"""


def is_valid_person_name(name: str) -> bool:
    """校验是否为有效的人名/自称名。

    过滤规则：
      - 长度 2~10 字符
      - 不是纯语气词（吗/嘛/呢/啊/呀/哦/嗯/哈）
      - 不是 Agent 本人（叶鹏祥）
      - 不是无意义的指代词（谁呀/什么/哪位）

    Args:
        name: 待校验的名字字符串

    Returns:
        True 表示是有效名字。
    """
    n = re.sub(r"\s+", "", (name or "").strip())
    if len(n) < 2 or len(n) > 10:
        return False
    if _INVALID_NAME_RE.match(n):
        return False
    if n in ("叶鹏祥", "叶鹏祥大侠", "谁呀", "谁啊", "什么", "哪位"):
        return False
    if re.fullmatch(r"[吗嘛呢啊呀哦嗯哈]+", n):
        return False
    return True


def is_asking_about_third_party(msg: str) -> bool:
    """判断用户是否在询问第三方人物（非自称）。

    区分"我是XXX"（自称）和"你认识XXX吗"（问第三方）。
    只有在没有自称信号且存在第三方问句模式时才返回 True。
    """
    text = msg.strip()
    if not text:
        return False
    if extract_self_name(text):  # 先检查是否自称
        return False
    return bool(_ASK_ABOUT_PERSON.search(text))


def extract_asked_person_name(msg: str) -> str | None:
    """从"你认识XXX吗"类问句中提取第三方人名。

    用于生成反幻觉提示时告知 LLM：用户在问 XXX，不是在自称 XXX。
    """
    m = _ASK_ABOUT_PERSON.search(msg.strip())
    if not m:
        return None
    for g in m.groups():
        if g and is_valid_person_name(g.strip()):
            return g.strip()
    return None


def extract_self_name(msg: str) -> str | None:
    """从"我是XXX"中提取用户自称名。

    这是最核心的身份识别函数之一，用于：
      - L3 检索 query 改写（用自称名替代代词做更精准的检索）
      - 反幻觉提示生成（判断此次自称是否已有记忆记录）
      - L0 即时捕获（capture_l0_from_user_message 调用）

    Args:
        msg: 用户原始消息

    Returns:
        提取到的名字字符串；无匹配或名字无效时返回 None。
    """
    text = msg.strip()
    m = _SELF_NAME.search(text)
    if not m:
        # 备选：以"我是"开头但正则未匹配的场景（如"我是刘远慧嘛"）
        if text.startswith("我是") and len(text) > 2:
            who = text[2:].strip("吗嘛呢啊呀 ")
            if is_valid_person_name(who):
                return who
        return None
    who = m.group(1).strip()
    if not is_valid_person_name(who):
        return None
    return who


def is_self_intro_only_fact(fact: str) -> bool:
    """判断是否为纯自称噪声——"本轮用户自称XXX"类模板文本。

    这类文本是 LLM 生成的套话格式，不能作为"认识此人"的依据。
    包含此类模板的 L3 记忆块不应被当作身份证据。
    """
    return bool(_SELF_INTRO_FACT.search(fact))


def is_noise_memory_for_l3(text: str) -> bool:
    """L3/Facts 噪声过滤：判断一段 L3 记忆文本是否为无意义的噪声。

    噪声类型包括：
      - 纯自称套话（"对话中用户自称XXX"）
      - 元指令文本（"仅用户说明为准/勿自动推断"）
      - 自称模板 + 短文本（< 80 字符）——内容太短不足以构成有意义的记忆

    这些噪声块不应参与 L3 召回结果。

    Args:
        text: L3 记忆文本

    Returns:
        True 表示是噪声，应被过滤掉。
    """
    t = (text or "").strip()
    if not t:
        return True
    if is_self_intro_only_fact(t):
        return True
    if "仅用户说明为准" in t or "勿自动推断" in t:
        return True
    if re.search(r"用户自称[「\"]", t) and len(t) < 80:
        return True
    return False


def prune_self_intro_noise(device_id: str) -> int:
    """清理 L3 存储中的自称噪声块（维护用）。

    扫描并删除所有包含"本轮用户自称/对话中用户自称"模板的 chunk。
    """
    n = 0
    for sub in ("本轮用户自称", "对话中用户自称"):
        for ref in store.l3_find_chunks_by_text(sub, device_id=device_id, limit=20):
            if store.l3_delete_chunk(ref["chunk_id"]):
                n += 1
    return n


def _name_in_text_blob(name: str, blob: str) -> bool:
    """子串匹配：名字是否出现在文本中。

    短名（2字）做精确匹配，长名（3+字）前两字出现即可视为相关。
    """
    if not name or len(name) < 2:
        return False
    if name in blob:
        return True
    if len(name) >= 3 and name[:2] in blob:
        return True
    return False


def name_in_memory_text(name: str, text: str) -> bool:
    """记忆文本中是否出现该人名。"""
    return _name_in_text_blob(name, text.strip())


def _profile_has_substance(profile: dict) -> bool:
    """判断画像是否有实质内容（非空 draft/provisional）。

    考虑多种老版数据结构：extend_custom.relationship_to_me、
    version 号、profile_relationship 等。
    """
    if profile.get("provisional"):
        return False
    if profile.get("known_in_memory") is False:
        return False
    from app.memory.profile import has_profile_archive_content, profile_relationship

    ext = profile.get("extend_custom") or {}
    if ext.get("relationship_to_me"):
        return True
    if has_profile_archive_content(profile):
        return True
    if str(profile.get("version") or "1") not in ("", "1", "draft"):
        return True
    rel = profile_relationship(profile)
    if rel:
        return True
    return False


def name_known_in_memory(
    device_id: str,
    name: str,
    memory: dict,
    *,
    person_profile: dict | None = None,
    person_id: str | None = None,
) -> bool:
    """检查记忆/画像中是否已有此人（不仅仅是本轮自称）。

    这是反幻觉的核心判断——决定了 LLM 是该"认识"此人还是"不认识"。

    检查层次：
      1. 画像匹配：person_profile 有实质内容且名字匹配 → 认识
      2. 记忆文本匹配：L2 摘要或 L3 语料中包含名字 + 关系类关键词 → 认识
      3. L3 数据库直接查询：按 person_id 查 L3，逐一排除噪声 → 有则认识

    Args:
        device_id:      设备标识
        name:           待检查的名字
        memory:         本轮召回的记忆数据（含 episodic 和 L3 文本）
        person_profile: 当前用户的画像（可选）
        person_id:      当前活跃的用户 ID（可选）

    Returns:
        True 表示记忆库中已有此人的实质信息。
    """
    name = name.strip()
    if not name:
        return False

    # 第一关：画像匹配（最可靠）
    if person_profile and _profile_has_substance(person_profile):
        from app.memory.profile import profile_matches_name, normalize_profile

        if profile_matches_name(normalize_profile(person_profile), name):
            return True

    # 第二关：记忆召回文本中找名字
    # 收集所有 L2/L3 文本片段
    blob_parts: list[str] = []
    blob_parts.extend(memory.get("episodic") or [])
    blob_parts.extend(memory_l3_texts(memory))
    for layer in (memory.get("matches") or {}).values():
        if isinstance(layer, list):
            for item in layer:
                if isinstance(item, dict):
                    blob_parts.append(str(item.get("text", "")))
                else:
                    blob_parts.append(str(item))
    blob = "\n".join(blob_parts)

    if _name_in_text_blob(name, blob):
        for part in blob_parts:
            if not _name_in_text_blob(name, part):
                continue
            # 排除自称噪声
            if is_self_intro_only_fact(part):
                continue
            # 必须出现关系标记或名字长度 ≥ 3（2字匹配过于宽松，需要关系标记佐证）
            if _KNOWN_PERSON_MARKERS.search(part) or "关系" in part or "在一起" in part:
                return True
            if len(name) >= 3:
                return True

    # 第三关：L3 数据库精确查询（更可靠的来源验证）
    pid = str(person_id or (person_profile or {}).get("person_id") or "").strip()
    if pid:
        for row in store.l3_list_person_memory(pid, device_id=device_id, limit=40):
            fact = str(row.get("text", ""))
            # 跳过噪声块
            if is_self_intro_only_fact(fact) or is_noise_memory_for_l3(fact):
                continue
            if _name_in_text_blob(name, fact) and (
                _KNOWN_PERSON_MARKERS.search(fact) or "关系" in fact or "是其" in fact
            ):
                return True

    return False


def user_message_hints(
    user_msg: str,
    *,
    memory: dict | None = None,
    person_profile: dict | None = None,
    device_id: str = "",
) -> str:
    """根据本轮用户消息生成反幻觉提示块，用于注入 system prompt。

    这是反幻觉的核心函数——分析用户消息的各种信号，生成针对性的提示文本，
    告诉 LLM 在生成回复时要遵守哪些事实约束。

    覆盖场景（按检测顺序）：
      1. 用户纠错（打错了/说错了） → 提示只接受纠正后信息
      2. 纠正长期记忆（记错了/没这回事） → 提示勿重复旧记忆
      3. 身份追问（我是谁/你记得我吗） → 提示优先 L0/L2，勿罗列矛盾名字
      4. 询问第三方（你认识XXX吗） → 提示"不是用户在自称XXX"
      5. 自称且记忆有记录 → 提示可正常对话但勿编造
      6. 自称但记忆无记录 → 提示"禁止假装认识"，触发访客对待
      7. 改名/澄清 → 提示以澄清为准
      8. 记忆没命中 → 提示坦诚不知道

    Args:
        user_msg:       用户原始消息
        memory:         本轮召回的记忆数据
        person_profile: 当前用户的画像（可选）
        device_id:      设备标识

    Returns:
        提示文本块（多行）；无特殊场景时返回空字符串。
    """
    msg = user_msg.strip()
    if not msg:
        return ""
    lines: list[str] = []
    mem = memory or {}

    # 场景 1：用户纠错信号（打错字/说错了）
    if _CORRECTION.search(msg):
        lines.append(
            "【本条】用户在纠正：只接受纠正后信息；禁止推断记忆未写的亲属/经历；"
            "禁止假装一直记得纠正前内容。"
        )

    # 场景 2：纠正长期记忆（记错了/没这回事）
    if re.search(
        r"记错了|记错|没这回事|不是这样的|长期记忆|纠正.*记忆|你记错|其实不是",
        msg,
    ):
        lines.append(
            "【本条】用户在纠正长期记忆：只认用户本条；勿重复已被否定的旧记忆。"
        )

    # 场景 3：身份追问（我是谁/你记得我吗）
    # 特殊性：不能用 L3 中的自称噪声（可能是不同人的名字），应优先 L0 + L2
    if is_identity_question(msg):
        lines.append(
            "【本条】用户在问「我是谁/你记得我吗」：优先依据 **L0 核心记忆**、"
            "**L2 近7天会话摘要**、已确认事实；"
            "无则坦诚不记得并请对方说明；禁止罗列互相矛盾的多个自称名。"
        )

    # 场景 4：询问第三方人物
    # 例如："你认识刘远航吗"——用户是在问刘远航，不是在自称刘远航
    if is_asking_about_third_party(msg):
        asked = extract_asked_person_name(msg)
        if asked:
            corpus_known = any(
                name_in_memory_text(asked, str(s)) for s in memory_l3_texts(mem)
            )
            if corpus_known:
                lines.append(
                    f"【本条】用户在问「认不认识 {asked}」：记忆库中有关于 {asked} 的内容，"
                    f"按记忆回答；**不是**用户在自称 {asked}。"
                )
            else:
                lines.append(
                    f"【本条】用户在问「认不认识 {asked}」：按记忆库回答对该人的了解；"
                    f"**不是**用户在自称 {asked}；禁止回复「你谁」混淆成对说话人追问。"
                    f"若无记忆则说对 {asked} 没印象/不太熟，勿编造。"
                )

    # 场景 5/6：自称名检查（最重要）
    who = extract_self_name(msg)
    if who:
        known = name_known_in_memory(device_id, who, mem, person_profile=person_profile)
        if not known:
            # 二次确认：直接搜索 L3 文本中是否有该名字
            for s in memory_l3_texts(mem):
                if name_in_memory_text(who, str(s)):
                    known = True
                    break
        if known:
            # 场景 5：记忆中有记录 → 可以自然对话
            lines.append(
                f"【本条】用户自称「{who}」：检索到与此人相关的记忆，可自然对话但勿编造记忆未写明的细节。"
            )
        else:
            # 场景 6：记忆中没有记录 → 执行访客模式对待
            # 这是反幻觉最关键的场景：禁止假装认识
            lines.append(
                f"【本条】用户自称「{who}」，但 L2/L3/已入库事实/画像中**此前均无此人**。"
                f"**禁止**假装认识（好久不见、记得你、老熟人、咱上次…）；"
                f"可口语表示没印象，让对方自我介绍一下；"
                f"勿编造任何与 {who} 相关的经历、关系、约定。"
            )

    # 场景 7：改名/澄清
    if _RENAME.search(msg):
        lines.append(
            "【本条】用户在澄清「不是…而是…」：以澄清为准；勿编造澄清未涉及的关系与经历。"
        )

    # 场景 8：身份相关的疑问词
    if re.search(r"你谁|你是谁|我是谁|叫什么", msg):
        lines.append(
            "【本条】身份相关：只许用记忆库信息；没有就说不太记得或请对方说明。"
        )

    # 场景 9：记忆未命中但用户在追问
    if re.search(r"记得吗|还记得|有没有印象|认识我吗|你知道我", msg):
        if not memory_l3_hit(mem) and not mem.get("l2_hit"):
            lines.append(
                "【本条】用户在问你是否记得某事/某人，但本轮记忆检索偏弱："
                "须承认不太记得或请对方补充，禁止编造共同经历。"
            )

    if mem.get("memory_miss") and query_needs_memory_answer(msg):
        lines.append(
            "【本条 · 硬性约束】记忆库未检索到与用户问题相关的内容："
            "只能说不记得/没印象并请对方补充，禁止编造任何细节。"
        )

    return "\n".join(lines)


def format_stored_facts(
    device_id: str, person_id: str | None = None, limit: int = 12
) -> str:
    """格式化已入库 L3 记忆为 prompt 列表（过滤噪声与纯自称）。

    用于 system prompt 或调试信息中展示该用户已掌握的长期记忆。

    Args:
        device_id: 设备标识
        person_id: 用户 ID（None 或未实名返回提示文本）
        limit:     最多展示条数

    Returns:
        Markdown 列表格式的记忆清单文本。
    """
    pid = str(person_id or "").strip()
    if not pid:
        return "（尚未绑定对话人，无已入库记忆；勿编造）"
    rows = store.l3_list_person_memory(pid, device_id=device_id, limit=limit)
    lines: list[str] = []
    for r in rows:
        fact = str(r.get("text", ""))
        if is_noise_memory_for_l3(fact):
            continue
        # 区分纯自称噪声和有效记忆：前者标注出来提醒 LLM
        if is_self_intro_only_fact(fact):
            lines.append(f"- {fact}（**仅本轮自称，勿当作早已认识的老朋友**）")
        else:
            lines.append(f"- {fact}")
    if not lines:
        return "（暂无已确认长期记忆；勿编造）"
    return "\n".join(lines)


def capture_user_stated_facts(
    device_id: str, person_id: str, session_id: str, user_msg: str
) -> list[str]:
    """用户纠错/澄清语料 → 直接写入 L3 长期记忆。

    当用户说"其实我叫XXX"、"搞错了，不是我"等纠错性表述时，
    将完整的用户原话加上结构化的元信息写入 L3，作为修正记录。

    如果用户画像是 draft/provisional 状态，写入后尝试触发画像转正。

    Args:
        device_id:  设备标识
        person_id:  用户 ID
        session_id: 会话标识
        user_msg:   用户原始消息

    Returns:
        已写入 L3 的语料文本列表。
    """
    from app.memory.l3 import ingest_l3_text

    msg = user_msg.strip()
    if not msg or not str(person_id or "").strip():
        return []

    # 构建结构化的语料块：原话 + 涉及的姓名 + 关系说明
    corpus_parts: list[str] = [f"【用户原话】{msg}"]

    if _CORRECTION.search(msg):
        names = _NAME_TOKEN.findall(msg)
        names = [n for n in names if n not in ("刚才", "输错", "打错", "说错", "搞错", "不是")]
        if names:
            corpus_parts.append(f"涉及姓名：{'、'.join(dict.fromkeys(names[:4]))}")

    m2 = _RENAME.search(msg)
    if m2:
        who = (m2.group(1) or m2.group(2) or "").strip()
        if who:
            corpus_parts.append(f"用户澄清身份/姓名为「{who}」")

    # 检测关系说明："XXX是我女朋友"等
    m3 = re.search(r"([一-鿿·]{2,8})\s*是我\s*(女朋友|女友|老婆)", msg)
    if m3:
        corpus_parts.append(f"用户说明：{m3.group(1)}是其{m3.group(2)}")

    # 只有确实包含纠错/澄清/关系信息时才写入
    if len(corpus_parts) <= 1 and not (
        _CORRECTION.search(msg) or _RENAME.search(msg) or m3
    ):
        return []

    corpus = "\n".join(corpus_parts)
    cid = ingest_l3_text(
        device_id,
        person_id,
        corpus,
        source="user_stated",
        source_session=session_id,
        category="correction",
    )

    # 纠错写入后，尝试推动临时画像转正
    if cid:
        raw = store.get_person_profile(person_id)
        if raw and raw.get("provisional"):
            from app.memory.profile import try_promote_provisional_profile

            try_promote_provisional_profile(device_id, person_id, raw)
    return [corpus] if cid else []


def girlfriend_tone_active(
    user_message: str,
    memory: dict,
    person_profile: dict | None = None,
    *,
    device_id: str = "",
) -> bool:
    """判断是否启用"女友"口吻——即刘远慧/刘大炮的调侃亲密风格。

    业务语义：当系统检测到当前对话对象确认为女朋友身份时，
    启用更亲密、调侃的语气（而非通用礼貌风格）。

    判断依据（三层确认）：
      1. 画像关系为"女朋友/女友/老婆"且画像有实质内容 → 直接启用
      2. 用户消息中提到刘远慧/刘大炮等名字 + 记忆中有此人 → 启用
      3. L3 语义记忆中出现刘远慧/刘大炮 + 关系标记词 → 启用

    Args:
        user_message:   用户消息文本
        memory:         本轮召回的记忆数据
        person_profile: 当前用户画像
        device_id:      设备标识

    Returns:
        True 表示应启用女友口吻。
    """
    # 第一层：画像关系确认（最可靠）
    if person_profile and not person_profile.get("provisional"):
        from app.memory.profile import profile_relationship

        rel = profile_relationship(person_profile)
        if rel in ("女朋友", "女友", "老婆") and _profile_has_substance(person_profile):
            return True

    # 第二层：消息中提到特定名字 + 记忆库有实质记录
    for name in ("刘远慧", "刘大炮", "秋雨", "远慧"):
        if name in user_message and name_known_in_memory(
            device_id, name, memory, person_profile=person_profile
        ):
            return True

    # 第三层：L3 语义记忆中有相关人名 + 关系标记
    blob = " ".join(memory.get("semantic", []))
    if re.search(r"刘远慧|刘大炮", blob) and _KNOWN_PERSON_MARKERS.search(blob):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 主动话题校验
# ══════════════════════════════════════════════════════════════════════════════

# 负面情绪信号词 —— 用户表达难过/生气/疲惫等情绪时不应追问
_NEGATIVE_EMOTION = re.compile(
    r"难过|伤心|好累|疲惫|崩溃|气死|烦躁|焦虑|抑郁|想哭|"
    r"别烦|不想说话|别吵|好烦|累死|心累|不想聊|别问了|"
    r"不开心|难受|委屈|生气|发火|受不了"
)

# 用户提问信号 —— 用户正在提问时不应再追问
_USER_ASKING = re.compile(r"[?？吗呢啥嘛么]$")

# 用户拒聊信号
_USER_REJECT = re.compile(r"别烦我|不想聊|不想说话|别吵|累了|睡了|先不说了|回头再说|拜拜|再见|晚安")

# 用户单字/极短回复 —— 需要主动话题挽救对话
_SINGLE_WORD = re.compile(r"^[嗯哦好行对是]+[.。!！…]*$")

# 禁止的话题开头模式（自问自答特征）
_FORBIDDEN_TOPIC_STARTS = (
    "是的", "对呀", "没错", "我也觉得", "对啊", "对的", "是呀",
    "我也", "确实", "就是", "嗯嗯", "哈哈",
)

# 角色混淆特征 —— 主动话题中不应出现
_ROLE_CONFUSION_PATTERNS = re.compile(
    r"(你呢|你咋样|你那边|你怎么样|你想我没|你在干嘛|你最近|你还好吗)"
)


def is_user_negative_emotion(msg: str) -> bool:
    """检测用户是否在表达负面情绪，此类消息不应生成主动话题。"""
    return bool(_NEGATIVE_EMOTION.search(msg.strip()))


def is_user_asking(msg: str) -> bool:
    """检测用户是否正在提问，提问时不应再追问。"""
    return bool(_USER_ASKING.search(msg.strip()))


def is_user_rejecting(msg: str) -> bool:
    """检测用户是否表达不想继续聊天的意图。"""
    return bool(_USER_REJECT.search(msg.strip()))


def is_single_word_reply(msg: str) -> bool:
    """检测用户是否为单字/极简回复，此类场景应主动找话题。"""
    q = msg.strip()
    return len(q) <= 3 and bool(_SINGLE_WORD.match(q))


def should_suppress_active_topic(user_msg: str) -> bool:
    """综合判断是否应该禁止生成主动话题。

    以下场景禁止：
      - 用户表达负面情绪
      - 用户正在提问
      - 用户表示不想聊天
    """
    q = user_msg.strip()
    if not q:
        return True
    if is_user_negative_emotion(q):
        return True
    if is_user_asking(q):
        return True
    if is_user_rejecting(q):
        return True
    return False


def should_force_active_topic(user_msg: str, main_reply: str) -> bool:
    """综合判断是否应该强制生成主动话题。

    以下场景强制生成：
      - 用户连续单字回复
      - 主回复很短（≤5字），对话可能冷场
    """
    if is_single_word_reply(user_msg.strip()):
        return True
    if len(main_reply.strip()) <= 5:
        return True
    return False


def validate_active_topic(topic: str) -> str | None:
    """校验并清洗主动话题，不合法返回 None。

    三道纯规则防线：
      1. 必须包含"你"且以问号结尾
      2. 禁止以自问自答模式开头
      3. 长度不超过 20 字
      4. 不含角色混淆特征（你呢/你咋样等）
    """
    t = topic.strip()
    if not t:
        return None
    # 长度限制
    if len(t) > 20:
        # 尝试截断到最后一个问号
        last_q = max(t.rfind("?"), t.rfind("？"))
        if last_q > 0:
            t = t[:last_q + 1]
        if len(t) > 20:
            return None
    # 必须包含"你"
    if "你" not in t:
        return None
    # 必须以问号结尾
    if not t.endswith("?") and not t.endswith("？"):
        return None
    # 禁止自问自答开头
    for prefix in _FORBIDDEN_TOPIC_STARTS:
        if t.startswith(prefix):
            return None
    # 禁止角色混淆特征
    if _ROLE_CONFUSION_PATTERNS.search(t):
        return None
    return t
