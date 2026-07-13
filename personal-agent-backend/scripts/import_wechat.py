#!/usr/bin/env python3
"""微信聊天记录导入工具：TXT 导出 → persona/import/drafts/ 草稿 → corpus/ 语料。

用途：将微信导出的 .txt 聊天记录（单聊或群聊）转换为陪伴机器人可识别的语料格式，
包括口吻范例草稿和按月汇总的长期记忆摘要，最终写入 persona/corpus/ 目录供 长期记忆 入库。

数据流：
  微信 TXT → 解析消息行 → 合并为对话块 → 提取 Q→A 对（口吻草稿）
                                          → 按月汇总 → LLM 总结 → corpus/（长期记忆）

输出文件（单聊）：
  - persona/corpus/wechat_memory.md         # 日常记忆（按月汇总）
  - persona/corpus/intimate.md              # 亲密记忆（女友私聊，需 --private）
  - persona/import/drafts/style_daily.md    # 日常口吻草稿
  - persona/import/drafts/style_intimate.md # 亲密口吻草稿（需 --private）

输出文件（群聊，需 --group）：
  - persona/corpus/wechat_group_friends.md       # 好友群记忆
  - persona/import/drafts/style_group_friends.md # 群聊口吻草稿

典型用法：
    # 单聊导入（日常模式）
    python scripts/import_wechat.py --input persona/import/raw/wechat/clean_chat_memory.txt

    # 单聊导入（含亲密模式）
    python scripts/import_wechat.py --input xxx.txt --private

    # 群聊导入
    python scripts/import_wechat.py --group

    # 不调用 LLM（仅用关键词 fallback 生成摘要）
    python scripts/import_wechat.py --input xxx.txt --no-llm

前置条件：
    - 微信聊天记录需先导出为 TXT（通过第三方工具或手工整理）
    - TXT 格式：每行一条消息，格式为 "YYYY-MM-DD HH:MM '昵称': 消息文本"
    - 群聊格式略有不同，见 parse_group_file 函数说明
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import chat_completion

# ---------- 路径常量 ----------

PERSONA_DIR = Path(__file__).resolve().parents[2] / "persona"
WECHAT_RAW_DIR = PERSONA_DIR / "import" / "raw" / "wechat"
DEFAULT_INPUT = WECHAT_RAW_DIR / "clean_chat_memory.txt"
DRAFT_DIR = PERSONA_DIR / "import" / "drafts"
CORPUS_DIR = PERSONA_DIR / "corpus"

# 输出文件路径
MEMORY_DAILY = CORPUS_DIR / "wechat_memory.md"       # 日常长期记忆
MEMORY_INTIMATE = CORPUS_DIR / "intimate.md"          # 亲密长期记忆
MEMORY_GROUP = CORPUS_DIR / "wechat_group_friends.md" # 群聊长期记忆

# ---------- 群聊相关常量 ----------

# 叶鹏祥在群里的微信昵称
GROUP_SELF_NICK = "一捧雪"
# 群聊好友：微信昵称 → 真实姓名
GROUP_FRIENDS: dict[str, str] = {
    "计算机原理": "伍钰涛",
    "唐凯": "唐凯",
    "袁子翔": "袁子翔",
}
# 群聊中需要跳过的无关昵称（表情包账号等）
GROUP_SKIP_NICKS = frozenset({"🐔🐔💦💦💧💧"})

# ---------- 正则表达式预编译 ----------

# 单聊消息行解析：时间 + 昵称 + 消息正文
LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+'([^']+)':\s*(.*)$")
# 群聊消息头解析：时间 + 昵称（消息正文在后续行中）
GROUP_HEAD_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'([^']+)'\s*$")
# 微信多媒体占位符（需过滤的无意义消息）
JUNK_TEXT = re.compile(
    r"^\[(表情包|图片|语音|视频|文件|其他消息|位置|名片|链接|红包|转账|撤回)"
)

# ---------- 文本特征常量 ----------

# 口语化特征词（用于评估回复质量，含这些词的回复通常更自然、更像本人说话）
COLLOQUIAL = ("你妈", "得了", "再说吧", "我去", "666", "笑死", "嗯嗯", "好吧", "逆天", "嘻嘻", "okok")
# 亲密关系特征词（用于判定一条消息是否属于亲密对话，影响分类和打分）
INTIMATE_HINTS = (
    "老公", "老婆", "宝贝", "宝宝", "亲亲", "抱抱", "想你", "害羞", "调皮", "爱爱", "乖", "爱你",
)
# 过于简短/无信息量的回复，在做 Q→A 对提取时跳过
SKIP_ANSWER = re.compile(r"^(嗯|哦|好|行|ok|OK|好哒|好的|收到|1|111)$", re.I)
# 默认身份信息
DEFAULT_SELF_REAL = "叶鹏祥"
DEFAULT_PARTNER_REAL = "刘远慧"
# 女友在微信中的多种昵称
PARTNER_NICKNAMES = ("刘大炮", "秋雨", "远慧")

# 重大事件 / 情绪关键词
# 当一条消息包含这些词时，它属于"值得记忆的高信号内容"，
# 不仅在月度摘要中会被优先保留，其附近的上下文（前后 4 条）也会被纳入
_SIGNIFICANCE_KEYWORDS = (
    "生日", "生气", "吵架", "吵", "分手", "和好", "冷战", "对不起", "抱歉", "原谅",
    "爱你", "想你", "宝贝", "老公", "老婆", "见面", "约会", "旅行", "出去", "回来",
    "住院", "生病", "考试", "实习", "工作", "面试", "搬家", "礼物", "红包",
    "纪念", "周年", "圣诞", "跨年", "过年", "情人节", "七夕", "求婚", "戒指",
    "怀孕", "结婚", "见家长", "介绍", "误会", "拉黑", "删了", "不理", "哄",
)


# ============================================================================
# 消息质量判断
# ============================================================================

def is_junk(text: str) -> bool:
    """判断一条消息是否为无价值的噪声内容。

    噪声包括：
    - 空消息或过短消息（< 2 字符）
    - 微信多媒体占位符（[图片]、[语音] 等）
    - 这些内容没有语义价值，不应参与语料构建

    Args:
        text: 消息文本

    Returns:
        True 表示是无价值内容
    """
    t = text.strip()
    return not t or len(t) < 2 or bool(JUNK_TEXT.match(t))


def is_intimate_text(text: str) -> bool:
    """判断消息是否包含亲密关系特征词。

    用于区分日常对话和亲密对话，在提取 Q→A 对和生成记忆时
    决定消息应该被归类到哪一类（日常/亲密）。

    Args:
        text: 消息文本

    Returns:
        True 表示含亲密特征词，应归入亲密分类
    """
    return any(h in text for h in INTIMATE_HINTS)


# ============================================================================
# 文件查找
# ============================================================================

def find_group_txt() -> Path | None:
    """在 raw/wechat/ 目录下查找群聊 TXT 文件。

    按文件名 glob "群聊*.txt" 搜索，取第一个匹配项。

    Returns:
        找到的群聊文件路径；无匹配时返回 None
    """
    files = sorted(WECHAT_RAW_DIR.glob("群聊*.txt"))
    return files[0] if files else None


# ============================================================================
# 消息解析
# ============================================================================

def parse_file(path: Path, partner: str, self_name: str) -> list[dict]:
    """解析单聊微信 TXT 文件为结构化消息列表。

    支持的格式（每行一条消息）：
        YYYY-MM-DD HH:MM '昵称': 消息文本

    解析后的每条消息包含：时间戳、发言人、文本、是否为自己、
    是否亲密内容等字段，供后续的合并、打分、分类使用。

    Args:
        path: 单聊 TXT 文件路径
        partner: 对话对象的微信昵称
        self_name: 自己的微信昵称

    Returns:
        结构化消息字典列表，按文件中的出现顺序排列
    """
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            who, text = m.group(2), m.group(3).strip()
            # 只保留自己和指定对话对象的消息
            if who not in (partner, self_name) or is_junk(text):
                continue
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
            except ValueError:
                ts = None
            rows.append({
                "ts": ts,
                "who": who,
                "text": text,
                "is_self": who == self_name,
                "intimate": is_intimate_text(text),
            })
    return rows


def parse_group_file(path: Path) -> list[dict]:
    """解析多行格式的群聊微信 TXT 文件。

    群聊 TXT 的特殊格式：
    - 每条消息由两行组成：第一行是"时间 昵称"（消息头），第二行是消息正文
    - 可能有多行正文（换行消息）

    示例：
        2024-01-15 20:30:00 '一捧雪'
        今天打游戏不？
        2024-01-15 20:31:00 '计算机原理'
        来啊 我上线

    解析逻辑：
    1. 遇到消息头时 flush 前一条消息
    2. 收集后续行作为正文，直到遇到下一个消息头
    3. 跳过非群友的无关消息（GROUP_SKIP_NICKS）

    Args:
        path: 群聊 TXT 文件路径

    Returns:
        结构化群聊消息字典列表
    """
    allowed = {GROUP_SELF_NICK, *GROUP_FRIENDS.keys()}
    rows: list[dict] = []
    who: str | None = None
    ts: datetime | None = None
    body: list[str] = []

    def flush() -> None:
        """将当前缓冲的 who/ts/body 组装为一条消息并存入列表。"""
        nonlocal who, ts, body
        if not who or who not in allowed:
            who, ts, body = None, None, []
            return
        text = "\n".join(body).strip()
        if is_junk(text):
            who, ts, body = None, None, []
            return
        rows.append({
            "ts": ts,
            "who": who,
            "text": text,
            "is_self": who == GROUP_SELF_NICK,
            "intimate": is_intimate_text(text),
        })
        who, ts, body = None, None, []

    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            m = GROUP_HEAD_RE.match(line.strip())
            if m:
                flush()
                nick = m.group(2).strip()
                # 跳过无关群成员的消息（表情包账号等）
                if nick in GROUP_SKIP_NICKS:
                    who, ts, body = None, None, []
                    continue
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts = None
                who = nick
                body = []
                continue
            if who and line.strip():
                body.append(line.strip())
        flush()
    return rows


# ============================================================================
# 消息合并为对话块
# ============================================================================

def merge_blocks(rows: list[dict]) -> list[dict]:
    """将同一发言人的连续消息合并为对话块。

    微信聊天中同一个人经常连续发多条消息（如"在吗""跟你说个事""今天..."），
    这些在语义上应视为一段完整表达。合并后便于提取更完整的 Q→A 对。

    合并策略：
    - 遍历排序后的消息列表，连续 same-who 的消息视为同一个 block
    - 每个 block 保留发言人、是否为自己、时间戳、亲密标记
    - block 的 text 字段是所有消息文本的空格连接

    Args:
        rows: 按时间排序的原始消息列表

    Returns:
        合并后的对话块列表，同一发言人的连续消息已合并
    """
    if not rows:
        return []
    blocks = [{"who": rows[0]["who"], "is_self": rows[0]["is_self"], "texts": [rows[0]["text"]],
               "ts": rows[0]["ts"], "intimate": rows[0].get("intimate", False)}]
    for r in rows[1:]:
        if r["who"] == blocks[-1]["who"]:
            # 同一发言人连续消息，追加到当前块
            blocks[-1]["texts"].append(r["text"])
            # 继承亲密标记：任一消息为亲密则块为亲密
            blocks[-1]["intimate"] = blocks[-1]["intimate"] or r.get("intimate", False)
        else:
            blocks.append({"who": r["who"], "is_self": r["is_self"], "texts": [r["text"]],
                           "ts": r["ts"], "intimate": r.get("intimate", False)})
    for b in blocks:
        b["text"] = " ".join(b["texts"])
    return blocks


# ============================================================================
# Q→A 对提取（口吻范例数据源）
# ============================================================================

def extract_qa_pairs(blocks: list[dict], *, max_answer_len: int = 120) -> list[dict]:
    """从对话块中提取（非我）→（我）的 Q→A 对，作为口吻范例。

    提取逻辑：
    - 寻找"对方发言 → 自己回复"的相邻块对
    - 过滤条件：问题/回答不为空，回答不是过于简短的敷衍（嗯、好、行等），
      回答长度在合理范围内（4～max_answer_len）
    - 保留问题原文的前 150 字符（防止超长截断）

    这些 Q→A 对构成了"我是怎么说话的"数据基础，最终用于构建 style/examples.md。

    Args:
        blocks: 已合并的对话块列表
        max_answer_len: 回答最大字符数，超出则跳过（长回答通常是叙述而非口吻范例）

    Returns:
        Q→A 对字典列表，每项含 q（问题）、a（回答）、ts（时间）、intimate（分类）
    """
    pairs = []
    for i, b in enumerate(blocks):
        # 仅提取"对方问 → 自己答"的对（b 是自己）
        if not b["is_self"] or i == 0 or blocks[i - 1]["is_self"]:
            continue
        q, a = blocks[i - 1]["text"], b["text"]
        if is_junk(q) or is_junk(a) or SKIP_ANSWER.match(a.strip()):
            continue
        if not (4 <= len(a) <= max_answer_len):
            continue
        intimate = blocks[i - 1].get("intimate") or b.get("intimate")
        pairs.append({"q": q[:150], "a": a, "ts": b.get("ts"), "intimate": intimate})
    return pairs


def score_pair(p: dict, *, private: bool) -> float:
    """对 Q→A 对进行质量评分，用于选出最具代表性的口吻范例。

    评分维度：
    - 长度分：8~55 字最佳（+2.0），80 字以内尚可（+1.0），过长扣分（-0.5）
    - 口语化特征词：每命中一个加 1.5 分（含"我去"、"666"等词更自然）
    - 亲密加成：在 private 模式下，含亲密词的对加 4.0 分

    Args:
        p: Q→A 对字典
        private: 是否为亲密隐私模式（影响亲密词加分逻辑）

    Returns:
        质量分数，越高表示越适合作为口吻范例
    """
    a = p["a"]
    s = 2.0 if 8 <= len(a) <= 55 else (1.0 if len(a) <= 80 else -0.5)
    s += sum(1.5 for w in COLLOQUIAL if w in a)
    if private and p.get("intimate"):
        s += 4.0
    return s


def dedupe_pairs(pairs: list[dict], limit: int, *, private: bool) -> list[dict]:
    """对 Q→A 对去重并选出 top-N。

    去重策略：以回答的前 40 个字符（去除空白后）作为去重键，
    保证不出现内容几乎相同的重复范例。

    Args:
        pairs: Q→A 对列表
        limit: 最多保留数量
        private: 是否亲密模式（传给 score_pair）

    Returns:
        去重后的 top-N Q→A 对列表
    """
    seen: set[str] = set()
    out = []
    # 从高分到低分排序
    for p in sorted(pairs, key=lambda x: score_pair(x, private=private), reverse=True):
        key = re.sub(r"\s+", "", p["a"])[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


# ============================================================================
# 口吻草稿输出
# ============================================================================

def write_style_draft(pairs: list[dict], path: Path, *, intimate: bool, limit: int, partner: str) -> None:
    """将筛选后的 Q→A 对写入口吻草稿文件。

    输出格式为 markdown 文件，包含标题和"问：.../答：..."格式的范例。
    草稿文件需要人工审核后手动合并到 style/examples.md 或 examples_intimate.md。

    Args:
        pairs: Q→A 对列表
        path: 输出文件路径
        intimate: 是否亲密模式（影响筛选和标题）
        limit: 最多写入数量
        partner: 对话对象描述（用于文件头标注）
    """
    pool = [p for p in pairs if bool(p.get("intimate")) == intimate]
    top = dedupe_pairs(pool, limit, private=intimate)
    target = "style/examples_intimate.md" if intimate else "style/examples.md"
    lines = [f"# {'亲密' if intimate else '日常'}口吻草稿 → 合并到 {target}", ""]
    for p in top:
        lines += [f"问：{p['q']}", f"答：{p['a']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# 月度汇总与长期记忆生成
# ============================================================================

def _month_key(ts: datetime | None) -> str:
    """将消息时间戳转换为月度分组键（YYYY-MM）。"""
    return ts.strftime("%Y-%m") if ts else "未标时间"


def group_rows_by_month(rows: list[dict]) -> dict[str, list[dict]]:
    """将消息列表按月份分组。

    Args:
        rows: 结构化消息列表

    Returns:
        月份键 → 该月消息列表 的字典
    """
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_month[_month_key(r.get("ts"))].append(r)
    return by_month


def _score_message(text: str) -> float:
    """对单条消息进行重要性评分。

    高信号内容（得分 >= 2.0）会被优先选入 LLM 摘要的上下文窗口：
    - 命中重大事件/情绪关键词：每个关键词 +2.0
    - 含亲密特征词：+1.5
    - 长度适中（40~180 字）：+0.5（太短无信息，太长可能是转发/系统消息）
    - 过短（< 4 字）：-1.0
    - 含系统模板残留："[你的姓名]"、"共计1500字" 等：-5.0（直接排除）

    Args:
        text: 消息文本

    Returns:
        重要性分数，越大表示越值得放入摘要上下文
    """
    t = text.strip()
    score = 0.0
    for kw in _SIGNIFICANCE_KEYWORDS:
        if kw in t:
            score += 2.0
    if is_intimate_text(t):
        score += 1.5
    if 40 <= len(t) <= 180:
        score += 0.5
    if len(t) < 4:
        score -= 1.0
    # 微信导出格式中的系统模板残留，直接排除
    if "[你的姓名]" in t or "共计1500字" in t:
        score -= 5.0
    return score


def _speaker_label(
    row: dict,
    *,
    self_wx: str,
    partner_wx: str,
    self_real: str,
    partner_real: str,
    group: bool,
) -> str:
    """生成消息的发言人标签字符串。

    格式示例：
    - 单聊（自己）："我(我·叶鹏祥)"
    - 单聊（对方）："打瓦女高手(刘远慧)"
    - 群聊（自己）："一捧雪(我·叶鹏祥)"
    - 群聊（好友）："伍钰涛(计算机原理)"

    Args:
        row: 消息字典
        self_wx: 自己微信昵称
        partner_wx: 对方微信昵称
        self_real: 自己真实姓名
        partner_real: 对方真实姓名
        group: 是否群聊

    Returns:
        发言人标签字符串
    """
    if group:
        nick = str(row.get("who", ""))
        if row.get("is_self"):
            return f"{GROUP_SELF_NICK}(我·{self_real})"
        real = GROUP_FRIENDS.get(nick, nick)
        return f"{real}({nick})"
    if row.get("is_self"):
        return f"{self_wx}(我·{self_real})"
    return f"{partner_wx}({partner_real})"


def build_month_transcript(
    month_rows: list[dict],
    *,
    self_wx: str,
    partner_wx: str,
    self_real: str = DEFAULT_SELF_REAL,
    partner_real: str = DEFAULT_PARTNER_REAL,
    max_chars: int = 14000,
    group: bool = False,
) -> str:
    """构建一个月的对话文字记录（transcript），用于发送给 LLM 总结。

    采样策略（在长长的聊天记录中选取最有代表性的部分）：
    1. 高信号采样：得分 >= 2.0 的消息，保留其前后各 4 条作为上下文窗口
    2. 均匀采样：每隔 N 条取 1 条（N = len/40），保证时间分布均匀
    3. 两种采样的结果去重合并，按时间排序
    4. 按 max_chars 总字符数截断（默认 14000，适配常见 LLM 上下文窗口）

    Args:
        month_rows: 该月的消息列表
        self_wx: 自己微信昵称
        partner_wx: 对方微信昵称
        self_real: 自己真实姓名
        partner_real: 对方真实姓名
        max_chars: 输出最大字符数
        group: 是否群聊

    Returns:
        该月的对话文字记录字符串，格式为 "时间 发言人: 消息"
    """
    usable = [r for r in month_rows if not is_junk(r["text"])]
    usable.sort(key=lambda r: r.get("ts") or datetime.min)
    if not usable:
        return ""

    # 对所有消息评分
    scored = [(i, _score_message(r["text"])) for i, r in enumerate(usable)]
    selected: set[int] = set()

    # 策略 1：高信号消息 + 上下文窗口（前后各 4 条）
    for i, s in scored:
        if s >= 2.0:
            for j in range(max(0, i - 4), min(len(usable), i + 5)):
                selected.add(j)

    # 策略 2：均匀采样，保证覆盖整个月份的时间范围
    step = max(1, len(usable) // 40)
    for i in range(0, len(usable), step):
        selected.add(i)

    # 构建最终输出文本
    parts: list[str] = []
    total = 0
    for i in sorted(selected):
        r = usable[i]
        who = _speaker_label(
            r,
            self_wx=self_wx,
            partner_wx=partner_wx,
            self_real=self_real,
            partner_real=partner_real,
            group=group,
        )
        ts_s = r["ts"].strftime("%m-%d %H:%M") if r.get("ts") else ""
        line = f"[{ts_s}] {who}: {r['text']}"
        # 超过字符上限时，如果已有足够行则停止，否则截断
        if total + len(line) + 1 > max_chars:
            if len(parts) < 24:
                parts.append(line[:240])
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts)


# ============================================================================
# 身份约定块（注入到 LLM prompt 中防止角色混淆）
# ============================================================================

def _single_chat_identity_block(
    *,
    self_wx: str,
    partner_wx: str,
    self_real: str,
    partner_real: str,
) -> str:
    """生成单聊身份约定文本，强制 LLM 遵守角色认知。

    这段文本会被注入到每个月的总结 prompt 中，防止 LLM：
    - 混淆"我"和对方的身份
    - 把不同的人名张冠李戴
    - 编造对话中没有的人物关系

    Args:
        self_wx: 自己微信昵称
        partner_wx: 对方微信昵称
        self_real: 自己真实姓名
        partner_real: 对方真实姓名

    Returns:
        身份约定的 markdown 格式文本
    """
    nicks = "、".join(PARTNER_NICKNAMES)
    return f"""- 叙述者「我」= {self_real}，微信里显示为「{self_wx}」
- 对话对象 = {partner_real}（女友），微信昵称「{partner_wx}」，也可称 {nicks}
- 禁止把 {partner_real} 与刘远航等其他姓名混淆成姐弟等关系，除非对话里明确这么说
- 禁止编造对话里没出现的人名、亲属关系、见面经历"""


def _group_identity_block() -> str:
    """生成群聊身份约定文本。"""
    friend_desc = "；".join(f"{real}(微信「{nick}」)" for nick, real in GROUP_FRIENDS.items())
    return f"""- 叙述者「我」= {DEFAULT_SELF_REAL}，群里微信昵称「{GROUP_SELF_NICK}」
- 群友：{friend_desc}，均为初中同学、高中同校
- 禁止混淆群友身份，不要把伍钰涛/唐凯/袁子翔的名字或事迹张冠李戴"""


# ============================================================================
# LLM 月度总结
# ============================================================================

def summarize_month_llm(
    month: str,
    transcript: str,
    *,
    mode: str,
    identity_block: str,
    partner_desc: str,
) -> str:
    """调用 LLM 将一个月的对话记录总结为第一人称回忆叙事。

    三种总结模式：
    - "daily"：日常交往记忆，重点写重大事件和情绪转折，省略亲密细节
    - "intimate"：亲密关系记忆，重点写甜言蜜语、调情、亲密矛盾和好
    - "group"：群聊记忆，重点写聚会、互怼、学校生活

    Args:
        month: 月份标签（如 2024-06）
        transcript: 该月的对话文字记录
        mode: 总结模式（daily/intimate/group）
        identity_block: 身份约定文本
        partner_desc: 对话来源描述

    Returns:
        LLM 生成的月度总结文本（中文、第一人称、150-350 字）
    """
    if not transcript.strip():
        return "这个月聊天记录太少，没有可总结的内容。"

    # 根据模式选择合适的写作侧重点
    if mode == "intimate":
        focus = (
            "重点写：甜言蜜语、亲密称呼、想对方、调情撒娇、浪漫时刻、"
            "因亲密话题产生的矛盾或和好。可点到重大节点，但不要写成日常流水账。"
            "省略纯生活琐事（吃饭通勤等）。"
        )
    elif mode == "group":
        focus = (
            "重点写：聚会、打游戏、互怼、学校生活、重要八卦、群里的情绪爆发或和好。"
            "日常灌水一笔带过。"
        )
    else:
        focus = (
            "重点写：重大事件（吵架和好、生日、见面约会、旅行、实习工作变动、重要决定）、"
            "情绪转折点、待跟进的事。日常闲聊可一笔带过。"
            "省略亲密私密细节（那些会单独存另一份记忆）。"
        )

    prompt = f"""根据以下微信对话，以叶鹏祥的第一人称「我」写 {month} 的回忆总结。

## 身份约定（必须严格遵守）
{identity_block}

## 写作要求
1. 只写对话里能佐证的内容，禁止编造
2. {focus}
3. 用 150～350 字叙述性中文，不要罗列原话、不要 markdown 列表
4. 提到人时用真实姓名，微信昵称可放括号里
5. 若该月确实只有闲聊，写一句说明即可

对话来源：{partner_desc}

对话记录：
{transcript}

只输出总结正文，不要标题、不要 JSON。"""

    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2).strip()
    # LLM 不可用时（API 未配置/返回异常），退回到关键词 fallback
    if not raw or "还没连上大模型" in raw or raw.startswith("DeepSeek"):
        return _fallback_month_summary(month, transcript, mode=mode)
    return raw


def _fallback_month_summary(month: str, transcript: str, *, mode: str) -> str:
    """无 LLM 时的简易 fallback：抽取含重要关键词的句子作为摘要。

    这在 LLM API 不可用时提供最低限度的可用摘要，便于检查和调试。
    """
    hits: list[str] = []
    for line in transcript.splitlines():
        if _score_message(line) >= 2.0:
            hits.append(line.split(": ", 1)[-1][:120])
        if len(hits) >= 6:
            break
    if not hits:
        return f"{month} 主要是日常微信闲聊，没有自动识别出的重大事件（建议配置 LLM_API_KEY 后重跑）。"
    label = {"intimate": "亲密", "group": "群里", "daily": "日常"}.get(mode, "")
    joined = "；".join(hits[:4])
    return f"{month} {label}记忆里印象较深的有：{joined}。"


# ============================================================================
# 写入 corpus 长期记忆文件
# ============================================================================

def write_memory_corpus(
    rows: list[dict],
    path: Path,
    *,
    mode: str,
    partner_desc: str,
    self_wx: str = "我",
    partner_wx: str = "",
    self_real: str = DEFAULT_SELF_REAL,
    partner_real: str = DEFAULT_PARTNER_REAL,
    preamble: list[str] | None = None,
    use_llm: bool = True,
    max_transcript_chars: int = 14000,
    group: bool = False,
) -> None:
    """将消息按月汇总为第一人称回忆叙事，写入 corpus/ 长期记忆文件。

    这是整个导入流程的核心输出步骤，生成的 .md 文件将被
    scripts/ingest.py 入库到 长期记忆 向量搜索引擎。

    处理流程：
    1. 将消息按月份分组
    2. 对每个月份：
       a. 从长聊天记录中采样代表性消息（build_month_transcript）
       b. 调用 LLM 生成第一人称月度总结（summarize_month_llm）
       c. 或使用关键词 fallback（_fallback_month_summary）
    3. 拼接所有月份总结 + 文件头，写入 markdown 文件

    Args:
        rows: 结构化消息列表
        path: 输出 .md 文件路径
        mode: 记忆类型（daily/intimate/group）
        partner_desc: 对话对象描述
        self_wx: 自己微信昵称
        partner_wx: 对方微信昵称
        self_real: 自己真实姓名
        partner_real: 对方真实姓名
        preamble: 额外的文件头说明文本
        use_llm: 是否调用 LLM 做总结（False 时用关键词 fallback）
        max_transcript_chars: 每月送入 LLM 的对话字符上限
        group: 是否群聊
    """
    by_month = group_rows_by_month(rows)
    title_map = {
        "daily": "微信日常记忆",
        "intimate": "微信亲密记忆",
        "group": "微信好友群记忆",
    }

    # 构建文件头（标题 + 前言 + 身份约定说明）
    lines = [f"# {title_map.get(mode, '微信记忆')}（长期记忆 长期记忆）", ""]
    if preamble:
        lines.extend(preamble)
        lines.append("")
    lines += [
        f"来源：{partner_desc}，由 import_wechat.py 按月总结生成。",
        f"叙述视角：{self_real} 第一人称；人名关系以对话与 persona 为准，禁止混淆。",
        "",
    ]

    identity = _group_identity_block() if group else _single_chat_identity_block(
        self_wx=self_wx,
        partner_wx=partner_wx or partner_desc,
        self_real=self_real,
        partner_real=partner_real,
    )

    # 逐月生成总结
    for month in sorted(by_month):
        transcript = build_month_transcript(
            by_month[month],
            self_wx=self_wx,
            partner_wx=partner_wx or partner_desc,
            self_real=self_real,
            partner_real=partner_real,
            max_chars=max_transcript_chars,
            group=group,
        )
        if not transcript.strip():
            continue
        print(f"  总结 {month} ({mode})…")
        if use_llm:
            summary = summarize_month_llm(
                month,
                transcript,
                mode=mode,
                identity_block=identity,
                partner_desc=partner_desc,
            )
        else:
            summary = _fallback_month_summary(month, transcript, mode=mode)
        lines += [f"## {month}", "", summary.strip(), ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ============================================================================
# 群聊导入入口
# ============================================================================

def import_group_chat(
    path: Path,
    *,
    use_llm: bool = True,
    max_transcript_chars: int = 14000,
) -> None:
    """群聊导入的完整流程入口。

    流程：
    1. 解析群聊 TXT → 消息列表
    2. 合并连续消息为对话块
    3. 提取 Q→A 对作为群聊口吻草稿
    4. 生成群聊月度长期记忆（按群友身份标注）

    Args:
        path: 群聊 TXT 文件路径
        use_llm: 是否调用 LLM 做月度总结
        max_transcript_chars: 每月送入 LLM 的对话字符上限
    """
    rows = parse_group_file(path)
    if not rows:
        sys.exit("未解析到群消息，请确认 TXT 格式与昵称（一捧雪/计算机原理/唐凯/袁子翔）")
    blocks = merge_blocks(rows)
    pairs = extract_qa_pairs(blocks, max_answer_len=120)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    # 构建群友描述文本，用于文件头说明
    friend_desc = "、".join(
        f"{real}（{nick}）" for nick, real in GROUP_FRIENDS.items()
    )
    preamble = [
        "叶鹏祥在群里微信昵称「一捧雪」。",
        f"群友：{friend_desc}，均为初中同学、高中同校。",
        "群里说话比跟女友更随意，互怼、打游戏、聊学校生活为主。",
    ]

    # 写入群聊口吻草稿
    write_style_draft(
        pairs,
        DRAFT_DIR / "style_group_friends.md",
        intimate=False,
        limit=100,
        partner="好友群",
    )

    # 生成群聊长期记忆
    write_memory_corpus(
        rows,
        MEMORY_GROUP,
        mode="group",
        partner_desc=f"好友群（{path.name}）",
        self_wx=GROUP_SELF_NICK,
        preamble=preamble,
        group=True,
        use_llm=use_llm,
        max_transcript_chars=max_transcript_chars,
    )

    # 统计各群友发言数量
    by_who = defaultdict(int)
    for r in rows:
        by_who[r["who"]] += 1
    print(f"group file: {path.name.encode('unicode_escape').decode()}")
    print(f"  messages {len(rows)}, qa pairs {len(pairs)}")
    for nick in [GROUP_SELF_NICK, *GROUP_FRIENDS]:
        print(f"  {nick}: {by_who.get(nick, 0)}")
    print("口吻草稿:", DRAFT_DIR / "style_group_friends.md")
    print("长期记忆:", MEMORY_GROUP)


# ============================================================================
# 主入口
# ============================================================================

def main() -> None:
    """命令行参数解析与流程分发。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="单聊 TXT 文件路径（默认读取 persona/import/raw/wechat/clean_chat_memory.txt）")
    ap.add_argument("--partner", default="打瓦女高手",
                    help="对话对象的微信昵称（默认 打瓦女高手）")
    ap.add_argument("--self-name", default="我",
                    help="自己的微信昵称（默认 我）")
    ap.add_argument("--private", action="store_true",
                    help="同时生成亲密记忆和口吻（含女友私聊内容）")
    ap.add_argument("--self-real-name", default=DEFAULT_SELF_REAL,
                    help=f"叙述者真实姓名（默认 {DEFAULT_SELF_REAL}）")
    ap.add_argument("--partner-real-name", default=DEFAULT_PARTNER_REAL,
                    help=f"单聊对象真实姓名（默认 {DEFAULT_PARTNER_REAL}）")
    ap.add_argument("--no-llm", action="store_true",
                    help="不调用大模型，仅用关键词 fallback 生成月度摘要")
    ap.add_argument("--max-transcript-chars", type=int, default=14000,
                    help="每月送入 LLM 的对话字符上限（默认 14000）")
    ap.add_argument("--group", action="store_true",
                    help="导入群聊 TXT（一捧雪=叶鹏祥）")
    args = ap.parse_args()
    use_llm = not args.no_llm

    # ---------- 群聊流程 ----------
    if args.group:
        path = args.input or find_group_txt()
        if not path or not path.is_file():
            sys.exit(f"找不到群聊 TXT，请放到 {WECHAT_RAW_DIR}/群聊*.txt")
        import_group_chat(
            path,
            use_llm=use_llm,
            max_transcript_chars=args.max_transcript_chars,
        )
        print("\n已写入 corpus/，执行: python scripts/ingest.py")
        return

    # ---------- 单聊流程 ----------
    path = args.input or DEFAULT_INPUT
    if not path.is_file():
        sys.exit(f"找不到: {path}")

    # 解析消息并提取 Q→A 对
    rows = parse_file(path, args.partner, args.self_name)
    pairs = extract_qa_pairs(merge_blocks(rows), max_answer_len=150 if args.private else 120)

    DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    # 写入日常口吻草稿
    write_style_draft(pairs, DRAFT_DIR / "style_daily.md", intimate=False, limit=100, partner=args.partner)
    print("生成日常长期记忆…")
    write_memory_corpus(
        rows,
        MEMORY_DAILY,
        mode="daily",
        partner_desc=f"与「{args.partner}」的单聊",
        self_wx=args.self_name,
        partner_wx=args.partner,
        self_real=args.self_real_name,
        partner_real=args.partner_real_name,
        use_llm=use_llm,
        max_transcript_chars=args.max_transcript_chars,
    )
    print("口吻草稿:", DRAFT_DIR / "style_daily.md")
    print("长期记忆:", MEMORY_DAILY)

    # 如果开启 --private，额外生成亲密记忆和口吻
    if args.private:
        write_style_draft(pairs, DRAFT_DIR / "style_intimate.md", intimate=True, limit=80, partner=args.partner)
        print("生成亲密长期记忆…")
        write_memory_corpus(
            rows,
            MEMORY_INTIMATE,
            mode="intimate",
            partner_desc=f"与「{args.partner}」的单聊",
            self_wx=args.self_name,
            partner_wx=args.partner,
            self_real=args.self_real_name,
            partner_real=args.partner_real_name,
            use_llm=use_llm,
            max_transcript_chars=args.max_transcript_chars,
        )
        print("亲密口吻草稿:", DRAFT_DIR / "style_intimate.md")
        print("亲密长期记忆:", MEMORY_INTIMATE)

    print("\n记忆已写入 corpus/，执行: python scripts/ingest.py")


if __name__ == "__main__":
    main()
