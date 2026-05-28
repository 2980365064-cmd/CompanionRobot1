"""Anti-hallucination: memory-only facts, user corrections → facts store."""

from __future__ import annotations

import re

from app.memory.semantic import semantic_memory
from app.session import store

# 用户当场纠正称呼/名字
_CORRECTION = re.compile(r"输错了|打错了|说错了|搞错了|打错字|写错了|刚才.*错")
# 我是XXX（自称，排除本人测试）
_SELF_NAME = re.compile(r"我是([\u4e00-\u9fff·]{2,10})")
# 不是A是B / 其实是B
_RENAME = re.compile(r"(?:不是|非).{0,12}?是([\u4e00-\u9fff·]{2,10})|其实是([\u4e00-\u9fff·]{2,10})")
# 提取句中 2～4 字中文名（纠正句）
_NAME_TOKEN = re.compile(r"[\u4e00-\u9fff]{2,4}")

ANTI_HALLUCINATION_RULES = """## 事实铁律（最高优先级，覆盖口吻与调侃）

**只允许**使用以下来源中的具体事实：Profile、L2 近期摘要、L3 长期记忆、下方「已入库事实」、用户**本条及上文亲口说的**内容。
- 记忆库**没有**的人名、关系、经历、亲属称呼 → **禁止编造**；用口语问一句，例如「不太记得了，你再说下」「XX 啥情况啊」
- **禁止**根据姓氏相近、谐音、名字像就**推断**关系（例：刘远航≠刘远慧的姐姐，除非记忆里明写「姐姐」且对应同一人）
- 用户纠正打错字/改名：只认**最新说法**，不要补全「你姐」「你们家」等记忆未写明的背景
- 禁止编造：亲戚关系、谁是谁的姐姐/同学、未提及的同住、未提及的见面经历
- 不确定时宁可短句追问，也不要「帮你圆故事」"""


def user_message_hints(user_msg: str) -> str:
    """Per-turn hints injected before LLM (same turn)."""
    msg = user_msg.strip()
    if not msg:
        return ""
    lines: list[str] = []
    if _CORRECTION.search(msg):
        lines.append(
            "【本条】用户在纠正名字/称呼：只接受纠正后的信息；"
            "禁止推断姐姐、亲戚、家人等记忆库未出现的关系；"
            "禁止用「你姐」「打错名字」这类记忆无依据的调侃。"
        )
    if _SELF_NAME.search(msg) and not re.search(r"我是叶鹏祥", msg):
        name = _SELF_NAME.search(msg).group(1).strip()
        lines.append(
            f"【本条】用户自称「{name}」：仅作当前对话称呼，勿与记忆中人名自动合并，勿推断亲属关系。"
        )
    if _RENAME.search(msg):
        lines.append(
            "【本条】用户在澄清「不是…而是…」：以澄清后的为准，勿编造澄清未涉及的关系。"
        )
    if re.search(r"你谁|你是谁|我是谁|叫什么", msg):
        lines.append(
            "【本条】身份相关：人物信息只来自记忆库；没有的就说不太记得或请对方说明。"
        )
    return "\n".join(lines)


def format_stored_facts(device_id: str, limit: int = 12) -> str:
    rows = store.list_facts(device_id, limit=limit)
    if not rows:
        return "（暂无；勿编造）"
    return "\n".join(f"- {r['fact']}" for r in rows)


def capture_user_stated_facts(device_id: str, session_id: str, user_msg: str) -> list[str]:
    """Persist high-confidence facts from user corrections / self-identification."""
    msg = user_msg.strip()
    if not msg:
        return []
    saved: list[str] = []

    def _add(fact: str, category: str = "person") -> None:
        semantic_memory.add_fact(device_id, fact, category, 0.92, session_id)
        saved.append(fact)

    if _CORRECTION.search(msg):
        names = _NAME_TOKEN.findall(msg)
        names = [n for n in names if n not in ("刚才", "输错", "打错", "说错", "搞错", "不是")]
        name_part = f"涉及：{'、'.join(dict.fromkeys(names[:4]))}。" if names else ""
        _add(
            f"用户已纠正称呼/名字错误；{name_part}"
            "以纠正后的说法为准；禁止根据相似姓名推断姐姐等亲属关系。",
            "correction",
        )

    m = _SELF_NAME.search(msg)
    if m and m.group(1).strip() not in ("叶鹏祥",):
        who = m.group(1).strip()
        _add(f"对话中用户自称「{who}」（仅用户说明为准，勿自动推断与他人的亲属关系）。")

    m2 = _RENAME.search(msg)
    if m2:
        who = (m2.group(1) or m2.group(2) or "").strip()
        if who:
            _add(f"用户澄清身份/姓名为「{who}」，以本条为准。")

    # 明确「XX 是我女朋友/女友」等
    m3 = re.search(r"([\u4e00-\u9fff·]{2,8})\s*是我\s*(女朋友|女友|老婆)", msg)
    if m3:
        _add(f"用户说明：{m3.group(1)}是其{m3.group(2)}。", "relationship")

    return saved


def girlfriend_tone_active(user_message: str, memory: dict) -> bool:
    if re.search(r"远慧|刘大炮|刘远慧|秋雨|老婆|宝贝|老公", user_message):
        return True
    blob = " ".join(memory.get("semantic", []))
    return bool(re.search(r"刘远慧|刘大炮", blob))
