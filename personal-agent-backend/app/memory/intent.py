"""L3 long-term memory recall intent."""

from __future__ import annotations

import re

# 强触发：明确回忆
_STRONG = [
    r"还记得",
    r"记不记得",
    r"以前(是不是|说过|提过)",
    r"那时候",
    r"上次.{0,12}(说|聊|提|去|见|吃)",
    r"你说过",
    r"很久以前",
    r"小时候",
    r"\d+年前",
    r"去年|前年|当初",
]

# 生活事实 / 关系（导入的 wechat_memory、my_corpus 多在这类问题里用到）
_PERSONAL = [
    r"女朋友|男友|老婆|老公|远慧|刘远慧|刘大炮|秋雨|刘远航",
    r"你谁|我是谁|你是谁|叫什么|输错|打错|姐姐|弟弟|亲戚",
    r"我俩|咱们|我们一起|咱俩",
    r"见面|约会|去哪|吃过|玩过",
    r"实习|杭州|南溪|爱琴海",
    r"喜欢什么|爱吃|忌口|喝什么",
    r"周末.*(干嘛|做什么|干啥)",
]

_FACT_QUERY = re.compile(
    r"(叫什么|叫什么名字|忌口|喜欢喝|猫叫|生日|老家)"
)


def needs_l3_recall(query: str) -> bool:
    q = query.strip()
    if not q:
        return False
    for pat in _STRONG + _PERSONAL:
        if re.search(pat, q):
            return True
    if _FACT_QUERY.search(q) and len(q) <= 48:
        return True
    return False
