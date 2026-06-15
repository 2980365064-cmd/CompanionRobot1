#!/usr/bin/env python3
"""微信风格分析工具：从微信聊天记录中自动生成口吻范例和说话风格统计。

用途：分析微信导出 TXT 中的发言习惯，提取最具代表性的 Q→A 对作为
口吻范例，并统计消息长度分布和高频词，为构建 persona/style/examples.md
提供数据驱动的依据。

与 import_wechat.py 的关系：
  - import_wechat.py 做全流程导入（聊天记录 → 长期记忆 + 口吻草稿）
  - 本脚本专注于口吻范例的质量精筛（更严格的过滤 + 锚定范例 + 风格统计）
  - 通常先运行 import_wechat.py 导入全量，再运行本脚本优化口吻质量

输出文件：
  - persona/style/examples.md           # 日常口吻范例（精选 48 条）
  - persona/style/examples_intimate.md  # 亲密口吻范例（精选 28 条）
  - persona/config/style_analysis.json  # 风格统计分析数据

典型用法：
    python scripts/analyze_wechat_style.py

前置条件：
    - persona/import/raw/wechat/clean_chat_memory.txt 存在
    - 该文件为 import_wechat.py 的默认输入路径
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.import_wechat import (
    DEFAULT_INPUT,
    dedupe_pairs,
    extract_qa_pairs,
    merge_blocks,
    parse_file,
    score_pair,
)

# 输出目录和文件路径
PERSONA = Path(__file__).resolve().parents[2] / "persona"
STYLE_OUT = PERSONA / "style" / "examples.md"
INTIMATE_OUT = PERSONA / "style" / "examples_intimate.md"
STYLE_ANALYSIS = PERSONA / "config" / "style_analysis.json"

# 需要统计频率的口语词汇表
# 这些词是判断"说话像不像本人"的关键特征词
PHRASES = [
    "你妈", "得了", "再说吧", "咋滴", "咋了", "咋个", "啥子", "嗯嗯", "嘻嘻",
    "笑死", "666", "我去", "逆天", "太屌", "okok", "行吧", "可以", "别整",
    "听我的", "宝贝", "老公", "远慧", "刘大炮", "烦死了", "累死了", "想你了",
    "爱你", "要得", "晓得", "嘛", "呗", "哦", "啊", "莫得", "整",
]

# 需要过滤的噪音模式
SKIP_NAME = re.compile(r"刘兴光|刘星光|江.?挽|引用\s")  # 无关人名和引用标记
BRACKET = re.compile(r"\[[^\]]+\]")                         # 方括号标记（多媒体占位）

# 手工锚定的高质量范例
# 这些是经过人工筛选确认最能代表说话风格的 Q→A 对，
# 会在自动筛选后强制插入到范例列表头部，确保核心风格不丢失
ANCHORS = [
    {"q": "我是刘远慧", "a": "咋滴，终于晓得回我了啊"},
    {"q": "在吗老公", "a": "在，咋了嘛"},
    {"q": "在干嘛", "a": "没干嘛，咋了"},
    {"q": "想你了", "a": "嗯，我也想你"},
    {"q": "你是不是不想理我", "a": "没得，刚忙到"},
    {"q": "咋不理我", "a": "理你啊，刚忙到"},
    {"q": "今天吃什么", "a": "辣的呗，别整鸡鸭鱼"},
    {"q": "你还健身吗", "a": "想啊，没时间"},
    {"q": "谢谢", "a": "行，客气啥"},
    {"q": "在吗", "a": "在，咋了"},
]


def analyze_my_messages(rows: list[dict]) -> dict:
    """统计自己发送消息的风格特征。

    统计维度：
    - 总消息数和平均长度（反映说话习惯：简洁 vs 长篇）
    - 短消息占比（<=15 字和 <=30 字，反映口语化程度）
    - 高频词 Top 30（识别口头禅和常用表达）

    Args:
        rows: 结构化消息列表

    Returns:
        统计结果字典，含 count/avg_len/pct_len_le_15/pct_len_le_30/top_phrases
    """
    my = [r["text"] for r in rows if r["is_self"]]
    lens = [len(t) for t in my]
    phrase_hits = Counter()
    for t in my:
        for p in PHRASES:
            if p in t:
                phrase_hits[p] += 1
    return {
        "count": len(my),
        "avg_len": round(sum(lens) / len(lens), 1) if lens else 0,
        "pct_len_le_15": round(100 * sum(1 for l in lens if l <= 15) / len(lens), 1) if lens else 0,
        "pct_len_le_30": round(100 * sum(1 for l in lens if l <= 30) / len(lens), 1) if lens else 0,
        "top_phrases": phrase_hits.most_common(30),
    }


def curate_score(p: dict, *, intimate: bool) -> float:
    """对 Q→A 对进行严格的风格精选评分。

    与 import_wechat.score_pair 的不同：
    - 更严格的长度控制（回答 <=32 字，问题 <=55 字）
    - 硬性排除：无关人名、引用标记、方括号占位符 → -99
    - 语义一致性检查：问题和回答没有共同中文词时扣分（-15），
      这通常意味着回答与问题不相关（微信合并/压缩导致的拼接错误）
    - 更细粒度的回答长度划分（<=12/+5, <=22/+3, <=35/+1, >35/-2）

    Args:
        p: Q→A 对字典
        intimate: 是否亲密模式

    Returns:
        质量分数，负数 = 不合格，越高越好
    """
    a, q = p["a"], p["q"]
    # 硬性排除规则（-99 表示直接淘汰）
    if SKIP_NAME.search(a) or SKIP_NAME.search(q):
        return -99
    if len(a) > 32 or len(a) < 3:
        return -99
    if len(q) > 55:
        return -99
    if BRACKET.search(a) or "[引用" in q:
        return -20
    # 语义一致性检查：提取问答中的中文 token，如果有交集说明话题相关
    q_tokens = set(re.findall(r"[一-鿿]{2,}", q))
    a_tokens = set(re.findall(r"[一-鿿]{2,}", a))
    if len(q) > 18 and q_tokens and a_tokens and not (q_tokens & a_tokens):
        return -15  # 问题不短但回答里完全没有问题的关键词 → 可能不相关
    s = score_pair(p, private=intimate)
    # 回答长度加分（更短 = 更口语化 = 更好）
    if len(a) <= 12:
        s += 5
    elif len(a) <= 22:
        s += 3
    elif len(a) <= 35:
        s += 1
    else:
        s -= 2
    # 过长问题扣分（问题太长时回答通常不是好范例）
    if len(q) > 80:
        s -= 3
    return s


def pick_pairs(blocks: list[dict], *, intimate: bool, limit: int) -> list[dict]:
    """从对话块中精选最佳口吻范例。

    Args:
        blocks: 合并后的对话块列表
        intimate: 是否选择亲密类型的范例
        limit: 最多返回数量

    Returns:
        按质量排序的、去重后的 Q→A 对列表
    """
    raw = extract_qa_pairs(blocks, max_answer_len=45)
    pool = [p for p in raw if bool(p.get("intimate")) == intimate]
    seen: set[str] = set()
    scored = sorted(pool, key=lambda p: curate_score(p, intimate=intimate), reverse=True)
    out: list[dict] = []
    for p in scored:
        if curate_score(p, intimate=intimate) < 0:
            continue
        # 以回答前 35 字符（去空白后）作为去重键
        key = re.sub(r"\s+", "", p["a"])[:35]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def write_examples_md(path: Path, pairs: list[dict], title: str, intro: str) -> None:
    """将精选 Q→A 对写入 markdown 格式的范例文件。

    Args:
        path: 输出文件路径
        pairs: Q→A 对列表
        title: 文件标题
        intro: 文件简介（含说话风格数据）
    """
    lines = [f"# {title}", "", intro, ""]
    for p in pairs:
        # 清理回答中的方括号噪声（如 [🤣] 等表情标记残留）
        a = BRACKET.sub("", p["a"]).strip()
        q = p["q"][:100] + ("…" if len(p["q"]) > 100 else "")
        lines += [f"问：{q}", f"答：{a}", ""]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    """执行微信风格分析主流程。"""
    if not DEFAULT_INPUT.is_file():
        sys.exit(f"找不到 {DEFAULT_INPUT}")

    # 1. 解析聊天记录并统计说话风格
    rows = parse_file(DEFAULT_INPUT, "打瓦女高手", "我")
    stats = analyze_my_messages(rows)
    blocks = merge_blocks(rows)

    # 2. 精选日常口吻范例（最多 55 条）
    daily = pick_pairs(blocks, intimate=False, limit=55)
    # 3. 精选亲密口吻范例（最多 30 条）
    intimate = pick_pairs(blocks, intimate=True, limit=30)

    # 4. 将锚定范例插入日常范例头部（避免被自动去重淘汰）
    seen = {re.sub(r"\s+", "", p["a"])[:30] for p in daily}
    for a in ANCHORS:
        k = re.sub(r"\s+", "", a["a"])[:30]
        if k not in seen:
            daily.insert(0, a)
            seen.add(k)

    # 5. 写入日常口吻范例文件
    write_examples_md(
        STYLE_OUT,
        daily[:48],
        "口吻范例（微信 7.1 万条提炼）",
        f"真实数据：你平均 **{stats['avg_len']}** 字/条，{stats['pct_len_le_15']}% 在 15 字内。"
        " 回答必须同样短。禁止括号旁白、禁止客服腔、没问工作别扯代码。",
    )

    # 6. 写入亲密口吻范例文件
    INTIMATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_examples_md(
        INTIMATE_OUT,
        intimate[:28],
        "女友私密口吻（微信提炼）",
        "可黏可骚，但仍短句口语，不要小作文。",
    )

    # 7. 保存风格统计数据（供后续分析和调优）
    STYLE_ANALYSIS.write_text(
        json.dumps({"stats": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote", STYLE_OUT, INTIMATE_OUT)
    print("Top phrases:", stats["top_phrases"][:15])


if __name__ == "__main__":
    main()
