#!/usr/bin/env python3
"""口吻范例清理工具：移除微信导入过程中产生的语义错乱 Q→A 对。

用途：微信聊天记录的自动导入（import_wechat.py）在合并连续消息和
提取 Q→A 对时，可能产生一些语义上不合理的范例对，例如：
- 问题与回答语义不相关（微信合并/压缩导致的拼接错误）
- 回答过长或包含多余空格（系统消息残留）
- 包含多媒体标记残留（[图片]、[表情]等）
- 纯重复字符的无意义回复（哈哈哈哈）
- 过于通用的敷衍回复（"还好吧 其实这个"等）

本脚本会对所有风格范例文件执行一次清理扫描，移除这些问题对，
保留高质量、有代表性的范例。

处理范围：
  - style/examples.md           # 跳过（主文件已人工 curated）
  - style/examples_intimate.md  # 清洗（strict 模式）
  - import/drafts/style_daily.md
  - import/drafts/style_intimate.md
  - import/drafts/style_group_friends.md

严格模式（strict）：
  - style/ 目录下的正式文件使用更严格的过滤规则（回答 <=32 字，问题 <=50 字）
  - drafts/ 目录下的草稿文件使用较宽松的规则（回答 <=48 字，问题 <=85 字）

典型用法：
    python scripts/clean_style_examples.py

运行时机：
    - 每次运行 import_wechat.py 生成口吻草稿后
    - 手动编辑过风格文件后，用于验证和清理
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PERSONA = Path(__file__).resolve().parents[2] / "persona"

# 需要清理的风格文件列表
STYLE_FILES = [
    PERSONA / "style" / "examples.md",
    PERSONA / "style" / "examples_intimate.md",
    PERSONA / "import" / "drafts" / "style_daily.md",
    PERSONA / "import" / "drafts" / "style_intimate.md",
    PERSONA / "import" / "drafts" / "style_group_friends.md",
]

# 正则：匹配 "问：..." 和 "答：..." 两种行首格式
PAIR_RE = re.compile(r"^问[:：](.*)$")
ANS_RE = re.compile(r"^答[:：](.*)$")
BRACKET = re.compile(r"\[[^\]]+\]")

# 已知的语义不良回答（固定的无效模板）
GENERIC_A = frozenset({
    "还好吧 其实这个", "嗯嗯对对对", "笑死了", "666", "6666", "嗯嗯对对",
    "有点逆天", "逆天", "好吧晚安", "嗯嗯嗯呢", "还好吧说实话 没那么严重",
    "今天瘦了这么多逆天",
})

# 需要强制保留的锚定范例（即使命中某些过滤规则也不删除）
# 这些是经人工确认最能代表说话风格的核心范例
KEEP_ANCHORS = [
    ("在吗", "在，咋了"),
    ("谢谢", "行，客气啥"),
    ("想你了", "嗯，我也想你"),
    ("在干嘛", "没干嘛，咋了"),
    ("在吗老公", "在，咋了嘛"),
    ("我是刘远慧", "咋滴，终于晓得回我了啊"),
    ("今天吃什么", "辣的呗，别整鸡鸭鱼"),
    ("你还健身吗", "想啊，没时间"),
    ("咋不理我", "理你啊，刚忙到"),
    ("你是不是不想理我", "没得，刚忙到"),
]


def parse_pairs(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """解析 markdown 格式的范例文件，分离文件头和 Q→A 对。

    解析策略：
    - 遇到第一个"问："之前的内容视为文件头（标题、说明等）
    - 之后按"问："→"答："顺序配对
    - 连续出现的"问："会覆盖上一个未配对的"问"

    Args:
        text: 范例文件的完整内容

    Returns:
        (文件头行列表, [(问题, 回答), ...]) 的元组
    """
    header: list[str] = []
    pairs: list[tuple[str, str]] = []
    q: str | None = None
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        mq = PAIR_RE.match(stripped)
        ma = ANS_RE.match(stripped)
        if mq:
            started = True
            q = mq.group(1).strip()
            continue
        if ma and q is not None:
            pairs.append((q, ma.group(1).strip()))
            q = None
            continue
        if not started and stripped:
            header.append(line)
    return header, pairs


def is_broken(q: str, a: str, *, strict: bool) -> bool:
    """判断一个 Q→A 对是否语义异常/格式损坏。

    检测规则（优先级从高到低）：
    1. 空内容或过短回答（<2 字）
    2. 长度超标（strict 模式更严格：回答 <=32/48 字，问题 <=50/85 字）
    3. 回答含过多空格（>=3 个，说明是拼接/合并残留）
    4. 问题含过多空格（>=5 个，说明是系统消息残留）
    5. 包含微信引用标记（"[引用"）
    6. 包含 HTML 实体残留（&#x20;）
    7. 包含纯重复字符（如"哈哈哈哈哈哈"）
    8. 包含过多方括号标记（>=3 个）
    9. 包含过多问号（>2 个，通常是自动生成的内容）
    10. 命中已知无效回答列表
    11. 问答语义不相关（没有共同中文词汇）
    12. 回答包含过多句子片段（>=3 个 >=4 字的片段）

    Args:
        q: 问题文本
        a: 回答文本
        strict: 是否启用严格模式（对 style/ 目录文件使用）

    Returns:
        True 表示该对应被移除
    """
    if not q or not a or len(a) < 2:
        return True
    # 长度限制：strict 模式更严格
    max_a, max_q = (32, 50) if strict else (48, 85)
    if len(a) > max_a or len(q) > max_q:
        return True
    # 过多空格 = 合并/拼接残留
    if len(a) > 18 and a.count(" ") >= 3:
        return True
    if len(q) > 28 and q.count(" ") >= 5:
        return True
    # 微信引用标记残留
    if "[引用" in q or "[引用" in a:
        return True
    # HTML 实体残留（微信导出格式）
    if "&#x20;" in q or "&#x20;" in a or "转发的聊天记录" in q:
        return True
    # 纯重复字符（如"哈哈哈哈哈"）
    if re.search(r"(.)\1{7,}", re.sub(r"\s+", "", a)):
        return True
    # 过多方括号标记
    if q.count("[") >= 3:
        return True
    # 过多问号（非自然对话）
    if q.count("?") + q.count("？") > 2:
        return True
    # 已知无效回答
    if a in GENERIC_A:
        return True
    if "嗯嗯对对" in a:
        return True
    if strict and len(q) > 42:
        return True
    if "还好吧" in a and "其实" in a:
        return True
    # 语义不相关检查：提取问答中的中文词，无交集 = 可能不相关
    q_tokens = set(re.findall(r"[一-鿿]{2,}", q))
    a_tokens = set(re.findall(r"[一-鿿]{2,}", a))
    if len(q) > 22 and len(a) > 10 and q_tokens and a_tokens:
        if len(q_tokens & a_tokens) == 0:
            return True
    # 过多句子片段（长回答被切分后的残留）
    chunks = [c for c in re.split(r"[\s，。！？]+", a) if len(c) >= 4]
    if len(chunks) >= 3 and len(a) > 22:
        return True
    return False


def clean_pairs(pairs: list[tuple[str, str]], *, strict: bool) -> list[tuple[str, str]]:
    """对 Q→A 对列表执行清理过滤。

    清理流程：
    1. 优先保留锚定范例（KEEP_ANCHORS 中的对不受过滤规则影响）
    2. 清理回答中的方括号噪声
    3. 按 is_broken 规则过滤异常对
    4. 以回答前 30 字符（去空白）作为键去重

    Args:
        pairs: 原始 Q→A 对列表
        strict: 是否使用严格过滤规则

    Returns:
        清理后的 Q→A 对列表
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for q, a in pairs:
        # 锚定范例无条件保留
        if (q, a) in KEEP_ANCHORS:
            key = re.sub(r"\s+", "", a)[:30]
            if key not in seen:
                seen.add(key)
                out.append((q, a))
            continue
        # 清理方括号噪声
        a_clean = BRACKET.sub("", a).strip()
        a_clean = re.sub(r"\s+", " ", a_clean).strip()
        if is_broken(q, a_clean, strict=strict):
            continue
        key = re.sub(r"\s+", "", a_clean)[:30]
        if key in seen:
            continue
        seen.add(key)
        out.append((q, a_clean))
    return out


def write_file(path: Path, header: list[str], pairs: list[tuple[str, str]]) -> None:
    """将清理后的范例写回文件。

    Args:
        path: 输出文件路径
        header: 文件头行列表（标题、说明等）
        pairs: 清理后的 Q→A 对列表
    """
    # 只保留标题段（到第一个"问："行或空行之前）
    head: list[str] = []
    for line in header:
        if line.strip().startswith("问：") or line.strip().startswith("答："):
            break
        head.append(line)
    # 去除尾部空行
    while head and not head[-1].strip():
        head.pop()
    lines = list(head)
    if lines:
        lines.append("")
    for q, a in pairs:
        lines += [f"问：{q}", f"答：{a}", ""]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    """执行批量清理流程。"""
    total_before = total_after = 0
    for path in STYLE_FILES:
        if not path.is_file():
            continue
        # style/ 目录下的正式文件使用 strict 严格模式
        strict = path.parent.name == "style"
        header, pairs = parse_pairs(path.read_text(encoding="utf-8"))
        # examples.md 已经过人工 curation，跳过自动清洗
        if path.name == "examples.md":
            continue
        kept = clean_pairs(pairs, strict=strict)
        total_before += len(pairs)
        total_after += len(kept)
        write_file(path, header, kept)
        print(f"{path.name}: {len(pairs)} -> {len(kept)}")
    print(f"total {total_before} -> {total_after}")


if __name__ == "__main__":
    main()
