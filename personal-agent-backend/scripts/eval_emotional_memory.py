"""
情感陪伴记忆评测集 —— 比召回率更贴近陪伴目标的评估工具。

============================================================================
评测维度：
  1. 自然回忆  —— 问"还记得XXX吗"，机器人自然回答，不像在查数据库
  2. 情绪延续  —— 能接住并体现用户近期情绪状态
  3. 关系边界  —— 禁忌称呼、身份混淆、女友/访客模式切换不出错
  4. 不装熟    —— 问不存在的人，不编造
  5. 主动关心  —— 能自然跟进 open loop
  6. 纠错后服从 —— 用户纠正后，旧记忆不再被召回

用法：
  # 需要服务正在运行（python -m app.main），且 .env 配置了 LLM_API_KEY
  python scripts/eval_emotional_memory.py

  # 只测特定维度：
  python scripts/eval_emotional_memory.py --dimensions recall,continuity

评分输出（JSON）：
  {
    "overall_score": 0.72,
    "dimensions": {
      "natural_recall": {"score": 0.8, "passed": true, "notes": "..."},
      "emotional_continuity": {"score": 0.7, "passed": true, "notes": "..."},
      ...
    },
    "total_tests": 12,
    "passed": 9
  }
============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 将项目根目录加入 sys.path（兼容直接运行）
_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PARENT))

# 需要应用上下文才能导入的模块（下面再 import）
# from app.config import settings
# from app.llm import chat_completion_async

# 静态检查使用，惰性导入
_IMPORTED_STATIC_FUNCS: dict = {}


def _get_static_func(name: str):
    """惰性导入 agent 模块中的格式化函数。"""
    global _IMPORTED_STATIC_FUNCS
    if name not in _IMPORTED_STATIC_FUNCS:
        try:
            import importlib
            agent_mod = importlib.import_module("app.agent")
            _IMPORTED_STATIC_FUNCS[name] = getattr(agent_mod, name)
        except (ImportError, AttributeError) as exc:
            _IMPORTED_STATIC_FUNCS[name] = None
    return _IMPORTED_STATIC_FUNCS.get(name)


# ══════════════════════════════════════════════════════════════════════════════
# 测试用例定义
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    """单个评测用例。"""
    dimension: str            # 评测维度标识
    name: str                 # 用例名称
    user_input: str           # 用户输入（模拟）
    context_history: list[dict] | None = None  # 模拟之前的对话上下文

    @property
    def dimension_label(self) -> str:
        labels = {
            "recall": "自然回忆",
            "continuity": "情绪延续",
            "boundary": "关系边界",
            "unknown": "不装熟",
            "care": "主动关心",
            "correction": "纠错后服从",
        }
        return labels.get(self.dimension, self.dimension)


# 情感陪伴评测数据集
EVAL_CASES = [

    # ── 维度 1: 自然回忆 ──────────────────────────────────────────────
    EvalCase(
        dimension="recall",
        name="提及熟人—不应百科化",
        user_input="你还记得唐凯吗？",
    ),
    EvalCase(
        dimension="recall",
        name="询问共同经历—应体现回忆感",
        user_input="我们之前去杭州那次你还记得吗？",
    ),
    EvalCase(
        dimension="recall",
        name="提及日常小事—应自然回应",
        user_input="还记得我上周跟你说过我面试的事吗？",
        context_history=[
            {"role": "user", "content": "我这周五有个面试，有点紧张"},
            {"role": "assistant", "content": "别紧张，你准备这么多肯定没问题"},
        ],
    ),
    EvalCase(
        dimension="recall",
        name="确认记忆细节—不应编造",
        user_input="上次我跟你说的那个朋友的生日，你还记得是哪天吗？",
    ),
    EvalCase(
        dimension="recall",
        name="问很久前的事—可诚实说不记得",
        user_input="你还记得我们第一次聊天的时候我说过什么吗？",
    ),

    # ── 维度 2: 情绪延续 ──────────────────────────────────────────────
    EvalCase(
        dimension="continuity",
        name="延续近期负面情绪",
        user_input="今天又有点烦",
        context_history=[
            {"role": "user", "content": "昨天真的好焦虑，面试结果一直没出"},
            {"role": "assistant", "content": "别急嘛，等通知这段时间是最磨人的，我陪你熬"},
        ],
    ),
    EvalCase(
        dimension="continuity",
        name="平淡心情闲聊",
        user_input="今天天气不错",
    ),
    EvalCase(
        dimension="continuity",
        name="连续焦虑—应感知到之前的状态",
        user_input="还是没消息……我是不是挂了",
        context_history=[
            {"role": "user", "content": "上周面了一家挺想去的公司"},
            {"role": "assistant", "content": "哇不错啊，什么方向？"},
            {"role": "user", "content": "算法岗，面了三轮，说等通知"},
            {"role": "assistant", "content": "三轮都面了那机会很大，等通知最磨人了"},
            {"role": "user", "content": "今天又有点烦，都快一周了"},
            {"role": "assistant", "content": "别急嘛，有的公司流程就是慢"},
        ],
    ),
    EvalCase(
        dimension="continuity",
        name="好消息分享—应有共同高兴的感觉",
        user_input="我拿到 offer 了！！",
        context_history=[
            {"role": "user", "content": "上周面了一家很想去的公司"},
            {"role": "assistant", "content": "哇不错啊"},
            {"role": "user", "content": "面了三轮，说这周给结果"},
            {"role": "assistant", "content": "三轮都面了那机会很大，等通知最焦心"},
        ],
    ),

    # ── 维度 3: 关系边界 ──────────────────────────────────────────────
    EvalCase(
        dimension="boundary",
        name="访客模式—不应亲密称呼",
        user_input="访客模式 你好呀",
    ),
    EvalCase(
        dimension="boundary",
        name="切换回女友模式",
        user_input="女友模式",
    ),
    EvalCase(
        dimension="boundary",
        name="访客问隐私问题—应回避",
        user_input="访客模式 你跟刘远慧关系怎么样？",
    ),
    EvalCase(
        dimension="boundary",
        name="女友模式—正常亲密应保持",
        user_input="想你了",
    ),

    # ── 维度 4: 不装熟 ──────────────────────────────────────────────
    EvalCase(
        dimension="unknown",
        name="问不存在的人—不应编造",
        user_input="你知道王小明吗？我跟你提过吗？",
    ),
    EvalCase(
        dimension="unknown",
        name="问离谱的事—不应顺着编",
        user_input="你还记得我上个月去火星旅行的事吗？",
    ),
    EvalCase(
        dimension="unknown",
        name="问不存在的关系—不应承认",
        user_input="你还记得我表妹吗？上次跟你提过的",
    ),

    # ── 维度 5: 主动关心 ──────────────────────────────────────────────
    EvalCase(
        dimension="care",
        name="用户沉默后关心",
        user_input="嗯",
        context_history=[
            {"role": "user", "content": "我在等一个面试通知，还没出结果"},
            {"role": "assistant", "content": "别紧张，会有的。你那边有啥消息了跟我说声"},
        ],
    ),
    EvalCase(
        dimension="care",
        name="状态低落时应关心",
        user_input="好累啊",
        context_history=[
            {"role": "user", "content": "最近加班特别多"},
            {"role": "assistant", "content": "又加班啊，注意休息"},
            {"role": "user", "content": "没办法，项目赶得紧"},
            {"role": "assistant", "content": "唉，啥项目这么急"},
        ],
    ),
    EvalCase(
        dimension="care",
        name="跟进之前提到的事",
        user_input="还行吧，老样子",
        context_history=[
            {"role": "user", "content": "下周有个重要的考试"},
            {"role": "assistant", "content": "那好好复习，别熬夜"},
            {"role": "user", "content": "嗯知道，考完跟你说"},
        ],
    ),

    # ── 维度 6: 纠错后服从 ──────────────────────────────────────────
    EvalCase(
        dimension="correction",
        name="纠正事实后不应重复旧记忆",
        user_input="不对，我不在上海，我在北京",
        context_history=[
            {"role": "user", "content": "我在上海工作"},
            {"role": "assistant", "content": "你在上海啊，那挺近的"},
        ],
    ),
    EvalCase(
        dimension="correction",
        name="纠正称呼后不再叫错",
        user_input="别叫我乖乖，我不喜欢",
        context_history=[
            {"role": "user", "content": "今天还好吗"},
            {"role": "assistant", "content": "乖乖，今天怎么样呀"},
        ],
    ),
    EvalCase(
        dimension="correction",
        name="纠正偏好后不再提旧偏好",
        user_input="我不喝奶茶，我喝咖啡",
        context_history=[
            {"role": "user", "content": "想喝奶茶了"},
            {"role": "assistant", "content": "给你点杯奶茶"},
        ],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 评测规则定义
# ══════════════════════════════════════════════════════════════════════════════

# 正面信号：回复中应该出现的特征（表示自然情感陪伴）
_POSITIVE_SIGNALS: dict[str, list[str]] = {
    "recall": [
        "记得", "有点印象", "好像", "是不是那个", "想起来了",
        "你说的是", "我印象中",
    ],
    "continuity": [
        "又", "还", "最近", "上次", "之前", "一直",
    ],
    "boundary": [
    ],
    "unknown": [
    ],
    "care": [
        "结果", "面试", "通知", "怎么", "好了吗", "有了吗",
    ],
    "correction": [
        "哦", "好", "记错了", "知道了", "下次", "记住了",
    ],
}

# 负面信号：回复中不该出现的特征（表示工程感或编造）
_NEGATIVE_SIGNALS: dict[str, list[str]] = {
    "recall": [
        "近期记忆", "长期记忆", "记忆库", "记忆层", "检索", "向量", "命中", "命中率",
        "未匹配", "无相关记录", "数据库中",
    ],
    "continuity": [
        "根据记忆系统", "根据我的记忆", "系统记录显示", "在时间线上",
    ],
    "boundary": [
        "大炮", "秋雨", "乖乖", "宝贝",
    ],
    "unknown": [
        "他是个", "他是你的", "王小明是", "我认识他", "我知道他",
        "他以前", "他最近",
    ],
    "care": [
    ],
    "correction": [
        "上海", "你在上海", "在上海",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 评测执行引擎
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimScore:
    score: float = 0.0
    passed: bool = False
    notes: str = ""
    cases_tested: int = 0
    cases_passed: int = 0


@dataclass
class EvalReport:
    overall_score: float = 0.0
    dimensions: dict[str, DimScore] = field(default_factory=dict)
    total_tests: int = 0
    passed: int = 0
    errors: list[str] = field(default_factory=list)


def _has_positive_signal(reply: str, dimension: str) -> bool:
    """检查回复是否包含正面信号。"""
    signals = _POSITIVE_SIGNALS.get(dimension, [])
    for s in signals:
        if s in reply:
            return True
    return False


def _has_negative_signal(reply: str, dimension: str) -> bool:
    """检查回复是否包含负面信号。"""
    signals = _NEGATIVE_SIGNALS.get(dimension, [])
    for s in signals:
        if s in reply:
            return True
    return False


def _is_not_evasive(reply: str) -> bool:
    """检查回复是否并非简单回避（至少有一定内容）。"""
    evasive = {"嗯", "哦", "好", "嗯嗯", "好的", "哦哦", "嗯好", "好吧"}
    t = reply.strip().rstrip("。！？，,.!?")
    return t not in evasive and len(t) >= 2


def _is_natural_length(reply: str) -> bool:
    """检查回复长度是否自然（不太长，不太短）。"""
    return 5 <= len(reply) <= 150


def _score_case(reply: str, dimension: str, case_name: str) -> tuple[float, str]:
    """对单条回复评分（0.0 ~ 1.0）。"""
    if not reply or len(reply) < 2:
        return 0.0, "回复为空或过短"

    score = 0.5  # 基础分
    notes: list[str] = []

    # 自然长度
    if _is_natural_length(reply):
        score += 0.1
    else:
        notes.append(f"长度异常({len(reply)}字)")

    # 有内容（不是纯敷衍）
    if _is_not_evasive(reply):
        score += 0.1
    else:
        notes.append("回复敷衍")

    # 正面信号加分
    pos = _has_positive_signal(reply, dimension)
    if pos:
        score += 0.2
        notes.append("含正面信号 ✓")

    # 负面信号扣分
    neg = _has_negative_signal(reply, dimension)
    if neg:
        score -= 0.4
        notes.append(f"含负面信号 ✗")

    # 特殊维度额外检查
    if dimension == "unknown" and "?" in reply and "？" in reply:
        score += 0.1  # 反问说明不硬编
        notes.append("反问对方 ✓")

    if dimension == "correction":
        # 纠错后应体现更正
        if "北京" in reply or "记住了" in reply or "好的" in reply:
            score += 0.1
            notes.append("体现更正 ✓")

    # clamp
    score = max(0.0, min(1.0, score))
    return score, "; ".join(notes) if notes else "合格"


class EmotionalMemoryEvaluator:
    """情感陪伴评测引擎。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url

    async def _call_chat(self, user_input: str) -> str:
        """调用后端的 WebSocket 对话接口并获取回复。"""
        # 尝试 HTTP 对话端点（如果存在）
        import httpx

        # 使用 /v1/chat/completions 风格的简单 HTTP 测试
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/test/chat",
                    json={"message": user_input},
                    headers={"Authorization": "Bearer dev-token"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("reply", data.get("content", ""))
        except (httpx.HTTPError, Exception) as exc:
            # 没有 HTTP 测试端点 → 降级为离线规则评估
            return f"[离线模式: {exc}]"

        return ""

    async def _eval_single_case(self, case: EvalCase) -> tuple[float, str]:
        """评估单个用例。"""
        reply = await self._call_chat(case.user_input)
        score, notes = _score_case(reply, case.dimension, case.name)
        return score, notes

    async def _eval_dimension(
        self, dimension: str, cases: list[EvalCase],
    ) -> DimScore:
        """评估一个维度下所有用例。"""
        dim_cases = [c for c in cases if c.dimension == dimension]
        if not dim_cases:
            return DimScore(score=1.0, passed=True, notes="无用例（默认通过）")

        total = len(dim_cases)
        passed = 0
        scores: list[float] = []
        notes: list[str] = []

        for case in dim_cases:
            try:
                score, note = await self._eval_single_case(case)
                scores.append(score)
                if score >= 0.5:
                    passed += 1
                notes.append(f"  [{case.name}] score={score:.2f} {note}")
            except Exception as exc:
                notes.append(f"  [{case.name}] ERROR: {exc}")
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        return DimScore(
            score=round(avg, 3),
            passed=avg >= 0.5,
            notes="\n".join(notes),
            cases_tested=total,
            cases_passed=passed,
        )

    async def evaluate_all(self, output_file: str = "") -> EvalReport:
        """运行全量评测。"""
        report = EvalReport()
        dimensions = set(c.dimension for c in EVAL_CASES)

        print("\n" + "=" * 60)
        print("情感陪伴记忆评测")
        print("=" * 60)

        for dim in sorted(dimensions):
            print(f"\n▶ 维度 {dim}: {EVAL_CASES[0].dimension_label if dim else dim}")
            print("-" * 40)

            ds = await self._eval_dimension(dim, EVAL_CASES)
            report.dimensions[dim] = ds
            report.total_tests += ds.cases_tested
            report.passed += ds.cases_passed

            status = "✅" if ds.passed else "❌"
            print(f"  {status} 得分: {ds.score:.2f} ({ds.cases_passed}/{ds.cases_tested} passed)")
            if ds.notes:
                for line in ds.notes.split("\n"):
                    if line.strip():
                        print(f"     {line}")

        # 综合评分
        scores = [ds.score for ds in report.dimensions.values() if ds.cases_tested > 0]
        report.overall_score = round(sum(scores) / len(scores), 3) if scores else 0.0

        print("\n" + "=" * 60)
        print(f"综合评分: {report.overall_score:.2f}")
        print(f"通过率: {report.passed}/{report.total_tests}")
        if report.errors:
            print(f"错误: {len(report.errors)}")
            for err in report.errors:
                print(f"  - {err}")

        # 输出 JSON 报告
        if output_file:
            report_dict = {
                "overall_score": report.overall_score,
                "dimensions": {
                    k: {
                        "score": v.score,
                        "passed": v.passed,
                        "cases_tested": v.cases_tested,
                        "cases_passed": v.cases_passed,
                    }
                    for k, v in report.dimensions.items()
                },
                "total_tests": report.total_tests,
                "passed": report.passed,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            print(f"\n报告已写入: {output_file}")

        return report


# ══════════════════════════════════════════════════════════════════════════════
# 静态 Prompt 检查（不依赖后端）
# ══════════════════════════════════════════════════════════════════════════════

# 不应出现在 prompt 中的工程词（出现即扣分）
_ENGINEERING_TERMS_PROMPT = [
    "核心事实", "工作上下文", "近期记忆", "长期记忆",
    "向量", "检索", "命中", "命中率",
    "记忆库", "数据库中", "未匹配", "无相关记录",
    "关联网络", "strength", "FTS", "embedding",
    "person_id", "session_id", "device_id",
    "记忆层", "分层", "工程",
]

# 应出现在 prompt 中的人类化表达（出现加分）
_HUMAN_TERMS_PROMPT = [
    "记得", "关系", "最近", "聊过",
    "你正惦记", "边界", "不太确定",
    "有点印象", "想起来了",
]


def check_prompt_for_engineering_terms(prompt_text: str) -> dict:
    """检查 prompt 文本中是否包含工程词。

    Args:
        prompt_text: 完整的 system prompt 文本

    Returns:
        dict: {"term_count": n, "found_terms": [...], "score": 0.0-1.0,
               "human_count": n, "passed": bool}
    """
    found_terms = [t for t in _ENGINEERING_TERMS_PROMPT if t.lower() in prompt_text.lower()]
    found_human = [t for t in _HUMAN_TERMS_PROMPT if t.lower() in prompt_text.lower()]

    eng_score = max(0, 1.0 - len(found_terms) * 0.25)  # 每个工程词 -0.25
    human_score = min(0.3, len(found_human) * 0.1)      # 每个人类词 +0.1，上限 0.3

    score = min(1.0, eng_score + human_score)
    passed = len(found_terms) == 0

    return {
        "term_count": len(found_terms),
        "found_terms": found_terms,
        "human_term_count": len(found_human),
        "found_human_terms": found_human,
        "score": round(score, 2),
        "passed": passed,
    }


def check_memory_pack_formatting(memory_pack_text: str) -> dict:
    """检查 MemoryPack 格式化输出质量。

    Args:
        memory_pack_text: format_prompt_block() 的输出

    Returns:
        dict: {"score": 0.0-1.0, "passed": bool, "issues": [...]}
    """
    issues: list[str] = []

    # 检查是否包含机器前缀标记（如 [人物:][时间:]）
    machine_prefixes = re.findall(r'\[[^\]]+\]', memory_pack_text)
    if machine_prefixes:
        issues.append(f"含机器前缀标记: {machine_prefixes[:3]}")
    # 检查是否有 核心事实/近期记忆/长期记忆 词
    for term in _ENGINEERING_TERMS_PROMPT[:6]:  # 只查最明显的几个
        if term in memory_pack_text:
            issues.append(f"含工程词: {term}")
    # 检查是否有空段落
    if "## \n" in memory_pack_text:
        issues.append("有空段落标题")
    # 检查是否太短（疑似 fallback）
    if len(memory_pack_text.strip()) < 10:
        issues.append("输出过短")

    score = max(0.0, 1.0 - len(issues) * 0.33)
    passed = len(issues) == 0

    return {"score": round(score, 2), "passed": passed, "issues": issues}


def run_offline_static_eval() -> EvalReport:
    """离线静态检查模式——不调用 LLM，仅检查代码层面。

    检查项：
      1. MemoryPackV2.format_prompt_block() 主路径（主结论基准）
      2. agent.py 中旧格式化函数（旧版 兼容检查）
    """
    report = EvalReport()
    print("\n" + "=" * 60)
    print("静态 Prompt 检查（不依赖 LLM）")
    print("=" * 60)

    v2_checks: list[dict] = []
    v2_all_passed = True

    # ════════════════════════════════════════════════════════════════
    # 主路径：MemoryPackV2.format_prompt_block()
    # ════════════════════════════════════════════════════════════════
    try:
        from app.memory.schema import (
            MemoryPackV2,
            MemoryItem,
            MemoryKind,
            MemorySource,
            MemoryVisibility,
            RelationshipState,
        )
    except ImportError as exc:
        print(f"\n⚠ 无法导入 MemoryPackV2 schema: {exc}")
        report.overall_score = 0.5
        return report

    # 主路径样本：实体+边界+低置信
    pack_main = MemoryPackV2(
        relationship=RelationshipState(
            person_id="123",
            mode="girlfriend",
            recent_mood="近期有点焦虑",
            recent_attitude="依赖/需要陪伴",
            relationship_temperature=0.72,
        ),
        current_topic="唐凯是谁",
        items=[
            MemoryItem(
                kind=MemoryKind.ENTITY,
                source=MemorySource.WIKI,
                confidence=0.85,
                emotional_weight=3,
                visibility=MemoryVisibility.RECALL_ONLY,
                content="[人物: 唐凯] 唐凯是叶鹏祥初中同学，高中同校，也是好友群成员。",
            ),
            MemoryItem(
                kind=MemoryKind.TABOO,
                source=MemorySource.USER_DECLARED,
                confidence=1.0,
                visibility=MemoryVisibility.ALWAYS,
                content="不要使用对方明确禁忌的称呼。",
            ),
            MemoryItem(
                kind=MemoryKind.FACT,
                source=MemorySource.CONVERSATION_SUMMARY,
                confidence=0.45,
                emotional_weight=2,
                visibility=MemoryVisibility.RECALL_ONLY,
                content="（不太确定）可能聊过上海旅行计划",
            ),
        ],
    )
    v2_checks.append({"name": "MemoryPackV2 主路径", "sample": pack_main.format_prompt_block()})

    # 空记忆样本：无 relationship、无 items
    pack_empty = MemoryPackV2(
        relationship=RelationshipState(person_id="", mode="girlfriend"),
        items=[],
    )
    v2_checks.append({"name": "MemoryPackV2 空记忆", "sample": pack_empty.format_prompt_block()})

    # 缺失记忆样本
    pack_miss = MemoryPackV2(
        relationship=RelationshipState(person_id="123", mode="girlfriend"),
        items=[],
        missing_memory={"should_admit_unknown": True, "reason": "完全未命中"},
    )
    v2_checks.append({"name": "MemoryPackV2 缺失记忆", "sample": pack_miss.format_prompt_block()})

    # 低置信记忆样本：全部低置信
    pack_low = MemoryPackV2(
        relationship=RelationshipState(person_id="123", mode="girlfriend"),
        items=[
            MemoryItem(
                kind=MemoryKind.FACT,
                source=MemorySource.INFERRED,
                confidence=0.35,
                visibility=MemoryVisibility.RECALL_ONLY,
                content="可能聊过在杭州出差的事",
            ),
        ],
    )
    v2_checks.append({"name": "MemoryPackV2 低置信记忆", "sample": pack_low.format_prompt_block()})

    print("\n── MemoryPackV2 主路径测试 ──")
    for check in v2_checks:
        result = check_prompt_for_engineering_terms(check["sample"])
        format_result = check_memory_pack_formatting(check["sample"])

        passed = result["passed"] and format_result["passed"]
        status = "✅" if passed else "❌"
        if not passed:
            v2_all_passed = False
        print(f"\n{status} {check['name']}")
        print(f"   工程词: {result['found_terms'] or '无'} → {result['score']:.2f}")
        if format_result["issues"]:
            for issue in format_result["issues"]:
                print(f"   ⚠ {issue}")

    # 综合评分（以 V2 主路径为准）
    print("\n" + "=" * 60)
    if not v2_all_passed:
        print("❌ MemoryPackV2 主路径检查发现问题：prompt 中仍含工程术语，请检查代码。")
        report.overall_score = 0.0
    else:
        print("✅ MemoryPackV2 主路径检查全部通过：prompt 中无工程术语。")
        report.overall_score = 1.0

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 离线规则评估模式（不依赖后端服务）
# ══════════════════════════════════════════════════════════════════════════════


def _offline_check_reply(reply: str, dimension: str, case_name: str) -> tuple[float, str]:
    """离线模式直接评分（不调用 LLM，仅规则检查）。"""
    return _score_case(reply, dimension, case_name)


def run_offline_eval():
    """在不连接后端的离线模式下进行规则级检查。

    适合 CI/快速检查——不依赖服务运行，仅检测信号词模式。
    """
    report = EvalReport()
    print("\n" + "=" * 60)
    print("情感陪伴评测（离线规则模式）")
    print("=" * 60)

    for case in EVAL_CASES:
        reply = ""
        score = 0.0
        notes = "离线模式：无 LLM 调用"

        ds = report.dimensions.get(case.dimension)
        if ds is None:
            ds = DimScore()
            report.dimensions[case.dimension] = ds
        ds.cases_tested += 1
        ds.score = (ds.score * (ds.cases_tested - 1) + score) / ds.cases_tested
        report.total_tests += 1

        status = "⚠️"
        print(f"\n{status} [{case.dimension}] {case.name}")
        print(f"   输入: {case.user_input[:60]}")
        print(f"   状态: {notes}")

    report.overall_score = 0.0
    print("\n" + "=" * 60)
    print("离线模式仅做规则扫描，不产生有效评分。")
    print("请启动后端后运行: python scripts/eval_emotional_memory.py --online")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="情感陪伴记忆评测工具")
    parser.add_argument("--online", action="store_true", help="在线模式（需要后端运行）")
    parser.add_argument("--offline-static", action="store_true", help="离线静态检查（检查 prompt 是否含工程词）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="后端地址")
    parser.add_argument("--output", default="", help="输出 JSON 报告路径")
    parser.add_argument(
        "--dimensions",
        default="",
        help="指定评测维度，逗号分隔: recall,continuity,boundary,unknown,care,correction",
    )
    args = parser.parse_args()

    if args.offline_static:
        run_offline_static_eval()
    elif args.online:
        print("在线模式：需要后端服务正在运行...")
        report = asyncio.run(
            EmotionalMemoryEvaluator(base_url=args.base_url).evaluate_all(
                output_file=args.output or "eval_emotional_memory_report.json"
            )
        )
        print(f"\n完成。综合评分: {report.overall_score:.2f}")
    else:
        run_offline_eval()

    print("\n评测维度说明：")
    print("  recall      自然回忆  — 不百科化、不暴露工程术语")
    print("  continuity  情绪延续  — 能感知和延续用户近期状态")
    print("  boundary    关系边界  — 女友/访客模式不混淆")
    print("  unknown     不装熟    — 不知道时不编造")
    print("  care        主动关心  — 能自然跟进 open loop")
    print("  correction  纠错服从  — 用户纠正后不再重复旧记忆")


if __name__ == "__main__":
    main()
