#!/usr/bin/env python3
"""记忆系统诊断工具。

用途：在完成微信导入（import_wechat.py）和语料入库（ingest.py）后，
对 长期记忆 长期记忆系统进行端到端健康检查，确保语料正确入库且检索功能正常。

检查项目：
1. Corpus 目录内容检测 —— 确认 persona/corpus/ 下有语料文件
2. 长期记忆 存储统计 —— 查看 corpus chunks 和 facts 的数量
3. Embedding 兼容性 —— 检查当前 embedding 提供商与已存储向量的维度是否匹配
4. 自动入库兜底 —— 如果 长期记忆 为空，自动尝试执行 ingest_directory
5. 检索有效性测试 —— 对一组中文查询执行 memory_router.recall，
   验证 needs_memory_recall 判断和实际命中数量是否正常

典型用法：
    python scripts/diagnose_memory.py     # 全面诊断
    python scripts/diagnose_memory.py     # ingest 后验证

关联脚本：
    - scripts/import_wechat.py  # 先导入微信聊天记录
    - scripts/ingest.py         # 将语料入库到 长期记忆
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.embed_meta import check_embed_compat, load_embed_meta
from app.llm import embed_provider_name
from app.memory.guard import needs_memory_recall
from app.memory.router import memory_router
from app.memory.long_term_memory import long_term_memory, _stored_dim
from app.persona.ingest import ingest_directory


def main() -> None:
    """执行记忆系统全面诊断。"""
    corpus = settings.resolved_corpus_dir()
    print("=== Memory diagnose ===\n")
    print(f"corpus dir: {corpus}")
    # 列出 corpus 目录中的实际语料文件（排除示例文件）
    print(f"corpus files: {[p.name for p in sorted(corpus.glob('*.md')) if not p.name.endswith('.example')]}")
    print(f"长期记忆 corpus chunks: {long_term_memory.count_chunks()}")
    print(f"长期记忆 facts: {0}")
    print(f"Embed provider: {embed_provider_name()}")
    print(f"Stored meta: {load_embed_meta()}")
    print(f"Stored corpus dim: {_stored_dim()}")
    ok, msg = check_embed_compat()
    print(f"Compat: {msg}")
    print(f"长期记忆 backend: {settings.search_backend} (corpus chunks: {long_term_memory.count_chunks()})")

    # embedding 模型不兼容时，提示用户执行 reset rebuild
    if not ok:
        print("\n>>> 请执行: python scripts/ingest.py --reset  然后重启 uvicorn")
        return

    # 长期记忆 为空时自动尝试入库，提供友好的中文错误提示
    if long_term_memory.count_chunks() == 0:
        print("\n[!] 长期记忆 语料为空。请执行:")
        print("    python scripts/ingest.py")
        print("    若换过 embedding: python scripts/ingest.py --reset")
        try:
            result = ingest_directory()
            print(f"    已尝试自动 ingest，入库文件: {result.get('files')}")
            print(f"    persona facts: {(result.get('fact_stats') or {}).get('facts', 0)}")
            print(f"    现在 corpus chunks: {long_term_memory.count_chunks()}")
        except Exception as e:
            print(f"    自动 ingest 失败: {e}")
        return

    # 对一组中文查询进行检索测试，覆盖日常、记忆、关系等场景
    queries = [
        "周末一般干嘛？",
        "还记得我们去哪见面吗？",
        "我女朋友叫什么？",
        "今天天气怎么样",
    ]
    print("\n--- 检索测试 ---")
    for q in queries:
        # memory_router.recall 会协调多个记忆层（工作上下文/近期记忆/长期记忆）的检索
        mem = memory_router.recall("default", "test", q, person_id="")
        print(
            f"\nQ: {q}"
        )
        diag = mem.get("diagnostics", {})
        items = mem.get("items", [])
        print(
            f"  needs_memory={needs_memory_recall(q)} "
            f"long_term={diag.get('has_long_term', 0)} "
            f"hits={len(items)} "
            f"miss={mem.get('memory_miss')}"
        )
        # 打印前 2 条检索到的记忆内容（截断超长文本）
        for i, s in enumerate(items[:2], 1):
            text = str(getattr(s, "content", "") or s.get("content", "")) if isinstance(s, dict) else str(s)
            print(f"  [{i}] {text[:100]}…" if len(text) > 100 else f"  [{i}] {text}")


if __name__ == "__main__":
    main()
