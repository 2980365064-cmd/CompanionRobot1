#!/usr/bin/env python3
"""记忆系统诊断工具。

用途：在完成微信导入（import_wechat.py）和语料入库（ingest.py）后，
对 L3 长期记忆系统进行端到端健康检查，确保语料正确入库且检索功能正常。

检查项目：
1. Corpus 目录内容检测 —— 确认 persona/corpus/ 下有语料文件
2. L3 存储统计 —— 查看 corpus chunks 和 facts 的数量
3. Embedding 兼容性 —— 检查当前 embedding 提供商与已存储向量的维度是否匹配
4. 自动入库兜底 —— 如果 L3 为空，自动尝试执行 ingest_directory
5. 检索有效性测试 —— 对一组中文查询执行 memory_router.recall，
   验证 needs_l3_recall 判断和实际命中数量是否正常

典型用法：
    python scripts/diagnose_memory.py     # 全面诊断
    python scripts/diagnose_memory.py     # ingest 后验证

关联脚本：
    - scripts/import_wechat.py  # 先导入微信聊天记录
    - scripts/ingest.py         # 将语料入库到 L3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.embed_meta import check_embed_compat, load_embed_meta
from app.llm import embed_provider_name
from app.memory.guard import needs_l3_recall
from app.memory.router import memory_router
from app.memory.l3 import semantic_memory, _stored_dim
from app.rag import ingest_directory


def main() -> None:
    """执行记忆系统全面诊断。"""
    corpus = settings.resolved_corpus_dir()
    print("=== Memory diagnose ===\n")
    print(f"corpus dir: {corpus}")
    # 列出 corpus 目录中的实际语料文件（排除示例文件）
    print(f"corpus files: {[p.name for p in sorted(corpus.glob('*.md')) if not p.name.endswith('.example')]}")
    print(f"L3 corpus chunks: {semantic_memory.corpus.count()}")
    print(f"L3 facts: {semantic_memory.facts.count()}")
    print(f"Embed provider: {embed_provider_name()}")
    print(f"Stored meta: {load_embed_meta()}")
    print(f"Stored corpus dim: {_stored_dim()}")
    ok, msg = check_embed_compat()
    print(f"Compat: {msg}")
    print(f"L3 backend: {settings.search_backend} (corpus chunks: {semantic_memory.corpus.count()})")

    # embedding 模型不兼容时，提示用户执行 reset rebuild
    if not ok:
        print("\n>>> 请执行: python scripts/ingest.py --reset  然后重启 uvicorn")
        return

    # L3 为空时自动尝试入库，提供友好的中文错误提示
    if semantic_memory.corpus.count() == 0:
        print("\n[!] L3 语料为空。请执行:")
        print("    python scripts/ingest.py")
        print("    若换过 embedding: python scripts/ingest.py --reset")
        try:
            result = ingest_directory()
            print(f"    已尝试自动 ingest，入库文件: {result.get('files')}")
            print(f"    persona facts: {(result.get('fact_stats') or {}).get('facts', 0)}")
            print(f"    现在 corpus chunks: {semantic_memory.corpus.count()}")
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
        # memory_router.recall 会协调多个记忆层（L1/L2/L3）的检索
        mem = memory_router.recall("default", "test", q, person_id="")
        print(
            f"\nQ: {q}"
        )
        print(
            f"  needs_memory={needs_l3_recall(q)} "
            f"L3={mem.get('l3_hit', mem.get('facts_hit'))} "
            f"hits={len(mem.get('l3') or mem.get('semantic') or [])} "
            f"miss={mem.get('memory_miss')}"
        )
        # 打印前 2 条检索到的记忆内容（截断超长文本）
        for i, s in enumerate((mem.get("l3") or mem.get("semantic") or [])[:2], 1):
            print(f"  [{i}] {s[:100]}…" if len(s) > 100 else f"  [{i}] {s}")


if __name__ == "__main__":
    main()
