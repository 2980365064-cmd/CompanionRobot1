#!/usr/bin/env python3
"""Check L3 ingest + recall (run after import_wechat + ingest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.embed_meta import check_embed_compat, load_embed_meta
from app.llm import embed_provider_name
from app.memory.intent import needs_l3_recall
from app.memory.router import memory_router
from app.memory.semantic import semantic_memory, _stored_dim
from app.rag import ingest_directory


def main() -> None:
    corpus = settings.resolved_corpus_dir()
    print("=== Memory diagnose ===\n")
    print(f"corpus dir: {corpus}")
    print(f"corpus files: {[p.name for p in sorted(corpus.glob('*.md')) if not p.name.endswith('.example')]}")
    print(f"L3 corpus chunks: {semantic_memory.corpus.count()}")
    print(f"L3 facts: {semantic_memory.facts.count()}")
    print(f"Embed provider: {embed_provider_name()}")
    print(f"Stored meta: {load_embed_meta()}")
    print(f"Stored corpus dim: {_stored_dim(semantic_memory.corpus)}")
    ok, msg = check_embed_compat()
    print(f"Compat: {msg}")
    print(f"L3 mode: {settings.l3_recall_mode} (light_k={settings.l3_light_top_k}, rag_k={settings.rag_top_k})")
    if not ok:
        print("\n>>> 请执行: python scripts/ingest.py --reset  然后重启 uvicorn")
        return

    if semantic_memory.corpus.count() == 0:
        print("\n[!] 向量库为空。请执行:")
        print("    python scripts/ingest.py")
        print("    若换过 embedding: python scripts/ingest.py --reset")
        try:
            n = ingest_directory()
            print(f"    已尝试自动 ingest，入库文件: {n}")
            print(f"    现在 corpus chunks: {semantic_memory.corpus.count()}")
        except Exception as e:
            print(f"    自动 ingest 失败: {e}")
        return

    queries = [
        "周末一般干嘛？",
        "还记得我们去哪见面吗？",
        "我女朋友叫什么？",
        "今天天气怎么样",
    ]
    print("\n--- 检索测试 ---")
    for q in queries:
        mem = memory_router.recall("default", "test", q)
        print(f"\nQ: {q}")
        print(f"  intent={needs_l3_recall(q)} L3={mem['l3_triggered']} hits={len(mem['semantic'])}")
        for i, s in enumerate(mem["semantic"][:2], 1):
            print(f"  [{i}] {s[:100]}…" if len(s) > 100 else f"  [{i}] {s}")


if __name__ == "__main__":
    main()
