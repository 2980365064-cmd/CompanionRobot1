#!/usr/bin/env python3
"""Ingest persona corpus into L3 search index."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest persona corpus into L3 memory index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空旧语料向量后再入库（换 embedding 模型后建议执行）",
    )
    args = parser.parse_args()

    from app.embed_meta import check_embed_compat, save_embed_meta
    from app.llm import embed_provider_name
    from app.memory.semantic import semantic_memory

    files = ingest_directory(reset=args.reset)
    if not files:
        print("No corpus files found. Add markdown under persona/corpus/")
        return
    meta = save_embed_meta()
    ok, msg = check_embed_compat()
    print("Sources:", ", ".join(files))
    print(f"Ingested {semantic_memory.corpus.count()} chunks (provider={embed_provider_name()}, dim={meta.get('dim')})")
    print(msg if ok else f"WARNING: {msg}")


if __name__ == "__main__":
    main()
