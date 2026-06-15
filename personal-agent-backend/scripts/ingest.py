#!/usr/bin/env python3
"""语料入库命令行工具。

用途：将 persona/corpus/ 目录下的 Markdown 语料文件经过清洗、分块后，
写入 L3 向量搜索引擎（SQLite + FTS5 + 向量 hybrid），使陪伴机器人
能够从长期记忆中检索相关知识来回答问题。

主要功能：
  - 扫描 corpus 目录，将所有 .md/.txt 语料入库到 L3
  - 支持 --reset 选项清空旧数据后重建（切换 embedding 模型后必需）
  - 支持 --no-facts 选项跳过结构化事实提取（仅入库原始语料块）
  - 入库后自动验证 embedding 兼容性并输出统计信息

典型用法：
    python scripts/ingest.py                    # 首次/增量入库
    python scripts/ingest.py --reset            # 清空后全量重建
    python scripts/ingest.py --reset --no-facts # 仅重建向量，不提取结构化事实

前置依赖：
    - persona/corpus/ 目录下需要有 .md 或 .txt 语料文件
    - LLM_API_KEY 和 EMBED_API_KEY 需要已配置（用于向量化和可选的事实提取）
    - 如果换过 embedding 模型，必须使用 --reset 清空旧向量

关联脚本：
    - scripts/import_wechat.py  # 先将微信聊天转为语料文件
    - scripts/diagnose_memory.py # 入库后验证 L3 健康状态
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import ingest_directory


def main() -> None:
    """解析命令行参数并执行语料入库流程。"""
    parser = argparse.ArgumentParser(description="Ingest persona corpus into L3 memory index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空旧语料向量后再入库（换 embedding 模型后建议执行）",
    )
    parser.add_argument(
        "--no-facts",
        action="store_true",
        help="仅入库 Corpus 向量，不批量提取 Facts / 建关联网",
    )
    args = parser.parse_args()

    from app.embed_meta import check_embed_compat, save_embed_meta
    from app.llm import embed_provider_name
    from app.memory.l3 import clear_persona_derived_memory, semantic_memory

    # 如果指定了 reset，先清空 corpus 和关联的 persona 派生数据（facts/relations）
    if args.reset:
        semantic_memory.reset_corpus()
        cleared = clear_persona_derived_memory()
        print(f"Cleared persona facts/relations: {cleared}")

    # 执行语料入库（reset 确保空库重建，extract_facts 由命令行参数控制）
    result = ingest_directory(reset=args.reset, extract_facts=not args.no_facts)
    files = result.get("files") or []
    if not files:
        print("No corpus files found. Add markdown under persona/corpus/")
        return

    # 保存当前 embedding 元信息（provider + 维度），用于后续兼容性检查
    meta = save_embed_meta()
    ok, msg = check_embed_compat()
    fs = result.get("fact_stats") or {}
    print("Sources:", ", ".join(files))
    print(
        f"Ingested {semantic_memory.corpus.count()} corpus chunks "
        f"(provider={embed_provider_name()}, dim={meta.get('dim')})"
    )
    print(
        f"Facts extracted: {fs.get('facts', 0)} from {fs.get('chunks', 0)} chunks "
        f"(skipped {fs.get('skipped', 0)} short chunks)"
    )
    print(msg if ok else f"WARNING: {msg}")


if __name__ == "__main__":
    main()
