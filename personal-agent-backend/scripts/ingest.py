#!/usr/bin/env python3
"""语料入库命令行工具。

用途：将 persona/corpus/ 目录下的 Markdown 语料文件经过清洗、分块后，
写入长期记忆（SQLite + FTS5 + 向量 hybrid），使陪伴机器人
能够从长期记忆中检索相关知识来回答问题。

主要功能：
  - 扫描 corpus 目录，将所有 .md/.txt 语料入库到长期记忆
  - 支持 --reset 选项清空旧数据后重建（切换 embedding 模型后必需）
  - 入库后自动验证 embedding 兼容性并输出统计信息

典型用法：
    python scripts/ingest.py                    # 首次/增量入库
    python scripts/ingest.py --reset            # 清空后全量重建

前置依赖：
    - persona/corpus/ 目录下需要有 .md 或 .txt 语料文件
    - LLM_API_KEY 和 EMBED_API_KEY 需要已配置
    - 如果换过 embedding 模型，必须使用 --reset 清空旧向量

关联脚本：
    - scripts/import_wechat.py  # 先将微信聊天转为语料文件
    - scripts/diagnose_memory.py # 入库后验证记忆健康状态
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.ingest import ingest_directory


def main() -> None:
    """解析命令行参数并执行语料入库流程。"""
    parser = argparse.ArgumentParser(description="Ingest persona corpus into memory index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空旧语料向量后再入库（换 embedding 模型后建议执行）",
    )
    args = parser.parse_args()

    from app.embed_meta import check_embed_compat, save_embed_meta
    from app.llm import embed_provider_name
    from app.memory.long_term_memory import clear_derived_memory, long_term_memory

    if args.reset:
        long_term_memory.reset_corpus()
        cleared = clear_derived_memory()
        print(f"Cleared persona data: {cleared}")

    result = ingest_directory(reset=args.reset, extract_facts=False)
    files = result.get("files") or []
    if not files:
        print("No corpus files found. Add markdown under persona/corpus/")
        return

    meta = save_embed_meta()
    ok, msg = check_embed_compat()
    print("Sources:", ", ".join(files))
    print(
        f"Ingested {long_term_memory.count_chunks()} chunks "
        f"(provider={embed_provider_name()}, dim={meta.get('dim')})"
    )
    print(msg if ok else f"WARNING: {msg}")


if __name__ == "__main__":
    main()
