#!/usr/bin/env python3
"""重建 corpus 记忆条目 —— 将旧脏数据按新规范全量或局部重建。

新规范（2026-07 收口版）：
  - source_table='corpus'
  - source_id='<relative_path>#s<section>p<part>' （稳定主键，支持幂等 upsert）
  - kind='wiki', source='wiki'
  - context_json 包含 source_path 和 month_key

清理规则（安全设计）：
  - 全量模式：删除 source_table='corpus' 的全部条目 + 旧脏数据
  - 局部模式（--only-path）：只删除目标文件相关的 corpus 条目
  - 绝不碰 person_id 非空的用户运行时记忆
  - 绝不碰 person_profiles / memory_relations / relationship_states

用法：
    python scripts/rebuild_corpus_memory_items.py              # 默认全量重建
    python scripts/rebuild_corpus_memory_items.py --dry-run    # 仅报告，不落库
    python scripts/rebuild_corpus_memory_items.py --backup     # 重建前备份 agent.db
    python scripts/rebuild_corpus_memory_items.py --only-path monthly/liu_yuanhui/2025-04.md  # 单文件局部重建
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.session import store


def _backup_db() -> Path | None:
    """备份 agent.db 到同目录下的 .bak 文件。"""
    db = settings.resolved_db_path()
    if not db.exists():
        print(f"[BACKUP] 数据库不存在: {db}，跳过备份")
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = db.with_name(f"agent_corpus_rebuild_{ts}.db.bak")
    shutil.copy2(str(db), str(backup))
    size_mb = backup.stat().st_size / 1024 / 1024
    print(f"[BACKUP] 已备份: {backup} ({size_mb:.1f} MB)")
    return backup


def _count_dirty_corpus() -> dict:
    """统计待清理的脏 corpus 行数和类型。"""
    import sqlite3
    db = settings.resolved_db_path()
    if not db.exists():
        return {"corpus_clean_rows": 0, "corpus_dirty_rows": 0, "total_memory_items": 0}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        new_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_items WHERE source_table='corpus'"
        ).fetchone()
        new_count = int(new_rows["n"]) if new_rows else 0

        dirty_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_items "
            "WHERE (person_id IS NULL OR person_id = '') AND kind='wiki'"
            "  AND (source_table IS NULL OR source_table = '')"
        ).fetchone()
        dirty_count = int(dirty_rows["n"]) if dirty_rows else 0

        total_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_items"
        ).fetchone()
        total_count = int(total_rows["n"]) if total_rows else 0
    finally:
        conn.close()

    return {
        "corpus_clean_rows": new_count,
        "corpus_dirty_rows": dirty_count,
        "total_memory_items": total_count,
    }


def _delete_dirty_corpus() -> dict:
    """删除全部 corpus 相关条目（新规范 + 旧脏数据）。

    绝不会删除：
      - person_id 非空的用户运行时记忆
      - person_profiles / memory_relations / relationship_states
    """
    import sqlite3
    db = settings.resolved_db_path()
    if not db.exists():
        return {"deleted_clean_corpus": 0, "deleted_dirty_corpus": 0, "total_deleted": 0}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id FROM memory_items WHERE source_table='corpus'"
        ).fetchall()
        ids1 = {str(r["id"]) for r in rows}

        rows2 = conn.execute(
            "SELECT id FROM memory_items "
            "WHERE (person_id IS NULL OR person_id = '') AND kind='wiki'"
            "  AND (source_table IS NULL OR source_table = '')"
        ).fetchall()
        ids2 = {str(r["id"]) for r in rows2}

        all_ids = list(ids1 | ids2)

        if all_ids:
            placeholders = ",".join("?" * len(all_ids))
            conn.execute(
                f"DELETE FROM memory_items_fts WHERE id IN ({placeholders})",
                all_ids,
            )
            conn.execute(
                f"DELETE FROM memory_items WHERE id IN ({placeholders})",
                all_ids,
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "deleted_clean_corpus": len(ids1),
        "deleted_dirty_corpus": len(ids2),
        "total_deleted": len(all_ids),
    }


def _delete_corpus_by_path(source_path: str) -> dict:
    """仅删除与指定源文件相关的 corpus 条目（新规范 + 旧脏数据）。

    绝不会删除其他文件或用户记忆。

    Returns:
        dict: {deleted_new: int, deleted_dirty: int, total_deleted: int}
    """
    import sqlite3
    db = settings.resolved_db_path()
    if not db.exists():
        return {"deleted_new": 0, "deleted_dirty": 0, "total_deleted": 0}

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        # 新规范：source_id LIKE '<path>#%'
        rows = conn.execute(
            "SELECT id FROM memory_items "
            "WHERE source_table='corpus' AND source_id LIKE ?",
            (f"{source_path}#%",),
        ).fetchall()
        ids_new = {str(r["id"]) for r in rows}

        # 旧脏数据：person_id='' AND kind='wiki' AND source_table=''
        # 尝试匹配 source 或 context_json 中的路径信息
        pattern = f"%{source_path}%"
        rows2 = conn.execute(
            "SELECT id FROM memory_items "
            "WHERE (person_id IS NULL OR person_id = '') AND kind='wiki'"
            "  AND (source_table IS NULL OR source_table = '')"
            "  AND (source LIKE ? OR context_json LIKE ?)",
            (pattern, pattern),
        ).fetchall()
        ids_dirty = {str(r["id"]) for r in rows2}

        all_ids = list(ids_new | ids_dirty)

        if all_ids:
            placeholders = ",".join("?" * len(all_ids))
            conn.execute(
                f"DELETE FROM memory_items_fts WHERE id IN ({placeholders})",
                all_ids,
            )
            conn.execute(
                f"DELETE FROM memory_items WHERE id IN ({placeholders})",
                all_ids,
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "deleted_new": len(ids_new),
        "deleted_dirty": len(ids_dirty),
        "total_deleted": len(all_ids),
    }


def _audit_single_file(source_path: str) -> dict:
    """单文件审计：只校验该文件的 expected/actual/duplicate 是否完整。"""
    from app.persona.ingest import build_corpus_chunk_specs, audit_corpus_sync_state

    corpus_dir = settings.resolved_corpus_dir()
    if not corpus_dir.exists():
        return {"file_found": False, "error": "corpus_dir not found"}

    # 扫描该文件的预期 specs
    all_specs = build_corpus_chunk_specs(corpus_dir)
    file_specs = [s for s in all_specs if s.get("source_path") == source_path]
    expected_ids = sorted(s["source_id"] for s in file_specs if s.get("source_id"))

    # DB 中该文件的实际条目
    actual_ids = sorted(store.list_corpus_source_ids_by_path(source_path))

    # 去重检测
    from collections import Counter
    actual_counts = Counter(actual_ids)
    duplicate_ids = sorted(sid for sid, cnt in actual_counts.items() if cnt > 1)

    missing = sorted(set(expected_ids) - set(actual_ids))
    stale = sorted(set(actual_ids) - set(expected_ids))

    return {
        "file_found": bool(file_specs),
        "source_path": source_path,
        "expected_count": len(expected_ids),
        "actual_count": len(set(actual_ids)),
        "actual_row_count": len(actual_ids),
        "missing_source_ids": missing,
        "stale_source_ids": stale,
        "duplicate_source_ids": duplicate_ids,
        "is_complete": (
            len(missing) == 0
            and len(stale) == 0
            and len(duplicate_ids) == 0
            and len(actual_ids) == len(expected_ids)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重建 corpus 记忆条目（全量 → 新规范）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅报告不落库",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="重建前备份 agent.db",
    )
    parser.add_argument(
        "--only-path",
        type=str,
        default="",
        help="单文件局部重建（仅删除并重建该文件的 corpus 条目）",
    )
    args = parser.parse_args()

    # ── 阶段 0：统计 ──
    counts = _count_dirty_corpus()
    print("=" * 60)
    print("corpus 重建前统计")
    print("=" * 60)
    print(f"  新规范 corpus 行:      {counts['corpus_clean_rows']}")
    print(f"  旧脏数据（wiki）行:     {counts['corpus_dirty_rows']}")
    print(f"  memory_items 总条数:    {counts['total_memory_items']}")
    print()

    corpus_dir = settings.resolved_corpus_dir()
    corpus_files = list(corpus_dir.rglob("*.md")) if corpus_dir.exists() else []
    print(f"  corpus 目录源文件数:    {len(corpus_files)} 个")
    print()

    if args.only_path:
        matched = [f for f in corpus_files if f.relative_to(corpus_dir).as_posix() == args.only_path]
        print(f"  目标文件: {args.only_path} {'✓ 存在' if matched else '✗ 不存在'}")
        print()

    if counts["total_memory_items"] == 0 and not args.only_path:
        print("[SKIP] memory_items 为空，无需重建")
        return

    # ══════════════════════════════════════════════════════════════════════
    # 局部模式（--only-path）：仅重建单个文件
    # ══════════════════════════════════════════════════════════════════════
    if args.only_path:
        print("=" * 60)
        print(f"局部重建: {args.only_path}")
        print("=" * 60)

        if args.dry_run:
            print()
            print("[DRY-RUN] 将删除该文件的相关 corpus 行:")
            # 统计该文件在 DB 中的条目
            path_new = len(store.list_corpus_source_ids_by_path(args.only_path))
            print(f"  新规范: {path_new} 条（source_id LIKE '{args.only_path}#%'）")
            print()
            print("[DRY-RUN] 将重建该文件的 specs 并同步")
            print()
            print("[DRY-RUN] 完成（未实际修改）")
            return

        # 阶段 1：局部备份
        if args.backup:
            backup_path = _backup_db()
            if backup_path:
                print()
        else:
            print("[SKIP] 未指定 --backup，跳过备份")
            print()

        # 阶段 2：局部删除
        del_result = _delete_corpus_by_path(args.only_path)
        print(f"  删除新规范行: {del_result['deleted_new']}")
        print(f"  删除旧脏数据: {del_result['deleted_dirty']}")
        print(f"  合计删除:     {del_result['total_deleted']}")
        print()

        # 阶段 3：局部重建
        from app.persona.ingest import build_corpus_chunk_specs, sync_corpus_chunk_specs

        all_specs = build_corpus_chunk_specs(corpus_dir)
        file_specs = [s for s in all_specs if s.get("source_path") == args.only_path]

        if not file_specs:
            print(f"[WARN] 未找到文件 {args.only_path} 的 specs，跳过同步")
        else:
            print(f"  为 {args.only_path} 重建 {len(file_specs)} 个 chunk...")
            sync_stats = sync_corpus_chunk_specs(file_specs, reset=False)
            print(f"  写入:    {sync_stats.get('written', 0)}")
            print(f"  错误:    {sync_stats.get('errors', 0)}")
            print(f"  最终行:  {sync_stats.get('final_corpus_rows', '?')}")
            print(f"  最终块:  {sync_stats.get('final_corpus_ids', '?')}")
            print()

        # 阶段 4：局部审计
        print("=" * 60)
        print("局部审计")
        print("=" * 60)
        audit = _audit_single_file(args.only_path)
        print(f"  文件: {audit.get('source_path', '?')}")
        print(f"  期望块数: {audit.get('expected_count', 0)}")
        print(f"  实际块数: {audit.get('actual_count', 0)}")
        print(f"  物理行数: {audit.get('actual_row_count', 0)}")
        print(f"  缺失: {len(audit.get('missing_source_ids', []))} 条")
        print(f"  多余: {len(audit.get('stale_source_ids', []))} 条")
        print(f"  重复: {len(audit.get('duplicate_source_ids', []))} 个 source_id")
        print(f"  is_complete: {audit.get('is_complete', False)}")
        print()

        if audit.get("is_complete"):
            print(f"[OK] 文件 {args.only_path} 重建完成，审计通过！")
        else:
            print(f"[WARN] 文件 {args.only_path} 重建后审计不完整")
            if audit.get("missing_source_ids"):
                for sid in audit["missing_source_ids"][:3]:
                    print(f"    缺失: {sid}")
            if audit.get("stale_source_ids"):
                for sid in audit["stale_source_ids"][:3]:
                    print(f"    多余: {sid}")
            sys.exit(1)

        return

    # ══════════════════════════════════════════════════════════════════════
    # 全量模式
    # ══════════════════════════════════════════════════════════════════════

    if counts["total_memory_items"] == 0:
        print("[SKIP] memory_items 为空，无需重建")
        return

    # ── 阶段 1：备份 ──
    if args.backup and not args.dry_run:
        backup_path = _backup_db()
        if backup_path:
            print()
    elif args.backup and args.dry_run:
        print("[DRY-RUN] 跳过备份（--dry-run 模式）")
        print()

    # ── 阶段 2：删除旧 corpus（dry-run 下跳过） ──
    if args.dry_run:
        print("[DRY-RUN] 将删除:")
        print(f"  新规范 corpus 行: {counts['corpus_clean_rows']} 条")
        print(f"  旧脏数据行:       {counts['corpus_dirty_rows']} 条")
        print(f"  合计:             {counts['corpus_clean_rows'] + counts['corpus_dirty_rows']} 条")
        print()

        print("[DRY-RUN] 将重建:")
        print(f"  corpus 源文件:  {len(corpus_files)} 个")
        print()
        print("[DRY-RUN] 完成（未实际修改）")
        return

    # ── 阶段 2（实际执行）：删除 ──
    del_result = _delete_dirty_corpus()
    print("=" * 60)
    print("删除结果")
    print("=" * 60)
    print(f"  删除新规范 corpus:  {del_result['deleted_clean_corpus']} 条")
    print(f"  删除旧脏数据:       {del_result['deleted_dirty_corpus']} 条")
    print(f"  合计删除:           {del_result['total_deleted']} 条")
    print()

    # ── 阶段 3：重建 ──
    from app.persona.ingest import audit_corpus_sync_state, ingest_directory

    print("=" * 60)
    print("全量重建 corpus...")
    print("=" * 60)

    result = ingest_directory(reset=False)
    sync_stats = result.get("sync_stats", {})

    print(f"  写入:     {sync_stats.get('written', 0)}")
    print(f"  错误:     {sync_stats.get('errors', 0)}")
    print(f"  过期删除:  {sync_stats.get('stale_deleted', 0)}")
    print(f"  去重:     {sync_stats.get('dedup_removed', 0)} 行")
    print(f"  total:    {sync_stats.get('total_specs', 0)}")
    print(f"  最终行:   {sync_stats.get('final_corpus_rows', '?')}")
    print(f"  最终块:   {sync_stats.get('final_corpus_ids', '?')}")
    print(f"  源文件:   {len(result.get('files', []))} 个")
    print()

    # ── 阶段 4：审计验证 ──
    print("=" * 60)
    print("重建后审计")
    print("=" * 60)
    audit = audit_corpus_sync_state()
    print(f"  is_complete:          {audit['is_complete']}")
    print(f"  expected_chunks:      {audit['expected_chunk_count']}")
    print(f"  actual_chunks:        {audit['actual_chunk_count']}")
    print(f"  actual_rows:          {audit.get('actual_row_count', '?')}")
    if audit.get("duplicate_source_ids"):
        print(f"  重复 source_id:       {len(audit['duplicate_source_ids'])} 个")
    if audit["missing_source_ids"]:
        print(f"  缺失: {len(audit['missing_source_ids'])} 条")
        for sid in audit["missing_source_ids"][:5]:
            print(f"    - {sid}")
        if len(audit["missing_source_ids"]) > 5:
            print(f"    ... 还有 {len(audit['missing_source_ids']) - 5} 条")
    if audit["stale_source_ids"]:
        print(f"  多余: {len(audit['stale_source_ids'])} 条")
        for sid in audit["stale_source_ids"][:5]:
            print(f"    - {sid}")
        if len(audit["stale_source_ids"]) > 5:
            print(f"    ... 还有 {len(audit['stale_source_ids']) - 5} 条")
    print()

    if audit["is_complete"]:
        print("[OK] corpus 重建完成，审计通过！")
    else:
        print("[WARN] corpus 重建后审计不完整，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
