"""统一记忆完整性审计的行为测试。"""

import sys
from pathlib import Path

from scripts.audit_unified_memory_integrity import Finding, report, scan, scan_database


def _retired_table_name() -> str:
    return "l" + "3_chunks"


def _retired_layer_name() -> str:
    return "L" + "2"


def test_scan_classifies_retired_schema_in_application_as_blocker(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "sample.py").write_text(_retired_table_name())

    assert any(item.bucket == "runtime_blocker" for item in scan(tmp_path))


def test_scan_classifies_retired_layer_term_in_test_as_debt(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "sample.py").write_text(_retired_layer_name())

    assert any(item.bucket == "test_debt" for item in scan(tmp_path))


def test_strict_rejects_every_debt_bucket(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "sample.py").write_text(_retired_layer_name())
    from scripts.audit_unified_memory_integrity import main

    previous = sys.argv
    try:
        sys.argv = ["audit", "--strict", "--root", str(tmp_path)]
        assert main() == 1
    finally:
        sys.argv = previous


def test_report_exposes_all_gate_counts():
    data = report([Finding("tests/sample.py", 1, "test_debt", "layer_terminology", "retired", "x")])

    assert data["summary"]["test_debt"] == 1
    assert data["summary"]["runtime_blocker"] == 0


def test_scan_classifies_retired_term_in_persona_as_documentation_debt(tmp_path: Path):
    path = tmp_path / "persona" / "config"
    path.mkdir(parents=True)
    (path / "persona.md").write_text(_retired_layer_name())

    assert any(item.bucket == "documentation_debt" for item in scan(tmp_path))


def test_database_scan_rejects_retired_table(tmp_path: Path):
    database_dir = tmp_path / "personal-agent-backend"
    database_dir.mkdir()
    database = database_dir / "agent.db"
    import sqlite3

    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE memory_items (source_table TEXT, source_id TEXT)")
        conn.execute("CREATE TABLE memory_relations (from_id TEXT, to_id TEXT)")
        conn.execute("CREATE TABLE " + _retired_table_name() + " (id TEXT)")

    assert any(item.rule == "database" for item in scan_database(tmp_path))


def test_scan_rejects_retired_memory_path_name(tmp_path: Path):
    path = tmp_path / "persona" / "corpus" / "archive"
    path.mkdir(parents=True)
    filename = "notes_" + "legacy.md"
    (path / filename).write_text("归档内容")

    assert any(item.rule == "retired_path" for item in scan(tmp_path))
