"""后台备份与恢复。"""

from __future__ import annotations

import shutil
import sqlite3
import tarfile
from datetime import datetime
from pathlib import Path

from app.config import settings

BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"


def _backup_id(kind: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kind}"


def list_backups() -> dict:
    BACKUP_DIR.mkdir(exist_ok=True)
    items = []
    for path in sorted(BACKUP_DIR.iterdir(), reverse=True):
        if path.is_file() and path.suffix in (".db", ".tgz"):
            items.append({
                "id": path.stem,
                "name": path.name,
                "kind": "persona" if path.suffix == ".tgz" else "database",
                "size": path.stat().st_size,
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            })
    return {"backups": items, "dir": str(BACKUP_DIR)}


def create_backup(kind: str = "all") -> dict:
    kind = (kind or "all").lower()
    BACKUP_DIR.mkdir(exist_ok=True)
    made = []
    if kind in ("all", "database", "db"):
        db = settings.resolved_db_path()
        out = BACKUP_DIR / f"{_backup_id('agent')}.db"
        if db.exists():
            src = sqlite3.connect(str(db))
            dst = sqlite3.connect(str(out))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            made.append(str(out))
    if kind in ("all", "persona"):
        persona = settings.resolved_persona_dir()
        out = BACKUP_DIR / f"{_backup_id('persona')}.tgz"
        with tarfile.open(out, "w:gz") as tar:
            tar.add(persona, arcname="persona")
        made.append(str(out))
    return {"created": made, **list_backups()}


def restore_backup(backup_id: str) -> dict:
    BACKUP_DIR.mkdir(exist_ok=True)
    candidates = [p for p in BACKUP_DIR.iterdir() if p.stem == backup_id]
    if not candidates:
        raise FileNotFoundError("backup not found")
    path = candidates[0]
    safety = create_backup("all")
    if path.suffix == ".db":
        shutil.copy2(path, settings.resolved_db_path())
        kind = "database"
    elif path.suffix == ".tgz":
        persona = settings.resolved_persona_dir()
        if persona.exists():
            shutil.rmtree(persona)
        persona.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(persona.parent)
        kind = "persona"
    else:
        raise ValueError("unsupported backup")
    return {"restored": backup_id, "kind": kind, "safety_backup": safety.get("created", [])}
