"""后台智能体自我认知管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import settings
from app.persona.card import clear_profile_cache


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text("utf-8").strip()


def _write_text(path: Path, content: str) -> None:
    body = str(content or "").strip()
    if len(body) < 4:
        raise ValueError("content is too short")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", "utf-8")


def _file_meta(path: Path, content: str) -> dict:
    return {
        "path": str(path),
        "content": content,
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "updated_at": path.stat().st_mtime if path.exists() else None,
    }


@dataclass
class AgentSelfCognitionStore:
    """只管理 persona.md 与 profile_card.md 两个白名单文件。"""

    persona_path: Path
    profile_card_path: Path
    clear_cache: Callable[[], None] = clear_profile_cache

    def load(self) -> dict:
        persona = _read_text(self.persona_path)
        profile_card = _read_text(self.profile_card_path)
        return {
            "persona": _file_meta(self.persona_path, persona),
            "profile_card": _file_meta(self.profile_card_path, profile_card),
            "hint": "persona.md 是源设定；profile_card.md 是每轮对话实际优先注入的摘要卡。",
        }

    def save(
        self,
        *,
        persona_text: str | None = None,
        profile_card_text: str | None = None,
    ) -> dict:
        if persona_text is None and profile_card_text is None:
            raise ValueError("persona_text or profile_card_text required")
        if persona_text is not None:
            _write_text(self.persona_path, persona_text)
        if profile_card_text is not None:
            _write_text(self.profile_card_path, profile_card_text)
        self.clear_cache()
        return self.load()


def default_self_cognition_store() -> AgentSelfCognitionStore:
    return AgentSelfCognitionStore(
        persona_path=settings.resolved_persona_path(),
        profile_card_path=settings.resolved_profile_card_path(),
    )
