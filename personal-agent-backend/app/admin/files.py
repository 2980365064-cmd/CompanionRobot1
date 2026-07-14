"""后台 Persona/语料文件管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import settings


@dataclass
class AdminFileStore:
    roots: dict[str, Path]

    def __post_init__(self) -> None:
        self.roots = {k: v.resolve() for k, v in self.roots.items()}

    def resolve(self, virtual_path: str) -> Path:
        clean = str(virtual_path or "").strip().replace("\\", "/").lstrip("/")
        if "/" not in clean:
            raise ValueError("invalid file path")
        root_key, rel = clean.split("/", 1)
        root = self.roots.get(root_key)
        if not root:
            raise ValueError("unknown file root")
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            raise ValueError("path traversal rejected")
        return target

    def list_files(self) -> list[dict]:
        items: list[dict] = []
        for key, root in self.roots.items():
            if not root.exists():
                continue
            if root.is_file():
                items.append(_file_item(key, root, root.parent))
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".md", ".txt", ".json"):
                    items.append(_file_item(key, path, root))
        return items

    def read(self, virtual_path: str) -> dict:
        path = self.resolve(virtual_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError("file not found")
        return {
            "path": virtual_path,
            "name": path.name,
            "content": path.read_text("utf-8"),
            "size": path.stat().st_size,
            "updated_at": path.stat().st_mtime,
        }

    def write(self, virtual_path: str, content: str) -> dict:
        path = self.resolve(virtual_path)
        if path.suffix.lower() not in (".md", ".txt", ".json"):
            raise ValueError("unsupported file type")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content or ""), "utf-8")
        return self.read(virtual_path)

    def delete(self, virtual_path: str) -> dict:
        path = self.resolve(virtual_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError("file not found")
        path.unlink()
        return {"deleted": True, "path": virtual_path}


# ── 分类规则 ──────────────────────────────────────────────────

CLASSIFICATION_RULES: list[tuple[str, str, str, str, int]] = [
    # (path_prefix, group_key, group_name, description, sort_order)
    ("config/persona.md",      "persona",       "人格设定",   "机器人身份、性格、表达边界定义", 1),
    ("config/profile_card.md", "profile_card",  "Profile Card", "每轮对话固定注入的人设摘要", 2),
    ("style/",                 "style",         "口吻风格",   "语气、句式、聊天习惯样例", 3),
    ("corpus/people/",         "people",        "人物语料",   "实名用户和第三方人物的长期资料", 4),
    ("corpus/preferences/",    "preferences",   "偏好禁忌",   "喜好、雷区、价值观与原则", 5),
    ("corpus/open_loops/",     "open_loops",    "开放事项",   "未完成事项、持续关注点", 6),
    ("corpus/archive/",        "archive",       "历史归档",   "旧语料、历史总结、迁移保留内容", 7),
]

FALLBACK_RULES: list[tuple[str, str, str, str, int]] = [
    (".example",               "templates",     "示例模板",   "参考格式，不一定参与真实记忆", 8),
    (".json",                  "json_config",   "配置数据",   "结构化分析或系统配置数据", 9),
]

SAMPLE_PATTERNS = ("sample_", "template", "example_")


def classify_virtual_path(virtual_path: str, name: str) -> dict:
    """根据虚拟路径决定文件分类元数据。"""
    for prefix, gk, gn, desc, order in CLASSIFICATION_RULES:
        if virtual_path.startswith(prefix) or virtual_path == prefix.rstrip("/"):
            return {"group": gn, "group_key": gk, "description": desc, "sort_order": order}

    for suffix, gk, gn, desc, order in FALLBACK_RULES:
        if name.endswith(suffix):
            return {"group": gn, "group_key": gk, "description": desc, "sort_order": order}

    if any(name.lower().startswith(p) for p in SAMPLE_PATTERNS):
        return {"group": "templates", "group_key": "templates",
                "description": "参考格式，不一定参与真实记忆", "sort_order": 8}

    return {"group": "其他", "group_key": "other",
            "description": "未分类文件", "sort_order": 99}


def _file_item(root_key: str, path: Path, base: Path) -> dict:
    rel = path.relative_to(base)
    virtual = f"{root_key}/{rel.as_posix()}"
    cls = classify_virtual_path(virtual, path.name)
    return {
        "path": virtual,
        "name": path.name,
        "root": root_key,
        "ext": path.suffix.lower(),
        "size": path.stat().st_size,
        "updated_at": path.stat().st_mtime,
        **cls,
    }


def default_file_store() -> AdminFileStore:
    persona_dir = settings.resolved_persona_dir()
    return AdminFileStore({
        "config": persona_dir / "config",
        "style": settings.resolved_style_dir(),
        "corpus": settings.resolved_corpus_dir(),
    })
