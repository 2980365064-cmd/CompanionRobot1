"""Track embedding model/dimension used for Chroma ingest."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.llm import embed_provider_name, embed_texts


def _meta_path() -> Path:
    return Path(settings.chroma_path) / ".embed_meta.json"


def save_embed_meta(sample_text: str = "dimension probe") -> dict:
    emb = embed_texts([sample_text])[0]
    meta = {
        "dim": len(emb),
        "provider": embed_provider_name(),
        "model": settings.embed_model,
    }
    path = _meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_embed_meta() -> dict | None:
    path = _meta_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def current_query_dim() -> int:
    return len(embed_texts(["probe"])[0])


def check_embed_compat() -> tuple[bool, str]:
    meta = load_embed_meta()
    qdim = current_query_dim()
    if not meta:
        return True, f"未记录入库维度（当前查询 dim={qdim}），建议执行 python scripts/ingest.py --reset"
    sdim = int(meta.get("dim", 0))
    if sdim and sdim != qdim:
        return False, (
            f"向量维度不一致：入库 dim={sdim} ({meta.get('provider')}/{meta.get('model')})，"
            f"当前查询 dim={qdim} ({embed_provider_name()}/{settings.embed_model})。"
            f"请执行: python scripts/ingest.py --reset"
        )
    return True, f"维度一致 dim={qdim}"
