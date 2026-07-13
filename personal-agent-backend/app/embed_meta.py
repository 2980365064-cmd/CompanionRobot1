"""向量元数据管理 —— 记录并校验当前向量模型的维度和提供商。

本模块的角色：
  长期记忆 长期记忆入库时会调用 embedding API 生成向量。如果后续用户换了向量模型
  （比如从阿里云 DashScope text-embedding-v3 换到其他模型），新旧向量的维度不一致，
  会导致向量检索分数异常甚至报错。本模块在每次语料入库时记录当时的模型信息和维度，
  启动时校验一致性，不匹配则提示用户重建索引。

使用方法：
  - save_embed_meta()  入库后调用，持久化向量元数据到 .embed_meta.json
  - check_embed_compat() 启动时/health 检查时调用，返回 (是否兼容, 说明)
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.llm import embed_provider_name, embed_texts


def _meta_path() -> Path:
    """返回向量元数据文件的路径（与 agent.db 同目录，名为 .embed_meta.json）。"""
    base = settings.resolved_db_path()
    return base.parent / ".embed_meta.json"


def save_embed_meta(sample_text: str = "dimension probe") -> dict:
    """保存向量元数据：用一条探针文本调用 embedding API，记录维度、提供商、模型名。

    参数:
        sample_text: 用于探测的文本，默认 "dimension probe"
    返回:
        dict: {"dim": 向量维度, "provider": 提供商名, "model": 模型名}
    """
    # 用一条固定文本做探针调用，拿到当前 embedding API 的实际输出维度
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
    """加载已保存的向量元数据，文件不存在或格式错误返回 None。"""
    path = _meta_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def current_query_dim() -> int:
    """获取当前配置下 embedding API 输出的向量维度（用于兼容性检查）。"""
    return len(embed_texts(["probe"])[0])


def check_embed_compat() -> tuple[bool, str]:
    """检查入库时的向量维度与当前查询维度是否一致。

    返回:
        (True, msg)  兼容，或虽无元数据但可继续
        (False, msg) 维度不一致，需要重建索引（python scripts/ingest.py --reset）

    为什么需要这个检查：
      如果用户在 .env 中切换了 EMBED_API_KEY 或 EMBED_MODEL，向量维度可能变化，
      导致 sqlite/chroma 中已入库的向量与后续查询向量无法正确比较余弦相似度。
    """
    meta = load_embed_meta()
    qdim = current_query_dim()
    if not meta:
        # 从未记录过入库维度：可能是首次使用或元数据文件被删
        return True, f"未记录入库维度（当前查询 dim={qdim}），建议执行 python scripts/ingest.py --reset"
    sdim = int(meta.get("dim", 0))
    if sdim and sdim != qdim:
        # 维度不匹配：必须重建，否则向量检索结果无意义
        return False, (
            f"向量维度不一致：入库 dim={sdim} ({meta.get('provider')}/{meta.get('model')})，"
            f"当前查询 dim={qdim} ({embed_provider_name()}/{settings.embed_model})。"
            f"请执行: python scripts/ingest.py --reset"
        )
    return True, f"维度一致 dim={qdim}"
