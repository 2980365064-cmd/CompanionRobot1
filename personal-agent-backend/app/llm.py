"""Embedding and LLM clients."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Iterable

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# 阿里云 DashScope 兼容模式
DASHSCOPE_EMBED_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_EMBED_MODEL = "text-embedding-v3"


def _embed_client() -> OpenAI:
    if not settings.embed_api_key:
        raise ValueError("EMBED_API_KEY is not set")
    base = settings.embed_base_url or DASHSCOPE_EMBED_BASE
    return OpenAI(api_key=settings.embed_api_key, base_url=base)


def _llm_client() -> OpenAI:
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url or None)


_EMBED_BATCH_SIZE = 10  # DashScope compatible-mode limit


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not _should_use_remote_embed():
        logger.warning("EMBED_API_KEY not set, using local fallback vectors (re-run ingest after配置阿里云 Key)")
        return [_fallback_embed(t) for t in texts]
    try:
        client = _embed_client()
        model = settings.embed_model or DASHSCOPE_EMBED_MODEL
        out: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            resp = client.embeddings.create(model=model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out
    except Exception as e:
        err = str(e)
        if "401" in err or "Authentication" in err or "InvalidApiKey" in err:
            logger.error("DashScope API Key 无效，请检查 .env 的 EMBED_API_KEY")
        elif "402" in err or "余额" in err or "Arrearage" in err:
            logger.error("DashScope 账户余额不足，请到 dashscope.console.aliyun.com 充值")
        else:
            logger.error("DashScope embedding failed: %s", e)
        return [_fallback_embed(t) for t in texts]


def _should_use_remote_embed() -> bool:
    """配置了阿里云 EMBED_API_KEY 时使用远程向量，否则本地 fallback。"""
    return bool(settings.embed_api_key)


def embed_provider_name() -> str:
    if settings.embed_api_key and "dashscope" in (settings.embed_base_url or DASHSCOPE_EMBED_BASE):
        return "aliyun_dashscope"
    if settings.embed_api_key:
        return "custom"
    return "local_fallback"


def _fallback_embed(text: str, dim: int = 256) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = digest[i % len(digest)]
        vals.append((b / 127.5) - 1.0)
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def chat_completion(messages: list[dict], *, temperature: float | None = None) -> str:
    if not settings.llm_api_key:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"嗯，我听到了。关于「{user[:40]}」，我现在还没连上大模型，请在 .env 配置 DeepSeek API Key 并重启服务。"
    client = _llm_client()
    temp = temperature if temperature is not None else 0.7
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temp,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        err = str(e)
        if "402" in err or "Insufficient Balance" in err or "余额" in err:
            return "DeepSeek 账户余额不足，请到 platform.deepseek.com 充值后再试。"
        if "401" in err or "Authentication" in err:
            return "DeepSeek API Key 无效，请检查 .env 里的 LLM_API_KEY 是否正确。"
        if "429" in err:
            return "DeepSeek 请求太频繁，请稍后再试。"
        return f"调用 DeepSeek 失败：{err[:120]}"


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(x * x for x in bv)) or 1.0
    return dot / (na * nb)
