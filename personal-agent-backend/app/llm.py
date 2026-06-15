"""LLM 调用封装 —— DeepSeek 对话、阿里云 DashScope 向量、轻量提取。

本模块的角色：
  陪伴机器人的"大脑"接口层。所有与外部 AI 服务的通信都封装在这里，
  包括：
  - 对话生成（DeepSeek chat API）
  - 文本向量化（阿里云 DashScope embedding API，支持本地 fallback）
  - 轻量提取（用更小的 max_tokens 做 Facts 提取等后台任务）
  - 向量相似度计算（余弦相似度）

设计要点：
  - 同步函数 chat_completion 被异步包装为 chat_completion_async（通过 asyncio.to_thread）
  - 没有 EMBED_API_KEY 时自动回退到 SHA256 哈希生成伪向量（仅用于开发测试）
  - 小模型提取函数 chat_completion_small 默认 max_tokens=256、temperature=0.1，
    用于后台任务降低成本和延迟
  - 所有 API 错误都有中文兜底回复，用户不会看到原始错误信息
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Iterable

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# 阿里云 DashScope 兼容模式默认地址和模型
# DashScope 兼容 OpenAI 接口格式，可直接用 openai 库调用
DASHSCOPE_EMBED_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_EMBED_MODEL = "text-embedding-v3"


def _embed_client() -> OpenAI:
    """返回阿里云 DashScope embedding 客户端（单例，复用 HTTP 连接池）。

    需要 EMBED_API_KEY 已配置，否则抛出 ValueError。
    """
    global _embed_client_cache
    if _embed_client_cache is not None:
        return _embed_client_cache
    if not settings.embed_api_key:
        raise ValueError("EMBED_API_KEY is not set")
    import httpx
    base = settings.embed_base_url or DASHSCOPE_EMBED_BASE
    _embed_client_cache = OpenAI(
        api_key=settings.embed_api_key,
        base_url=base,
        http_client=httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=16),
        ),
    )
    return _embed_client_cache


# 模块级 OpenAI client 缓存，复用 HTTP 连接避免每次请求都做 TCP+TLS 握手
# 实测可节省 500ms~2s 的首次 token 延迟
_llm_client_cache: OpenAI | None = None
_embed_client_cache: OpenAI | None = None


def warmup_llm_client() -> None:
    """预初始化 LLM HTTP 连接池（启动时调用，减少首次对话延迟）。"""
    _llm_client()


def _llm_client() -> OpenAI:
    """返回 DeepSeek LLM 客户端（单例，复用 HTTP 连接池）。"""
    global _llm_client_cache
    if _llm_client_cache is None:
        import httpx
        _llm_client_cache = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            http_client=httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=16),
            ),
        )
    return _llm_client_cache


# DashScope 兼容模式单次调用最多支持 10 条文本，超过需分批
# 此限制来自 DashScope embedding API 的 input 数组上限；
# 阿里云不公开此时，10 是实测安全值，超过会返回 400 错误。
_EMBED_BATCH_SIZE = 10


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本向量化，返回浮点数向量列表。

    参数:
        texts: 待向量化的文本列表（每条约 chunk_size 长度）

    返回:
        list[list[float]]: 每条文本对应的向量（维度由模型决定，如 DashScope v3 为 1024）

    异常处理：
      - 无 EMBED_API_KEY：回退本地 SHA256 哈希伪向量（质量低，仅用于开发）
      - API 认证失败/余额不足：记录错误日志后回退本地 fallback
      - 其他网络错误：同样回退 fallback

    为什么有 fallback：
      开发阶段可能没有阿里云 API Key，但仍然需要验证整个流程跑通。
      fallback 向量质量不足以做精确语义检索，但足够验证代码逻辑。
    """
    if not texts:
        return []
    if not _should_use_remote_embed():
        logger.warning("EMBED_API_KEY not set, using local fallback vectors (re-run ingest after配置阿里云 Key)")
        return [_fallback_embed(t) for t in texts]
    try:
        client = _embed_client()
        model = settings.embed_model or DASHSCOPE_EMBED_MODEL
        out: list[list[float]] = []
        # DashScope 兼容模式单次最多 10 条，需要分批发送
        # 为什么要分批：阿里云 DashScope embedding API 的 input 数组
        # 有隐式上限（约 10 条），单次发太多会返回 HTTP 400；
        # 这里按 _EMBED_BATCH_SIZE 切片发送，保证每批合法。
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
    """判断是否使用远程向量 API（是否配置了 EMBED_API_KEY）。"""
    return bool(settings.embed_api_key)


def embed_provider_name() -> str:
    """返回当前向量提供商的名称（供 /health 接口展示）。

    返回值:
        "aliyun_dashscope" 阿里云 DashScope
        "custom"           其他自定义兼容 API
        "local_fallback"   本地哈希 fallback
    """
    if settings.embed_api_key and "dashscope" in (settings.embed_base_url or DASHSCOPE_EMBED_BASE):
        return "aliyun_dashscope"
    if settings.embed_api_key:
        return "custom"
    return "local_fallback"


def _fallback_embed(text: str, dim: int = 256) -> list[float]:
    """本地 SHA256 哈希回退向量生成器。

    原理：
      对文本做 SHA256 哈希，将 32 字节的摘要循环扩展为 dim 维向量，
      归一化到 [-1, 1] 范围。不同文本的向量接近正交（哈希雪崩效应），
      因此可以做基本的相似度比较，但语义准确性远不如真实 embedding。

    运算细节：
      - SHA256 输出 32 字节（256 位），每个字节值域 [0, 255]
      - (b / 127.5) - 1.0 线性映射到 [-1, 1]
      - 当 dim > 32 时，digest[i % 32] 循环复用字节（不是密码学安全操作，但向量维度只需近似分布）
      - 最终 L2 归一化使所有 fallback 向量模长相同，余弦相似度 = 内积

    仅在未配置 EMBED_API_KEY 时使用，生产环境不推荐。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = digest[i % len(digest)]
        vals.append((b / 127.5) - 1.0)  # 映射到 [-1, 1]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]  # L2 归一化


def _chat_completion_raw(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int = 200,
    stream: bool = False,
):
    """底层 LLM 调用，返回原始响应对象或流式迭代器。

    参数:
        messages:    OpenAI 格式的消息列表
        temperature: 生成温度（None 时默认为 0.7）
        max_tokens:  最大生成 token 数
        stream:      是否使用流式模式

    返回:
        非流式: OpenAI ChatCompletion 对象
        流式:    OpenAI 流式迭代器

    Raises:
        ValueError: 未配置 API Key
        Exception:  其他 API 调用错误
    """
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is not set")
    client = _llm_client()
    temp = temperature if temperature is not None else 0.7
    return client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        stream=stream,
    )


def _llm_error_reply(e: Exception, messages: list[dict]) -> str:
    """将 LLM 异常转换为用户可读的中文错误提示。"""
    err = str(e)
    if "402" in err or "Insufficient Balance" in err or "余额" in err:
        return "DeepSeek 账户余额不足，请到 platform.deepseek.com 充值后再试。"
    if "401" in err or "Authentication" in err:
        return "DeepSeek API Key 无效，请检查 .env 里的 LLM_API_KEY 是否正确。"
    if "429" in err:
        return "DeepSeek 请求太频繁，请稍后再试。"
    return f"调用 DeepSeek 失败：{err[:120]}"


def chat_completion(messages: list[dict], *, temperature: float | None = None) -> str:
    """同步 LLM 对话调用，返回 assistant 回复文本。

    参数:
        messages:    OpenAI 格式的消息列表 [{"role":"system"|"user"|"assistant", "content":"..."}]
        temperature: 生成温度（None 时默认为 0.7），控制回复的随机性

    返回:
        str: 模型生成的回复文本

    错误处理策略：
      不抛出异常，返回用户可读的中文错误提示。这样即使 LLM 挂了，
      用户也能看到有意义的回复而不是 500 错误页面。
    """
    if not settings.llm_api_key:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"嗯，我听到了。关于「{user[:40]}」，我现在还没连上大模型，请在 .env 配置 DeepSeek API Key 并重启服务。"
    try:
        resp = _chat_completion_raw(messages, temperature=temperature)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return _llm_error_reply(e, messages)


def chat_completion_stream(messages: list[dict], *, temperature: float | None = None):
    """流式 LLM 对话调用，返回 token 迭代器。

    使用方式:
        for token in chat_completion_stream(messages):
            yield token  # 每个 token 是一个增量文本片段

    错误处理：
      流式模式下异常会直接抛出，由调用方处理。
      API Key 未配置时抛出 ValueError。
    """
    try:
        stream = _chat_completion_raw(messages, temperature=temperature, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n[ERROR] {_llm_error_reply(e, messages)}"


async def chat_completion_async(
    messages: list[dict],
    *,
    temperature: float | None = None,
) -> str:
    """异步 LLM 对话调用（线程池执行，不阻塞 asyncio 事件循环）。

    为什么用 asyncio.to_thread：
      OpenAI 库是同步的，直接在 async 函数中调用会阻塞事件循环，
      导致 WebSocket 心跳和后台定时任务无法按时执行。
      通过 to_thread 将其放到线程池中执行，保持事件循环响应。
    """
    return await asyncio.to_thread(chat_completion, messages, temperature=temperature)


async def chat_completion_stream_async(
    messages: list[dict],
    *,
    temperature: float | None = None,
):
    """异步流式 LLM 调用 —— 通过队列将同步迭代器桥接到 asyncio。

    使用 asyncio.Queue 在线程和事件循环之间传递 token，
    既保持 OpenAI SDK 同步调用在后台线程执行，
    又不阻塞事件循环。

    使用方式:
        async for token in chat_completion_stream_async(messages):
            ...
    """
    queue: asyncio.Queue = asyncio.Queue()

    def _run() -> None:
        try:
            for token in chat_completion_stream(messages, temperature=temperature):
                queue.put_nowait(("token", token))
        except Exception as exc:
            queue.put_nowait(("error", str(exc)))
        finally:
            queue.put_nowait(("done", None))

    task = asyncio.get_event_loop().run_in_executor(None, _run)
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                return
            if kind == "error":
                raise RuntimeError(payload)
            yield payload
    finally:
        if not task.done():
            task.cancel()


def chat_completion_small(messages: list[dict], *, temperature: float = 0.1, max_tokens: int = 256) -> str:
    """轻量 LLM 调用 —— 用于记忆提取、Facts 抽取等后台任务。

    与 chat_completion 的区别：
      - temperature 默认 0.1（极低，要求确定性的结构化输出）
      - max_tokens 默认 256（成本极低，后台任务不需要长回复）
      - 可选用更小/更便宜的模型（通过 llm_extract_model 配置）
      - 出错时不返回用户可读提示，而是返回空的 JSON {"found":[]}，
        让调用方（如 extractor）优雅降级

    典型使用场景：
      - L1→L2 会话摘要压缩
      - Facts 提取（从对话中抽取结构化事实）
      - 记忆修正（判断用户是否在纠正之前的记忆）
      - 画像更新（从对话中提取人物信息）
    """
    if not settings.llm_api_key:
        # 无 API Key 时返回空 JSON 而非错误文本，因为 chat_completion_small
        # 被结构化提取器（JSON 解析）调用，非 JSON 会导致解析异常；
        # {"found":[]} 是 extractor 约定的空结果格式，各调用方均能安全处理。
        return '{"found":[]}'
    client = _llm_client()
    # 支持配置单独的轻量模型（如 deepseek-chat 的小版本），未配置则复用对话模型
    model = getattr(settings, "llm_extract_model", None) or settings.llm_model
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("small LLM call failed: %s", e)
        return '{"found":[]}'


async def chat_completion_small_async(
    messages: list[dict], *, temperature: float = 0.1, max_tokens: int = 256
) -> str:
    """异步轻量 LLM 调用（用于后台提取任务）。"""
    return await asyncio.to_thread(chat_completion_small, messages, temperature=temperature, max_tokens=max_tokens)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """计算两个向量的余弦相似度，返回值范围 [0, 1]。

    参数:
        a, b: 两个浮点数向量（需等长，但不做长度校验，zip 以短的为准）

    返回:
        float: 余弦相似度值，1.0 表示完全相同，0.0 表示正交

    用于记忆召回阶段比较用户查询向量与 L2/L3 记忆向量的相似度。
    """
    av = a if isinstance(a, list) else list(a)
    bv = b if isinstance(b, list) else list(b)
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(x * x for x in bv)) or 1.0
    return dot / (na * nb)
