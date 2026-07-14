"""
百度语音识别（ASR）服务 —— 后端统一 ASR 适配。

职责：
  1. 将固件上传的 16kHz 16bit 单声道 PCM 提交百度 ASR REST API
  2. 处理 access_token 获取/刷新
  3. 错误重试、超时处理、降级
  4. 统一的日志和耗时观测

一期：整句 PCM → 百度 REST API（全量上传再识别）
二期：边收边传 → 实时 ASR 通道（预留 transcribe_streaming 接口）

百度 ASR REST API 文档：
  https://ai.baidu.com/ai-doc/SPEECH/Vk38lxily
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 百度 ASR OAuth Token 端点
BAIDU_OAUTH_URL = "https://openapi.baidu.com/oauth/2.0/token"
# 百度 ASR REST API（pro 版，支持长音频和远场）
BAIDU_ASR_URL = "https://vop.baidu.com/pro_api"
# 百度 ASR REST API（标准版）
BAIDU_ASR_SERVER_API_URL = "https://vop.baidu.com/server_api"

# 持久化 HTTP 客户端
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=8),
        )
    return _http_client


class ASRError(Exception):
    """ASR API 返回错误。"""


class ASRTimeout(Exception):
    """ASR 请求超时。"""


class BaiduASRService:
    """百度语音识别适配器。

    使用方式：
        asr = BaiduASRService()
        text = await asr.transcribe(pcm_data)
    """

    def __init__(
        self,
        app_id: str = "",
        api_key: str = "",
        secret_key: str = "",
        dev_pid: int = 0,
    ) -> None:
        self.app_id = app_id or settings.baidu_asr_app_id
        self.api_key = api_key or settings.baidu_asr_api_key
        self.secret_key = secret_key or settings.baidu_asr_secret_key
        self.dev_pid = dev_pid or settings.asr_dev_pid
        self.cuid = "sparkbot-backend-asr"

        self._token: str = ""
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        """ASR 是否已配置（三个凭证都非空）。"""
        return bool(self.api_key and self.secret_key)

    async def _fetch_token(self) -> str:
        """从百度 OAuth 端点获取 access_token。

        Token 有效期约 30 天，缓存到过期前 5 分钟。
        使用 asyncio.Lock 防并发重复获取。
        """
        async with self._token_lock:
            now = time.monotonic()
            if self._token and now < self._token_expires_at:
                return self._token

            client = _get_client()
            params = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            }
            try:
                resp = await client.get(BAIDU_OAUTH_URL, params=params)
                data = resp.json()
            except Exception as exc:
                raise ASRError(f"百度 OAuth 请求失败: {exc}") from exc

            if "access_token" not in data:
                err_msg = data.get("error_description", data.get("error", "unknown"))
                raise ASRError(f"百度 OAuth 返回错误: {err_msg}")

            self._token = data["access_token"]
            expires_in = int(data.get("expires_in", 2592000))  # 默认 30 天
            self._token_expires_at = now + expires_in - 300  # 提前 5 分钟过期
            logger.info(
                "百度 ASR token 获取成功，有效期 %d 秒", expires_in,
            )
            return self._token

    async def transcribe(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        *,
        retry_count: int = 2,
        timeout: float = 25.0,
    ) -> str:
        """将 PCM 音频提交百度 ASR REST API，返回识别文本。

        Args:
            pcm_data:    16kHz 16bit 单声道 PCM 数据
            sample_rate: 采样率（默认 16000）
            retry_count: 失败重试次数
            timeout:     单次请求超时（秒）

        Returns:
            识别的文本字符串

        Raises:
            ASRError:   API 返回错误或配置缺失
            ASRTimeout: 请求超时
        """
        if not self.configured:
            raise ASRError("百度 ASR 未配置（缺少 api_key 或 secret_key）")

        token = await self._fetch_token()
        client = _get_client()

        url = f"{BAIDU_ASR_URL}?dev_pid={self.dev_pid}&cuid={self.cuid}&token={token}"

        last_error: Exception | None = None
        for attempt in range(1 + retry_count):
            try:
                resp = await client.post(
                    url,
                    content=pcm_data,
                    headers={"Content-Type": f"audio/pcm;rate={sample_rate}"},
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = ASRTimeout(f"ASR 请求超时 (attempt {attempt + 1}): {exc}")
                if attempt < retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))
                continue
            except Exception as exc:
                last_error = ASRError(f"ASR 请求异常 (attempt {attempt + 1}): {exc}")
                if attempt < retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))
                continue

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                last_error = ASRError(f"ASR 响应非 JSON: {resp.text[:200]}")
                if attempt < retry_count:
                    continue
                break

            err_no = data.get("err_no", -1)
            if err_no != 0:
                err_msg = data.get("err_msg", f"err_no={err_no}")
                if err_no in (3301, 3302):  # token 失效
                    self._token = ""
                    self._token_expires_at = 0.0
                    if attempt < retry_count:
                        token = await self._fetch_token()
                        url = f"{BAIDU_ASR_URL}?dev_pid={self.dev_pid}&cuid={self.cuid}&token={token}"
                        continue
                last_error = ASRError(f"百度 ASR 返回错误: {err_msg}")
                if attempt < retry_count:
                    continue
                break

            result = data.get("result", [])
            if result:
                text = result[0].strip()
                logger.info(
                    "ASR 成功: len(pcm)=%d text=%s",
                    len(pcm_data), text[:40],
                )
                return text
            # 静音或空结果
            logger.info("ASR 返回空结果（可能为静音）: len(pcm)=%d", len(pcm_data))
            return ""

        # 所有重试均失败
        raise last_error or ASRError("ASR 请求失败（未知错误）")

    async def transcribe_streaming(self, pcm_generator, sample_rate: int = 16000):
        """预留：边收边传的实时 ASR 接口（二期实现）。

        Args:
            pcm_generator: 异步生成器，逐次 yield PCM 分片
            sample_rate:   采样率

        Yields:
            (阶段, 文本) 二元组，阶段可以是 "partial" 或 "final"
        """
        raise NotImplementedError("实时 ASR 属二期功能，尚未实现")


# 模块级单例
asr_service = BaiduASRService()
