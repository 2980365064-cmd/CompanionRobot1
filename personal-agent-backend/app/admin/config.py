"""后台配置中心：读取、展示、更新 .env 中的运维配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

BACKEND_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_ROOT / ".env"

SENSITIVE_KEYS = {
    "LLM_API_KEY",
    "EMBED_API_KEY",
    "API_TOKEN",
    "ES_API_KEY",
    "ES_PASSWORD",
    "TTS_API_KEY",
    "BAIDU_API_KEY",
    "TTS_CLONE_VOICE_ID",
}


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    category: str
    value_type: str = "str"
    restart_required: bool = False
    sensitive: bool = False
    description: str = ""
    impact: str = ""
    safe_hint: str = ""


FIELDS: list[ConfigField] = [
    ConfigField("LLM_API_KEY", "DeepSeek API Key", "模型",
                sensitive=True, restart_required=True,
                description="DeepSeek 对话模型的 API 访问密钥。所有对话和后台提取调用依赖此密钥。",
                impact="无有效密钥时对话无法产生回复，后台任务也无法执行。",
                safe_hint="建议通过环境变量注入，不要提交到版本控制。生产环境使用只读权限的子密钥。"),
    ConfigField("LLM_BASE_URL", "DeepSeek API 地址", "模型",
                restart_required=True,
                description="DeepSeek 兼容 OpenAI 格式的 API 端点地址。",
                impact="切换地址会改变所有模型调用目标，离线环境可指向本地代理。",
                safe_hint="保持官方地址不变。如需代理中转，确保协议和路径完整。"),
    ConfigField("LLM_MODEL", "对话模型", "模型",
                restart_required=True,
                description="对话使用的 LLM 模型名称，同时也是后台提取模型的默认值。",
                impact="切换模型后回复风格、速度、成本和质量都可能变化。",
                safe_hint="建议使用 deepseek-chat 或同等能力的指令跟随模型。"),

    ConfigField("EMBED_API_KEY", "DashScope API Key", "向量",
                sensitive=True, restart_required=True,
                description="阿里云 DashScope 向量模型 API 密钥。用于将文本转为向量用于语义检索。",
                impact="未配置时自动 fallback 到本地伪向量（仅用于联调，检索质量大幅下降）。",
                safe_hint="与 EMBED_BASE_URL 配套使用。首次配置后需重启服务。"),
    ConfigField("EMBED_BASE_URL", "向量 API 地址", "向量",
                restart_required=True,
                description="DashScope 向量模型的 API 端点地址。",
                impact="切换后所有向量检索使用新端点。",
                safe_hint="保持官方 DashScope 地址，仅在代理中转时修改。"),
    ConfigField("EMBED_MODEL", "向量模型", "向量",
                restart_required=True,
                description="文本向量化模型名称，决定了向量维度和语义表达能力。",
                impact="切换模型后向量维度可能变化，已有向量的检索效果可能受影响。",
                safe_hint="建议使用 text-embedding-v3。更换后建议执行全量重入库。"),

    ConfigField("CHAT_TEMPERATURE", "对话温度", "对话", "float",
                description="LLM 回复的随机性参数。值越高回复越有创意但可能偏离事实，越低越确定。",
                impact="直接影响回复风格：高温度更活泼/随机，低温度更严谨/重复。",
                safe_hint="建议 0.7-0.9。需要创造力时用上限，需要准确性时用下限。"),
    ConfigField("MAX_REPLY_CHARS", "回复最大字数", "对话", "int",
                description="单轮对话回复的最大字符数。超过时会被截断。",
                impact="过短回复不完整，过长不适合语音场景",
                safe_hint="语音场景建议 60-120 字。文本场景可放宽到 200-300 字。"),
    ConfigField("FOLLOW_UP_ENABLED", "主动追问", "对话", "bool",
                description="是否允许机器人主动发起追问或开启新话题。",
                impact="启用后对话更自然，但可能偏离用户当前关注点。",
                safe_hint="建议保持开启以增强陪伴感。"),
    ConfigField("FOLLOW_UP_PROBABILITY", "主动追问概率", "对话", "float",
                description="每轮对话发起主动追问的概率，0-1 之间的浮点数。",
                impact="值越大机器人越主动追问，值越小越安静等待用户发言。",
                safe_hint="建议 0.3-0.6。高活跃用户场景可适当调低。"),

    ConfigField("WORKING_CONTEXT_TURNS", "压缩触发轮数", "记忆", "int",
                description="工作上下文窗口超过多少轮后触发压缩。超过此轮数的早期消息被摘要化。",
                impact="值越小摘要越频繁但上下文越紧凑，值越大保留细节越多但 token 消耗增加。",
                safe_hint="建议 20-40。低内存设备可设小，高配置可设大。"),
    ConfigField("CONTEXT_COMPACTION_BATCH_TURNS", "上下文压缩每批轮数", "记忆", "int",
                description="每次上下文压缩处理的对话轮数。",
                impact="影响压缩粒度和单次处理耗时。",
                safe_hint="建议与 WORKING_CONTEXT_TURNS 一致或略低。"),
    ConfigField("RECENT_MEMORY_RETENTION_DAYS", "近期记忆保留天数", "记忆", "int",
                description="近期情景摘要的保留天数。超过此期限的摘要会被归档到长期记忆。",
                impact="值越小归档越快但丢失近期上下文，值越大保留细节越多但数据库增长。",
                safe_hint="建议 7-14 天。需要更长时间上下文可设 30 天。"),
    ConfigField("RECENT_MEMORY_RECALL_RECENT", "近期记忆最近召回数", "记忆", "int",
                description="检索近期摘要时，按时间顺序返回的最近摘要数量。",
                impact="值越大注入 prompt 的上下文越多但 token 增加，值越小可能遗漏重要信息。",
                safe_hint="建议 3-8。"),
    ConfigField("RECENT_MEMORY_TOP_K", "近期记忆语义召回数", "记忆", "int",
                description="通过向量相似度检索近期情景摘要时返回的最大结果数。",
                impact="值越大检索覆盖面越广但噪声可能增加。",
                safe_hint="建议 5-15。"),
    ConfigField("RECENT_MEMORY_SIM_THRESHOLD", "近期记忆相似度阈值", "记忆", "float",
                description="近期记忆向量检索的相似度最低阈值。低于此值的摘要不会被召回。",
                impact="阈值越低召回越多但无关结果增加，阈值越高过滤越严但可能漏掉相关记忆。",
                safe_hint="建议 0.45-0.6。"),
    ConfigField("LONG_TERM_MEMORY_SIM_THRESHOLD", "长期记忆相似度阈值", "记忆", "float",
                description="长期记忆向量检索的相似度最低阈值。",
                impact="影响长期记忆的召回精度。",
                safe_hint="建议 0.4-0.55。长期记忆数据量大，适当降低阈值可避免遗漏。"),
    ConfigField("RAG_TOP_K", "RAG 召回数", "记忆", "int",
                description="知识库检索（RAG）返回的最大文档片段数量。",
                impact="影响知识问答的准确性和回复长度。",
                safe_hint="建议 3-8。"),
    ConfigField("MEMORY_RELATION_MIN_STRENGTH", "关联记忆强度阈值", "记忆", "float",
                description="记忆关联图中，低于此强度的关系不会被注入系统提示。",
                impact="影响记忆关联扩展的召回范围。",
                safe_hint="建议 0.3-0.5。"),

    ConfigField("PERSON_PROFILE_AUTO_UPDATE", "画像自动更新", "画像", "bool",
                description="是否允许系统自动从对话中提取信息更新人物画像。",
                impact="启用后画像更动态但不稳定，关闭后仅手动更新。",
                safe_hint="建议保持开启，辅以人工审核。"),
    ConfigField("PROFILE_BATCH_INTERVAL_HOURS", "画像归档间隔小时", "画像", "int",
                description="批量更新人物画像的时间间隔（小时）。",
                impact="间隔越短画像更新越及时但系统负载增加。",
                safe_hint="建议 2-6 小时。"),
    ConfigField("PROFILE_BATCH_LOOKBACK_HOURS", "画像回溯小时", "画像", "int",
                description="每次画像更新时回溯多少小时内的对话数据。",
                impact="回溯范围越大画像更新越全面但处理量越大。",
                safe_hint="建议 24-72 小时。"),
    ConfigField("DEFAULT_OWNER_PERSON_ID", "默认实名用户 ID", "身份",
                description="启动时默认绑定的实名用户 ID。当没有指定用户时使用此 ID。",
                impact="修改后新会话默认绑定到不同的用户。",
                safe_hint="建议与机器人绑定的主要用户一致。"),
    ConfigField("DEFAULT_OWNER_DISPLAY_NAME", "默认实名用户名", "身份",
                description="默认实名用户的可读显示名。",
                impact="不会影响功能，仅用于日志和界面显示。",
                safe_hint="与 PERSON_ID 对应即可。"),
    ConfigField("GUEST_IDENTITY_REMINDER_EVERY", "访客实名提醒轮数", "身份", "int",
                description="访客模式下，每隔多少轮对话提醒用户实名注册一次。",
                impact="值越小提醒越频繁，值越大访客体验越连贯。",
                safe_hint="建议 5-15 轮。"),

    ConfigField("CONSOLE_LOG_MODE", "日志模式", "日志",
                description="控制台输出的详细级别：silent/normal/debug/trace。",
                impact="silent 只显示错误，trace 显示全部记忆召回底层数据。",
                safe_hint="日常使用 normal，排查问题时切换到 debug 或 trace。"),
    ConfigField("CONSOLE_LOG_MEMORY_DETAIL", "显示记忆摘要", "日志", "bool",
                description="是否在控制台显示每轮记忆召回的简短摘要。",
                impact="开启后有助于排查记忆召回问题，但会增加日志量。",
                safe_hint="排查问题时临时开启。"),
    ConfigField("CONSOLE_LOG_PROMPT_PREVIEW", "显示 Prompt 预览", "日志", "bool",
                description="是否在控制台显示每轮构建的 prompt 首段预览。",
                impact="开启后可检查 prompt 结构，但不显示完整内容。",
                safe_hint="调试提示词问题时开启。"),
    ConfigField("CONSOLE_LOG_TIMING", "显示耗时", "日志", "bool",
                description="是否在控制台显示各阶段耗时统计。",
                impact="开启后有助于性能分析。",
                safe_hint="排查性能问题时开启。"),

    ConfigField("API_TOKEN", "后台与设备 Token", "安全",
                sensitive=True, restart_required=True,
                description="管理后台和 ESP32 设备的访问令牌。通过 X-API-Token 请求头传递。",
                impact="修改后所有使用旧 Token 的客户端和固件需同步更新。",
                safe_hint="生产环境务必修改默认值。建议 32 位以上随机字符串。值为 'dev-token' 时触发安全提醒。"),

    ConfigField("PERSONA_INGEST_ON_STARTUP", "启动同步语料", "语料", "bool",
                description="启动时是否自动将 persona/ 目录下的文件入库到长期记忆。",
                impact="开启后修改 persona 文件后重启即可生效，无需手动执行 ingest 脚本。",
                safe_hint="开发环境建议开启，生产环境根据变更频率决定。"),
    ConfigField("PERSONA_INGEST_RESET_ON_STARTUP", "启动全量重建", "语料", "bool",
                restart_required=True,
                description="启动时是否全量清空长期记忆后重新入库。会丢失所有长期记忆。",
                impact="全量重建后所有之前积累的长期记忆会丢失，仅保留语料库内容。",
                safe_hint="仅在需要彻底重置记忆时开启，日常保持关闭。"),

    ConfigField("PERSONA_PATH", "persona.md 路径", "路径",
                restart_required=True,
                description="机器人角色设定文件（persona.md）的路径。",
                impact="修改后机器人将使用不同的人设文件。",
                safe_hint="保持默认值，如需变更使用绝对路径。"),
    ConfigField("PROFILE_CARD_PATH", "profile_card.md 路径", "路径",
                restart_required=True,
                description="机器人人格卡片（profile_card.md）的路径，每轮对话固定注入。",
                impact="修改后对话注入的人设摘要将改变。",
                safe_hint="保持默认值。"),
    ConfigField("CORPUS_DIR", "corpus 目录", "路径",
                restart_required=True,
                description="语料文件存放目录路径（corpus/）。",
                impact="修改后 ingest 脚本和启动同步将扫描新目录。",
                safe_hint="保持默认值。"),
    ConfigField("DB_PATH", "数据库路径", "路径",
                restart_required=True,
                description="SQLite 数据库文件路径。",
                impact="更换后切换到新数据库文件，旧数据不会自动迁移。",
                safe_hint="修改后需手动迁移数据文件。定期备份此文件。"),

    ConfigField("HOST", "监听地址", "服务",
                restart_required=True,
                description="FastAPI 服务监听的 IP 地址。0.0.0.0 表示监听所有网络接口。",
                impact="修改后服务可访问的网络范围变化。",
                safe_hint="生产环境建议绑定具体内网 IP，不要暴露到公网。"),
    ConfigField("PORT", "监听端口", "服务",
                "int", restart_required=True,
                description="HTTP 服务监听端口。",
                impact="修改后需要更新反向代理和固件中的连接地址。",
                safe_hint="建议使用 1024 以上端口避免权限问题。"),

    ConfigField("TTS_API_KEY", "百度 TTS API Key", "语音",
                sensitive=True, restart_required=True,
                description="百度语音合成（TTS）API 密钥，用于将文字转为语音播报。",
                impact="未配置时 TTS 功能不可用。",
                safe_hint="与 BAIDU_API_KEY 配合使用。"),
    ConfigField("BAIDU_API_KEY", "百度云 API Key", "语音",
                sensitive=True, restart_required=True,
                description="百度云平台的通用 API 访问密钥，用于 TTS 等服务的鉴权。",
                impact="未配置时百度云相关服务不可用。",
                safe_hint="从百度云控制台获取，注意与 Secret Key 配套使用。"),
    ConfigField("TTS_CLONE_VOICE_ID", "复刻音色 ID", "语音",
                sensitive=True, restart_required=True,
                description="百度 TTS 声音复刻功能中使用的音色 ID。通过声音复刻 API 获取。",
                impact="不使用复刻音色时可不配置，使用默认音色。",
                safe_hint="通过百度 TTS 控制台上传声音样本后获取。"),
]

FIELD_BY_KEY = {f.key: f for f in FIELDS}


def mask_secret(value: str) -> str:
    value = str(value or "")
    if not value:
        return ""
    if len(value) <= 8:
        return value[:1] + "*" * max(1, len(value) - 2) + value[-1:]
    return value[:4] + "******" + value[-3:]


def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if "\n" in text:
        text = text.replace("\n", "\\n")
    return text


def apply_env_patch(original: str, updates: dict[str, Any]) -> str:
    pending = {str(k).upper(): _format_env_value(v) for k, v in updates.items()}
    lines: list[str] = []
    seen: set[str] = set()
    for line in original.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in pending:
                lines.append(f"{key}={pending[key]}")
                seen.add(key)
                continue
        lines.append(line)
    for key, value in pending.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    return "\n".join(lines).rstrip() + "\n"


def _coerce_value(field: ConfigField, value: Any) -> Any:
    if value is None:
        return ""
    if field.value_type == "int":
        return int(value)
    if field.value_type == "float":
        return float(value)
    if field.value_type == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on", "是")
    return str(value).strip()


def _setting_attr(key: str) -> str:
    return key.lower()


def list_config() -> dict:
    text = ENV_PATH.read_text("utf-8") if ENV_PATH.exists() else ""
    env_values = _parse_env(text)
    fields: list[dict] = []
    for field in FIELDS:
        attr = _setting_attr(field.key)
        raw = env_values.get(field.key, getattr(settings, attr, ""))
        display = mask_secret(str(raw)) if field.sensitive else raw
        fields.append({
            "key": field.key,
            "label": field.label,
            "category": field.category,
            "type": field.value_type,
            "value": "" if field.sensitive else raw,
            "masked_value": display,
            "configured": bool(str(raw or "").strip()),
            "sensitive": field.sensitive,
            "restart_required": field.restart_required,
            "description": field.description,
            "impact": field.impact,
            "safe_hint": field.safe_hint,
        })
    return {
        "env_path": str(ENV_PATH),
        "exists": ENV_PATH.exists(),
        "fields": fields,
        "categories": sorted({f.category for f in FIELDS}),
    }


def update_config(updates: dict[str, Any]) -> dict:
    accepted: dict[str, Any] = {}
    restart_required: list[str] = []
    hot_updated: list[str] = []
    for raw_key, raw_value in updates.items():
        key = str(raw_key).upper()
        field = FIELD_BY_KEY.get(key)
        if not field:
            continue
        if field.sensitive and not str(raw_value or "").strip():
            continue
        value = _coerce_value(field, raw_value)
        accepted[key] = value
        if field.restart_required:
            restart_required.append(key)
        else:
            setattr(settings, _setting_attr(key), value)
            os.environ[key] = _format_env_value(value)
            hot_updated.append(key)
    original = ENV_PATH.read_text("utf-8") if ENV_PATH.exists() else ""
    ENV_PATH.write_text(apply_env_patch(original, accepted), "utf-8")
    return {
        "updated": sorted(accepted),
        "hot_updated": sorted(hot_updated),
        "restart_required": sorted(restart_required),
        "config": list_config(),
    }


def test_config(kind: str = "all") -> dict:
    kind = (kind or "all").lower()
    checks: list[dict] = []
    if kind in ("all", "llm"):
        checks.append({
            "name": "DeepSeek",
            "ok": bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model),
            "detail": settings.llm_model if settings.llm_api_key else "未配置 LLM_API_KEY",
        })
    if kind in ("all", "embed"):
        checks.append({
            "name": "DashScope",
            "ok": bool(settings.embed_api_key and settings.embed_base_url and settings.embed_model),
            "detail": settings.embed_model if settings.embed_api_key else "未配置 EMBED_API_KEY，将使用 fallback",
        })
    if kind in ("all", "db"):
        db = settings.resolved_db_path()
        checks.append({"name": "SQLite", "ok": db.exists(), "detail": str(db)})
    return {"checks": checks, "ok": all(c["ok"] for c in checks)}
