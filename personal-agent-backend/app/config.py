"""全局配置 —— 基于 .env 文件的陪伴机器人全部运行参数。

本模块的角色：
  采用 pydantic-settings 自动加载项目根目录的 .env 文件，
  提供类型安全的配置访问。所有模块通过 `from app.config import settings` 引用全局单例。

配置文件位置：项目根目录的 .env 文件
示例配置见：.env.example

配置分类：
  1. LLM 配置（DeepSeek）
  2. Embedding 配置（阿里云 DashScope）
  3. 安全配置（API Token）
  4. 路径配置（persona 语料/持久化）
  5. 记忆参数（工作上下文 / 近期记忆 / 长期记忆）
  6. 画像参数（临时画像转正/归档）
  7. 运行时参数（端口、温度等）
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（personal-agent-backend/）
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """陪伴机器人全局配置类。

    所有字段可从 .env 文件读取，未配置时使用默认值。
    extra="ignore" 表示 .env 中多余的键不会导致报错（兼容未来添加的字段）。
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # .env 中的未知键不报错，允许渐进式添加配置项
    )

    # ============================
    # LLM 对话模型配置（DeepSeek）
    # ============================
    # DeepSeek API Key（从 platform.deepseek.com 获取）
    llm_api_key: str = ""
    # DeepSeek API 地址（默认使用官方地址）
    llm_base_url: str = "https://api.deepseek.com/v1"
    # 对话模型名称
    llm_model: str = "deepseek-chat"

    # ============================
    # Embedding 向量模型配置（阿里云 DashScope）
    # ============================
    # 阿里云 DashScope API Key（从 dashscope.console.aliyun.com 获取）
    # 不配置时使用本地 SHA256 哈希 fallback 向量（质量低，仅用于开发测试）
    embed_api_key: str = ""
    # DashScope 兼容模式地址
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 向量模型名称
    embed_model: str = "text-embedding-v3"

    # ============================
    # 安全配置
    # ============================
    # API Token：HTTP 和 WebSocket 接口的认证密钥
    # 开发环境默认 "dev-token"，生产环境应在 .env 中修改
    api_token: str = "dev-token"

    # ============================
    # Persona 语料路径配置
    # 详见 persona/README.md
    # ============================
    # persona 目录根路径（相对于项目根目录）
    persona_dir: str = "../persona"
    # 机器人人格描述文件（persona.md）
    persona_path: str = "../persona/config/persona.md"
    # Profile Card 模板文件（定义对话对象画像结构）
    profile_card_path: str = "../persona/config/profile_card.md"
    # 风格目录（包含语气、口癖等 style 描述）
    style_dir: str = "../persona/style"
    # 语料目录（包含长期知识 .md 文件，启动时自动入库到长期记忆）
    corpus_dir: str = "../persona/corpus"

    # ============================
    # 持久化路径
    # ============================
    # ChromaDB 向量数据存储路径（search_backend=chroma 时使用）
    chroma_path: str = "./chroma_data"
    # SQLite 数据库路径（存储会话、消息、画像、核心事实、近期记忆、长期记忆等）
    db_path: str = "./agent.db"

    # ============================
    # 搜索后端配置
    # ============================
    # 长期记忆检索后端：sqlite（推荐，2G 机器友好）或 es（Elasticsearch，数据量大时推荐）
    search_backend: str = "sqlite"
    # Elasticsearch 连接配置（search_backend=es 时生效）
    # ES 用于长期记忆的混合检索（全文+向量），替代默认的 SQLite FTS
    es_url: str = "http://127.0.0.1:9200"
    es_api_key: str = ""                  # API Key 认证（与 username/password 二选一）
    es_username: str = ""                 # 用户名认证
    es_password: str = ""                 # 密码认证
    es_index_prefix: str = "sparkbot"     # 索引名前缀，生成 sparkbot_long_term 等索引名
    es_timeout_sec: int = 8               # 请求超时（秒）
    es_keyword_candidates: int = 32       # 关键词搜索候选数（第一阶段粗排，bigram 需更多候选）
    es_vector_candidates: int = 32        # 向量搜索候选数（第一阶段粗排）
    es_rerank_top_n: int = 16             # 重排序后最终保留条数（第二阶段精排）
    es_min_recall_score: float = 0.35     # 最低召回相似度阈值（修复 RRF 后不再有虚假加分）


    # ============================
    # 当前会话上下文（Working Context）
    # ============================
    # 上下文压缩触发阈值（轮数）
    working_context_turns: int = 30
    # 每批压缩轮数
    context_compaction_batch_turns: int = 18

    # ============================
    # 近期记忆（Recent Memory）
    # ============================
    # 近期记忆保留天数
    recent_memory_retention_days: int = 14
    # 时间倒序召回条数
    recent_memory_recall_recent: int = 3
    # 向量嵌入池大小
    recent_memory_embed_pool: int = 15
    # 向量相似度最低阈值
    recent_memory_sim_threshold: float = 0.6
    # 注入 prompt 条数
    recent_memory_top_k: int = 3

    # ============================
    # 长期记忆（Long-Term Memory）
    # ============================
    # 向量相似度最低阈值
    long_term_memory_sim_threshold: float = 0.55
    # 入库去噪
    long_term_memory_denoise_enabled: bool = True
    # 噪音文件模式
    long_term_memory_noise_file_patterns: str = ""
    # RAG 检索条数
    rag_top_k: int = 5

    # ============================
    # 画像（Profile）参数
    # ============================
    # 临时画像是否自动更新
    person_profile_auto_update: bool = False
    # 画像履历归档间隔（小时）：每 N 小时扫描近期/长期记忆，更新性格/履历描述
    profile_batch_interval_hours: int = 3
    # 归档回溯窗口（小时）：只取最近 N 小时内的近期/长期记忆内容
    profile_batch_lookback_hours: int = 3
    # 临时画像转正模式：any=任一规则满足即转正，all=全部规则满足才转正
    profile_promotion_mode: str = "any"
    # 临时画像转正规则列表（逗号分隔），详见 memory/pipeline/promotion.py
    profile_promotion_rules: str = (
        "relationship_declared,substantive_fact,recent_episode,"
        "profile_long_memory,user_remember_intent"
    )
    # 记忆自动修正：用户说"不对/不是/记错了"时触发 LLM 纠错
    memory_auto_correct: bool = True

    # ============================
    # 访客与身份识别参数
    # ============================
    # 访客每 N 轮对话提醒一次实名（如"怎么称呼你呀"）
    guest_identity_reminder_every: int = 3

    # 默认对话对象（开机/新会话默认女友模式）
    default_owner_person_id: str = ""
    default_owner_display_name: str = "刘远慧"
    # ============================
    # 记忆关联图参数
    # ============================
    # 关联图召回最低强度阈值（只有 strength >= 此值的关联才注入 prompt）
    memory_relation_min_strength: float = 0.6

    # ============================
    # Persona 语料导入参数
    # ============================
    # 启动时是否自动同步 persona/corpus/ 到长期记忆
    persona_ingest_on_startup: bool = True
    # 启动时是否全量重建语料索引（true=每次重启都清空重建，很慢，仅维护用）
    persona_ingest_reset_on_startup: bool = False
    # 语料入库时是否同步提取 Facts
    persona_ingest_extract_facts: bool = False
    # 语料 Facts 绑定的 person_id（系统级知识用固定 ID 存储，不绑定真实用户）
    persona_fact_person_id: str = "persona_global"

    # ============================
    # 会话管理参数
    # ============================
    # 会话空闲超时（分钟）：超过此时间无活动则触发 session_end
    session_idle_minutes: int = 10

    # ============================
    # 对话生成参数
    # ============================
    # 回复最大字数（用于 prompt 中限制 LLM 输出长度和硬截断）
    # 微信实测平均 8 字/条，设为 80 给偶尔长回复留空间
    max_reply_chars: int = 80
    # LLM 对话温度（0.88 偏高，让回复更自然不机械，但不过于随机）
    chat_temperature: float = 0.88

    # ============================
    # 主动追问（Follow-up）
    # ============================
    # 是否启用主动追问：回复用户后以一定概率判断是否继续聊，生成一条新话题
    follow_up_enabled: bool = True
    # 主动追问触发概率（0.0～1.0）：概率命中后 LLM 再判断是否真的适合追话
    follow_up_probability: float = 0.6

    # ============================
    # ============================
    # TTS 语音合成配置（百度）
    # ============================
    # 百度 TTS API Key（用于旧版 text2audio 接口）
    # 不配置时 TTS 功能关闭，只返回文本 bubbles
    tts_api_key: str = ""
    # 发音人 ID（4100=度小乔，4103=度小贤，4117=度小鹿 等）
    tts_speaker_voice: str = "4100"
    # 语速 0-15，默认 6
    tts_speed: int = 6
    # 音调 0-15，默认 5
    tts_pitch: int = 5
    # 音量 0-15，默认 8
    tts_volume: int = 8
    # 音频格式：pcm = raw PCM (aue=4), wav = WAV (aue=6)
    tts_audio_format: str = "pcm"

    # ============================
    # 声音复刻配置（百度大模型声音复刻）
    # ============================
    # 百度云 API Key，在 https://console.bce.baidu.com/iam/#/iam/apikey/list 创建
    baidu_api_key: str = ""
    # 复刻音色 ID，运行 scripts/create_voice.py 上传音频后获得
    tts_clone_voice_id: str = ""

    # ============================
    # 百度语音识别（ASR）配置
    # ============================
    # 百度 ASR APP ID（语音识别应用标识符）
    baidu_asr_app_id: str = ""
    # 百度 ASR API Key
    baidu_asr_api_key: str = ""
    # 百度 ASR Secret Key（与 api_key 配合获取 access_token）
    baidu_asr_secret_key: str = ""
    # 是否在 v2 音频协议中启用 ASR（true=语音转文字，false=仅传文本不识别）
    asr_enabled: bool = False
    # 百度 ASR 模型：15372=普通话(远场), 15373=粤语, 1737=英语
    asr_dev_pid: int = 15372

    # ============================
    # 控制台日志配置（阶段 3.0）
    # ============================
    # 日志模式：silent=仅启动/错误, normal=核心链路, debug=详细(含MemoryPack/Prompt), trace=全部底层
    console_log_mode: str = "normal"
    # 是否在 normal 模式显示记忆包摘要行
    console_log_memory_detail: bool = True
    # debug/trace 模式是否预览 prompt 内容
    console_log_prompt_preview: bool = False
    # 是否显示各阶段耗时
    console_log_timing: bool = True
    # 控制台分隔线宽度
    console_log_width: int = 100

    # ============================
    # 运行时参数
    # ============================
    host: str = "0.0.0.0"  # 监听地址
    port: int = 8001        # 监听端口

    # ============================
    # 路径解析辅助方法
    # ============================
    def _resolve(self, p: str) -> Path:
        """将相对路径解析为基于项目根目录的绝对路径。

        为什么需要：.env 中配置的路径是相对于项目根目录的，
        但进程可能在任何目录启动，需要转换为绝对路径。
        """
        path = Path(p)
        if not path.is_absolute():
            path = _BACKEND_ROOT / path
        return path

    def resolved_persona_dir(self) -> Path:
        """返回 persona 根目录的绝对路径。"""
        return self._resolve(self.persona_dir)

    def resolved_persona_path(self) -> Path:
        """返回 persona.md 文件的绝对路径。"""
        return self._resolve(self.persona_path)

    def resolved_profile_card_path(self) -> Path:
        """返回 profile_card.md 文件的绝对路径。"""
        return self._resolve(self.profile_card_path)

    def resolved_style_dir(self) -> Path:
        """返回风格目录的绝对路径。"""
        return self._resolve(self.style_dir)

    def resolved_corpus_dir(self) -> Path:
        """返回语料目录的绝对路径（启动时扫描 .md 文件入库）。"""
        return self._resolve(self.corpus_dir)

    def resolved_db_path(self) -> Path:
        """返回 SQLite 数据库绝对路径（避免从错误 cwd 启动时读到空库）。"""
        return self._resolve(self.db_path)

    def resolved_data_dir(self) -> Path:
        """向后兼容：语料入库根目录（等同于 resolved_corpus_dir）。"""
        return self.resolved_corpus_dir()


# 全局单例：所有模块通过 `from app.config import settings` 引用
settings = Settings()
