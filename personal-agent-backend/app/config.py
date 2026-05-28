"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    embed_api_key: str = ""
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_model: str = "text-embedding-v3"

    api_token: str = "dev-token"

    # persona/ 目录布局（见 persona/README.md）
    persona_dir: str = "../persona"
    persona_path: str = "../persona/config/persona.md"
    profile_card_path: str = "../persona/config/profile_card.md"
    style_dir: str = "../persona/style"
    corpus_dir: str = "../persona/corpus"

    chroma_path: str = "./chroma_data"
    db_path: str = "./agent.db"
    search_backend: str = "es"
    es_url: str = "http://127.0.0.1:9200"
    es_api_key: str = ""
    es_username: str = ""
    es_password: str = ""
    es_index_prefix: str = "sparkbot"
    es_timeout_sec: int = 8
    es_keyword_candidates: int = 24
    es_vector_candidates: int = 24
    es_rerank_top_n: int = 16
    es_min_recall_score: float = 0.22

    working_memory_turns: int = 30
    l1_compress_batch_turns: int = 12
    l2_retention_days: int = 7
    l2_recall_recent: int = 3
    l2_recall_query_k: int = 2
    episodic_top_k: int = 3
    rag_top_k: int = 5
    l3_denoise_enabled: bool = True
    l3_noise_file_patterns: str = "wechat_memory.md,wechat_group_*.md,intimate.md"
    # L3: intent | hybrid (default) | always
    l3_recall_mode: str = "hybrid"
    l3_light_top_k: int = 3
    auto_extract_facts: bool = False

    session_idle_minutes: int = 10
    max_reply_chars: int = 80
    chat_temperature: float = 0.88

    host: str = "0.0.0.0"
    port: int = 8000

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = _BACKEND_ROOT / path
        return path

    def resolved_persona_dir(self) -> Path:
        return self._resolve(self.persona_dir)

    def resolved_persona_path(self) -> Path:
        return self._resolve(self.persona_path)

    def resolved_profile_card_path(self) -> Path:
        return self._resolve(self.profile_card_path)

    def resolved_style_dir(self) -> Path:
        return self._resolve(self.style_dir)

    def resolved_corpus_dir(self) -> Path:
        return self._resolve(self.corpus_dir)

    def resolved_data_dir(self) -> Path:
        """Backward compat: corpus ingest root."""
        return self.resolved_corpus_dir()


settings = Settings()
