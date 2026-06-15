"""RAG（检索增强生成）向后兼容重导出模块。

本模块本身不包含新的业务逻辑，仅从 app.persona.ingest 重导出关键函数，
目的是保持旧代码的导入路径（如 `from app.rag import ingest_directory`）仍然可用，
避免因模块重组导致的破坏性变更。

实际功能说明：
  - 语料入库、降噪、分块等逻辑全部在 app.persona.ingest 中实现
  - 本模块纯粹是一个兼容性适配层

历史原因：
  - 早先版本将 RAG/语料相关逻辑放在 app/rag.py 中
  - 后来 persona 模块重构，将语料处理迁移到 app/persona/ingest.py
  - 保留本模块避免 scripts/ 和外部引用需要批量修改导入路径
"""

from app.persona.ingest import (  # noqa: F401
    chunk_text,
    denoise_text,
    ingest_directory,
    startup_ingest_corpus,
    strip_dialogue_examples,
)

# 从 app.memory.l3 重导出，保持旧代码兼容
from app.memory.l3 import (  # noqa: F401
    batch_extract_facts_from_persona_chunks,
    clear_persona_derived_memory,
)
