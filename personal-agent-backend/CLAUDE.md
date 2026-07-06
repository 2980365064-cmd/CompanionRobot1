# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

SparkBot Personal Agent 的 Python FastAPI 后端。通过 WebSocket 长连接接收 ESP32 硬件端的语音/文本消息，调用 DeepSeek 大模型生成对话回复，具备多层记忆系统、人物画像、情感追踪和 TTS 语音合成。

> 父级 `../CLAUDE.md` 有项目整体架构、固件和 persona 配置的完整说明。本文件只覆盖后端子项目的特有内容。

## 常用命令

```bash
# 安装与配置
pip install -r requirements.txt
cp .env.example .env   # 填入 LLM_API_KEY（DeepSeek）和 EMBED_API_KEY（阿里云 DashScope）

虚拟环境：python3 -m venv venv
source venv/bin/activate

# 启动（推荐方式——仅显示 Agent 监控日志）
python -m app.main

# 或直接 uvicorn（开发时可用 --reload）
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 语料管理
python scripts/ingest.py             # 增量入库 persona/corpus/ → L3
python scripts/ingest.py --reset     # 全量重建语料索引
python scripts/compress_profile.py   # 生成/更新 profile_card.md

# 测试脚本
python scripts/test_persona.py       # 口吻 + 记忆召回本地抽测
python scripts/test_ws_client.py     # WebSocket 模拟机器人联调
python scripts/smoke_test.py         # 冒烟测试（HTTP + WebSocket 全链路）
python scripts/eval_agent.py         # Agent 端到端评估
python scripts/diagnose_memory.py    # 记忆系统诊断（检查各层数据一致性）
python scripts/e2e_pipeline_test.py  # 端到端管线测试
python scripts/test_scheme_a_chain.py # 对比测试方案 A 链路

# 情感陪伴评测（覆盖 6 维度 × 26 测试用例）
python scripts/eval_emotional_memory.py                  # 离线规则模式
python scripts/eval_emotional_memory.py --online          # 在线模式（需后端运行）
python scripts/eval_emotional_memory.py --offline-static  # 静态 Prompt 检查（无工程词检查）

# 微信数据导入
python scripts/import_wechat.py --private    # 微信单聊导入（隐私数据，不提交）
python scripts/analyze_wechat_style.py       # 从微信提炼口吻特征

# Docker
docker compose up -d --build
```

## 架构核心

### 单轮对话流水线 (`app/agent.py`)

```
用户消息 → get_or_create_session → resolve_interlocutor_before_memory（身份门控）
→ memory_router.recall（L0全量+L2向量+L3混合检索+关联扩展）
→ load_profile_card（加载机器人人格）
→ build_messages（拼装 system prompt）
→ chat_completion_async（DeepSeek 单次调用，||| 分隔主回复和主动话题）
→ _parse_reply → 校验/截断 → 写入 L1 → 异步入库
```

关键文件：`app/agent.py:696 handle_chat()` 是主入口，`build_messages()` 是 prompt 组装核心。

### 记忆分层体系

| 层 | 存储 | 注入方式 | 模块 |
|---|---|---|---|
| **L0 Core** | `l0_core_memories` 表 | 全量注入每轮 system prompt | `memory/l0.py` |
| **L1 Working** | `messages` 表（本会话窗口） | 最近 N 轮拼入 messages | `memory/l1.py` |
| **L2 Episodic** | `episodic_memories` 表 | 向量检索后注入，14天过期归档（含重要性加权） | `memory/l2.py` |
| **L3 Corpus** | `l3_chunks` + FTS5 | 混合检索（向量+全文），永久存储 | `memory/l3.py` |
| **Profile** | `person_profiles` 表（JSON） | 格式化后注入 prompt | `memory/profile.py` |

> **2026-07-02 重构：L2 保留期从 7→14 天，增加 importance(1-5) 和 people 字段。高重要性事件(>=4)不受过期限制。**

**记忆流转**：L1(本轮) → L1压缩→ L2(14天摘要，高重要性永久) → 到期归档→ L3(永久) → 高频提升→ L0
| **L3 Corpus** | `l3_chunks` + FTS5 | 混合检索（向量+全文），永久存储 | `memory/l3.py` |
| **Profile** | `person_profiles` 表（JSON） | 格式化后注入 prompt | `memory/profile.py` |

**记忆流转**：L1(本轮) → L1压缩→ L2(7天摘要) → 过期归档→ L3(永久) → 高频提升→ L0

### 身份门控 (`memory/identity.py`, `memory/interlocutor.py`)

- **访客模式**（`tmp_*` person_id 或 `MODE_VISITOR`）：仅 L1，禁止访问 L0/L2/L3。每 N 轮口语提醒实名
- **已实名模式**（verified person_id）：全量记忆注入
- 口语实名格式：`名字 ID`（如"刘远慧 123"），由 `parse_identity_credentials()` 解析
- 对话角色切换：用户说"访客模式"/"女友模式"触发 `resolve_interlocutor_before_memory()`

### 数据库 (`app/session.py`)

单文件 SQLite `agent.db`，WAL 模式，8MB 缓存。全局单例 `store = SessionStore()`。所有 DB 操作是同步的，调用方通过 `asyncio.to_thread()` 在后台线程执行。

核心表：`sessions`, `messages`, `episodic_memories`, `l3_chunks` + `l3_chunks_fts`(FTS5), `l0_core_memories`, `person_profiles`, `memory_relations`, `l3_recall_stats`

Schema 迁移是增量的：各 `_migrate_*` 方法检测列是否存在再 ALTER TABLE，不需要手动迁移。

### 配置系统 (`app/config.py`)

pydantic-settings 自动加载 `.env`，全局单例 `settings`。所有路径配置相对于 `personal-agent-backend/` 解析。关键配置分类见 `.env.example` 注释。

### LLM 策略 (`app/llm.py`)

- **对话**：`chat_completion_async()` — DeepSeek `deepseek-chat`，temperature=0.88
- **流式对话**：`chat_completion_stream_async()` — 逐步 yield token，供 WebSocket 实时推送
- **后台提取**：`chat_completion_small_async()` — 同模型但 temperature=0.1、max_tokens=256，用于 Facts 抽取/记忆修正/会话压缩
- **向量**：`embed_texts()` — 阿里云 DashScope `text-embedding-v3`，未配置时 fallback 本地 SHA256 伪向量
- HTTP 连接池在启动时预热（`warmup_llm_client()`），减少首次对话延迟

### WebSocket 协议 (`app/ws_handler.py`)

消息类型：`hello`(握手) → `chat`(对话，流式 TTS bubbles) → `session_end`(收尾)。支持 `abort`(打断 TTS)、`new_session`(强制新会话)、`ping`/`pong`(心跳)。

回复以 bubbles 为单位（按句末标点切分），每个 bubble 独立做 TTS 合成。

### 后台定时任务 (在 `app/main.py` lifespan 中启动)

- `idle_session_sweeper` — 每分钟清理空闲超时会话
- `l2_rollup_sweeper` — 每小时归档过期 L2 → L3
- `profile_batch_sweeper` — 每 N 小时从 L2+L3 增量更新人物画像

### 控制台监控 (`app/monitor.py`)

**阶段 3.0**：升级为"记忆测试驾驶舱"，支持 4 种日志模式：

| 模式 | 行为 |
|------|------|
| `silent` | 仅显示启动和错误信息 |
| `normal` | 每轮 box-drawing 核心链路（身份/记忆/回复/耗时/后台事件） |
| `debug` | 详细展开 MemoryPackV2、Consolidator 裁决、Prompt 摘要 |
| `trace` | 显示底层召回候选项 |

配置方式（`.env`）：
```
CONSOLE_LOG_MODE=debug
CONSOLE_LOG_MEMORY_DETAIL=true
CONSOLE_LOG_PROMPT_PREVIEW=false
CONSOLE_LOG_TIMING=true
CONSOLE_LOG_WIDTH=100
```

`python -m app.main` 启动后每轮对话打印：
- normal 模式：`╭─ Turn #42 · 14:03:18 · web-admin` 风格输出
- debug 模式：额外显示 MemoryPackV2 细节、Prompt 工程词检查、Consolidator 完整结果

## 模块速查

| 模块 | 作用 |
|------|------|
| `app/main.py` | FastAPI 入口，HTTP/WebSocket 路由，后台定时任务 |
| `app/agent.py` | 对话编排引擎，单轮处理 + prompt 组装 + 异步入库 |
| `app/config.py` | pydantic-settings 全局配置 |
| `app/session.py` | SQLite 持久层，SessionStore 单例 |
| `app/llm.py` | DeepSeek 对话 + DashScope 向量 + 轻量提取 |
| `app/ws_handler.py` | WebSocket 协议处理（hello/chat/session_end 等） |
| `app/monitor.py` | 控制台监控输出（记忆命中、耗时、后台事件） |
| `app/tts.py` | 百度 TTS 语音合成 + 声音复刻 |
| `app/memory/router.py` | MemoryRouter 记忆召回调度中心（底层） |
| `app/memory/schema.py` | **记忆统一语义层** — MemoryItem/MemoryKind/MemoryPackV2 等核心数据结构，所有新模块基于此 schema |
| `app/memory/orchestrator.py` | **Memory Orchestrator — 记忆编排器**，将 L0/L2/L3 统一转换为 MemoryPack（5 类产品记忆），支持 V2 转换 |
| `app/memory/relationship_state.py` | **关系状态持久化** — 每轮更新关系温度/情绪趋势/态度/关心点/避雷点 |
| `app/memory/open_loops.py` | **结构化 Open Loop 管理器** — 待办事项检测/创建/解决/冷却，替代 L2 硬解析 |
| `app/memory/emotional_events.py` | **情感事件抽取器** — 从 L2 和对话中提取高重要性情感事件 |
| `app/memory/l0.py` | L0 核心记忆提取与格式化 |
| `app/memory/l1.py` | L1 工作记忆（消息滑动窗口） |
| `app/memory/l2.py` | L2 情景记忆向量检索 |
| `app/memory/l3.py` | L3 长期记忆混合检索（FTS5 + 向量） |
| `app/memory/guard.py` | 反幻觉规则、主动话题校验、用户意图识别 |
| `app/memory/extractor.py` | L1→L2 压缩、Facts 提取、过期 L2 归档 |
| `app/memory/identity.py` | 身份识别（名字+ID 解析、访客/实名的判断） |
| `app/memory/interlocutor.py` | 对话角色管理（女友/访客模式切换） |
| `app/memory/profile.py` | 人物画像读写与格式化 |
| `app/memory/relations.py` | 记忆关联图（语义关系网络） |
| `app/memory/emotion.py` | 情感轨迹追踪 |
| `app/memory/contacts.py` | 第三方人物画像管理 |
| `app/memory/correction.py` | 记忆自动修正（用户纠错触发） |
| `app/memory/consolidator.py` | **MemoryConsolidator — 统一写入仲裁器**，每轮分类后裁决记忆沉淀 + 关系状态 + Open Loop |
| `app/memory/self_state.py` | 机器人自我状态管理 |
| `app/persona/card.py` | 机器人人格卡片加载（persona.md） |
| `app/persona/ingest.py` | 语料分块/入库/降噪 |
| `app/store/chunks.py` | L3 文本分块和 FTS 查询构造 |
| `app/memory_admin.py` | L0/Profile/L2 管理 API 后端 |
| `app/person_admin.py` | 用户身份管理 API 后端 |

## 代码约定

- 所有注释、docstring、日志输出使用中文
- 数据库操作全是同步函数，在 `asyncio.to_thread()` 中调用
- `store`（SessionStore）、`settings`（Settings）、`agent_monitor`（AgentMonitor）是模块级全局单例
- _post_process 中所有操作都是 fire-and-forget（失败不阻塞主回复）
- 路径配置全部通过 `settings.resolved_*()` 方法解析为绝对路径
- 回复长度有 `max_reply_chars` 限制（.env 配置默认 80），在代码层用硬截断做双重保险
