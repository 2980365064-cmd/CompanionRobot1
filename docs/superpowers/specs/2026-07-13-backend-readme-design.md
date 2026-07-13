# 后端展示型 README 设计规格

## 目标

将 `personal-agent-backend/README.md` 升级为面向 GitHub 访客与潜在贡献者的中文项目主页：让读者在不阅读源代码的前提下理解陪伴机器人后端能做什么、如何运行，以及统一记忆系统为何可靠。

## 受众与成功标准

- **首要受众**：GitHub 访客、潜在贡献者与技术评估者。
- **阅读结果**：读者能在首页识别系统边界、对话路径、隐私策略和关键模块，并可按最短步骤在本地启动服务。
- **真实性约束**：所有功能、命令、数据表、协议名称与配置项必须以当前后端代码和 `.env.example` 为准；不得重新引入已移除的分层记忆表或兼容接口。

## 内容结构

1. 项目简介、能力概览和技术栈。
2. GitHub Mermaid 系统架构图：设备/网页入口、FastAPI、对话编排、统一记忆、外部模型与 SQLite 的边界。
3. Mermaid 单轮时序图：握手与消息进入后，身份门控、工作上下文、召回规划、Prompt、流式回复与 TTS、异步沉淀的顺序。
4. Mermaid 统一记忆流图与核心数据模型图。
5. 三项设计亮点：身份门控的访客隔离、FTS5 与向量混合检索的时间/关系增强、低延迟主链路与非阻塞后台沉淀。
6. WebSocket 消息类型、快速开始、配置、验证命令与模块地图。
7. 安全边界、部署文档链接和扩展方向。

## 架构表述约定

- `messages` 只表示当前会话的 Working Context；`memory_items` 是长期写入和召回的唯一记忆来源。
- `person_profiles`、`relationship_states`、`open_loops` 是状态寄存器；`memory_relations` 只连接 `memory:<uuid>`、`entity:` 和 `relationship:` 节点。
- 未验证访客只读取本会话上下文，不触发长期召回或 embedding；已验证身份可使用完整记忆包。
- 首次回复路径中的数据库同步操作通过 `asyncio.to_thread()` 迁出事件循环；记忆提取、压缩、画像更新等工作不阻塞回复。

## 文档范围

- 修改：`personal-agent-backend/README.md`。
- 不新增运行时依赖、图片文件、API 或代码逻辑。
- 图表使用 GitHub 原生 Mermaid，确保仓库页面可直接渲染。

## 验证

- 检查 Markdown 标题层级、内部链接、Mermaid 代码块和命令路径。
- 对照 `app/main.py`、`app/agent.py`、`app/memory/`、`app/session.py`、`.env.example` 与测试/审计脚本进行事实核验。
- 对 README 改动进行文档审查：术语一致性、敏感信息泄露、不可执行命令与误导性架构表述。
