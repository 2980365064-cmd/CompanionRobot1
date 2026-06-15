# SparkBot Personal Agent Backend

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 LLM_API_KEY、EMBED_API_KEY（默认 SEARCH_BACKEND=sqlite，无需 ES）
python scripts/compress_profile.py   # config + style → config/profile_card.md
python scripts/ingest.py             # 可选；启动时会自动入库
python -m app.main                   # 推荐：控制台仅显示 Agent 监控日志
# 或: uvicorn app.main:app --host 0.0.0.0 --port 8001 --log-level warning --no-access-log
```

**测试页**：浏览器打开 http://127.0.0.1:8001/（若 `.env` 未设 `PORT`，以启动日志里的端口为准）

## 记忆架构

| 层 | 说明 |
|----|------|
| **Profile** | `persona/config/profile_card.md`，每轮固定注入（不进向量库） |
| **L1** | 20 轮滑动窗口；满 20 轮压缩最老 12 轮到 L2 |
| **L2** | 7 天摘要，每轮召回；过期自动汇总到 L3 |
| **L3** | SQLite FTS5+向量（`agent.db`） | 长期语料混合检索 |

详见 [../persona/README.md](../persona/README.md)

## 常用脚本

| 脚本 | 作用 |
|------|------|
| `scripts/compress_profile.py` | 生成/更新 `profile_card.md` |
| `scripts/ingest.py` | 语料入库 L3 |
| `scripts/test_persona.py` | 本地口吻 + 记忆抽测 |

See [deploy/README.md](deploy/README.md) for cloud deployment.
