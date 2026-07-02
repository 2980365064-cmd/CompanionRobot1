#!/bin/bash
# 清空服务端所有对话/记忆缓存（L1/L2/L3/facts），保留 .env
# 用法: bash /opt/sparkbot/personal-agent-backend/deploy/reset_memory.sh
set -e

BACKEND=/opt/sparkbot/personal-agent-backend

echo "==> Stop backend"
systemctl stop sparkbot-agent 2>/dev/null || true

echo "==> Stop Elasticsearch (optional, no longer needed)"
docker stop sparkbot-elasticsearch 2>/dev/null || true
docker rm sparkbot-elasticsearch 2>/dev/null || true

echo "==> Remove memory files"
rm -f "$BACKEND/agent.db"
rm -rf "$BACKEND/chroma_data"
rm -f "$BACKEND/chroma_data/.embed_meta.json" 2>/dev/null || true

echo "==> Cleared. Start backend and run ingest:"
echo "    systemctl start sparkbot-agent"
echo "    cd $BACKEND && source .venv/bin/activate"
echo "    python scripts/compress_profile.py"
echo "    python scripts/ingest.py --reset"
