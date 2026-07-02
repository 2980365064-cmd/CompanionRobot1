#!/bin/bash
# Run on ECS: bash /opt/sparkbot/personal-agent-backend/deploy/setup_es.sh
set -e

COMPOSE_FILE="$(dirname "$0")/docker-compose.elasticsearch.yml"

echo "==> Check swap (recommended on 2GB ECS)"
if ! swapon --show | grep -q /swapfile; then
  if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile 2>/dev/null || true
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
free -h

echo "==> Start Elasticsearch"
cd "$(dirname "$0")"
if ! docker-compose -f docker-compose.elasticsearch.yml up -d 2>/dev/null; then
  echo "Compose pull failed, trying Bitnami image..."
  docker stop sparkbot-elasticsearch 2>/dev/null || true
  docker rm sparkbot-elasticsearch 2>/dev/null || true
  docker run -d \
    --name sparkbot-elasticsearch \
    -p 127.0.0.1:9200:9200 \
    -e ELASTICSEARCH_ENABLE_SECURITY=false \
    -e ELASTICSEARCH_HEAP_SIZE=512m \
    -v es_data:/bitnami/elasticsearch/data \
    --restart unless-stopped \
    --memory 768m \
    bitnami/elasticsearch:8.14.0
fi

echo "==> Wait for ES (up to 90s)"
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:9200 >/dev/null 2>&1; then
    echo "Elasticsearch is up."
    curl -s http://127.0.0.1:9200 | head -c 120
    echo
    exit 0
  fi
  sleep 3
done
echo "ES not ready. Check: docker logs sparkbot-elasticsearch"
exit 1
