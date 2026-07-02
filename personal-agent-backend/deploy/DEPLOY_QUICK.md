# 服务器上传与启动（快速）

仓库根目录需包含 **`persona/`** 与 **`personal-agent-backend/`** 同级。

## 一、上传到服务器

### 方式 A：Git（推荐）

```bash
# 服务器上
cd /opt/sparkbot   # 或你的目录
git clone <你的仓库地址> .
# 已有仓库则:
git pull
```

### 方式 B：本机 rsync（Windows 可用 WSL / Git Bash）

```bash
# 在本机项目上一级目录执行，将 陪伴机器人 同步到服务器
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'agent.db' --exclude '.env' \
  ./陪伴机器人/ user@你的服务器IP:/opt/sparkbot/
```

### 方式 C：只打包后端 + persona

```bash
tar czf sparkbot-deploy.tgz persona personal-agent-backend \
  --exclude='personal-agent-backend/.venv' \
  --exclude='personal-agent-backend/agent.db' \
  --exclude='personal-agent-backend/__pycache__'
scp sparkbot-deploy.tgz user@服务器:/opt/
ssh user@服务器 "cd /opt && tar xzf sparkbot-deploy.tgz"
```

**不要上传** `.env`（含密钥）；在服务器上单独创建。

---

## 二、配置环境变量

```bash
cd /opt/sparkbot/personal-agent-backend
cp .env.example .env
nano .env   # 或 vim
```

至少填写：

- `LLM_API_KEY`
- `EMBED_API_KEY`（推荐，否则向量质量差）
- `API_TOKEN`（与机器人/测试页一致）
- `PORT=8001`（与 docker / 防火墙一致）

`PERSONA_DIR=../persona` 保持默认即可（相对 backend 目录）。

首次有语料时（可选，启动也会尝试自动 ingest）：

```bash
cd /opt/sparkbot/personal-agent-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py
```

---

## 三、启动方式（二选一）

### 1. Docker（省事）

```bash
cd /opt/sparkbot/personal-agent-backend
docker compose up -d --build
docker compose logs -f agent
curl http://127.0.0.1:8001/health
```

浏览器：`http://服务器IP:8001/`（需安全组/防火墙放行 **8001**）

### 2. systemd + 虚拟环境（无 Docker）

```bash
cd /opt/sparkbot/personal-agent-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

sudo cp deploy/sparkbot-agent.service /etc/systemd/system/
# 编辑 service 里 WorkingDirectory / User / ExecStart 路径与你的安装目录一致
sudo nano /etc/systemd/system/sparkbot-agent.service

sudo systemctl daemon-reload
sudo systemctl enable sparkbot-agent
sudo systemctl start sparkbot-agent
sudo systemctl status sparkbot-agent
```

默认 service 监听 `127.0.0.1:8000`，若用 `.env` 的 `PORT=8001`，请把 `ExecStart` 改为 `--port 8001`，前面加 nginx 反代 443。

---

## 四、对外 HTTPS（机器人 wss）

1. 域名解析到服务器  
2. 按 [nginx.conf](nginx.conf) 配置反代到 `127.0.0.1:8001`  
3. `certbot --nginx -d agent.你的域名.com`  
4. 固件 WebSocket：`wss://agent.你的域名.com/ws/v1/chat`

---

## 五、验证

```bash
curl http://127.0.0.1:8001/health
# llm.configured / embed.configured 应为 true

cd personal-agent-backend && source .venv/bin/activate
python scripts/test_ws_client.py --url ws://127.0.0.1:8001/ws/v1/chat --token 你的API_TOKEN
```

---

## 六、更新代码后

```bash
cd /opt/sparkbot && git pull
cd personal-agent-backend
docker compose up -d --build    # Docker
# 或
sudo systemctl restart sparkbot-agent
```

数据库 `agent.db` 在 volume/本地文件中，更新代码**不会**自动清库；需要重置见 `deploy/reset_memory.sh`。
