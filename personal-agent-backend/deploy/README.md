# Cloud deployment for SparkBot Personal Agent

## 1. Server requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+ or Docker
- Domain with DNS A record pointing to server (for wss)

## 2. Docker deploy

```bash
cd personal-agent-backend
cp .env.example .env   # edit keys
docker compose up -d --build
curl http://127.0.0.1:8001/health
```

## 3. nginx + Let's Encrypt (wss)

Use [nginx.conf](nginx.conf) as `/etc/nginx/sites-available/sparkbot-agent`.

```bash
sudo certbot --nginx -d agent.example.com
sudo nginx -t && sudo systemctl reload nginx
```

Firmware menuconfig WS URI: `wss://agent.example.com/ws/v1/chat`

## 4. systemd (non-Docker)

Copy [sparkbot-agent.service](sparkbot-agent.service) to `/etc/systemd/system/`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable sparkbot-agent
sudo systemctl start sparkbot-agent
```

## 5. Firewall

```bash
sudo ufw allow 22
sudo ufw allow 443
sudo ufw enable
```

## 6. Firmware flash

See [../../docs/SETUP.md](../../docs/SETUP.md).

After deploy, verify WebSocket:

```bash
# install wscat: npm i -g wscat
wscat -c wss://agent.example.com/ws/v1/chat
> {"type":"hello","device_id":"test","token":"YOUR_TOKEN","session_id":""}
```
