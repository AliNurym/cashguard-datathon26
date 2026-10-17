#!/bin/bash
# Установка CashGuard на VM Google Cloud с GPU (Ubuntu, образ Deep Learning VM с драйвером NVIDIA).
# Запуск из распакованного архива:  bash deploy/gcp_setup.sh
# Повторный запуск безопасен: что уже сделано, пропускается.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KEY_FILE=/etc/cashguard/api_key
step() { printf '\n=== %s\n' "$1"; }

step "1/7 GPU"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "Драйвер NVIDIA не найден. Образ Deep Learning VM предлагает установить его при первом входе по SSH:"
  echo "выйдите, зайдите снова, ответьте y на вопрос про драйвер NVIDIA, затем запустите этот скрипт ещё раз."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step "2/7 uv и Python 3.12"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$APP_DIR"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv

step "3/7 Зависимости"
uv pip install --python .venv/bin/python -r requirements-service.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch не видит CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

step "4/7 Веса моделей (~10 ГБ, один раз)"
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_models.py

step "5/7 Ключ API"
if [ ! -f "$KEY_FILE" ]; then
  sudo mkdir -p /etc/cashguard
  openssl rand -hex 16 | sudo tee "$KEY_FILE" >/dev/null
fi
sudo chmod 644 "$KEY_FILE"

step "6/7 Служба cashguard (systemd)"
sudo tee /etc/systemd/system/cashguard.service >/dev/null <<EOF
[Unit]
Description=CashGuard
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment=HF_HUB_OFFLINE=1
ExecStart=/bin/sh -c 'CASHGUARD_API_KEY=\$(cat $KEY_FILE) exec $APP_DIR/.venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8000'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now cashguard
sudo systemctl restart cashguard

step "7/7 HTTPS (Caddy, адрес вида <ip>.sslip.io)"
IP=$(curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
HOST="${IP//./-}.sslip.io"
if ! command -v caddy >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y caddy
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$HOST {
    request_body {
        max_size 200MB
    }
    reverse_proxy 127.0.0.1:8000
}
EOF
sudo systemctl enable --now caddy
sudo systemctl reload caddy || sudo systemctl restart caddy

echo; echo "Жду, пока воркер загрузит модели..."
for i in $(seq 1 90); do
  if curl -s http://127.0.0.1:8000/api/v1/health | grep -q '"ready"'; then break; fi
  sleep 2
done
curl -s http://127.0.0.1:8000/api/v1/health; echo

cat <<EOF

Готово.
  Интерфейс:  https://$HOST/
  API:        https://$HOST/api/v1/analyze   и   https://$HOST/api/v1/jobs/{job_id}
  Ключ:       $(cat $KEY_FILE)        (заголовок X-API-Key)
Логи: journalctl -u cashguard -f      Перезапуск: sudo systemctl restart cashguard
EOF
