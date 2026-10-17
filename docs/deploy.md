# План деплоя на публичный URL

Требования ТЗ: публичный URL с интерфейсом и API, ключ `X-API-Key`, результат по каждому видео не позже 5 минут, модели только локальные (облачные AI-API запрещены; арендованный сервер, на котором наши модели работают сами, этому не противоречит). Машина должна работать всё время проверки.

## Варианты

| | A. Арендованный GPU-сервер (рекомендуется) | B. Этот Mac (M4 Max) + туннель |
|---|---|---|
| Скорость | быстрее MPS (VLM и детекторы на CUDA) | проверено: см. замеры ниже |
| Надёжность | дата-центр, стабильная сеть, не засыпает | зависит от ноутбука, питания и домашней сети (во время работы сеть дважды обрывалась) |
| Адрес | постоянный IP/домен, HTTPS через Caddy | постоянный только с именованным туннелем (нужен аккаунт и домен в Cloudflare) |
| Что нужно от команды | аккаунт у провайдера, оплата | аккаунт Cloudflare + домен, держать Mac включённым |
| Время развёртывания | ~1 час | ~20 минут |

Рекомендация: **A как основной, B как запасной** (поднят параллельно, второй URL держим наготове).

## A. GPU-сервер

Для Google Cloud есть пошаговая инструкция без командной строки на Mac: [deploy_gcp.md](deploy_gcp.md) (машина g2-standard-8 с NVIDIA L4, установка одной командой `deploy/gcp_setup.sh`). Ниже — то же вручную для любого провайдера.

Минимум: NVIDIA с 16 ГБ видеопамяти (Qwen3-VL-4B ~9 ГБ в fp16 + детекторы и DINOv2 ~2 ГБ), 32 ГБ ОЗУ, 40 ГБ диска, Ubuntu 22.04/24.04, драйвер NVIDIA и CUDA 12.

```bash
# 1. Код и окружение
git clone <репозиторий> cashguard && cd cashguard          # или scp папки без data/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-service.txt   # torch подтянет CUDA-сборку для Linux
# 2. Веса — один раз (~9 ГБ); дальше сервис работает без интернета (HF_HUB_OFFLINE=1 в воркере)
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; [d(r) for r in ('PekingU/rtdetr_v2_r50vd','google/owlv2-base-patch16-ensemble','facebook/dinov2-base','Qwen/Qwen3-VL-4B-Instruct')]"
# 3. Классификаторы: скопировать models/safe_clf.joblib и models/money_clf.joblib
# 4. Ключ для жюри
openssl rand -hex 16 > /etc/cashguard.key
```

Служба systemd `/etc/systemd/system/cashguard.service`:

```ini
[Unit]
Description=CashGuard
After=network.target

[Service]
WorkingDirectory=/opt/cashguard
ExecStart=/bin/sh -c 'CASHGUARD_API_KEY=$(cat /etc/cashguard.key) exec /opt/cashguard/.venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8000'
Restart=always

[Install]
WantedBy=multi-user.target
```

HTTPS через Caddy (сертификат выпускается автоматически), `/etc/caddy/Caddyfile`:

```
cashguard.<домен> {
    request_body max_size 200MB
    reverse_proxy 127.0.0.1:8000
}
```

```bash
systemctl enable --now cashguard caddy
curl https://cashguard.<домен>/api/v1/health      # "worker": "ready"
```

## B. Этот Mac + Cloudflare Tunnel

```bash
brew install cloudflared
cloudflared tunnel login                         # аккаунт Cloudflare, домен в нём
cloudflared tunnel create cashguard
cloudflared tunnel route dns cashguard cashguard.<домен>
CASHGUARD_API_KEY=$(cat <файл ключа>) .venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8000 &
cloudflared tunnel run --url http://127.0.0.1:8000 cashguard
caffeinate -dimsu &                              # не засыпать; питание от сети
```

Быстрый туннель без аккаунта (`cloudflared tunnel --url http://127.0.0.1:8000`) даёт случайный адрес `*.trycloudflare.com`, который меняется при перезапуске, — для сдачи не годится, только для проверки.

## Пропускная способность и лимит 5 минут

Очередь обрабатывается одним воркером по порядку, поэтому время ответа на последнее видео пачки ≈ сумма времён обработки всех видео перед ним.

Замеры на Apple M4 Max (MPS), детекторы 2.5/1 кадр/с, все правила включены (`scripts/api_e2e.py`):

| Сценарий | Результат |
|---|---|
| 40 роликов по одному | 40/40 `done`, медиана 12 с, максимум 75 с (ролик 143 с, 29 МБ), всего 652 с |
| 6 самых длинных роликов одновременно | последний готов через 205 с |
| Оценка для видео ~29 МБ (~140 с) | ~75 с каждое → в 5 минут укладываются ~4 одновременно отправленных |

Если жюри отправит пачку больше, чем успевает обработать один воркер за 5 минут:

1. GPU-сервер (вариант A) сокращает время одного видео.
2. `CASHGUARD_OWL_FPS=0.5` — ещё ~15% времени; качество на обучающих роликах то же (проверено прореживанием кэша, сквозным прогоном — нет).
3. `CASHGUARD_VLM=0` экономит 0–5 вопросов VLM на ролик, но выключает правила L и H — только как крайняя мера.
4. Второй экземпляр сервиса на второй машине за балансировщиком (Caddy `reverse_proxy` с двумя адресами) — задачи привязаны к экземпляру, поэтому нужен sticky-routing по `job_id`; на хакатон проще ограничиться одной быстрой машиной.

## Перед отправкой адреса жюри

- [ ] `GET /api/v1/health` → `"worker": "ready"`
- [ ] `scripts/api_e2e.py --base https://<адрес> --key-file <ключ> --mode seq --only X-15,X-3,X-28,X-32` — схема без ошибок, время < 5 мин
- [ ] запрос без ключа → 401
- [ ] интерфейс открывается с другого устройства и из другой сети
- [ ] сервис переживает перезапуск (systemd / повторный запуск), задачи на диске
- [ ] выключить исходящий интернет на сервере и прогнать один ролик (проверка локальности)
