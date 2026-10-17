# Развёртывание в Google Cloud: пошагово

Нужна одна виртуальная машина с видеокартой NVIDIA. Всё делается в веб-консоли <https://console.cloud.google.com>, командную строку на Mac ставить не нужно.

## Какой сервер

| Параметр | Значение |
|---|---|
| Тип машины | **g2-standard-8**: 1 × NVIDIA L4 (24 ГБ видеопамяти), 8 vCPU, 32 ГБ ОЗУ |
| Диск | 100 ГБ, образ **Deep Learning on Linux** (в нём уже есть CUDA и установщик драйвера NVIDIA) |
| Регион | любой, где есть свободные L4: например `europe-west4` (Нидерланды) или `us-central1` |
| Режим | обычный (не Spot: Spot-машину Google может выключить в любой момент, в том числе во время проверки) |
| Цена | порядка 1 $ в час; точную оценку консоль показывает справа при создании. После проверки машину остановить |

Запасной вариант, если L4 недоступна: `n1-standard-8` + 1 × NVIDIA T4 (16 ГБ). Он медленнее; кроме того, правила L и H на T4 не проверялись.

## 0. Перед началом (это может занять время — сделать сразу)

1. **Платёжный аккаунт без пробного режима.** На Free Trial видеокарты недоступны: нужен переход на платный аккаунт (кнопка Upgrade / «Активировать полный аккаунт» в Billing).
2. **Квота на GPU.** У новых проектов квота на видеокарты часто нулевая.
   - Откройте IAM & Admin → **Quotas & System Limits**.
   - Найдите `GPUs (all regions)` и запросите **1**.
   - Найдите `NVIDIA L4 GPUs` в выбранном регионе и тоже запросите **1**.
   - Одобрение обычно приходит от нескольких минут до нескольких часов. Без него машину с GPU создать нельзя.

## 1. Создать машину

Compute Engine → **VM instances** → **Create instance**:

1. Name: `cashguard`. Region/Zone: регион из таблицы, любая зона.
2. Machine configuration → вкладка **GPUs** → GPU type `NVIDIA L4`, 1 шт. → машина `g2-standard-8`.
3. Boot disk → **Change** → Operating system `Deep Learning on Linux`, версия с CUDA 12 → размер **100 ГБ**.
4. Firewall: отметить **Allow HTTP traffic** и **Allow HTTPS traffic**.
5. **Create**.

## 2. Закрепить адрес

VPC network → **IP addresses** → найти внешний адрес машины `cashguard` → **Promote to static address**. Иначе адрес сменится после перезапуска машины, и выданная жюри ссылка перестанет работать.

## 3. Установить сервис

1. На Mac: `sh deploy/make_bundle.sh` → файл `dist/cashguard_bundle.tar.gz` (~170 КБ).
2. В консоли напротив машины нажать **SSH**: откроется терминал в браузере.
3. При первом входе образ спросит, установить ли драйвер NVIDIA. Ответить `y`, дождаться окончания.
4. В окне SSH: ⚙ → **Upload file** → выбрать `cashguard_bundle.tar.gz`.
5. Выполнить:

```bash
mkdir -p ~/cashguard && tar -xzf ~/cashguard_bundle.tar.gz -C ~/cashguard
bash ~/cashguard/deploy/gcp_setup.sh
```

Скрипт ставит Python и зависимости, качает веса (~10 ГБ, 5–15 минут), создаёт ключ API, запускает сервис как службу (переживает перезагрузку) и включает HTTPS. В конце печатает:

- адрес интерфейса вида `https://34-12-56-78.sslip.io/` — сервис `sslip.io` превращает IP в доменное имя, свой домен покупать не нужно;
- адрес API;
- ключ для жюри.

## 4. Проверить с Mac

```bash
.venv/bin/python scripts/api_e2e.py --base https://<адрес> --key-file <файл с ключом> --mode seq --only X-15,X-3,X-28,X-32
```

Ожидается: `done` у всех, `schema_bad: []`, 401 без ключа. Затем открыть интерфейс с телефона через мобильный интернет: это проверка, что адрес виден снаружи.

## 5. Примеры на стартовом экране (по желанию)

Загрузить 2–3 ролика через интерфейс (или `POST /ui/api/analyze`), дождаться результата и прописать их `job_id`:

```bash
sudo mkdir -p /etc/systemd/system/cashguard.service.d
printf "[Service]\nEnvironment=CASHGUARD_EXAMPLES=<id1>,<id2>,<id3>\n" | sudo tee /etc/systemd/system/cashguard.service.d/examples.conf
sudo systemctl daemon-reload && sudo systemctl restart cashguard
curl -s https://<адрес>/ui/api/examples
```

Задачи и видео примеров лежат в `~/cashguard/data/service`: пока эта папка цела, примеры переживают перезапуск.

## Если что-то пошло не так

| Симптом | Что сделать |
|---|---|
| Скрипт пишет «Драйвер NVIDIA не найден» | выйти из SSH, зайти снова, ответить `y` на вопрос про драйвер, запустить скрипт ещё раз |
| При создании машины «quota exceeded» или «no resources» | дождаться одобрения квоты (шаг 0) или выбрать другую зону или регион |
| HTTPS не открывается, по `http://<IP>/` тоже пусто | `sudo journalctl -u caddy -n 50`; проверить галочки HTTP/HTTPS в настройках машины |
| Сервис не отвечает | `sudo journalctl -u cashguard -n 100`; перезапуск: `sudo systemctl restart cashguard` |
| Сертификат для sslip.io не выпустился | в `/etc/caddy/Caddyfile` заменить первую строку на `:80`, выполнить `sudo systemctl reload caddy`, выдать жюри `http://<IP>/` |

После окончания проверки: VM instances → `cashguard` → **Stop**, чтобы не платить за простой. Статический адрес тоже стоит денег, пока не освобождён.
