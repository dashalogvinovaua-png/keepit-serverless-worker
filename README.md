# Serverless-воркер (видео-capable) — сборка и деплой

Стоковый `runpod/worker-comfyui` не отдаёт видео. Этот образ = стоковый + наш `handler.py`,
который возвращает mp4. Модели НЕ включаем в образ — они на сетевом томе (`/runpod-volume/models`).

## Файлы
- `Dockerfile` — `FROM runpod/worker-comfyui:5.8.6-base` + наш handler.
- `handler.py` — собирает любой выходной файл (SaveVideo → mp4) как base64.

---

## Вариант A — RunPod собирает из GitHub (без Docker на Mac) ⭐ рекомендую
1. Создай на GitHub репозиторий (напр. `keepit-serverless-worker`), залей в него ЭТУ папку
   (`Dockerfile` + `handler.py`) в корень.
2. RunPod → **Serverless** → **New Endpoint** → вкладка **Import Git Repository** (или "GitHub Repo").
3. Подключи свой GitHub, выбери репозиторий и ветку. RunPod соберёт образ сам.
4. Дальше настройки endpoint (том, GPU) — см. `../SERVERLESS_SETUP.md`, раздел «Endpoint».

## Вариант B — собрать локально и запушить в Docker Hub (нужен Docker Desktop)
```bash
cd serverless-worker
# ВАЖНО: RunPod GPU = linux/amd64 (Mac arm → нужен buildx с эмуляцией)
docker buildx build --platform linux/amd64 \
  -t <твой_dockerhub_логин>/keepit-comfyui-video:latest --push .
```
Затем в endpoint укажи образ `<логин>/keepit-comfyui-video:latest`.

---

## После сборки — создание endpoint
- Регион: **EU-NL-1** (там наш сетевой том с моделями).
- **Network Volume**: прицепить наш том (тот, что примонтирован к поду).
- GPU: **L40S** (или L40 / RTX 4090 — с нативным fp8; НЕ A40/A100).
- Workers: Min 0, Max 5.
- Deploy → скопировать **Endpoint ID**.

Дай мне `RUNPOD_API_KEY` + `RUNPOD_SERVERLESS_ENDPOINT` — запускаю тест:
```
node serverless_test.mjs data/<клип>.mp4 wan
```
