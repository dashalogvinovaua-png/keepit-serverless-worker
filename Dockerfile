# Кастомный serverless-воркер = стоковый worker-comfyui + наш handler, отдающий видео (mp4).
# Базовый образ ставит ComfyUI latest (ноды Wan VACE / видео уже есть) и умеет монтировать
# модели с сетевого тома (/runpod-volume/models). Мы лишь заменяем handler.py.
FROM runpod/worker-comfyui:5.8.6-base

# Наш handler умеет собирать ЛЮБОЙ выходной файл (SaveVideo → mp4), а не только images.
COPY handler.py /handler.py

# Наши модели на томе лежат под /runpod-volume/comfyui/models — указываем ComfyUI искать их там
# (базовый ищет в /runpod-volume/models, где у нас только битый абсолютный симлинк).
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# requests уже есть в базовом образе (использует стоковый handler). Больше ничего не нужно.
