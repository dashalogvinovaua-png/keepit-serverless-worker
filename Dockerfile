# Кастомный serverless-воркер = стоковый worker-comfyui + наш handler, отдающий видео (mp4).
# Базовый образ ставит ComfyUI latest (ноды Wan VACE / видео уже есть) и умеет монтировать
# модели с сетевого тома (/runpod-volume/models). Мы лишь заменяем handler.py.
FROM runpod/worker-comfyui:5.8.6-base

# Наш handler умеет собирать ЛЮБОЙ выходной файл (SaveVideo → mp4), а не только images.
COPY handler.py /handler.py

# Наши модели на томе лежат под /runpod-volume/comfyui/models — указываем ComfyUI искать их там
# (базовый ищет в /runpod-volume/models, где у нас только битый абсолютный симлинк).
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# Патч float.py от OOM на тяжёлых моделях (как на поде): выключаем stochastic rounding fp8,
# который даёт пик памяти и роняет воркер на Wan/LTX/SCAIL 14B+.
RUN if [ -f /comfyui/comfy/float.py ] && grep -q "_CK_STOCHASTIC_ROUNDING_AVAILABLE = True" /comfyui/comfy/float.py; then \
      sed -i "0,/_CK_STOCHASTIC_ROUNDING_AVAILABLE = True/s//_CK_STOCHASTIC_ROUNDING_AVAILABLE = False  # patched OOM/" /comfyui/comfy/float.py; \
    fi

# ── КАСТОМ-НОДЫ КАЧЕСТВА (макс улучшение результата) ──
# facerestore_cf — восстановление лиц (CodeFormer): чинит ЗУБЫ, ГЛАЗА, кожу на AI-выходе (Wan/SCAIL).
# ВНИМАНИЕ: comfyui_controlnet_aux УБРАН — ронял ComfyUI на старте воркера (unhealthy → джобы не берутся).
#   Нужен был только для control=pose / hand-refiner (сейчас НЕ используются — движение идёт через control=raw).
# Модели к ним кладём на СЕТЕВОЙ ТОМ (codeformer.pth, dwpose, upscale) — см. scripts/dl_quality_models.py.
# БЕЗ || true на clone → если клон упадёт, билд УПАДЁТ видимо (а не «зелёный» без ноды).
# ВАЖНО: ставим requirements САМОЙ facerestore_cf (без них нода падает на импорте → ComfyUI её не видит).
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/mav-rik/facerestore_cf.git \
 && pip install --no-cache-dir facexlib onnxruntime opencv-python-headless \
 && (pip install --no-cache-dir -r facerestore_cf/requirements.txt || true) \
 && python3 -c "import facexlib" \
 && ls facerestore_cf/*.py

# Модели качества ВШИВАЕМ В ОБРАЗ (не на том) — не нужен под для скачки, работает сразу после ребилда.
# CodeFormer+facexlib (лица), RealESRGAN (апскейл), DWPose (движение). ~600МБ, приемлемо.
RUN set -e; cd /comfyui/models; \
 mkdir -p facerestore_models facedetection upscale_models; \
 (wget -q -O facerestore_models/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth || true); \
 (wget -q -O facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth || true); \
 (wget -q -O facedetection/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth || true); \
 (wget -q -O upscale_models/RealESRGAN_x4plus.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth || true)

# ── ReActor: ТОЧНАЯ пересадка лица (face-swap, InsightFace inswapper) → 100% совпадение лица блогера ──
# После генерации копии вставляем ТОЧНОЕ лицо блогера в каждый кадр (не «похожее», а его).
# insightface==0.7.3 собирается ИЗ ИСХОДНИКОВ (нет готового wheel) → нужны компилятор + заголовки + numpy/cython.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev cmake unzip \
 && rm -rf /var/lib/apt/lists/*
# onnxruntime-gpu вместо cpu (facerestore ставил cpu) — иначе свап медленный. Отдельным шагом.
RUN pip uninstall -y onnxruntime 2>/dev/null || true; \
    pip install --no-cache-dir onnxruntime-gpu
# numpy<2 + cython ДО insightface (иначе сборка падает), затем сам insightface.
RUN pip install --no-cache-dir "numpy<2" cython \
 && pip install --no-cache-dir insightface==0.7.3
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git \
 && (pip install --no-cache-dir -r ComfyUI-ReActor/requirements.txt || true) \
 && python3 -c "import insightface, onnxruntime; print('reactor deps ok')" \
 && ls ComfyUI-ReActor/*.py
# Модели ReActor В ОБРАЗ: inswapper (свап-модель) + buffalo_l (детекция/распознавание лиц).
RUN set -e; mkdir -p /comfyui/models/insightface/models; cd /comfyui/models/insightface; \
 (wget -q -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx || true); \
 (cd models && wget -q -O buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
   && mkdir -p buffalo_l && (cd buffalo_l && unzip -oq ../buffalo_l.zip) && rm -f buffalo_l.zip || true)

# requests уже есть в базовом образе (использует стоковый handler).
