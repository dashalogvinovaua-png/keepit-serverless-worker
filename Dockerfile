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
# ЗАКАЛКА: весь блок в guard. Если клон/деп/импорт упадёт — нода УДАЛЯЕТСЯ, ComfyUI грузится чисто
# (иначе повторяем историю controlnet_aux: полуустановленная нода → unhealthy → джобы не берутся).
# НЕ ставим facerestore_cf/requirements.txt вслепую — он тянет basicsr, который ломает torchvision
# импорт и роняет ВЕСЬ ComfyUI на старте. Ставим только безопасные явные зависимости.
RUN cd /comfyui/custom_nodes \
 && ( git clone --depth 1 https://github.com/mav-rik/facerestore_cf.git \
      && pip install --no-cache-dir facexlib onnxruntime opencv-python-headless \
      && python3 -c "import facexlib, onnxruntime" \
      && ls facerestore_cf/*.py \
    ) || ( echo "!! facerestore_cf установка провалилась — удаляю ноду, чтобы ComfyUI грузился" \
      && rm -rf facerestore_cf )

# Модели качества ВШИВАЕМ В ОБРАЗ (не на том) — не нужен под для скачки, работает сразу после ребилда.
# CodeFormer+facexlib (лица), RealESRGAN (апскейл), DWPose (движение). ~600МБ, приемлемо.
RUN set -e; cd /comfyui/models; \
 mkdir -p facerestore_models facedetection upscale_models; \
 (wget -q -O facerestore_models/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth || true); \
 (wget -q -O facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth || true); \
 (wget -q -O facedetection/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth || true); \
 (wget -q -O upscale_models/RealESRGAN_x4plus.pth https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth || true)

# requests уже есть в базовом образе (использует стоковый handler).
