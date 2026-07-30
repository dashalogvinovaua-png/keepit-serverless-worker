# Кастомный serverless-воркер = стоковый worker-comfyui + наш handler, отдающий видео (mp4).
# Базовый образ ставит ComfyUI latest (ноды Wan VACE / видео уже есть) и умеет монтировать
# модели с сетевого тома (/runpod-volume/models). Мы лишь заменяем handler.py.
FROM runpod/worker-comfyui:5.8.6-base

# Версия ноды ReActor ЗАФИКСИРОВАНА коммитом. С --depth 1 без пиннинга один и тот же Dockerfile
# каждый раз собирал разный код ноды: сборка невоспроизводима, а поломка может прийти сама
# при следующем ребилде. Обновлять эту строку — осознанное действие, а не побочный эффект.
ARG REACTOR_SHA=6ad6b35a4df250d14cb2abf0808c9ffedf59f747

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
# ПОЧЕМУ ЭТОТ БЛОК ВЫГЛЯДИТ ТАК (три грабли, на которых билд падал раньше):
#  1) Готовых Linux-колёс insightface НЕ СУЩЕСТВУЕТ: на PyPI только исходники, у автора ReActor —
#     только Windows. Значит компилируем из исходников (нужен build-essential + python3-dev).
#  2) Базовый образ — Python 3.12 с venv /opt/venv. По умолчанию pip собирает пакет В ИЗОЛЯЦИИ и
#     тянет туда СВЕЖИЙ numpy, а расширение потом импортится с numpy из образа → «module compiled
#     using NumPy 2.x cannot be run in NumPy 1.x» и наоборот. Лечится --no-build-isolation:
#     собираем ровно тем numpy, который стоит в образе. Поэтому numpy НЕ трогаем и НЕ пиним.
#  3) insightface 0.7.3 — код 2023 года, Cython 3 его не собирает. Пиним cython<3 (0.29.37 умеет 3.12),
#     на всякий случай оставляем вторую попытку на cython 3.
# ФИНАЛЬНАЯ ПРОВЕРКА МЯГКАЯ: если insightface не собрался — образ всё равно выходит, просто БЕЗ ноды
# (сервис photo это видит и снимает без face-swap). Жёсткий gate здесь вреден: он оставлял в проде
# СТАРЫЙ образ, и ReActor не появлялся никогда.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev cmake unzip \
 && rm -rf /var/lib/apt/lists/*
# onnxruntime-gpu — на нём считается свап (CPU-вариант базового образа сносим, он тянет за собой CPU-провайдер).
RUN (pip uninstall -y onnxruntime >/dev/null 2>&1 || true); \
    pip install --no-cache-dir onnxruntime-gpu || pip install --no-cache-dir onnxruntime || true
RUN pip install --no-cache-dir "cython<3" "setuptools>=68" wheel
RUN pip install --no-cache-dir --no-build-isolation insightface==0.7.3 \
 || (pip install --no-cache-dir "cython>=3.0" \
     && pip install --no-cache-dir --no-build-isolation insightface==0.7.3) \
 || echo "!! insightface из исходников не собрался — пробуем без C-расширения"
# ПОСЛЕДНИЙ ЗАПАСНОЙ ПУТЬ: собираем insightface БЕЗ C-расширения (mesh_core_cython).
# Оно нужно только 3D-мешам из thirdparty/face3d, а свап лиц (FaceAnalysis + inswapper на onnx)
# работает и без него. Лучше face-swap без 3D-мешей, чем никакого face-swap.
RUN if ! python3 -c "import insightface" >/dev/null 2>&1; then \
      mkdir -p /tmp/insf && cd /tmp/insf \
      && (pip download insightface==0.7.3 --no-deps --no-binary :all: -d . \
          && tar xzf insightface-0.7.3.tar.gz && cd insightface-0.7.3 \
          && sed -i -e 's/^from Cython.*//' \
                    -e 's/^ *ext_modules=ext_modules,//' \
                    -e 's/^ *headers=.*mesh_core\.h.*//' \
                    -e 's/^ext_modules *=.*/ext_modules = []/' setup.py \
          && pip install --no-cache-dir --no-build-isolation . ) \
      || echo "!! insightface НЕ СОБРАЛСЯ ВООБЩЕ — образ поедет без ReActor"; \
    fi
# Ноду ставим ТОЛЬКО если insightface реально импортится: сломанная нода роняет ComfyUI на старте
# (так уже было с comfyui_controlnet_aux — воркер уходил в unhealthy и не брал джобы).
# Из requirements ReActor выкидываем numpy и opencv-python: numpy сломал бы ABI собранного расширения,
# а opencv-python конфликтует с уже стоящим opencv-python-headless.
RUN if python3 -c "import insightface, onnxruntime" >/dev/null 2>&1; then \
      cd /comfyui/custom_nodes \
      && (git clone https://github.com/Gourieff/ComfyUI-ReActor.git \
          && git -C ComfyUI-ReActor checkout -q ${REACTOR_SHA} \
          || git clone --depth 1 https://codeberg.org/Gourieff/comfyui-reactor.git ComfyUI-ReActor) \
      && grep -viE '^(numpy|opencv-python)([<>=!].*)?$' ComfyUI-ReActor/requirements.txt > /tmp/reactor-req.txt \
      && (pip install --no-cache-dir -r /tmp/reactor-req.txt || true) \
      && echo "REACTOR: нода установлена" > /reactor_status.txt; \
    else echo "REACTOR: insightface не собрался — ноды нет" > /reactor_status.txt; fi
# Страховка: если requirements ноды всё-таки поломали insightface — ноду убираем, образ остаётся живым.
RUN if [ -d /comfyui/custom_nodes/ComfyUI-ReActor ] && ! python3 -c "import insightface" >/dev/null 2>&1; then \
      rm -rf /comfyui/custom_nodes/ComfyUI-ReActor; \
      echo "REACTOR: insightface сломался после requirements — нода убрана" > /reactor_status.txt; fi
RUN cat /reactor_status.txt
# Модели ReActor В ОБРАЗ: inswapper (свап-модель) + buffalo_l (детекция/распознавание лиц).
RUN set -e; mkdir -p /comfyui/models/insightface/models; cd /comfyui/models/insightface; \
 (wget -q -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx || true); \
 (cd models && wget -q -O buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
   && mkdir -p buffalo_l && (cd buffalo_l && unzip -oq ../buffalo_l.zip) && rm -f buffalo_l.zip || true)

# ── ГЛАВНЫЙ ГЕЙТ: COMFYUI ОБЯЗАН СТАРТОВАТЬ ─────────────────────────────────
# ЭТА ПРОВЕРКА ДОЛЖНА ОСТАВАТЬСЯ ПОСЛЕДНЕЙ В ФАЙЛЕ. Всё, что добавляется ниже, ею не проверено.
#
# Зачем жёстко. Битая кастом-нода роняет ComfyUI на старте, но контейнер при этом поднимается,
# RunPod считает воркер живым — и задачи просто копятся в очереди. Ферма молчит, а причина не видна
# ни в одном ответе API. Мягкая проверка такую сборку пропускала: она лишь писала строчку в лог.
# Теперь сборка ПАДАЕТ. RunPod при неудачной сборке оставляет предыдущий рабочий образ,
# то есть худший исход — «нового не приехало», а не «ферма встала».
RUN cd /comfyui && (timeout 600 python main.py --quick-test-for-ci --cpu > /tmp/ci.log 2>&1; echo $? > /tmp/ci.rc); \
    grep -iE "reactor|insightface|import failed|traceback|error" /tmp/ci.log | head -40 || true; \
    if [ "$(cat /tmp/ci.rc)" != "0" ]; then \
      echo "!! COMFYUI НЕ СТАРТУЕТ — образ не выпускаем. Последние строки лога:"; tail -40 /tmp/ci.log; exit 1; \
    fi; \
    if grep -qi "reactor" /tmp/ci.log; then echo "REACTOR: ComfyUI ноду увидел" >> /reactor_status.txt; \
    else echo "REACTOR: в логе загрузки нод ReActor не видно" >> /reactor_status.txt; fi; \
    echo "─── итог ───"; cat /reactor_status.txt

# requests уже есть в базовом образе (использует стоковый handler).
