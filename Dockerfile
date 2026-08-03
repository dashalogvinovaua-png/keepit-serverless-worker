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
      && (git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git \
          || git clone --depth 1 https://codeberg.org/Gourieff/comfyui-reactor.git ComfyUI-ReActor) \
      && (cd ComfyUI-ReActor && git fetch -q --depth 1 origin 6ad6b35a4df2 && git checkout -q FETCH_HEAD && cd .. || cd /comfyui/custom_nodes) \
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

# ── ЗВУК: ноды речи и музыки для сервиса audio (platform/services/audio) ─────
# Речь  — ComfyUI-Qwen3-TTS (Qwen3-TTS 1.7B, Apache 2.0): готовые тембры, голос по описанию, клон.
# Музыка — ComfyUI-RT-HeartMuLa (HeartMuLa-3B, Apache 2.0): песня с вокалом по тексту и тегам.
# Веса (~16 ГБ) в образ НЕ кладём — они на сетевом томе, см. dl_audio_models.py.
#
# ЭТО ВТОРОЙ ЗАХОД. Первый (f33ac72) валидатор принял, но сборка упала через десять минут внутри
# docker, и вероятная причина названа в откате b40d6bb: авторская проверка «torch не сдвинулся»
# стояла как assert и обязана была валить сборку. Здесь та же цель достигнута иначе:
#   1) requirements ставятся С ОГРАНИЧЕНИЯМИ, а если резолвер упирается — вторым заходом с
#      --no-deps: так пакет ноды встаёт, но НИЧЕГО не тянет за собой и сдвинуть torch не может;
#   2) вместо assert идёт ВОЗВРАТ зафиксированных версий: строка constraints ставится как
#      requirements, и torch/numpy/transformers возвращаются к тем, что были до аудио-нод;
#   3) все шаги мягкие. Сломанная аудио-нода не должна ронять сборку — на этом образе живут
#      video и photo, и лучше выпустить образ без звука, чем оставить ферму без картинки.
# Форма закрепления коммитов оставлена ровно та, которую валидатор RunPod принял: поверхностный
# клон плюс дозагрузка коммита. Полный `git clone` он отвергает целиком.
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/DarioFT/ComfyUI-Qwen3-TTS.git \
 && (cd ComfyUI-Qwen3-TTS && git fetch -q --depth 1 origin 17c22adb80a6 && git checkout -q FETCH_HEAD \
     || echo "?? закреплённый коммит Qwen3-TTS не дозагрузился — остаюсь на ветке по умолчанию") \
 && cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/monnky/ComfyUI-RT-HeartMuLa.git \
 && (cd ComfyUI-RT-HeartMuLa && git fetch -q --depth 1 origin 64e5419bf4aa && git checkout -q FETCH_HEAD \
     || echo "?? закреплённый коммит HeartMuLa не дозагрузился — остаюсь на ветке по умолчанию") \
 && echo "AUDIO: ноды склонированы"

# Что было ДО аудио-нод — записываем, этим же файлом потом и восстановим.
RUN pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers)==' > /tmp/audio_con.txt; \
    echo "── зафиксировано перед аудио-нодами ──"; cat /tmp/audio_con.txt

# Речь. Сначала честная установка с ограничениями, при отказе резолвера — без зависимостей.
RUN grep -viE '^(torch|torchvision|torchaudio|numpy|transformers)([<>=!].*)?$' \
      /comfyui/custom_nodes/ComfyUI-Qwen3-TTS/requirements.txt > /tmp/req-tts.txt; \
    pip install --no-cache-dir -c /tmp/audio_con.txt -r /tmp/req-tts.txt \
      || pip install --no-cache-dir --no-deps -r /tmp/req-tts.txt \
      || echo "!! requirements Qwen3-TTS встали не полностью — проверка старта покажет, жива ли нода"

# Музыка. Тот же порядок.
RUN grep -viE '^(torch|torchvision|torchaudio|numpy|transformers)([<>=!].*)?$' \
      /comfyui/custom_nodes/ComfyUI-RT-HeartMuLa/requirements.txt > /tmp/req-mula.txt; \
    pip install --no-cache-dir -c /tmp/audio_con.txt -r /tmp/req-mula.txt \
      || pip install --no-cache-dir --no-deps -r /tmp/req-mula.txt \
      || echo "!! requirements HeartMuLa встали не полностью — проверка старта покажет, жива ли нода"

# ВОЗВРАТ ВЕРСИЙ вместо падения сборки. Если что-то всё же подвинуло torch/numpy/transformers —
# ставим обратно те, что были. Сдвиг torch кладёт Wan, LTX, SCAIL и ReActor разом, поэтому шаг
# обязателен; но он ЧИНИТ, а не валит, и потому безопасен для чужих веток.
RUN pip install --no-cache-dir -r /tmp/audio_con.txt || echo "!! вернуть зафиксированные версии не удалось"
RUN python3 -c "import torch; print('torch после аудио-нод:', torch.__version__)"

# Папки, куда ноды смотрят за весами на томе (extra_model_paths.yaml уже указывает на том).
RUN mkdir -p /comfyui/models/Qwen3-TTS /comfyui/models/HeartMuLa

# ── ПРАВДА О СБОРКЕ В ЛОГЕ ───────────────────────────────────────────────────
# Поднимаем ComfyUI в режиме CI (только загрузка нод, без сервера) и смотрим, зарегистрировалась ли
# нода. Проверка МЯГКАЯ — билд не валит, но в логе сборки сразу видно, поедет face-swap или нет.
RUN (cd /comfyui && timeout 600 python main.py --quick-test-for-ci --cpu > /tmp/ci.log 2>&1 || true); \
    if grep -qi "reactor" /tmp/ci.log; then \
      echo "REACTOR: ComfyUI ноду увидел" >> /reactor_status.txt; \
    else \
      echo "REACTOR: в логе загрузки нод ReActor не видно" >> /reactor_status.txt; \
    fi; \
    grep -iE "reactor|insightface|import failed|traceback" /tmp/ci.log | head -30 || true; \
    echo "─── итог ───"; cat /reactor_status.txt; \
    echo "─── аудио-ноды ───"; \
    grep -qi "qwen3" /tmp/ci.log && echo "AUDIO: ComfyUI увидел ноды Qwen3-TTS" || echo "AUDIO: нод Qwen3-TTS в логе НЕТ"; \
    grep -qi "heartmula" /tmp/ci.log && echo "AUDIO: ComfyUI увидел ноды HeartMuLa" || echo "AUDIO: нод HeartMuLa в логе НЕТ"; \
    grep -iE "qwen3|heartmula" /tmp/ci.log | head -20 || true

# requests уже есть в базовом образе (использует стоковый handler).
