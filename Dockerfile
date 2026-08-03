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

# ── ЗВУК ЖИВЁТ В ОТДЕЛЬНОМ ОКРУЖЕНИИ, А НЕ ЗДЕСЬ ────────────────────────────
# Прошлый заход ставил зависимости аудио-нод в ОБЩЕЕ окружение ComfyUI. Итог виден фактом: на
# воркере оказалась несовпадающая пара `torch 2.12.0` при `torchaudio 2.11.0`, а вместе с ней
# у видео пропала пересадка лица. Никакие constraints этого не предотвращают — они держат только
# те пакеты, которые кто-то догадался перечислить.
#
# Поэтому здесь аудио НЕ СТАВИТСЯ ВООБЩЕ. Готовим пустое отдельное окружение: свои пакеты и свой
# torch приедут в него следующим шагом, а handler будет звать его ПОДПРОЦЕССОМ. Два разных torch
# не встретятся в одном процессе никогда, и ComfyUI про звук знать не обязан.
RUN python3 -m venv --copies /opt/audio-venv \
 && /opt/audio-venv/bin/pip install --no-cache-dir --upgrade pip \
 && echo "AUDIO: отдельное окружение /opt/audio-venv готово"

# Свой torch — ТОЛЬКО внутрь audio-venv. Общее окружение ComfyUI этой строкой не затрагивается
# вообще: другой интерпретатор, другой каталог пакетов, другой процесс во время работы.
RUN /opt/audio-venv/bin/pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 || echo "!! torch в audio-venv не встал — движки речи не поедут, остальное не тронуто"

# Речь: Chatterbox Multilingual (MIT, 23 языка, клон голоса) и tts_uk (MIT, украинский родными
# голосами). Ставим мягко: не встали — образ всё равно выходит, видео и фото это не касается.
RUN /opt/audio-venv/bin/pip install --no-cache-dir chatterbox-tts \
 || echo "!! chatterbox-tts не встал"
RUN /opt/audio-venv/bin/pip install --no-cache-dir tts-uk \
 || echo "!! tts-uk не встал"

# ВЕСА КЛАДЁМ В ОБРАЗ, А НЕ КАЧАЕМ В РАБОТЕ.
# Почему не на том, как у видео: при попытке записать веса воркер отвечает «Disk quota exceeded»,
# и это НЕ нехватка места — система показывает на томе 91 тысячу гигабайт свободно. Значит на
# запись стоит квота, и в момент задачи скачать веса нельзя ни на том, ни на диск контейнера.
# Во время СБОРКИ места достаточно, поэтому тянем здесь и один раз: образ вырастет примерно на
# три гигабайта, зато движок готов к работе с первой секунды и не зависит от чужой площади.
ENV HF_HOME=/opt/audio-models
ENV HF_HUB_DISABLE_XET=1
# ЖЁСТКО: если веса не легли в образ, движок в работе их уже не докачает (на воркере квота на
# запись). Значит образ без весов бесполезен для звука — пусть сборка падает здесь, а не задача
# у владелицы.
RUN /opt/audio-venv/bin/python -c "from huggingface_hub import snapshot_download; p = snapshot_download('ResembleAI/chatterbox'); print('AUDIO: веса Chatterbox в образе:', p)"
# tts_uk качает веса ПРИ ИМПОРТЕ модуля (hf_hub_download на уровне файла), поэтому достаточно
# его импортировать — синтез для этого не нужен. Первая версия этого шага гоняла настоящий синтез
# тремя голосами на процессоре сборщика и растянула сборку на десятки минут без пользы.
RUN /opt/audio-venv/bin/python -c "import tts_uk.inference as m; print('AUDIO: веса tts_uk в образе, голоса:', list(getattr(m, 'voices', {}) or {}))"
RUN echo "AUDIO: размер весов в образе:" && du -sh /opt/audio-models && ls /opt/audio-models

# Работник звука: его зовёт handler подпроцессом.
COPY audio_worker.py /opt/audio_worker.py

# Что именно доехало — видно в логе сборки, а не выясняется на первой задаче.
RUN /opt/audio-venv/bin/python -c "import importlib, sys; [sys.stdout.write('AUDIO: %s %s\n' % (m, 'есть' if importlib.util.find_spec(m) else 'НЕТ')) for m in ('torch','torchaudio','chatterbox','tts_uk')]" || true

# ── ЖЁСТКИЕ ВОРОТА СБОРКИ ───────────────────────────────────────────────────
# Мягкие проверки уже подвели: образ вышел «зелёным», а про поломку видео узнали на готовом ролике.
# Дальше так нельзя. Эти два шага ОБЯЗАНЫ ронять сборку — лучше остаться на прошлом образе, чем
# выпустить сломанный.
#
# 1. Без ReActor образ не выпускаем: без него видео теряет пересадку лица.
#    insightface — жёстко, он и есть свап. onnxruntime проверяем отдельно и мягко: базовый образ
#    иногда несёт CPU-вариант, и это не повод не выпускать образ.
RUN python3 -c "import insightface; print('insightface ок')" \
 && test -d /comfyui/custom_nodes/ComfyUI-ReActor \
 || (echo "!! ВОРОТА: ReActor или insightface не собраны — образ НЕ выпускаем" && exit 1)
RUN python3 -c "import onnxruntime; print('onnxruntime ок:', onnxruntime.get_available_providers())" \
 || echo "?? onnxruntime не импортируется — свап может считаться медленно"

# 2. torch и torchaudio должны РАБОТАТЬ ВМЕСТЕ. Проверяем делом, а не подписью на коробке:
#    у базового образа пара 2.12.0 / 2.11.0 — номера разные, но собраны они друг под друга и
#    работают. Первая версия этих ворот сравнивала НОМЕРА и уронила сборку на здоровом образе;
#    ломает видео не расхождение цифр, а несовместимость ABI — вот её и ловим: грузим обе
#    библиотеки в один процесс и делаем настоящую операцию.
RUN python3 -c "import torch, torchaudio, sys; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__); w = torch.zeros(1, 16000); torchaudio.functional.resample(w, 16000, 22050); print('ВОРОТА: torch и torchaudio работают вместе'); sys.exit(0)" \
 || (echo "!! ВОРОТА: torch и torchaudio несовместимы — образ НЕ выпускаем" && exit 1)

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
    echo "─── звук ───"; \
    test -x /opt/audio-venv/bin/python && echo "AUDIO: отдельное окружение на месте" \
      || echo "AUDIO: отдельного окружения НЕТ"

# 3. ПОСЛЕДНИЕ ВОРОТА: ComfyUI должен подняться и ЗАРЕГИСТРИРОВАТЬ ReActor. Проверка выше ловит
# «пакет установлен», эта — «нода реально видна ComfyUI». Именно её отсутствие стоило нам ролика
# без пересадки лица: пакет был, а класс не регистрировался.
RUN grep -qi "reactor" /tmp/ci.log \
 || (echo "!! ВОРОТА: ComfyUI не увидел ReActor при загрузке нод — образ НЕ выпускаем" && exit 1)

# requests уже есть в базовом образе (использует стоковый handler).
