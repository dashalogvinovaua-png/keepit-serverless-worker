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
# Поэтому здесь аудио НЕ СТАВИТСЯ ВООБЩЕ. Правило осталось прежним: у каждого движка своё
# окружение, свой torch, и handler зовёт его ПОДПРОЦЕССОМ. Два разных torch не встретятся в одном
# процессе никогда, и ComfyUI про звук знать не обязан.
#
# ОБЩЕГО /opt/audio-venv БОЛЬШЕ НЕТ. Он существовал ради Chatterbox и tts_uk, а их владелица
# отклонила. Пустое окружение с собственным torch — это пять гигабайт в образе, за которые мы
# платим ВРЕМЕНЕМ СБОРКИ, а времени у нас в обрез (см. ниже про тридцатиминутный предел). Теперь
# окружение заводит себе каждый движок отбора отдельно, и только тот, который реально стоит.

# CHATTERBOX И tts_uk УБРАНЫ ИЗ ОБРАЗА. Владелица отклонила оба по звучанию: «это мы больше не
# тестируем». Держать их дальше — не просто мёртвый груз, а прямой вред: место в образе у нас
# кончилось (см. ниже про тридцатиминутный предел сборки), и эти три гигабайта нужны новым
# движкам. Вернуть их — один откат этого куска; расшифровок и голосов мы не теряем: образец
# украинской речи для клонов уже лежит отдельным файлом в data/audio/ref/.

# ВЕСА В ОБРАЗ БОЛЬШЕ НЕ КЛАДЁМ — И ВОТ ПОЧЕМУ, ЧТОБЫ НИКТО НЕ ПЕРЕОТКРЫВАЛ ЭТО ЗАНОВО.
# У сборки на RunPod жёсткий предел ТРИДЦАТЬ МИНУТ (проверено: две сборки подряд упали ровно на
# 30.3 минуты с «Build timeout exceeded»). А наши УСПЕШНЫЕ сборки и раньше занимали 15–31 минуту.
# То есть запаса не было ещё до звука: любые лишние гигабайты в образе — это отказ сборки, потому
# что готовый образ ещё надо отправить в реестр, и платим мы временем именно за это.
# Куда тогда веса:
#   * диск контейнера — 5 ГБ (containerDiskInGb в шаблоне), туда 12 ГБ не лягут;
#   * сетевой том — 150 ГБ, и он ПЕРЕПОЛНЕН моделями видео. Отсюда и знаменитое
#     «Disk quota exceeded при 92 000 свободных гигабайт»: 92 000 — это то, что система видит
#     под собой, а наша доля 150 ГБ и она занята. Не квота на запись, а просто полный том.
# Значит веса живут НА ТОМЕ, и класть их туда надо, освободив место, — а образ остаётся лёгким.
ENV HF_HOME=/opt/audio-models
ENV HF_HUB_DISABLE_XET=1

# ── HIGGS AUDIO 2 — В СВОЁМ ОТДЕЛЬНОМ ОКРУЖЕНИИ ─────────────────────────────
# ПОЧЕМУ ОТДЕЛЬНОЕ, а не общее /opt/audio-venv. Каждый движок из отбора тянет свою версию
# transformers, и они несовместимы между собой: код Higgs v2 берёт из transformers внутренности
# llama и whisper (`LLAMA_ATTENTION_CLASSES` убран после 4.47) и держится на >=4.45,<4.47, а тот
# же Chatterbox требовал 5.2. В общем окружении pip разрешит конфликт сдвигом версии — и молча
# сломает соседа. Этот урок ферма уже оплатила дважды: один раз звуком в окружении ComfyUI.
# Поэтому у каждого движка свой интерпретатор, а handler зовёт нужный (см. ENGINE_PY в handler.py).
# Встретиться они не могут физически: разные процессы.
RUN python3 -m venv --copies /opt/higgs-venv \
 && /opt/higgs-venv/bin/pip install --no-cache-dir --upgrade pip \
 && echo "HIGGS: отдельное окружение /opt/higgs-venv готово"

# torch под transformers 4.46. В базовом образе Python 3.12 — колёса 2.5.1+cu124 для него есть.
# Ступеньки вниз на случай, если версию уберут: лучше движок на процессоре, чем пустое место.
RUN /opt/higgs-venv/bin/pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
 || /opt/higgs-venv/bin/pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 || /opt/higgs-venv/bin/pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 || echo "!! HIGGS: torch не встал — движок не поедет, остальное не тронуто"

# transformers ПРИБИТ к 4.46.3: это последняя версия, где ещё есть внутренности, на которые
# опирается код v2. numpy держим ниже двойки — под ним собран весь этот слой.
RUN /opt/higgs-venv/bin/pip install --no-cache-dir \
      "transformers==4.46.3" "numpy<2" "accelerate>=0.26.0" \
      librosa dacite pandas loguru vector_quantize_pytorch omegaconf pydantic \
      json_repair langid jieba pydub click \
 || echo "!! HIGGS: зависимости не встали"

# Код Higgs v2 ПРИБИТ К КОММИТУ. В `main` репозиторий уже развёрнут на v3 (последний коммит так
# и называется: «point README to Higgs Audio v3, archive v2 guide»), и код v2 там держится на
# честном слове — в любой день его могут убрать. Ставим --no-deps: список зависимостей автора
# тянет boto3, s3fs и descript-audio-codec, которые в нашем пути исполнения не нужны вовсе
# (кодек внутри пакета свой, вложенный), а лишние пакеты — лишний риск сдвинуть torch.
#
# СТАВИМ ДЕРЕВОМ ИСХОДНИКОВ, А НЕ ПАКЕТОМ — и это не вкусовщина, это проверено отказом.
# `pip install` из репозитория собирает пакет по setup.cfg с `packages = find:`, а find берёт
# только те папки, где есть __init__.py. У автора его НЕТ ни в `serve/`, ни в `audio_processing/`,
# ни внутри вложенного кодека — он запускает код прямо из корня репозитория, где такие папки
# работают как namespace-пакеты. В итоге pip ставит обрубок: `import boson_multimodal` проходит,
# а `boson_multimodal.serve` не находится, и узнаёшь ты об этом уже на прогоне.
# Поэтому кладём репозиторий целиком и показываем его окружению файлом .pth.
ARG HIGGS_CODE=05a145bb490501b534563bf51bf2f7aa2326b271
RUN git init /opt/higgs-audio \
 && cd /opt/higgs-audio \
 && git remote add origin https://github.com/boson-ai/higgs-audio.git \
 && git fetch --depth 1 origin "${HIGGS_CODE}" \
 && git checkout FETCH_HEAD \
 && rm -rf /opt/higgs-audio/.git \
 && echo /opt/higgs-audio > "$(/opt/higgs-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/higgs-audio.pth" \
 && /opt/higgs-venv/bin/python -c "from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine; print('HIGGS: код v2 на месте, serve виден')" \
 || echo "!! HIGGS: код не встал — проверь, не вырезали ли v2 из репозитория"

# Сам скрипт весов кладём в образ, но НЕ ЗАПУСКАЕМ здесь: 12,8 ГБ не переживут тридцатиминутный
# предел сборки (см. выше). Он рассчитан на два места: его гоняют на поде, чтобы разложить веса
# по сетевому тому, и он же может добрать недостающее в работе, если на томе появится место.
COPY dl_higgs.py /opt/dl_higgs.py

# ДВЕ МАЛЕНЬКИЕ МОДЕЛИ, КОТОРЫЕ ТОКЕНИЗАТОР ТЯНЕТ САМ, ПО ИМЕНИ, ИЗНУТРИ — их в коде не видно,
# находятся только чтением: `bosonai/hubert_base` (377 МБ) и процессор whisper-large-v3-turbo
# (десятки мегабайт, без весов). Вот они как раз в образе: вместе меньше полугигабайта, а без них
# движок встанет колом уже на воркере, где сеть закрыта.
RUN HIGGS_DIR=/opt/audio-models/higgs /opt/higgs-venv/bin/python -c "\
from transformers import AutoModel, AutoProcessor;\
AutoModel.from_pretrained('bosonai/hubert_base', trust_remote_code=True);\
print('HIGGS: смысловая модель hubert_base в образе');\
AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo');\
print('HIGGS: процессор whisper в образе')" \
 || echo "!! HIGGS: спрятанные модели не легли в образ — движок упрётся в них на воркере"

# Что доехало — видно здесь, а не выясняется на первой задаче за деньги.
RUN /opt/higgs-venv/bin/python -c "import importlib, sys; [sys.stdout.write('HIGGS: %s %s\n' % (m, 'есть' if importlib.util.find_spec(m) else 'НЕТ')) for m in ('torch','torchaudio','transformers','boson_multimodal','librosa','vector_quantize_pytorch')]" || true
RUN echo "AUDIO: размер того, что в образе:" && du -sh /opt/audio-models 2>/dev/null || echo "AUDIO: весов в образе нет"

# ── COSYVOICE 3 — ПОБЕДИТЕЛЬ ОТБОРА, ЕСЛИ ВЛАДЕЛИЦА ЕГО ПРИМЕТ ───────────────
# ЗАЧЕМ ИМЕННО ОН. Из четырёх проб это единственная модель с РОДНЫМ русским (девять языков,
# русский в списке, Apache 2.0 — проверено по карточке). Разница слышна и видна числом: у неё
# русский идёт 12.7 знака в секунду, как у живого человека, а у CosyVoice 2 выходило 26.4 —
# слова просто не произносились, и владелица отвергла его с первого прослушивания.
# Украинского не заявляет НИ ОДНА открытая модель, поэтому украинский здесь — только клон
# по образцу (data/audio/ref/uk-voice-25s.wav), и это свойство рынка, а не нашей сборки.
#
# ВСЁ НИЖЕ ПРОВЕРЕНО РУКАМИ НА ПОДЕ, а не выведено из документации. Три ловушки, каждая из
# которых иначе стоила бы отдельной тридцатиминутной сборки:
#   1. `openai-whisper` из их requirements прибит к версии, которая НЕ СОБИРАЕТСЯ на python 3.11
#      («Getting requirements to build wheel did not run successfully») — ставим свежую отдельно;
#   2. CosyVoice пинит torch 2.3.1, а его же transformers 4.51 требует 2.4+ и падает на
#      `torch.library.register_fake` — возвращаем пару 2.4.1 последним шагом;
#   3. пакет ставится ДЕРЕВОМ ИСХОДНИКОВ с подмодулем Matcha-TTS: сам автор кладёт его в sys.path,
#      а без подмодуля движок не заводится вовсе.
RUN python3 -m venv --copies /opt/cosy3-venv \
 && /opt/cosy3-venv/bin/pip install --no-cache-dir --upgrade pip \
 && echo "COSY3: отдельное окружение готово"

ARG COSY3_CODE=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
RUN git clone --recursive -q https://github.com/FunAudioLLM/CosyVoice.git /opt/cosyvoice \
 && cd /opt/cosyvoice && git checkout -q ${COSY3_CODE} && git submodule update --init --recursive -q \
 && rm -rf /opt/cosyvoice/.git \
 && echo "COSY3: код и подмодуль Matcha-TTS на месте"

# Тяжёлое и ненужное для счёта выкидываем: tensorrt, deepspeed, gradio, tensorboard и веб-морда
# нужны обучению и демо, а нам — только синтез. Это и место в образе, и минуты сборки.
RUN cd /opt/cosyvoice \
 && grep -vE "tensorrt|deepspeed|gradio|tensorboard|fastapi|uvicorn|grpcio|openai-whisper" requirements.txt > /tmp/req.txt \
 && /opt/cosy3-venv/bin/pip install --no-cache-dir openai-whisper \
 && /opt/cosy3-venv/bin/pip install --no-cache-dir -r /tmp/req.txt \
 && /opt/cosy3-venv/bin/pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu124 \
 && echo /opt/cosyvoice > "$(/opt/cosy3-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/cosyvoice.pth" \
 && echo /opt/cosyvoice/third_party/Matcha-TTS >> "$(/opt/cosy3-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/cosyvoice.pth" \
 || echo "!! COSY3: зависимости не встали"

# Веса. ~10 ГБ — это МНОГО при тридцатиминутном пределе сборки. Базовая сборка сейчас укладывается
# в семь минут, поэтому запас есть, но он не бесконечен: если сборка начнёт падать по таймауту,
# веса переезжают на сетевой том, а этот шаг убирается (работник умеет искать их в обоих местах).
ENV HF_HUB_DISABLE_XET=1
RUN /opt/cosy3-venv/bin/python -c "\
from huggingface_hub import snapshot_download;\
p=snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/opt/audio-models/cosy3');\
print('COSY3: веса в образе:', p)" \
 || echo "!! COSY3: веса не легли в образ"

# ЕЩЁ ОДНА СПРЯТАННАЯ ЗАВИСИМОСТЬ, как hubert у Higgs: нормализатор текста `wetext` тянет свои
# ресурсы с modelscope В МОМЕНТ ПЕРВОЙ ЗАГРУЗКИ МОДЕЛИ. На поде это было видно в логе. Греем
# заранее, иначе первый холодный старт у владелицы полезет в чужую сеть за файлами.
RUN /opt/cosy3-venv/bin/python -c "\
from modelscope import snapshot_download;\
print('COSY3: ресурсы wetext:', snapshot_download('pengzhendong/wetext'))" \
 || echo "?? COSY3: ресурсы wetext не прогрелись — доберёт в работе"

# setuptools — ПОСЛЕДНИМ ШАГОМ, И ЭТО НЕ СЛУЧАЙНО.
# Сам по себе он нужен вот зачем: свежий venv его больше не кладёт, а `cosyvoice.flow.
# flow_matching` импортирует `pkg_resources`, который живёт именно в setuptools. На поде это не
# всплыло — там код шёл на системном Python, где setuptools был. В чистом окружении движок падал
# на первой же фразе, и поймала это ПРОБА ГОЛОСОМ, а не диагноз: диагноз показывал всё зелёным.
#
# А СТОИТ ОН ЗДЕСЬ, В ХВОСТЕ, потому что первая попытка починки встала ровно в этом месте:
# правка была в шаге создания окружения, то есть в САМОМ НАЧАЛЕ, и обесценила все слои после
# себя — включая скачивание десяти гигабайт весов. Сборка не уложилась в предел и упала на
# 35-й минуте. Дешёвые правки идут ПОСЛЕ тяжёлых слоёв, тогда пересборка трогает только хвост.
# ВЕРСИЯ ПРИБИТА НЕ ОТ ОСТОРОЖНОСТИ, А ПО ОТКАЗУ. Простой `pip install setuptools` притащил
# свежую ветку (81+), где `pkg_resources` уже ВЫРЕЗАН — и проверка ниже уронила всю сборку на
# четырнадцатой минуте. Нам нужен именно тот setuptools, в котором pkg_resources ещё живёт.
#
# Проверка теперь МЯГКАЯ (|| echo). Жёсткий gate здесь стоит дороже, чем помогает: он платит
# целой пересборкой за то, что проба голосом ловит за секунды. Настоящий воротарь у нас —
# прогон живого голоса на ферме, он уже поймал этот самый pkg_resources, когда диагноз молчал.
RUN /opt/cosy3-venv/bin/pip install --no-cache-dir "setuptools<81" wheel \
 && echo "COSY3: setuptools поставлен"

# ── ЖЁСТКИЕ ВОРОТА COSYVOICE 3 ──────────────────────────────────────────────
# ГДЕ ГЕЙТ, А ГДЕ ЕГО БЫТЬ НЕ ДОЛЖНО. Мягкие шаги сегодня дважды выпустили «зелёный» образ,
# который не умел говорить, а жёсткая проверка на ВЕРСИЮ ПАКЕТА (`import pkg_resources`) зря
# уронила сборку на четырнадцатой минуте: версия — это не то, ради чего мы собираем образ.
# Поэтому гейт стоит на том, что действительно важно: ДВИЖОК ОБЯЗАН ИМПОРТИРОВАТЬСЯ В САМОМ
# ОБРАЗЕ. Тогда сборка падает на настоящей причине, а не на имени зависимости.
#
# Импортируем ИМЕННО ДВА места, а не одно:
#   * `AutoModel` — точка входа, её зовёт работник;
#   * `cosyvoice.flow.flow_matching` — модуль, который РЕАЛЬНО сломался на ферме. Его тянет не
#     импорт входа, а загрузка модели по конфигу, поэтому проверять только вход недостаточно:
#     ровно так мы и получили образ, где диагноз зелёный, а первая же фраза падает.
# Весов для импорта не нужно — модель здесь не поднимается, только код.
RUN /opt/cosy3-venv/bin/python -c "\
from cosyvoice.cli.cosyvoice import AutoModel;\
import cosyvoice.flow.flow_matching;\
print('COSY3: движок импортируется в образе — ворота пройдены')" \
 || (echo "!! ВОРОТА: CosyVoice 3 не импортируется в образе — образ НЕ выпускаем" && exit 1)

RUN /opt/cosy3-venv/bin/python -c "import importlib, sys; [sys.stdout.write('COSY3: %s %s\n' % (m, 'есть' if importlib.util.find_spec(m) else 'НЕТ')) for m in ('torch','torchaudio','transformers','cosyvoice','matcha')]" || true

# Работник звука: его зовёт handler подпроцессом.
COPY audio_worker.py /opt/audio_worker.py

# Что именно доехало по звуку — видно выше, в проверке окружения Higgs. Общего окружения больше
# нет, и проверять в нём нечего.

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
    test -x /opt/higgs-venv/bin/python && echo "AUDIO: окружение Higgs на месте" \
      || echo "AUDIO: окружения Higgs НЕТ"

# 3. ПОСЛЕДНИЕ ВОРОТА: ComfyUI должен подняться и ЗАРЕГИСТРИРОВАТЬ ReActor. Проверка выше ловит
# «пакет установлен», эта — «нода реально видна ComfyUI». Именно её отсутствие стоило нам ролика
# без пересадки лица: пакет был, а класс не регистрировался.
RUN grep -qi "reactor" /tmp/ci.log \
 || (echo "!! ВОРОТА: ComfyUI не увидел ReActor при загрузке нод — образ НЕ выпускаем" && exit 1)

# requests уже есть в базовом образе (использует стоковый handler).
