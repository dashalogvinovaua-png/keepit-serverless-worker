# Как поднять ФЕРМУ по звуку (речь Qwen3-TTS и музыка HeartMuLa)

Готовый блок для `Dockerfile`. Сам образ **не пересобран**: он общий с video и photo, и пересборка
во время их прогонов — не моё решение. Здесь всё, что нужно, чтобы её сделать одним заходом.

## Что сейчас (проверено живой пробой 2026-07-31, не по коду)

Отправлены два минимальных графа на боевой эндпоинт `emqtm5lhnhmcqp`:

```
Qwen3Loader     → FAILED  "Node 'Qwen3Loader' not found. The custom node may not be installed."
HeartMuLaLoader → FAILED  "Node 'HeartMuLaLoader' not found. The custom node may not be installed."
```

Значит: **флаг `AUDIO_FARM=1` сам по себе ничего не поднимет.** В образе нет ни одной аудио-ноды
(в `Dockerfile` их и не было никогда — grep по `Qwen3|TTS|HeartMuLa` пуст). Включить флаг без нод
= каждая задача сначала ждёт GPU, потом падает на валидации графа, и мы платим за это время.

Порядок ровно такой: **ноды в образ → веса на том → `AUDIO_FARM=1`**.

## 1. Ноды в образ

```dockerfile
# ── ЗВУК: речь Qwen3-TTS и музыка HeartMuLa ──
# Классы, которые ждёт сервис audio (src/engine.js):
#   речь   — Qwen3Loader, Qwen3CustomVoice, Qwen3VoiceDesign, Qwen3VoiceClone
#   музыка — HeartMuLaLoader, HeartMuLaGenerator
# Обе ноды ставим МЯГКО (|| true) и проверяем в CI-прогоне ниже: сломанная нода роняет ComfyUI на
# старте и уводит воркер в unhealthy — так уже было с comfyui_controlnet_aux, и тогда встало ВСЁ,
# включая видео и фото.
RUN cd /comfyui/custom_nodes \
 && (git clone --depth 1 https://github.com/DarioFT/ComfyUI-Qwen3-TTS.git \
     && (pip install --no-cache-dir -r ComfyUI-Qwen3-TTS/requirements.txt || true) || true) \
 && (git clone --depth 1 https://github.com/monnky/ComfyUI-RT-HeartMuLa.git \
     && (pip install --no-cache-dir -r ComfyUI-RT-HeartMuLa/requirements.txt || true) || true)

# Правда о сборке в логе: видно сразу, поедет звук или нет.
RUN (cd /comfyui && timeout 900 python main.py --quick-test-for-ci --cpu > /tmp/ci-audio.log 2>&1 || true); \
    for n in Qwen3Loader Qwen3VoiceDesign HeartMuLaLoader HeartMuLaGenerator; do \
      if grep -q "$n" /tmp/ci-audio.log; then echo "AUDIO: $n есть"; else echo "AUDIO: $n НЕ ЗАРЕГИСТРИРОВАН"; fi; \
    done; \
    grep -iE "qwen|heartmula|import failed|traceback" /tmp/ci-audio.log | head -40 || true
```

Если какая-то нода не встанет — образ всё равно выйдет, а сервис audio увидит это по ошибке
`missing_node_type` и уйдёт в облако с пометкой (уже реализовано, ничего не сломается).

## 2. Веса на том

Скрипт уже написан: `dl_audio_models.py` (запускать на CPU-поде с примонтированным томом).

```sh
pip install -U "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 python3 dl_audio_models.py          # речь ~7 ГБ + музыка ~9 ГБ
HF_HUB_ENABLE_HF_TRANSFER=1 SKIP_MUSIC=1 python3 dl_audio_models.py   # только речь, если мало места
```

Пути должны совпасть с тем, что ждёт сервис:
`/runpod-volume/comfyui/models/Qwen3-TTS/<имя репозитория без владельца>`.

## 3. Флаги сервиса

```sh
AUDIO_FARM=1          # речь на ферме
AUDIO_FARM_MUSIC=1    # музыка на ферме (отдельно: веса музыки тяжелее и могли не доехать)
```

`GET /health` → `engines.farm.ready: true` и `engines.farm.whyNot.speech: null`.
Пока это не так, `whyNot` прямым текстом говорит, что именно мешает.

## Зачем это (цена вопроса)

| работа | ферма | облако | разница |
|---|---|---|---|
| минута речи (~900 знаков) | **120 токенов** | 810 токенов Qwen / 2 700 токенов ElevenLabs | в 7–22 раза |
| минута музыки | **350 токенов** | 8 000 токенов ElevenLabs Music | в 23 раза |

Речь zh/en/ja/ko и вся музыка обязаны считаться у себя: это ровно та же модель Qwen3-TTS, что и в
облаке. Облако остаётся законным только для uk/ru/es — там у Qwen нет родных тембров, и живой голос
носителя есть только у мультиязычной модели.


---

# ПЛАН ПЕРЕСБОРКИ (для директора фермы, 2026-08-03)

Сборку без слова директора не запускаю. Ниже — что именно надо доложить, сколько это весит и
сколько занимает по времени.

## 1. Что доложить в образ

| что | откуда | зачем | вес в образе |
|---|---|---|---|
| `ComfyUI-Qwen3-TTS` | `github.com/DarioFT/ComfyUI-Qwen3-TTS` | ноды `Qwen3Loader`, `Qwen3CustomVoice`, `Qwen3VoiceDesign`, `Qwen3VoiceClone` — вся речь | ~5 МБ кода + зависимости |
| `ComfyUI-RT-HeartMuLa` | `github.com/monnky/ComfyUI-RT-HeartMuLa` | ноды `HeartMuLaLoader`, `HeartMuLaGenerator` — вся музыка | ~8 МБ кода + зависимости |
| зависимости обеих нод | pip | `transformers`, `torchaudio`, `soundfile`, `descript-audio-codec` | ~300–600 МБ слоем |

Имена классов совпадают с тем, что сервис уже отправляет (`src/engine.js`), — граф менять не надо,
работать начнёт сразу после появления нод.

## 2. Веса — на ТОМ, не в образ

Скрипт готов: `dl_audio_models.py`.

| модель | размер | обязательна |
|---|---|---|
| Qwen3-TTS CustomVoice (готовые тембры) | ~2,3 ГБ | да, это основной режим |
| Qwen3-TTS VoiceDesign (голос по описанию) | ~2,3 ГБ | да, языки без родного тембра |
| Qwen3-TTS Base (клон по образцу) | ~2,3 ГБ | нет, только для продукта «клон голоса» |
| HeartMuLa-RL-oss-3B + HeartCodec | ~9 ГБ | для музыки на ферме |

Итого: **речь ~7 ГБ, музыка ~9 ГБ**. Можно ставить частями: `SKIP_MUSIC=1` — сначала только речь.

## 3. Сколько времени

| шаг | время |
|---|---|
| сборка образа (два клона + pip + CI-проверка нод) | 10–15 мин |
| push нового слоя в реестр | 3–8 мин (зависит от канала) |
| холодный старт воркера на новом образе | +40–60 с однократно |
| скачка весов на том (CPU-под, `hf_transfer`) | речь ~15 мин, музыка ~20 мин |

Всего от «поехали» до первой задачи на ферме: **около часа**, если веса качать параллельно сборке.

## 4. Чем рискуем и как подстелено

- **Образ общий с video и photo.** Сломанная нода роняет ComfyUI на старте — так уже было с
  `comfyui_controlnet_aux`, и тогда встало всё. Поэтому обе ноды ставятся МЯГКО (`|| true`), а в
  конце сборки идёт CI-прогон, который прямо в логе печатает, зарегистрировались классы или нет.
  Если нода не встала — образ выходит без неё, видео и фото не замечают разницы.
- **Место на томе.** 16 ГБ под аудио. Перед скачкой обязательно `df -h /runpod-volume` на поде:
  если тесно, ставим `SKIP_MUSIC=1` и живём с музыкой в облаке.
- **Сервис к этому готов.** Флаги раздельные (`AUDIO_FARM` — речь, `AUDIO_FARM_MUSIC` — музыка),
  так что речь можно включить раньше музыки. Если нод всё-таки не окажется, конвейер это увидит
  по ошибке валидации и уйдёт в облако с пометкой — задачи не потеряются.

## 5. Порядок включения после сборки

```sh
curl -sX POST localhost:8115/farm/probe -H "x-internal-token: $TOKEN"   # ноды приехали?
# в ответе nodes.speech=false и nodes.music=false означает «нод не хватает НЕТ», то есть всё на месте
AUDIO_FARM=1 AUDIO_FARM_MUSIC=1   # в окружение сервиса, затем перезапуск
curl -s localhost:8115/health     # engines.farm.ready=true, money.canDo.speech="да"
```

## 6. Что это даёт

| работа | сейчас (облако) | на ферме | экономия |
|---|---|---|---|
| минута речи | 1 000 токенов–0.27 | 120 токенов | в 8–22 раза |
| минута музыки | 12 000 токенов | 350 токенов | в 34 раза |

Приёмка 13→33 по нынешним ценам стоит 159 700 токенов в облаке; на ферме те же изделия — около 6 000 токенов.
