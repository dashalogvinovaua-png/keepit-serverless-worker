# ВЕСА HIGGS AUDIO 2 — КЛАДЁМ В ОБРАЗ ПРИ СБОРКЕ.
#
# В работе качать нельзя: на воркере запись упирается в квоту («Disk quota exceeded» при
# 92 тысячах свободных гигабайт на томе). Значит всё, что модели понадобится, должно приехать
# внутри образа — иначе первая же задача упрётся в сеть при HF_HUB_OFFLINE=1.
#
# ГЛАВНАЯ ЛОВУШКА, из-за которой этот файл существует отдельно: и код, и обе карточки моделей
# ПЕРЕЕХАЛИ на версию 3, и «свежий» main для нас сломан.
#   * карточка модели переименована: higgs-audio-v2-generation-3B-base → higgs-tts-2-3b-base;
#   * 2026-04-04 обе карточки перевели на формат transformers: у токенизатора вместо `model.pth`
#     появился `model.safetensors` и совсем другой config.json. А код v2 читает именно
#     `model.pth` и передаёт config прямо в конструктор — с новым форматом он падает.
# Поэтому обе ревизии ПРИБИТЫ ГВОЗДЯМИ к эпохе v2 (июль 2025). Не поднимать их «на свежее»,
# не проверив, что код v2 эти файлы ещё понимает.
#
# И вторая ловушка: токенизатор звука тянет ЕЩЁ ДВЕ модели по имени, внутри себя —
# `bosonai/hubert_base` (смысловая часть) и процессор `openai/whisper-large-v3-turbo`. В коде
# они не видны, их находишь только чтением. Они маленькие и лежат ПРЯМО В ОБРАЗЕ (см. Dockerfile),
# поэтому здесь их нет: этот файл занимается только двумя тяжёлыми репозиториями.
#
# ГДЕ ЗАПУСКАЮТ. В образ веса не помещаются — сборка на RunPod обрывается на тридцатой минуте.
# Поэтому скрипт гоняют туда, где место есть, задав HIGGS_DIR:
#   на поде с примонтированным сетевым томом:  HIGGS_DIR=/workspace/higgs python dl_higgs.py
#   на воркере, если на томе освободится место: HIGGS_DIR=/runpod-volume/higgs
import os
import sys

from huggingface_hub import snapshot_download

HIGGS_DIR = os.environ.get("HIGGS_DIR", "/opt/audio-models/higgs")

MODEL_REPO = "bosonai/higgs-tts-2-3b-base"
MODEL_REV = "10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84"      # 2025-07-28, до перевода на v3
TOKENIZER_REPO = "bosonai/higgs-audio-v2-tokenizer"
TOKENIZER_REV = "9d4988fbd4ad07b4cac3a5fa462741a41810dbec"  # 2025-07-22, ещё с model.pth


def main():
    # Модель: только веса и настройки. Картинки и демо-ролик из карточки (15 МБ) в образе не нужны.
    model_dir = snapshot_download(
        MODEL_REPO,
        revision=MODEL_REV,
        local_dir=os.path.join(HIGGS_DIR, "model"),
        allow_patterns=["*.json", "*.safetensors"],
    )
    print("HIGGS: веса модели в образе:", model_dir)

    tok_dir = snapshot_download(
        TOKENIZER_REPO,
        revision=TOKENIZER_REV,
        local_dir=os.path.join(HIGGS_DIR, "tokenizer"),
        allow_patterns=["config.json", "model.pth"],
    )
    print("HIGGS: токенизатор звука в образе:", tok_dir)

    # Проверяем делом: код v2 ищет именно эти два файла и падает без них.
    need = [os.path.join(tok_dir, "config.json"), os.path.join(tok_dir, "model.pth")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("!! HIGGS: токенизатор приехал не в том виде, нет файлов:", missing)
        sys.exit(1)

    print("HIGGS: готово, всё лежит в", HIGGS_DIR)


if __name__ == "__main__":
    main()
