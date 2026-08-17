# Скачка весов LTX-2.5 на сетевой том RunPod (запускать на поде с примонтированным томом).
#
# ЗАЧЕМ. Слово владелицы 17.08.2026: «так и делаем на серверлесс». Речь про переезд фермы на
# LTX-2.5 — открытые веса, вышли 11 августа. Всё, что отделяет нас от неё, — два шага:
# свежее ядро ComfyUI в образе (сделано отдельно) и ЭТИ ФАЙЛЫ на томе.
#
# ПОЧЕМУ НА ТОМ, А НЕ В ОБРАЗ. Serverless-воркер эфемерный: его диск умирает вместе с
# контейнером. 49 ГБ на каждом холодном старте — это разорение и минуты ожидания у человека.
# На томе модель лежит один раз. Том у нас `coloured_tan_mosquito`, 150 ГБ, US-WA-1 — тот же,
# что примонтирован к воркеру `keepit-serverless-worker`.
#
# МЕСТА ХВАТАЕТ, И ЭТО ПРОВЕРЕНО ЧИСЛОМ, А НЕ НА ГЛАЗ. Область животных предлагала снести с
# тома Wan Fun VACE (28 ГБ) и SCAIL (14 ГБ), чтобы освободить место. Директор запретил: оба
# движка работают сегодня — по последним 25 работам фермы ltx 20, wan+vace 5, а SCAIL это
# единственный движок копий трендов. Сносить живое ради непроверенного — не размен, а потеря.
# Скрипт сам считает свободное место перед скачиванием и отказывается, если его мало.
#
# Запуск на поде (GPU не нужен — хватает дешёвого процессорного пода с томом):
#   pip install -U "huggingface_hub[hf_transfer]"
#   HF_TOKEN=<ключ владелицы> python3 dl_ltx25_models.py
import os
import shutil
import sys

# Xet-транспорт HuggingFace рвётся на сетевом томе RunPod («Internal Writer Error») и оставляет
# модель НЕДОКАЧАННОЙ — на этом уже сломался шардированный HeartMuLa. Выключаем до импорта.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = os.environ.get('COMFY_MODELS', '/runpod-volume/comfyui/models')

# РЕПОЗИТОРИЙ ЗАКРЫТЫЙ: качается только с ключом. Ключ владелицы лежит на сервере фермы в
# infra/.env как HF_TOKEN — передавать его сюда переменной окружения, в файл не вписывать.
LTX = 'Lightricks/LTX-2.5'
GEMMA = 'Comfy-Org/gemma-4'

# Размеры настоящие, из API Hugging Face (сверено 14.08.2026 областью видео).
FILES = [
    (LTX, 'diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors',
     'diffusion_models', 21.50, 'сама модель'),
    (LTX, 'text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors',
     'text_encoders', 15.37, 'текстовый кодировщик с проекцией'),
    (LTX, 'vae/ltx-2.5-video-vae-bf16.safetensors', 'vae', 1.47, 'декодер картинки'),
    (LTX, 'vae/ltx-2.5-audio-vae-bf16.safetensors', 'vae', 0.36, 'декодер звука'),
    (LTX, 'model_patches/ltx-2.5-duration-head-bf16.safetensors', 'model_patches', 0.005,
     'голова длительности: модель сама предсказывает длину плана по промпту'),
    # ВТОРОЙ КОДИРОВЩИК ЛЕЖИТ В ЧУЖОМ РЕПОЗИТОРИИ, И ОН ОТКРЫТЫЙ. Числится обязательным во
    # всех трёх рабочих схемах (T2V, I2V, FLF2V) — поэтому стоит в списке, а не в примечаниях.
    (GEMMA, 'gemma4_e2b_it_bf16.safetensors', 'text_encoders', 10.28, 'второй кодировщик'),
]

НУЖНО_ГБ = sum(f[3] for f in FILES)
ЗАПАС_ГБ = 5          # том не заполняем под ноль: недокачанный файл хуже отсутствующего


def свободно(путь):
    try:
        os.makedirs(путь, exist_ok=True)
        return shutil.disk_usage(путь).free / 1e9
    except Exception as e:
        print(f'!! не читается место на {путь}: {e}')
        return None


def главное():
    if not os.environ.get('HF_TOKEN'):
        print('!! нет HF_TOKEN — репозиторий LTX-2.5 закрытый, без ключа не отдаст')
        sys.exit(2)

    св = свободно(ROOT)
    if св is None:
        sys.exit(2)
    print(f'том {ROOT}: свободно {св:.1f} ГБ, нужно {НУЖНО_ГБ:.1f} ГБ + запас {ЗАПАС_ГБ} ГБ')
    if св < НУЖНО_ГБ + ЗАПАС_ГБ:
        # НЕЗНАНИЕ МЕСТА — ЗАПРЕТ, А НЕ СКИДКА. Качать 49 ГБ «а вдруг влезет» значит получить
        # обрубок и не узнать об этом до первой съёмки.
        print('!! места не хватает. Ничего не качаю и ничего не удаляю сам:')
        print('   что сносить с тома — решение владелицы, а не скрипта.')
        sys.exit(3)

    итог = []
    for repo, файл, куда, гб, зачем in FILES:
        цель = os.path.join(ROOT, куда)
        os.makedirs(цель, exist_ok=True)
        имя = os.path.basename(файл)
        путь = os.path.join(цель, имя)
        # УЖЕ СКАЧАННОЕ НЕ КАЧАЕМ ВТОРОЙ РАЗ, но и «файл есть» признаком не считаем: сверяем
        # размер с ожидаемым. Обрубок от прошлой попытки — обычное дело на сетевом томе.
        если_есть = os.path.exists(путь) and os.path.getsize(путь) > гб * 0.95e9
        if если_есть:
            print(f'  ✓ уже на месте: {имя} ({гб} ГБ) — {зачем}')
            итог.append((имя, 'был'))
            continue
        print(f'  ↓ качаю {имя} ({гб} ГБ) — {зачем}')
        try:
            got = hf_hub_download(repo_id=repo, filename=файл, local_dir=цель,
                                  local_dir_use_symlinks=False,
                                  token=os.environ['HF_TOKEN'])
            # hf кладёт файл, сохраняя вложенные папки репозитория — переносим к плоскому имени,
            # которое ждёт ComfyUI.
            if os.path.abspath(got) != os.path.abspath(путь):
                shutil.move(got, путь)
            print(f'    готово: {os.path.getsize(путь)/1e9:.2f} ГБ')
            итог.append((имя, 'скачан'))
        except Exception as e:
            print(f'    !! НЕ СКАЧАЛОСЬ: {e}')
            итог.append((имя, f'ОШИБКА: {e}'))

    print('\nИТОГ:')
    for имя, что in итог:
        print(f'  {что:>8}  {имя}')
    плохо = [x for x in итог if x[1].startswith('ОШИБКА')]
    print(f'\nсвободно после: {свободно(ROOT):.1f} ГБ')
    if плохо:
        print('!! набор НЕПОЛНЫЙ — LTX-2.5 не запустится. Это не «почти готово», это «не готово».')
        sys.exit(4)
    print('набор полный.')


if __name__ == '__main__':
    главное()
