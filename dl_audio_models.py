# Скачка АУДИО-моделей на сетевой том RunPod (запускать на поде с примонтированным томом).
# Нужны сервису platform/services/audio: речь Qwen3-TTS, музыка HeartMuLa.
#
# Почему на ТОМ, а не «пусть нода скачает сама»: serverless-воркер эфемерный — HF-кэш умирает
# вместе с контейнером, и каждый холодный старт заново тянул бы ~7 ГБ. На томе модель лежит один раз.
#
# Запуск на поде (GPU не нужен, хватает CPU-пода с примонтированным томом):
#   pip install -U "huggingface_hub[hf_transfer]"
#   HF_HUB_ENABLE_HF_TRANSFER=1 python3 dl_audio_models.py
#
# Место: ~7 ГБ речь (три варианта Qwen3-TTS) + ~9 ГБ музыка (HeartMuLa 3B + кодек).
# Можно скачать частями: SKIP_MUSIC=1 или SKIP_SPEECH=1.
import os
import shutil

# Xet-транспорт HuggingFace рвётся на сетевом томе RunPod («Internal Writer Error: Background
# writer channel closed») и оставляет модель недокачанной — у нас на этом уже сломался шардированный
# HeartMuLa. Выключаем его ДО импорта huggingface_hub: качаем обычным HTTP, медленнее, но целиком.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from huggingface_hub import snapshot_download

M = os.environ.get('MODELS_DIR', '/workspace/comfyui/models')

# (repo_id, целевой относительный путь под models/, зачем)
REPOS = []

if os.environ.get('SKIP_SPEECH') != '1':
    REPOS += [
        # Готовые тембры — основной режим озвучки (нода Qwen3CustomVoice).
        ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', 'Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice',
         'речь: 9 готовых голосов, 10 языков (русский в списке)'),
        # Голос по описанию словами — нода Qwen3VoiceDesign. Клон не нужен, лицензионно чисто.
        ('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', 'Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
         'речь: голос собирается по описанию'),
        # База — под клон по образцу (нода Qwen3VoiceClone).
        ('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base',
         'речь: клон голоса по образцу 3–10 с'),
    ]

if os.environ.get('SKIP_MUSIC') != '1':
    REPOS += [
        # Песня с вокалом по тексту и тегам. RL-версия от 23.01.2026 — лучшая из открытых.
        ('HeartMuLa/HeartMuLa-RL-oss-3B-20260123', 'HeartMuLa/HeartMuLa-RL-oss-3B-20260123',
         'музыка: песня с вокалом до 240 с'),
        # Кодек, которым модель декодирует латент в звук 48 кГц. Без него генератор не запустится.
        ('HeartMuLa/HeartCodec-oss-20260123', 'HeartMuLa/HeartCodec-oss-20260123',
         'музыка: аудио-кодек 12.5 Гц → 48 кГц'),
    ]

def size_gb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e9


for repo, rel, why in REPOS:
    dst = os.path.join(M, rel)
    os.makedirs(dst, exist_ok=True)
    # ВСЕГДА зовём snapshot_download, даже если папка не пуста. Проверка «есть хоть один .safetensors»
    # обманывает на шардированных моделях: у HeartMuLa-3B четыре шарда, и прерванная скачка оставляет
    # один файл — папка «выглядит готовой», а модель не грузится. snapshot_download сам пропускает
    # уже целые файлы и дотягивает недостающие, так что повторный запуск дешёвый и честный.
    print(f'проверяю/качаю: {repo} → {rel} ({why})', flush=True)
    # STAGE_DIR — качать на локальный диск пода, а на том класть уже готовое.
    # Зачем: huggingface_hub держит рядом служебный .cache, то есть на пике нужен ДВОЙНОЙ объём
    # модели. На забитом томе это даёт «Disk quota exceeded» на ровном месте.
    stage = os.path.join(os.environ['STAGE_DIR'], rel) if os.environ.get('STAGE_DIR') else dst
    os.makedirs(stage, exist_ok=True)
    # Три попытки: сеть тома иногда обрывает большой файл, а докачка дешёвая — целые файлы пропускаются.
    for attempt in range(1, 4):
        try:
            snapshot_download(repo_id=repo, local_dir=stage, max_workers=4)
            if stage != dst:
                shutil.rmtree(os.path.join(stage, '.cache'), ignore_errors=True)
                for name in os.listdir(stage):
                    src, tgt = os.path.join(stage, name), os.path.join(dst, name)
                    if os.path.exists(tgt):
                        (shutil.rmtree if os.path.isdir(tgt) else os.remove)(tgt)
                    shutil.move(src, tgt)
                shutil.rmtree(stage, ignore_errors=True)
            else:
                shutil.rmtree(os.path.join(dst, '.cache'), ignore_errors=True)
            print('  ok', round(size_gb(dst), 2), 'ГБ')
            break
        except Exception as e:
            print(f'  попытка {attempt}/3 не вышла:', str(e)[:160], flush=True)

print('── итог по папкам ──')
for _, rel, _ in REPOS:
    dst = os.path.join(M, rel)
    files = [f for _, _, fs in os.walk(dst) for f in fs]
    print(f'  {rel}: {len(files)} файлов, {round(size_gb(dst), 2)} ГБ')
print('AUDIO MODELS DONE')
