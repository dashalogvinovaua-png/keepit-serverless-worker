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

for repo, rel, why in REPOS:
    dst = os.path.join(M, rel)
    os.makedirs(dst, exist_ok=True)
    # Уже скачано (есть веса) — не трогаем: повторный запуск скрипта должен быть дешёвым.
    have = any(f.endswith(('.safetensors', '.bin', '.pt')) for _, _, fs in os.walk(dst) for f in fs)
    if have:
        print('уже есть:', rel)
        continue
    print(f'качаю: {repo} → {rel} ({why})', flush=True)
    try:
        snapshot_download(repo_id=repo, local_dir=dst, max_workers=8)
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dst) for f in fs)
        print('  ok', round(size / 1e9, 2), 'ГБ')
    except Exception as e:
        print('  ✗ ошибка:', str(e)[:160])

print('AUDIO MODELS DONE')
