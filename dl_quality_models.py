# Скачка моделей КАЧЕСТВА на сетевой том RunPod (запускать на поде с примонтированным томом).
# Кладём: CodeFormer (лица), RealESRGAN (апскейл), DWPose (движение) — под ноды facerestore_cf,
# ImageUpscaleWithModel, comfyui_controlnet_aux. Пути — стандартные для ComfyUI на нашем томе.
# Запуск на поде:  HF_HUB_DISABLE_XET=1 python3 dl_quality_models.py
import os, urllib.request
M = os.environ.get('MODELS_DIR', '/workspace/comfyui/models')

# (url, целевой относительный путь под models/)
FILES = [
    # ── CodeFormer — восстановление лиц (зубы/глаза/кожа) ──
    ('https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
     'facerestore_models/codeformer.pth'),
    ('https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth',
     'facerestore_models/GFPGANv1.4.pth'),
    # facexlib-зависимости (детекция+парсинг лица) — иначе facerestore качает их в рантайме (медленно/офлайн)
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
     'facedetection/detection_Resnet50_Final.pth'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
     'facedetection/parsing_parsenet.pth'),
    # ── Апскейл 480p→1080p (детализация лица/еды) ──
    ('https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth',
     'upscale_models/RealESRGAN_x4plus.pth'),
    # ── DWPose — точный контроль движения для Wan VACE (control=pose) ──
    ('https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx',
     'controlnet_aux/ckpts/yzd-v/DWPose/yolox_l.onnx'),
    ('https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx',
     'controlnet_aux/ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx'),
]

for url, rel in FILES:
    dst = os.path.join(M, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 100000:
        print('уже есть:', rel); continue
    try:
        print('качаю:', rel, '…', flush=True)
        urllib.request.urlretrieve(url, dst)
        print('  ok', round(os.path.getsize(dst) / 1e6, 1), 'МБ')
    except Exception as e:
        print('  ✗ ошибка:', str(e)[:100])
print('QUALITY MODELS DONE')
