# РАБОТНИК ЗВУКА. Живёт в ОТДЕЛЬНОМ окружении /opt/audio-venv и запускается ПОДПРОЦЕССОМ.
#
# Почему так, а не нодой ComfyUI. Один раз мы уже поставили зависимости звука в общее окружение —
# и получили разъехавшийся стек рядом с Wan, LTX, SCAIL и ReActor. Здесь у звука свой Python, свой
# torch и свои пакеты; встретиться с чужими они не могут физически, потому что живут в разных
# процессах. Побочная выгода: ComfyUI про звук не знает, и регистрация классов нод ни на что не
# влияет — движок работает, даже если ноды не поднялись.
#
# Разговор простой: JSON на вход в stdin, JSON на выход в stdout.
#   вход:  {"engine":"chatterbox"|"ttsuk", "text":"...", "language":"uk", "ref_audio":"<base64>",
#           "exaggeration":0.5, "stability":0.5, "speed":1.0}
#   выход: {"ok":true, "audio":"<base64 wav>", "sample_rate":24000, "engine":"...", "seconds":12.3}
#          либо {"ok":false, "error":"человеческим языком"}
#
# Веса лежат НА ТОМЕ (HF_HOME указывает туда), поэтому качаются один раз, а не на каждый холодный
# старт контейнера.
import base64
import io
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/runpod-volume/hf-cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet рвётся на сетевом томе, проверено на весах

_cache = {}


def _wav_bytes(wave, sr):
    """Тензор → wav в память. Файлы на диске воркера не нужны никому."""
    import torch
    import torchaudio
    if not isinstance(wave, torch.Tensor):
        wave = torch.tensor(wave)
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, wave.cpu(), sr, format="wav")
    return buf.getvalue()


def run_chatterbox(job):
    """Chatterbox Multilingual (Resemble AI, MIT): 23 языка, клон голоса по образцу."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = _cache.get("chatterbox")
    if model is None:
        model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        _cache["chatterbox"] = model
    ref = None
    if job.get("ref_audio"):
        ref = "/tmp/ref.wav"
        with open(ref, "wb") as f:
            f.write(base64.b64decode(job["ref_audio"]))
    wave = model.generate(
        job["text"],
        language_id=job.get("language", "en"),
        audio_prompt_path=ref,
        exaggeration=float(job.get("exaggeration", 0.5)),
        cfg_weight=float(job.get("stability", 0.5)),
    )
    return wave, model.sr


def run_ttsuk(job):
    """tts_uk (RAD-TTS++, MIT): украинский родными голосами. Клона нет, голоса три.

    Возвращает тройку (мел-спектрограммы, волна, статистика) — нам нужна волна; частота 44 100.
    """
    from tts_uk.inference import synthesis
    voice = job.get("voice") or "tetiana"          # tetiana, lada — женские; mykyta — мужской
    _mels, wave, stats = synthesis(
        text=job["text"],
        voice=voice,
        n_takes=int(job.get("takes", 1)),
        use_latest_take=False,
    )
    _cache["ttsuk_stats"] = stats
    return wave, 44100


ENGINES = {"chatterbox": run_chatterbox, "ttsuk": run_ttsuk}


def main():
    started = time.time()
    try:
        job = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "вход не разобрать: %s" % e}))
        return
    engine = job.get("engine")
    fn = ENGINES.get(engine)
    if not fn:
        print(json.dumps({"ok": False,
                          "error": "нет такого движка: %s (есть: %s)" % (engine, ", ".join(ENGINES))}))
        return
    if not (job.get("text") or "").strip():
        print(json.dumps({"ok": False, "error": "пустой текст"}))
        return
    try:
        wave, sr = fn(job)
        data = _wav_bytes(wave, sr)
        print(json.dumps({
            "ok": True, "engine": engine, "sample_rate": sr,
            "seconds": round(len(data) / (sr * 2), 2),
            "took": round(time.time() - started, 1),
            "audio": base64.b64encode(data).decode(),
        }))
    except Exception as e:  # noqa: BLE001
        import traceback
        print(json.dumps({"ok": False, "engine": engine,
                          "error": str(e)[:400],
                          "trace": traceback.format_exc().strip().splitlines()[-4:]}))


if __name__ == "__main__":
    main()
