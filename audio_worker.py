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

# ЗАПАСНОЙ ПУТЬ ПО ЖЕЛЕЗУ. Отдельный torch в нашем окружении оказался собран без ядер под эту
# карту: «CUDA error: no kernel image is available for execution on the device». Пересобирать
# стек под каждую модель карты — долго и ненадёжно, поэтому handler при такой ошибке зовёт нас
# второй раз с FORCE_CPU=1. На процессоре речь считается медленнее, но считается, и владелица
# слышит звук сегодня, а не после ещё одной пересборки.
if os.environ.get("FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ГДЕ ЛЕЖАТ ВЕСА. Сначала сетевой том: он переживает холодный старт, и качать надо один раз.
# Но том общий с видео и фото, и он бывает полон — тогда вместо весов приезжает «Disk quota
# exceeded». В этом случае уходим на диск контейнера: веса будут качаться при каждом холодном
# старте (дороже по времени), зато движок работает, а не отказывает целиком.
# ВЕСА ЛЕЖАТ В ОБРАЗЕ. На томе и на диске контейнера запись упирается в квоту («Disk quota
# exceeded» при 91 тысяче свободных гигабайт), поэтому качать в работе нельзя вовсе — веса
# приезжают вместе с образом, собранные заранее.
IMAGE_CACHE = "/opt/audio-models"
VOLUME_CACHE = "/runpod-volume/hf-cache"
LOCAL_CACHE = "/root/.cache/huggingface"


def _writable(path, need_gb=3):
    """МОЖНО ЛИ ТУДА ПИСАТЬ — проверяем делом, а не свободным местом.

    Ошибка, на которой мы потеряли два часа: система показывает на томе 92 тысячи гигабайт
    свободных, а запись отвечает «Disk quota exceeded». Свободное место и право писать — разные
    вещи, поэтому спрашиваем не «сколько осталось», а «получится ли»: создаём и удаляем файл.
    """
    import shutil
    try:
        free = shutil.disk_usage(path).free / 1e9
    except Exception:  # noqa: BLE001
        return False, None
    if free < need_gb:
        return False, round(free, 1)
    probe = os.path.join(path, ".write-probe")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "wb") as f:
            f.write(b"x" * 1024)
        os.remove(probe)
        return True, round(free, 1)
    except Exception:  # noqa: BLE001
        return False, round(free, 1)


# Порядок: веса из образа → том, если туда РЕАЛЬНО пишется → диск контейнера.
_vol_ok, _free = _writable(VOLUME_CACHE)
_local_ok, _local_free = _writable(LOCAL_CACHE)
if os.path.isdir(IMAGE_CACHE) and os.listdir(IMAGE_CACHE):
    _where = IMAGE_CACHE
    os.environ.setdefault("HF_HUB_OFFLINE", "1")     # веса в образе, в сеть ходить незачем
elif _vol_ok:
    _where = VOLUME_CACHE
elif _local_ok:
    _where = LOCAL_CACHE
else:
    _where = LOCAL_CACHE                              # писать некуда — отказ объяснит сам движок
os.environ.setdefault("HF_HOME", _where)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet рвётся на сетевом томе, проверено на весах

_cache = {}


def _wav_bytes(wave, sr):
    """Тензор → wav в память. Файлы на диске воркера не нужны никому.

    Пишем 16 битами, а не тем float32, что отдают движки. Причины две, обе практические:
    файл выходит вдвое легче (а он едет к нам по сети в base64, где вес удваивается ещё раз),
    и поле `seconds` наконец считается правильно — оно всегда делило на два байта на отсчёт,
    то есть для float32 показывало вдвое больше, чем звучало на самом деле.
    """
    import torch
    import torchaudio
    if not isinstance(wave, torch.Tensor):
        wave = torch.tensor(wave)
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    wave = wave.cpu()
    if wave.is_floating_point():
        # Обрезаем по краям диапазона: иначе редкий выброс за единицу превратится в щелчок.
        wave = (wave.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
    buf = io.BytesIO()
    torchaudio.save(buf, wave, sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    return buf.getvalue()


def run_chatterbox(job):
    """Chatterbox Multilingual (Resemble AI, MIT): 23 языка, клон голоса по образцу."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = _cache.get("chatterbox")
    if model is None:
        import torch
        dev = "cuda" if (os.environ.get("FORCE_CPU") != "1" and torch.cuda.is_available()) else "cpu"
        model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
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


TTSUK_DIR = "/opt/audio-models/ttsuk"


def run_ttsuk(job):
    """tts_uk (RAD-TTS++, MIT): украинский родными голосами. Клона нет, голоса три.

    Возвращает тройку (мел-спектрограммы, волна, статистика) — нам нужна волна; частота 44 100.
    """
    # Пакет качает и ищет веса в ТЕКУЩЕМ каталоге, а не в HF_HOME. Поэтому переходим туда, где
    # они лежат (или куда их можно положить), и только потом импортируем.
    for cand in (TTSUK_DIR, LOCAL_CACHE + "/ttsuk", "/tmp/ttsuk"):
        try:
            os.makedirs(cand, exist_ok=True)
            os.chdir(cand)
            break
        except Exception:  # noqa: BLE001
            continue
    from tts_uk.inference import synthesis
    voice = job.get("voice") or "tetiana"          # tetiana, lada — женские; mykyta — мужской
    # Полная подпись: модель принимает не только текст и голос, но и рычаги просодии — темп,
    # высоту, громкость и разброс. Значения по умолчанию взяты нейтральными: 1.0 для темпа,
    # ноль для сдвигов высоты и энергии, обычные сигмы RAD-TTS++. Их можно править снаружи —
    # именно ими и будет отличаться манера у блогеров, которым достался один голос.
    _mels, wave, stats = synthesis(
        text=job["text"],
        voice=voice,
        n_takes=int(job.get("takes", 1)),
        use_latest_take=False,
        token_dur_scaling=float(job.get("speed_scale", 1.0)),
        f0_mean=float(job.get("f0_mean", 0.0)),
        f0_std=float(job.get("f0_std", 0.0)),
        energy_mean=float(job.get("energy_mean", 0.0)),
        energy_std=float(job.get("energy_std", 0.0)),
        sigma_decoder=float(job.get("sigma", 0.8)),
        sigma_token_duration=float(job.get("sigma_dur", 0.666)),
        sigma_f0=float(job.get("sigma_f0", 1.0)),
        sigma_energy=float(job.get("sigma_energy", 1.0)),
    )
    _cache["ttsuk_stats"] = stats
    return wave, 44100


# ГДЕ ЛЕЖАТ ВЕСА HIGGS. В образе их нет и быть не может: 12,8 ГБ не переживают тридцатиминутный
# предел сборки на RunPod. Ищем по порядку — сетевой том, потом образ (вдруг когда-нибудь влезут).
# Пути проверяем ДЕЛОМ, по наличию model.pth: код v2 читает именно его, и без него движок падает
# уже после загрузки одиннадцати гигабайт, то есть за наши деньги.
HIGGS_PLACES = ("/runpod-volume/higgs", "/opt/audio-models/higgs")


def _higgs_paths():
    for base in HIGGS_PLACES:
        model, tok = os.path.join(base, "model"), os.path.join(base, "tokenizer")
        if os.path.isdir(model) and os.path.exists(os.path.join(tok, "model.pth")):
            return model, tok
    places = ", ".join(HIGGS_PLACES)
    raise RuntimeError(
        "весов Higgs нет ни в одном из мест (%s). В образ они не влезают (предел сборки 30 минут), "
        "в работе их не скачать (диск контейнера 5 ГБ, сетевой том на 150 ГБ занят моделями видео). "
        "Их кладут на том с пода скриптом dl_higgs.py — сначала на томе нужно освободить место." % places
    )


def run_higgs(job):
    """Higgs Audio 2 (Boson AI, лицензия Boson Community на базе Meta Llama 3).

    Живёт в ДРУГОМ окружении — /opt/higgs-venv: ему нужен transformers 4.46, а Chatterbox рядом
    требует 5.2. Этот же файл запускается обоими интерпретаторами, потому что все импорты движков
    сделаны внутри функций: чужой движок не импортируется и мешать не может.

    Украинского и русского в родных языках модели НЕТ (заявлены en, zh, de, ko). Поэтому наши
    языки берём КЛОНОМ: даём образец живой речи (`ref_audio`) вместе с его расшифровкой
    (`ref_text`) — и модель продолжает говорить тем же голосом уже наш текст. Расшифровка
    обязательна: без неё модель не знает, что именно звучит в образце, и клон разваливается.
    """
    from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    import torch

    eng = _cache.get("higgs")
    if eng is None:
        model_dir, tok_dir = _higgs_paths()
        dev = "cuda" if (os.environ.get("FORCE_CPU") != "1" and torch.cuda.is_available()) else "cpu"
        eng = HiggsAudioServeEngine(
            model_dir, tok_dir, device=dev,
            # На процессоре половинная точность считается медленно и местами не считается вовсе.
            torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        )
        _cache["higgs"] = eng

    # Описание сцены — это то, чем модель задаёт манеру записи. «Тихая комната» держит её ближе
    # к обычной речи, без студийного эха, — ровно то, что нам нужно для съёмки «на телефон».
    scene = job.get("scene") or "Audio is recorded from a quiet room."
    messages = [Message(
        role="system",
        content="Generate audio following instruction.\n\n<|scene_desc_start|>\n%s\n<|scene_desc_end|>" % scene,
    )]
    if job.get("ref_audio"):
        # Клон голоса разговором: реплика человека — расшифровка образца, ответ модели — сам
        # образец. Дальше идёт наш текст, и модель отвечает тем же голосом.
        # audio_url='placeholder' — именно так движок берёт звук из raw_audio, а не читает файл
        # с диска (на воркере писать некуда).
        messages.append(Message(role="user", content=job.get("ref_text") or ""))
        messages.append(Message(role="assistant",
                                content=AudioContent(audio_url="placeholder", raw_audio=job["ref_audio"])))
    messages.append(Message(role="user", content=job["text"]))

    out = eng.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        # Тридцать секунд речи — это около 750 звуковых шагов; две тысячи с запасом хватает.
        max_new_tokens=int(job.get("max_new_tokens", 2048)),
        temperature=float(job.get("temperature", 0.3)),
        top_p=float(job.get("top_p", 0.95)),
        top_k=int(job.get("top_k", 50)),
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        # Модель умеет ответить и текстом вместо звука — здесь это была бы пустая трата прогона.
        force_audio_gen=True,
        seed=job.get("seed"),
    )
    if out.audio is None:
        raise RuntimeError("Higgs ответил текстом, а не звуком")
    _cache["higgs_text"] = getattr(out, "generated_text", None)
    return torch.from_numpy(out.audio), out.sampling_rate


ENGINES = {"chatterbox": run_chatterbox, "ttsuk": run_ttsuk, "higgs": run_higgs}


def _one(job, engine, fn, started):
    """Одна дорожка: посчитать и завернуть в ответ. Ошибку объясняем словами, а не следом стека."""
    try:
        wave, sr = fn(job)
        data = _wav_bytes(wave, sr)
        return {
            "ok": True, "engine": engine, "sample_rate": sr,
            "device": "cpu" if os.environ.get("FORCE_CPU") == "1" else "gpu",
            "weights_at": os.environ.get("HF_HOME"),
            "seconds": round(len(data) / (sr * 2), 2),
            "took": round(time.time() - started, 1),
            "audio": base64.b64encode(data).decode(),
        }
    except Exception as e:  # noqa: BLE001
        import traceback
        msg = str(e)
        if "quota" in msg.lower() or "no space" in msg.lower():
            msg = ("веса некуда положить: том %s (свободно %s ГБ), диск контейнера %s "
                   "(свободно %s ГБ). Свободное место есть, но запись запрещена квотой — "
                   "веса должны приезжать внутри образа." % (
                       "пишется" if _vol_ok else "НЕ пишется", _free,
                       "пишется" if _local_ok else "НЕ пишется", _local_free))
        return {"ok": False, "engine": engine, "error": msg[:400],
                "trace": traceback.format_exc().strip().splitlines()[-4:]}


def main():
    started = time.time()
    try:
        job = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "вход не разобрать: %s" % e}))
        return

    # НЕСКОЛЬКО ДОРОЖЕК ЗА ОДИН ЗАХОД. Handler поднимает нас НОВЫМ ПРОЦЕССОМ на каждый запрос,
    # а значит и веса читаются заново: у Higgs это 11,5 ГБ, минута-полторы работы карты, и мы
    # платим за неё как за генерацию. Три языка тремя запросами — это три таких загрузки на
    # ровном месте. Со списком `items` модель поднимается один раз и читает все фразы подряд.
    # Общее у списка — движок и всё, что задано в корне; своё у дорожки — текст, язык, образец.
    if isinstance(job.get("items"), list):
        engine = job.get("engine")
        fn = ENGINES.get(engine)
        if not fn:
            print(json.dumps({"ok": False, "error": "нет такого движка: %s (есть: %s)"
                                                    % (engine, ", ".join(ENGINES))}))
            return
        common = {k: v for k, v in job.items() if k != "items"}
        out = []
        for item in job["items"]:
            piece = dict(common)
            piece.update(item if isinstance(item, dict) else {})
            t0 = time.time()
            if not (piece.get("text") or "").strip():
                out.append({"ok": False, "error": "пустой текст", "id": piece.get("id")})
                continue
            res = _one(piece, engine, fn, t0)
            res["id"] = piece.get("id")
            out.append(res)
        print(json.dumps({"ok": any(r.get("ok") for r in out), "engine": engine,
                          "items": out, "took": round(time.time() - started, 1)}))
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
    res = _one(job, engine, fn, started)
    # Где лежат веса и куда вообще можно писать — нужно только при разборе поломок, поэтому
    # добавляем к одиночному ответу, а список этим не засоряем.
    res.update({"volume_free_gb": _free, "volume_writable": _vol_ok,
                "container_free_gb": _local_free, "container_writable": _local_ok})
    print(json.dumps(res))


if __name__ == "__main__":
    main()
