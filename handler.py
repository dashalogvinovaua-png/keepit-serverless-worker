"""
Кастомный serverless-handler для ComfyUI, умеющий отдавать ВИДЕО (mp4), а не только картинки.
Стоковый runpod/worker-comfyui собирает лишь output-ключ "images" и игнорирует SaveVideo →
Wan/LTX не работают. Здесь собираем ЛЮБОЙ выходной файл (images/videos/gifs) как base64.

Вход job.input:
  { "workflow": <ComfyUI API graph>, "images": [ {"name": "kf_ctl.mp4", "image": "<base64>"}, ... ] }
Выход:
  { "images": [ {"filename": "...mp4", "type": "base64", "data": "<base64>"} ] }
(клиент src/steps/runpod_serverless.js рекурсивно находит самый большой base64-блоб = видео.)
"""
import os, time, base64, urllib.parse
import requests
import runpod

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_TIMEOUT = int(os.environ.get("COMFY_TIMEOUT", "1500"))  # сек на один рендер


def _url(path):
    return f"http://{COMFY_HOST}{path}"


def wait_comfy(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(_url("/"), timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"ComfyUI ({COMFY_HOST}) недоступен")


def upload_inputs(images):
    for it in images or []:
        name = it["name"]
        data = base64.b64decode(it["image"])
        files = {"image": (name, data), "overwrite": (None, "true"), "type": (None, "input")}
        r = requests.post(_url("/upload/image"), files=files, timeout=180)
        r.raise_for_status()


def queue(workflow):
    r = requests.post(_url("/prompt"), json={"prompt": workflow}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt {r.status_code}: {r.text[:400]}")
    return r.json()["prompt_id"]


def view_bytes(fn, sub, typ):
    q = urllib.parse.urlencode({"filename": fn, "subfolder": sub or "", "type": typ or "output"})
    r = requests.get(_url(f"/view?{q}"), timeout=600)
    r.raise_for_status()
    return r.content


def collect_outputs(rec):
    """Собираем все выходные файлы из истории, независимо от ключа (images/videos/gifs)."""
    out = []
    for _node_id, node_out in (rec.get("outputs") or {}).items():
        if not isinstance(node_out, dict):
            continue
        for _key, items in node_out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                fn = it.get("filename")
                if not fn or it.get("type") == "temp":
                    continue
                try:
                    b = view_bytes(fn, it.get("subfolder"), it.get("type"))
                    out.append({"filename": fn, "type": "base64", "data": base64.b64encode(b).decode()})
                except Exception as e:
                    out.append({"filename": fn, "error": str(e)})
    return out


def handler(job):
    inp = job.get("input") or {}
    workflow = inp.get("workflow")
    if not workflow:
        return {"error": "нет 'workflow' во input"}
    wait_comfy()
    try:
        upload_inputs(inp.get("images"))
    except Exception as e:
        return {"error": f"upload inputs: {e}"}
    try:
        pid = queue(workflow)
    except Exception as e:
        return {"error": f"queue: {e}"}

    t0 = time.time()
    while time.time() - t0 < COMFY_TIMEOUT:
        try:
            rec = requests.get(_url(f"/history/{pid}"), timeout=30).json().get(pid)
        except Exception:
            time.sleep(2)
            continue
        if rec:
            st = rec.get("status", {})
            if st.get("status_str") == "error":
                return {"error": "workflow execution error", "status": st}
            if rec.get("outputs"):
                data = collect_outputs(rec)
                if data:
                    return {"images": data}
        time.sleep(2)
    return {"error": "таймаут ожидания выходов ComfyUI"}


runpod.serverless.start({"handler": handler})
