import os, subprocess, time, uuid
from pathlib import Path
import anthropic
from flask import Flask, jsonify, request, send_from_directory
from . import asr, llm
from .audio import to_waveform

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
ACCESS_CODE = os.getenv("ACCESS_CODE", "")


@app.before_request
def gate():
    if ACCESS_CODE and request.path.startswith("/api/") and request.headers.get("X-Access-Code") != ACCESS_CODE:
        return jsonify(error="Access code required."), 401


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", device=asr.DEVICE, models=asr.loaded())


@app.post("/api/reset")
def reset():
    llm.reset((request.get_json(silent=True) or {}).get("session_id", ""))
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    t0 = time.perf_counter()
    sid = request.form.get("session_id") or uuid.uuid4().hex
    text = (request.form.get("text") or "").strip()
    out, timings = {}, {}
    try:
        f = request.files.get("audio")
        if f:
            wav = to_waveform(f.read(), Path(f.filename or "").suffix or ".webm")
            r = asr.transcribe(wav, request.form.get("language", "auto"))
            out.update(transcript=r["text"], language=r["language"],
                       language_name=r["language_name"], confidence=r["confidence"])
            timings.update(detect=round(r["detect_s"], 2), asr=round(r["asr_s"], 2))
            text = f"{text} {r['text']}".strip()
        images = [i.read() for i in request.files.getlist("images")[:3]]
        if not text and not images:
            return jsonify(error="Nothing was heard. Tap the mic and try again."), 400
        t1 = time.perf_counter()
        reply = llm.chat(sid, text, images)
        timings.update(llm=round(time.perf_counter() - t1, 2), total=round(time.perf_counter() - t0, 2))
        return jsonify(reply=reply, timings=timings, **out)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except subprocess.CalledProcessError:
        return jsonify(error="That audio couldn't be read. Try recording again."), 400
    except anthropic.APIError as e:
        return jsonify(error=f"Claude API error: {getattr(e, 'message', e)}"), 502
    except Exception as e:
        app.logger.exception("chat failed")
        return jsonify(error=f"Server error: {e}"), 500
