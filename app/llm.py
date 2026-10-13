import base64, io, os, threading
import anthropic
from PIL import Image

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_MESSAGES = 12

SYSTEM = """You are an experienced agricultural advisor for smallholder and commercial farmers
in Nigeria and West Africa. You know local crops (cassava, yam, maize, rice, sorghum, cocoa,
tomato, pepper, poultry, fish farming), seasons and rainfall patterns, common pests and
diseases, soil management, and market realities.

Rules:
- Give practical, step-by-step advice suited to local conditions and budgets.
- If a photo is provided, describe what you see first, then give likely causes and what to check.
- Be honest about uncertainty. Don't invent exact dosages, prices or product names you aren't
  sure of, and recommend confirming chemical use with a local extension officer or agrovet.
- The user's text may come from speech recognition and contain errors, or mix English with
  Yoruba, Hausa or Pidgin. Interpret it charitably, and ask a short question if it's unclear.
- Keep answers concise. Use short paragraphs and simple "-" lists. No headings or tables."""

_sessions, _lock, _client = {}, threading.Lock(), None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _image_block(raw):
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise ValueError("One of the images couldn't be read. Use a JPG or PNG photo.")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def _trim(msgs):
    msgs = msgs[-MAX_MESSAGES:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


def reset(session_id):
    with _lock:
        _sessions.pop(session_id, None)


def chat(session_id, text, images=()):
    content = [_image_block(b) for b in images]
    content.append({"type": "text", "text": text or "Please look at this photo and advise me."})
    with _lock:
        history = list(_sessions.get(session_id, []))
    history.append({"role": "user", "content": content})
    resp = _get_client().messages.create(
        model=MODEL, max_tokens=1024, system=SYSTEM, messages=_trim(history))
    reply = "".join(b.text for b in resp.content if b.type == "text").strip()
    history.append({"role": "assistant", "content": reply})
    with _lock:
        _sessions[session_id] = _trim(history)
    return reply
