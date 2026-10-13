# Farm Advisor: voice, text and photo assistant

Speak (English, Yoruba, Hausa), type, or attach a photo. NCAIR ASR models transcribe speech, a Whisper
language detector routes it to the right model, and Claude answers as an agricultural advisor.

```
browser -> Flask /api/chat -> ffmpeg -> language router -> NCAIR ASR -> Claude -> reply
```

## Before deploying
1. On Hugging Face, accept the terms on all three model pages (they are gated):
   NCAIR1/NigerianAccentedEnglish, NCAIR1/Yoruba-ASR, NCAIR1/Hausa-ASR.
2. Have an Anthropic API key with credit, and a Hugging Face token (read access).

## Environment variables (set these on the GPU platform, never commit them)
| Name | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (required) |
| `HF_TOKEN` | Hugging Face token (required for the gated models) |
| `CLAUDE_MODEL` | default `claude-sonnet-5-5` |
| `PORT` | default `8000` |
| `TUNNEL` | `cloudflare` prints a public https link; `none` if the platform exposes the port itself |
| `ACCESS_CODE` | optional; visitors must enter it before the app answers |

## Start command for the platform
```
pip install -r requirements.txt && python run.py
```
Needs Python 3.10+, an NVIDIA GPU and a CUDA build of PyTorch (most GPU images include it).
ffmpeg is used if installed, otherwise a bundled copy is used automatically.
The first start downloads the models and takes a few minutes. Wait for the `DEMO LINK` banner in the logs.

## Run locally
```
cp .env.example .env   # fill in the keys
pip install -r requirements.txt
python run.py
```
Open http://localhost:8000 (the mic works on localhost and on https links).

## Checks
- `GET /health` returns the device and loaded models.
- Text-only test: `curl -F text="My maize leaves are yellow" localhost:8000/api/chat`

## Notes
- Run a single process: models are loaded once into GPU memory. Do not start multiple workers.
- Conversations are held in memory per browser tab and are lost on restart.
- The NCAIR models have research / non-commercial licence terms. Check them before any commercial use.
- Anyone with the public link can use your Claude credit. Set `ACCESS_CODE` and stop the process after the demo.
