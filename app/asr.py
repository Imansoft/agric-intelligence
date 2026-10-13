"""Three NCAIR ASR models plus a Whisper language detector, all loaded and warmed up at startup."""
import threading, time, warnings
import numpy as np
import torch
from huggingface_hub import login
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODELS = {
    "en": "NCAIR1/NigerianAccentedEnglish",
    "yo": "NCAIR1/Yoruba-ASR",
    "ha": "NCAIR1/Hausa-ASR",
}
NAMES = {"en": "Nigerian English", "yo": "Yoruba", "ha": "Hausa"}
LANGS = list(MODELS)
GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"
DTYPE = torch.float16 if GPU else torch.float32

_pipes, _lid, _lock = {}, {}, threading.Lock()


def loaded():
    return list(_pipes)


def load_all(hf_token=None):
    if hf_token:
        login(hf_token)
    print(f"Device: {DEVICE}", flush=True)
    for lang, model_id in MODELS.items():
        t0 = time.perf_counter()
        pipe = pipeline("automatic-speech-recognition", model=model_id,
                        device=0 if GPU else -1, torch_dtype=DTYPE)
        pipe(np.zeros(16000, dtype=np.float32),
             generate_kwargs={"language": lang, "task": "transcribe"})  # warm-up
        _pipes[lang] = pipe
        print(f"  {model_id} ready in {time.perf_counter() - t0:.1f}s", flush=True)

    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", torch_dtype=DTYPE).to(DEVICE).eval()
    tok = proc.tokenizer
    _lid.update(
        proc=proc, model=model,
        sot=tok.convert_tokens_to_ids("<|startoftranscript|>"),
        ids=[tok.convert_tokens_to_ids(f"<|{l}|>") for l in LANGS],
    )
    detect_language(np.zeros(16000, dtype=np.float32))
    print("  language detector ready", flush=True)


def detect_language(audio):
    feats = _lid["proc"](audio, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(DEVICE, dtype=DTYPE)
    dec = torch.tensor([[_lid["sot"]]], device=DEVICE)
    with torch.no_grad():
        logits = _lid["model"](feats, decoder_input_ids=dec).logits[0, -1]
    probs = torch.softmax(logits[_lid["ids"]].float(), dim=-1).cpu().tolist()
    return dict(zip(LANGS, probs))


def transcribe(audio, language="auto", min_conf=0.5):
    if audio.size < 1600:
        raise ValueError("The recording was too short. Hold the mic a little longer.")
    if language not in LANGS:
        language = "auto"
    t0 = time.perf_counter()
    probs = None
    with _lock:  # one GPU job at a time
        if language == "auto":
            probs = detect_language(audio)
            language = max(probs, key=probs.get)
            if probs[language] < min_conf:
                language = "en"
        t1 = time.perf_counter()
        text = _pipes[language](audio, generate_kwargs={"language": language, "task": "transcribe"})["text"].strip()
        t2 = time.perf_counter()
    return {
        "text": text, "language": language, "language_name": NAMES[language],
        "confidence": probs[language] if probs else None,
        "detect_s": t1 - t0, "asr_s": t2 - t1,
    }
