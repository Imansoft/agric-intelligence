import os, shutil, subprocess, tempfile
import numpy as np

SR = 16000

def _ffmpeg():
    return shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()

def to_waveform(data: bytes, suffix=".webm", max_seconds=30):
    """Any browser audio (webm/ogg/m4a) -> mono 16 kHz float32 numpy array."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        out = subprocess.run(
            [_ffmpeg(), "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
            capture_output=True, check=True,
        ).stdout
    finally:
        os.unlink(path)
    return np.frombuffer(out, dtype=np.float32)[: SR * max_seconds].copy()
