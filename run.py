"""Single entry point: loads models, starts the web server, opens a public link."""
import os, platform, re, shutil, stat, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.getenv("PORT", "8000"))
TUNNEL = os.getenv("TUNNEL", "cloudflare").lower()


def banner(url):
    print("\n" + "=" * 62 + f"\n  DEMO LINK: {url}\n" + "=" * 62 + "\n", flush=True)


def ensure_cloudflared():
    found = shutil.which("cloudflared")
    if found:
        return found
    local = Path("bin/cloudflared")
    if not local.exists():
        arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        print("Downloading cloudflared...", flush=True)
        local.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, local)
        local.chmod(local.stat().st_mode | stat.S_IEXEC)
    return str(local)


def start_tunnel():
    proc = subprocess.Popen(
        [ensure_cloudflared(), "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            banner(m.group(0))
            break
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    return proc


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to the platform's environment variables or .env")
    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set. The NCAIR models are gated and may fail to download.")

    from app import asr
    print("Loading models (this takes a few minutes the first time)...", flush=True)
    asr.load_all(os.getenv("HF_TOKEN"))

    from app.server import app
    from waitress import serve
    threading.Thread(target=lambda: serve(app, host="0.0.0.0", port=PORT, threads=4), daemon=True).start()
    time.sleep(1)
    print(f"Server running on port {PORT}", flush=True)

    if TUNNEL == "cloudflare":
        start_tunnel()
    else:
        print(f"No tunnel started. Open the URL your platform gives you for port {PORT}.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
