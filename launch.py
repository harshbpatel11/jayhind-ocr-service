"""Start the OCR server + a public tunnel, then print the URL and API key.

    python launch.py

Loads the models (via `import server`), serves on $PORT (default 8000), opens a
public HTTPS tunnel, and prints PUBLIC_URL + API_KEY. Leave it running.

Tunnel: Cloudflare quick tunnel by default (no signup, new URL each run). Set
NGROK_TOKEN to use ngrok instead (supports a reserved domain = stable URL).
"""
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

import uvicorn

import server  # noqa: E402  (import triggers model loading)

PORT = int(os.getenv("PORT", "8000"))
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def _serve():
    uvicorn.run(server.app, host="0.0.0.0", port=PORT, log_level="warning")


def _wait_healthy(timeout=900) -> bool:
    import requests

    for _ in range(timeout):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _cloudflare_tunnel():
    cf = os.path.join(os.getcwd(), "cloudflared")
    if not os.path.exists(cf):
        urllib.request.urlretrieve(CF_URL, cf)
        os.chmod(cf, 0o755)
    proc = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        m = re.search(r"https://[-a-z0-9.]+trycloudflare\.com", line)
        if m:
            # keep draining so cloudflared's pipe never blocks
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
            return m.group(0), proc
    return None, proc


def main():
    threading.Thread(target=_serve, daemon=True).start()
    print("[launch] loading models & starting server (first run downloads several GB)...", flush=True)
    if not _wait_healthy():
        sys.exit("[launch] server did not become healthy in time")

    token = os.getenv("NGROK_TOKEN", "").strip()
    proc = None
    if token:
        # ngrok has no ~100s response cap (unlike a Cloudflare quick tunnel), so
        # the minutes-long PaddleOCR-VL + LLM parse can finish. Preferred engine.
        from pyngrok import ngrok

        ngrok.set_auth_token(token)
        url = ngrok.connect(PORT, "http").public_url
        # ngrok exposes both schemes on one host; always hand out the https URL.
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        tunnel = "ngrok"
    else:
        # Cloudflare quick tunnel: no signup, but its edge times out slow origins
        # at ~100s (HTTP 524). Fine for the fast loopback engine; set NGROK_TOKEN
        # for the hosted GPU engine, whose parses routinely exceed that.
        print("[launch] NGROK_TOKEN not set — using a Cloudflare quick tunnel "
              "(~100s response cap; set NGROK_TOKEN for the hosted GPU engine).", flush=True)
        url, proc = _cloudflare_tunnel()
        tunnel = "cloudflare"

    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  TUNNEL     = {tunnel}")
    print(f"  PUBLIC_URL = {url}")
    print(f"  API_KEY    = {server.API_KEY}")
    print(bar)
    print("  Put both in jayhind-admin-back/.env :")
    print(f"     OCR_SERVICE_URL={url}")
    print(f"     OCR_SERVICE_KEY={server.API_KEY}")
    print("  then:  dev restart admin-back")
    print(f"{bar}\n  (leave this running; stop the cell / Ctrl-C to shut down)\n", flush=True)

    try:
        if proc is not None:
            proc.wait()
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
