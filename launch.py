"""Start the OCR server + a public Cloudflare tunnel, then print the URL and API key.

    python launch.py

Loads the models (via `import server`), serves on $PORT (default 8000), opens a
public HTTPS tunnel with a Cloudflare quick tunnel (no signup, new URL each run),
and prints PUBLIC_URL + API_KEY. Leave it running.

Note: a Cloudflare quick tunnel times out any single request that runs longer
than ~100s at the edge (HTTP 524). That is fine for fast documents; a parse that
routinely exceeds it needs a faster engine or a tunnel without that cap.
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

    url, proc = _cloudflare_tunnel()
    if not url:
        sys.exit("[launch] could not obtain a Cloudflare tunnel URL")

    bar = "=" * 68
    print(f"\n{bar}")
    print("  TUNNEL     = cloudflare")
    print(f"  PUBLIC_URL = {url}")
    print(f"  API_KEY    = {server.API_KEY}")
    print(bar)
    print("  Put both in jayhind-admin-back/.env :")
    print(f"     OCR_SERVICE_URL={url}")
    print(f"     OCR_SERVICE_KEY={server.API_KEY}")
    print("  then:  dev restart admin-back")
    print(f"{bar}\n  (leave this running; stop the cell / Ctrl-C to shut down)\n", flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
