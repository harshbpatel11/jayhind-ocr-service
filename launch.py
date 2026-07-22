"""Start the OCR server + a public Cloudflare tunnel, then print the URL and API key.

    python launch.py

Loads the models (via `import server`), serves on $PORT (default 8000), opens a
public HTTPS tunnel via Cloudflare, and prints PUBLIC_URL + API_KEY. Leave it
running.

Two tunnel modes:
  * Quick tunnel (default): no signup, but a NEW random trycloudflare.com URL
    every run.
  * Named tunnel (set CF_TUNNEL_TOKEN): a STABLE URL bound to your own domain
    that never changes across restarts. Configure the tunnel + its public
    hostname once in the Cloudflare Zero Trust dashboard, then export
    CF_TUNNEL_TOKEN (and CF_TUNNEL_HOSTNAME so this script can print the URL).

Note: a Cloudflare tunnel still times out any single request that runs longer
than ~100s at the edge (HTTP 524) on free/pro plans — a named tunnel gives a
stable URL but does NOT lift that cap.
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
# Token is a SECRET — never hardcode it (this repo is public). Pass it in Colab.
CF_TUNNEL_TOKEN = os.getenv("CF_TUNNEL_TOKEN", "").strip()
# Hostname is not secret, so it is baked in as the default (override with the env
# var if you point the tunnel at a different subdomain).
CF_TUNNEL_HOSTNAME = os.getenv("CF_TUNNEL_HOSTNAME", "ocr.aakhaja.com").strip()
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


def _ensure_cloudflared() -> str:
    cf = os.path.join(os.getcwd(), "cloudflared")
    if not os.path.exists(cf):
        urllib.request.urlretrieve(CF_URL, cf)
        os.chmod(cf, 0o755)
    return cf


def _drain(proc) -> None:
    # keep reading cloudflared's output so its stdout pipe never blocks
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()


def _named_tunnel():
    """STABLE URL: run the dashboard-managed named tunnel via its token.
    The public hostname → http://localhost:PORT mapping is configured once in the
    Cloudflare dashboard, so nothing to parse here — the URL is your fixed hostname.
    """
    cf = _ensure_cloudflared()
    proc = subprocess.Popen(
        [cf, "tunnel", "--no-autoupdate", "run", "--token", CF_TUNNEL_TOKEN],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    _drain(proc)
    url = f"https://{CF_TUNNEL_HOSTNAME}" if CF_TUNNEL_HOSTNAME else "<your configured tunnel hostname>"
    return url, proc


def _quick_tunnel():
    """New random trycloudflare.com URL each run (no signup)."""
    cf = _ensure_cloudflared()
    proc = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        m = re.search(r"https://[-a-z0-9.]+trycloudflare\.com", line)
        if m:
            _drain(proc)
            return m.group(0), proc
    return None, proc


def main():
    threading.Thread(target=_serve, daemon=True).start()
    print("[launch] loading models & starting server (first run downloads several GB)...", flush=True)
    if not _wait_healthy():
        sys.exit("[launch] server did not become healthy in time")

    if CF_TUNNEL_TOKEN:
        url, proc = _named_tunnel()
        tunnel = "cloudflare (named — stable URL)"
    else:
        url, proc = _quick_tunnel()
        tunnel = "cloudflare (quick — new URL each run)"
    if not url:
        sys.exit("[launch] could not obtain a Cloudflare tunnel URL")

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
    if CF_TUNNEL_TOKEN:
        print("  (named tunnel: URL + key stay fixed — set them in .env ONCE and you")
        print("   never touch the hub again on a Colab restart)")
    print(f"{bar}\n  (leave this running; stop the cell / Ctrl-C to shut down)\n", flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
