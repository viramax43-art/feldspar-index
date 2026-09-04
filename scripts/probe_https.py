"""HTTPS reachability without printing secrets."""

from __future__ import annotations

import socket
import sys
import urllib.request

TARGETS = (
    ("api.telegram.org", 443),
    ("149.154.166.110", 443),
    ("huggingface.co", 443),
)


def tcp(host: str, port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(8)
    try:
        sock.connect((host, port))
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"
    finally:
        sock.close()


def https(url: str) -> str:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "voice-caller"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return f"status={resp.status}"
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for host, port in TARGETS:
        print(f"tcp {host}:{port} {tcp(host, port)}", flush=True)
    print("https telegram", https("https://api.telegram.org"), flush=True)
    print("https hf", https("https://huggingface.co"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
