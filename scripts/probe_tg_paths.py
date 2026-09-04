"""Probe Telegram paths without printing proxy host/credentials."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def tcp(host: str, port: int, timeout: float = 8) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return "ok"
    except Exception as exc:
        return type(exc).__name__
    finally:
        sock.close()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    proxy = ""
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("TELEGRAM_PROXY="):
                proxy = line.split("=", 1)[1].strip()
                break
    print("proxy_set", bool(proxy and "://" in proxy))
    if proxy:
        parsed = urlparse(proxy if "://" in proxy else "socks5://" + proxy)
        if parsed.hostname and parsed.port:
            print("socks_tcp", tcp(parsed.hostname, int(parsed.port), 12))
    for port in (443, 80, 8443, 8080):
        print("telegram", port, tcp("api.telegram.org", port))
    print("loopback_11080", tcp("127.0.0.1", 11080, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
