from __future__ import annotations

import socket
import sys


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sock = socket.socket()
    sock.settimeout(3)
    try:
        sock.connect(("127.0.0.1", 11080))
        print("loopback_11080 ok")
    except Exception as exc:
        print("loopback_11080", type(exc).__name__)
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
