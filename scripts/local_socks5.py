"""Minimal SOCKS5 CONNECT proxy for Bot API via reverse SSH hop. No auth."""

from __future__ import annotations

import select
import socket
import struct
import sys
import threading


LISTEN = ("127.0.0.1", 11081)


def _pipe(a: socket.socket, b: socket.socket) -> None:
    for sock in (a, b):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
    try:
        while True:
            # Не рвать длинные GetFile: простой — норма, пока TCP жив.
            readable, _, failed = select.select([a, b], [], [a, b], 3600)
            if failed:
                return
            if not readable:
                continue
            for src in readable:
                dst = b if src is a else a
                data = src.recv(65536)
                if not data:
                    return
                dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (a, b):
            try:
                sock.close()
            except OSError:
                pass


def _client(conn: socket.socket) -> None:
    try:
        hello = conn.recv(256)
        if len(hello) < 2 or hello[0] != 0x05:
            conn.close()
            return
        conn.sendall(b"\x05\x00")
        req = conn.recv(4)
        if len(req) < 4 or req[0] != 0x05 or req[1] != 0x01:
            conn.sendall(b"\x05\x07\x00\x01" + b"\x00" * 6)
            conn.close()
            return
        atyp = req[3]
        if atyp == 1:
            raw = conn.recv(4)
            host = socket.inet_ntoa(raw)
        elif atyp == 3:
            ln = conn.recv(1)[0]
            host = conn.recv(ln).decode("idna")
        elif atyp == 4:
            raw = conn.recv(16)
            host = socket.inet_ntop(socket.AF_INET6, raw)
        else:
            conn.close()
            return
        port = struct.unpack("!H", conn.recv(2))[0]
        remote = socket.create_connection((host, port), timeout=20)
        conn.sendall(b"\x05\x00\x00\x01" + socket.inet_aton("0.0.0.0") + b"\x00\x00")
        _pipe(conn, remote)
    except OSError:
        try:
            conn.close()
        except OSError:
            pass


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(LISTEN)
    srv.listen(32)
    print(f"socks5_listen {LISTEN[0]}:{LISTEN[1]}", flush=True)
    while True:
        conn, _addr = srv.accept()
        threading.Thread(target=_client, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
