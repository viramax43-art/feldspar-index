"""Discover an outer UPnP IGD on 192.168.0.x (double NAT)."""

from __future__ import annotations

import socket
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SSDP = """M-SEARCH * HTTP/1.1\r
HOST: 239.255.255.250:1900\r
MAN: "ssdp:discover"\r
MX: 2\r
ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r
\r
"""


def unicast_locations(host: str, timeout: float = 2.5) -> list[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.settimeout(timeout)
    sock.sendto(SSDP.encode("ascii"), (host, 1900))
    found: set[str] = set()
    try:
        while True:
            data, _addr = sock.recvfrom(65535)
            text = data.decode("utf-8", "replace")
            for line in text.split("\r\n"):
                if line.lower().startswith("location:"):
                    found.add(line.split(":", 1)[1].strip())
    except (TimeoutError, socket.timeout):
        pass
    finally:
        sock.close()
    return list(found)


def try_http(host: str, port: int, path: str, timeout: float = 2.0) -> str:
    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return f"{resp.status} {url} {resp.read(80)!r}"
    except Exception as exc:
        return f"fail {url} {exc}"


def main() -> int:
    for host in ("192.168.0.1", "192.168.0.10"):
        print("===", host)
        print("ssdp", unicast_locations(host))
        for port, path in (
            (80, "/"),
            (80, "/rootDesc.xml"),
            (1900, "/rootDesc.xml"),
            (8080, "/"),
            (52869, "/rootDesc.xml"),
        ):
            print(try_http(host, port, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
