"""Add a UPnP IGD port mapping so the studio is reachable on the WAN IP."""

from __future__ import annotations

import socket
import sys
import urllib.request
from xml.etree import ElementTree as ET

SSDP = """M-SEARCH * HTTP/1.1\r
HOST: 239.255.255.250:1900\r
MAN: "ssdp:discover"\r
MX: 2\r
ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r
\r
"""


def _ipconfig_ipv4() -> list[str]:
    import re
    import subprocess

    try:
        out = subprocess.check_output(
            "ipconfig",
            shell=True,
            encoding="cp866",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return re.findall(r"IPv4[^:\r\n]*:\s*([0-9.]+)", out)


def _local_ip() -> str:
    ranked: list[str] = []
    for ip in _ipconfig_ipv4():
        if ip.startswith("127.") or ip.startswith("169.254."):
            continue
        ranked.append(ip)
    ranked.sort(
        key=lambda ip: (
            0 if ip.startswith("192.168.") else 1 if ip.startswith("10.") else 2,
            ip,
        )
    )
    if ranked:
        return ranked[0]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _ssdp_locations(timeout: float = 3.0, bind_ip: str | None = None) -> list[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    if bind_ip:
        sock.bind((bind_ip, 0))
        sock.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_MULTICAST_IF,
            socket.inet_aton(bind_ip),
        )
    sock.sendto(SSDP.encode("ascii"), ("239.255.255.250", 1900))
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


def _ctrl_url(location: str) -> str | None:
    with urllib.request.urlopen(location, timeout=5) as resp:
        xml = resp.read()
    ns = {
        "d": "urn:schemas-upnp-org:device-1-0",
        "s": "urn:schemas-upnp-org:service-1-0",
    }
    root = ET.fromstring(xml)
    base = location.rsplit("/", 1)[0]
    for service in root.findall(".//{urn:schemas-upnp-org:device-1-0}service"):
        stype = (service.findtext("{urn:schemas-upnp-org:device-1-0}serviceType") or "")
        if "WANIPConnection" not in stype and "WANPPPConnection" not in stype:
            continue
        ctrl = service.findtext("{urn:schemas-upnp-org:device-1-0}controlURL") or ""
        if ctrl.startswith("http"):
            return ctrl
        if ctrl.startswith("/"):
            from urllib.parse import urlparse

            p = urlparse(location)
            return f"{p.scheme}://{p.netloc}{ctrl}"
        return f"{base}/{ctrl}"
    return None


def _soap(ctrl: str, action: str, body: str, service: str) -> bytes:
    payload = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    ).encode("utf-8")
    req = urllib.request.Request(
        ctrl,
        data=payload,
        headers={
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPAction": f'"{service}#{action}"',
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.read()


def _first_ctrl(ip: str) -> tuple[str, str]:
    locations = _ssdp_locations(bind_ip=ip)
    if not locations:
        raise RuntimeError("UPnP gateway not found")
    last_err: Exception | None = None
    for loc in locations:
        try:
            ctrl = _ctrl_url(loc)
            if not ctrl:
                continue
            for service in (
                "urn:schemas-upnp-org:service:WANIPConnection:1",
                "urn:schemas-upnp-org:service:WANPPPConnection:1",
            ):
                try:
                    _soap(
                        ctrl,
                        "GetExternalIPAddress",
                        f'<u:GetExternalIPAddress xmlns:u="{service}"></u:GetExternalIPAddress>',
                        service,
                    )
                    return ctrl, service
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    continue
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"UPnP control URL not found: {last_err}")


def dump_status(ip: str) -> None:
    ctrl, service = _first_ctrl(ip)
    xml = _soap(
        ctrl,
        "GetExternalIPAddress",
        f'<u:GetExternalIPAddress xmlns:u="{service}"></u:GetExternalIPAddress>',
        service,
    ).decode("utf-8", "replace")
    print("external-xml", xml.replace("\n", " ")[:1500])
    for port in (80, 8765, 2222, 2223, 8443, 23456):
        body = (
            f'<u:GetSpecificPortMappingEntry xmlns:u="{service}">'
            "<NewRemoteHost></NewRemoteHost>"
            f"<NewExternalPort>{port}</NewExternalPort>"
            "<NewProtocol>TCP</NewProtocol>"
            "</u:GetSpecificPortMappingEntry>"
        )
        try:
            out = _soap(ctrl, "GetSpecificPortMappingEntry", body, service).decode(
                "utf-8", "replace"
            )
            print(f"map {port}:", out.replace("\n", " ")[:400])
        except Exception as exc:  # noqa: BLE001
            print(f"map {port}: missing ({exc})")


def add_mapping(ext_port: int, int_port: int, ip: str) -> None:
    locations = _ssdp_locations(bind_ip=ip)
    if not locations:
        raise RuntimeError("UPnP gateway not found")
    last_err: Exception | None = None
    for loc in locations:
        try:
            ctrl = _ctrl_url(loc)
            if not ctrl:
                continue
            for service in (
                "urn:schemas-upnp-org:service:WANIPConnection:1",
                "urn:schemas-upnp-org:service:WANPPPConnection:1",
            ):
                body = (
                    f'<u:AddPortMapping xmlns:u="{service}">'
                    "<NewRemoteHost></NewRemoteHost>"
                    f"<NewExternalPort>{ext_port}</NewExternalPort>"
                    "<NewProtocol>TCP</NewProtocol>"
                    f"<NewInternalPort>{int_port}</NewInternalPort>"
                    f"<NewInternalClient>{ip}</NewInternalClient>"
                    "<NewEnabled>1</NewEnabled>"
                    "<NewPortMappingDescription>AppWeb</NewPortMappingDescription>"
                    "<NewLeaseDuration>0</NewLeaseDuration>"
                    "</u:AddPortMapping>"
                )
                try:
                    _soap(ctrl, "AddPortMapping", body, service)
                    print(f"UPnP OK {ext_port}->{ip}:{int_port} via {loc}")
                    return
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    continue
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"UPnP mapping failed: {last_err}")


def main() -> int:
    ip = _local_ip()
    if len(sys.argv) > 1 and sys.argv[1] in {"status", "--status"}:
        print(f"local={ip}")
        dump_status(ip)
        return 0
    ext = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    internal = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    ip = sys.argv[3] if len(sys.argv) > 3 else ip
    print(f"local={ip}")
    add_mapping(ext, internal, ip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
