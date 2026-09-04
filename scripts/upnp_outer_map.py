"""Map HTTP ports on the outer Realtek IGD (192.168.0.1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upnp_map import _ctrl_url, _soap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOCATION = "http://192.168.0.1:49152/rootDesc.xml"
INNER_WAN = "192.168.0.10"
MAPS = (
    (80, 80),
    (8765, 8765),
    (2223, 2223),
    (8443, 8443),
    (23456, 23456),
)


def main() -> int:
    ctrl = _ctrl_url(LOCATION)
    print("ctrl", ctrl)
    if not ctrl:
        print("no WANIPConnection")
        return 1
    last = None
    for ext, internal in MAPS:
        ok = False
        for service in (
            "urn:schemas-upnp-org:service:WANIPConnection:1",
            "urn:schemas-upnp-org:service:WANPPPConnection:1",
            "urn:schemas-upnp-org:service:WANIPConnection:2",
        ):
            body = (
                f'<u:AddPortMapping xmlns:u="{service}">'
                "<NewRemoteHost></NewRemoteHost>"
                f"<NewExternalPort>{ext}</NewExternalPort>"
                "<NewProtocol>TCP</NewProtocol>"
                f"<NewInternalPort>{internal}</NewInternalPort>"
                f"<NewInternalClient>{INNER_WAN}</NewInternalClient>"
                "<NewEnabled>1</NewEnabled>"
                "<NewPortMappingDescription>VoiceCallerOuter</NewPortMappingDescription>"
                "<NewLeaseDuration>0</NewLeaseDuration>"
                "</u:AddPortMapping>"
            )
            try:
                _soap(ctrl, "AddPortMapping", body, service)
                print(f"OUTER OK {ext}->{INNER_WAN}:{internal}")
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                continue
        if not ok:
            print(f"OUTER FAIL {ext}: {last}")
    ext_xml = None
    for service in (
        "urn:schemas-upnp-org:service:WANIPConnection:1",
        "urn:schemas-upnp-org:service:WANIPConnection:2",
    ):
        try:
            ext_xml = _soap(
                ctrl,
                "GetExternalIPAddress",
                f'<u:GetExternalIPAddress xmlns:u="{service}"></u:GetExternalIPAddress>',
                service,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
    print("external", (ext_xml or b"").decode("utf-8", "replace")[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
