"""Open an outbound SSH tunnel to localhost.run so the studio is public."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
TARGETS = [
    [
        SSH,
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=NUL",
        "-o",
        "ServerAliveInterval=30",
        "-R",
        "80:127.0.0.1:8765",
        "nokey@localhost.run",
    ],
    [
        SSH,
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=NUL",
        "-o",
        "ServerAliveInterval=30",
        "-R",
        "80:127.0.0.1:8765",
        "serveo.net",
    ],
]


def launch(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.write_bytes(b"")
    fh = log_path.open("ab")
    flags = (
        CREATE_BREAKAWAY_FROM_JOB
        | CREATE_NEW_PROCESS_GROUP
        | DETACHED_PROCESS
        | CREATE_NO_WINDOW
    )
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=False,
        creationflags=flags,
    )


def wait_url(log_path: Path, seconds: float = 35.0) -> str:
    deadline = time.time() + seconds
    while time.time() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for token in text.replace("\n", " ").replace("\r", " ").split():
            if token.startswith("https://") or (token.startswith("http://") and "127.0.0.1" not in token):
                if any(part in token for part in (".lhr.life", "localhost.run", "serveo.net", "loca.lt")):
                    return token.strip("',\"")
        time.sleep(1.2)
    return ""


def main() -> int:
    root = Path(r"C:\work\app")
    log_path = root / "logs" / "tunnel.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["taskkill", "/F", "/IM", "ssh.exe"], capture_output=True, check=False)
    url = ""
    for cmd in TARGETS:
        print("try", " ".join(cmd[-2:]))
        launch(cmd, log_path)
        url = wait_url(log_path)
        print((log_path.read_text(encoding="utf-8", errors="replace") or "")[-1500:])
        if url:
            (root / "logs" / "PUBLIC_URL.txt").write_text(url + "\n", encoding="ascii")
            print("PUBLIC", url)
            return 0
        subprocess.run(["taskkill", "/F", "/IM", "ssh.exe"], capture_output=True, check=False)
    print("no public url")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
