"""Start cloudflared in breakaway mode and print the public URL."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    exe = root / "bin" / "cloudflared.exe"
    log_path = root / "logs" / "cloudflared.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not exe.exists():
        print("missing", exe)
        return 1
    subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], capture_output=True, check=False)
    log_path.write_text("", encoding="utf-8")
    flags = (
        CREATE_BREAKAWAY_FROM_JOB
        | CREATE_NEW_PROCESS_GROUP
        | DETACHED_PROCESS
        | CREATE_NO_WINDOW
    )
    fh = log_path.open("ab")
    subprocess.Popen(
        [str(exe), "tunnel", "--url", "http://127.0.0.1:8765", "--no-autoupdate"],
        cwd=str(root),
        stdout=fh,
        stderr=subprocess.STDOUT,
        close_fds=False,
        creationflags=flags,
    )
    url = ""
    for _ in range(40):
        time.sleep(1.5)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for token in text.replace("\n", " ").split():
            if "trycloudflare.com" in token and token.startswith("https://"):
                url = token.strip("'\",")
                break
        if url:
            break
    if url:
        (root / "logs" / "PUBLIC_URL.txt").write_text(url + "\n", encoding="ascii")
        print("PUBLIC", url)
        return 0
    print("no url yet")
    print(log_path.read_text(encoding="utf-8", errors="replace")[-2500:])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
