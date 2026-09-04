"""Start local SOCKS + reverse SSH hop as Windows detached processes (survive Cursor)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    flags = (
        CREATE_BREAKAWAY_FROM_JOB
        | CREATE_NEW_PROCESS_GROUP
        | DETACHED_PROCESS
        | CREATE_NO_WINDOW
    )
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    jobs = (
        ("local_socks5", [py, "-u", str(root / "scripts" / "local_socks5.py")]),
        ("tg_tunnel", [py, "-u", str(root / "scripts" / "_ssh_tg_tunnel.py")]),
    )
    for name, cmd in jobs:
        log_fh = (log_dir / f"{name}.out.log").open("ab")
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=False,
            creationflags=flags,
        )
        print(f"detached {name} pid={proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
