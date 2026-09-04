"""Start app.main outside the current Windows job so it survives an SSH session."""

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
    log_path = root / "logs" / "app.main.out.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    conda = Path(r"C:\ProgramData\Anaconda3")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    extra = os.pathsep.join(
        [
            str(root / "bin" / "ffmpeg" / "bin"),
            str(root / "bin"),
            str(conda),
            str(conda / "Scripts"),
            str(conda / "Library" / "bin"),
        ]
    )
    env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    flags = (
        CREATE_BREAKAWAY_FROM_JOB
        | CREATE_NEW_PROCESS_GROUP
        | DETACHED_PROCESS
        | CREATE_NO_WINDOW
    )
    log_fh = log_path.open("ab")
    # Не использовать `python -m app.main`: на Windows spawn/torch
    # поднимает второй poller через base C:\Python312 → Telegram Conflict.
    proc = subprocess.Popen(
        [sys.executable, "-u", str(root / "run_bot.py")],
        cwd=str(root),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        close_fds=False,
        creationflags=flags,
    )
    print(f"detached pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
