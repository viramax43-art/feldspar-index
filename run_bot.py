"""Стабильный entrypoint бота (не `python -m app.main` — на Windows это плодит второй poller)."""

from __future__ import annotations

import asyncio
import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()
    # Windows spawn-воркеры / torch иначе поднимают C:\\Python312\\python.exe
    # и могут заново выполнить этот файл → второй poller → Telegram Conflict.
    if multiprocessing.current_process().name != "MainProcess":
        return
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        multiprocessing.set_executable(sys.executable)
        import multiprocessing.spawn as _spawn

        _spawn.set_executable(sys.executable)
    except Exception:
        pass
    from app.main import main as async_main

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
