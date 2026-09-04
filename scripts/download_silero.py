#!/usr/bin/env python3
"""Скачать offline-пакет Silero TTS v5_5_ru."""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
DEFAULT_OUT = Path("assets/tts/silero/v5_5_ru.pt")
MIN_BYTES = 20 * 1024 * 1024


def is_valid_silero_pt(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        return any("tts_models" in name.replace("\\", "/") for name in names)
    except (zipfile.BadZipFile, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Silero v5_5_ru.pt")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.force and is_valid_silero_pt(args.output):
        print(f"Уже скачано: {args.output} ({args.output.stat().st_size // 1024} KB)")
        return 0
    if args.output.exists() and not is_valid_silero_pt(args.output):
        print(
            f"Битый пакет ({args.output.stat().st_size} bytes), качаю заново",
            flush=True,
        )
        args.output.unlink()

    tmp = args.output.with_suffix(".pt.partial")
    print(f"Скачиваю {args.url} → {args.output}", flush=True)
    try:
        req = urllib.request.Request(
            args.url, headers={"User-Agent": "voice-caller-silero"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if not is_valid_silero_pt(tmp):
            size = tmp.stat().st_size if tmp.exists() else 0
            tmp.unlink(missing_ok=True)
            print(f"Скачанный файл битый ({size} bytes)", file=sys.stderr)
            return 1
        tmp.replace(args.output)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"Готово: {args.output} ({args.output.stat().st_size // 1024} KB)", flush=True)
    print("Лицензия RU v5: CC BY-NC-SA 4.0 (некоммерческое использование).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
