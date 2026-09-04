"""Download a portable FFmpeg build into bin/ffmpeg/bin (no admin)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "bin" / "ffmpeg"
URLS = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/GyanD/codexffmpeg/releases/download/8.0/ffmpeg-8.0-essentials_build.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
)


def _download(url: str) -> bytes:
    print("downloading", url)
    req = Request(url, headers={"User-Agent": "voice-caller-ffmpeg"})
    with urlopen(req, timeout=420) as resp:
        return resp.read()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    existing = DEST / "bin" / "ffmpeg.exe"
    if existing.is_file() and existing.stat().st_size > 1_000_000:
        print("already", existing)
        return 0
    last_error: Exception | None = None
    data = b""
    for url in URLS:
        try:
            data = _download(url)
            if len(data) > 5_000_000:
                print("zip bytes", len(data))
                break
            print("too small", len(data))
        except Exception as exc:
            last_error = exc
            print("failed", url, exc)
            data = b""
    if len(data) < 5_000_000:
        raise SystemExit(f"download failed: {last_error}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        wanted = ("bin/ffmpeg.exe", "bin/ffprobe.exe", "bin/ffplay.exe")
        for name in names:
            unix = name.replace("\\", "/")
            if not any(unix.endswith(w) for w in wanted):
                continue
            target = DEST / "bin" / Path(unix).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as out:
                out.write(src.read())
            print("wrote", target, target.stat().st_size)
    if not (DEST / "bin" / "ffmpeg.exe").is_file():
        raise SystemExit("ffmpeg.exe missing after extract")
    print("ok", DEST / "bin" / "ffmpeg.exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
