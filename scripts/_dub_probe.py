"""One-off dub probe for a local video file."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


async def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "video_2026-08-22_21-30-18.mp4"
    lang = sys.argv[2] if len(sys.argv) > 2 else "ru"
    if not video.exists():
        print("missing", video)
        return 1

    from app.config import get_settings
    from app.database import Database
    from app.services.gigachat import GigaChatService
    from app.services.synthesis import SynthesisService
    from app.services.transcription import TranscriptionService
    from app.services.video_dub import VideoDubService
    from app.text.accent import AccentService
    from app.main import build_engines

    settings = get_settings()
    db = Database(settings.db_path)
    primary, fallback = build_engines(settings)
    synthesis = SynthesisService(settings, db, primary, fallback)
    transcription = TranscriptionService(settings)
    accents = AccentService(settings)
    gigachat = GigaChatService(settings)
    dub = VideoDubService(settings, transcription, gigachat, synthesis, accents)

    user_id = int(settings.web_user_id or 1327953308)
    print("video", video, "lang", lang)

    segments, duration = await dub.analyze(video)
    print(f"segments={len(segments)} duration={duration:.2f}s")
    for i, s in enumerate(segments):
        print(
            f"  [{i}] {s.start:.2f}-{s.end:.2f} ({s.duration:.2f}s) "
            f"rms={s.rms:.3f} «{(s.text or '')[:60]}»"
        )

    translated = await dub.translate_segments(segments, lang)
    for i, t in enumerate(translated):
        print(f"  tr[{i}] {t[:80]}")

    async def prog(i, n, tip):
        if i == 0 or i == n or i % max(1, n // 4) == 0:
            print(f"  progress {i}/{n}: {tip[:50]}")

    result = await dub.render(
        user_id,
        video,
        segments,
        translated,
        lang,
        duration,
        on_progress=prog,
    )
    print("out", result.video_path)
    print("clone", result.clone_sec, "refs", len(result.clone_refs))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
