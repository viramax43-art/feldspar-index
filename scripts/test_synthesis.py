#!/usr/bin/env python3
"""Smoke-тест гибридного синтеза без Telegram."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio.postprocess import finalize_synthesis_output
from app.config import Settings
from app.text.preprocess import get_inference_params, prepare_text_for_tts
from app.tts.silero_engine import SileroEngine
from app.tts.xtts_engine import XTTSEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Тест синтеза XTTS / Silero / auto")
    parser.add_argument("--engine", default="auto", choices=["auto", "xtts", "silero"])
    parser.add_argument("--reference", type=Path, help="WAV-референс для XTTS")
    parser.add_argument("--text", required=True, help="Русский текст")
    parser.add_argument("--output", required=True, type=Path, help="Выходной WAV")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--language", default="ru")
    parser.add_argument("--silero-speaker", default=None)
    args = parser.parse_args()

    settings = Settings(
        TELEGRAM_BOT_TOKEN="test",
        DEVICE=args.device,
        TTS_ENGINE=args.engine,
    )
    speaker = args.silero_speaker or settings.silero_speaker

    use_silero = args.engine == "silero"
    if args.engine == "auto":
        use_silero = args.reference is None or not args.reference.exists()

    if use_silero:
        if not settings.silero_model_path.exists():
            print(
                f"Silero модель не найдена: {settings.silero_model_path}\n"
                "Скачайте: python scripts/download_silero.py"
            )
            return 1
        engine = SileroEngine(
            settings.silero_model_path,
            speaker=speaker,
            sample_rate=settings.output_sample_rate,
            device="cpu",
        )
        print(f"Загрузка Silero ({speaker})...")
        engine.load()
        engine.warmup()
        speaker_paths: list[Path] = []
        engine_name = "silero"
    else:
        if args.reference is None or not args.reference.exists():
            print("Для XTTS нужен --reference path/to.wav")
            return 1
        engine = XTTSEngine(settings.tts_model_name, args.device)
        print("Загрузка XTTS...")
        engine.load()
        speaker_paths = [args.reference]
        engine_name = "xtts"

    chunks = prepare_text_for_tts(
        args.text,
        max_chunk_chars=settings.phrase_max_chars,
        min_chunk_chars=settings.phrase_min_chars,
        soft_max_chunk_chars=settings.phrase_soft_max_chars,
        engine=engine_name,
    )
    params = get_inference_params("neutral")
    audio_chunks = []
    pauses = []
    conditioning = None
    t0 = time.perf_counter()
    ttfp = 0.0

    for i, chunk in enumerate(chunks):
        print(f"[{engine_name}] Синтез: {chunk.text[:60]}...")
        t_chunk = time.perf_counter()
        wav, conditioning = engine.synthesize_chunk(
            chunk.text,
            speaker_paths,
            args.language,
            params,
            conditioning,
        )
        if i == 0:
            ttfp = (time.perf_counter() - t0) * 1000
        print(f"  chunk {i}: {(time.perf_counter() - t_chunk) * 1000:.0f} ms")
        audio_chunks.append(wav)
        pauses.append(chunk.pause_after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    finalize_synthesis_output(
        audio_chunks,
        pauses,
        args.output,
        None,
        sample_rate=engine.sample_rate,
    )
    total = (time.perf_counter() - t0) * 1000
    print(f"Готово: {args.output}")
    print(f"Metrics: ttfp={ttfp:.0f}ms total={total:.0f}ms engine={engine_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
