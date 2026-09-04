#!/usr/bin/env python3
"""Пересборка голосового профиля пользователя с улучшенной обработкой.

Этот скрипт:
1. Перерабатывает raw-файлы (если есть) или существующие ref_*.wav с bandpass + denoise
2. Пересчитывает качество
3. Пересобирает профиль
4. Сбрасывает кеш conditioning (будет пересчитан при первом запросе)

Использование:
    python scripts/rebuild_profile.py --user-id 1327953308
    python scripts/rebuild_profile.py --user-id 1327953308 --no-reprocess
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def rebuild(user_id: int, reprocess_raw: bool = True) -> None:
    from app.audio.preprocess import (
        apply_bandpass,
        apply_denoise,
        load_audio_mono,
        normalize_loudness,
        preprocess_telegram_voice,
        remove_long_pauses,
        speech_ratio,
        detect_clipping,
    )
    from app.audio.quality import evaluate_reference
    from app.config import get_settings
    from app.database import Database
    from app.services.voice_profile import VoiceProfileService
    import numpy as np
    import soundfile as sf

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.users_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_path)
    await db.init()
    profile_service = VoiceProfileService(settings, db)

    user_dir = profile_service.user_dir(user_id)
    refs_dir = profile_service.references_dir(user_id)
    cache_dir = profile_service.cache_dir(user_id)

    if not refs_dir.exists():
        logger.error("Папка референсов не найдена: %s", refs_dir)
        return

    # Raw может лежать в user_dir или в references/
    raw_files = sorted(
        set(
            list(user_dir.glob("raw_*.oga"))
            + list(user_dir.glob("account_raw_*.oga"))
            + list(refs_dir.glob("raw_*.oga"))
            + list(refs_dir.glob("account_raw_*.oga"))
        )
    )
    ref_files = sorted(refs_dir.glob("ref_*.wav"))

    if reprocess_raw and raw_files:
        logger.info("Переобработка %d raw-файлов с bandpass + denoise...", len(raw_files))
        for old_ref in ref_files:
            old_ref.unlink(missing_ok=True)
        await db.clear_voice_references(user_id)

        for idx, raw in enumerate(raw_files):
            wav_path = refs_dir / f"ref_{idx:03d}.wav"
            try:
                metrics = preprocess_telegram_voice(
                    raw,
                    wav_path,
                    sample_rate=settings.reference_sample_rate,
                    enable_denoise=True,
                    enable_bandpass=True,
                )
                quality = evaluate_reference(metrics)
                if quality.accepted:
                    await db.add_voice_reference(
                        user_id, wav_path.name, metrics["duration_sec"], quality.to_dict()
                    )
                    logger.info(
                        "  %s → %s (%.1f сек, score=%.0f)",
                        raw.name,
                        wav_path.name,
                        metrics["duration_sec"],
                        quality.score,
                    )
                else:
                    wav_path.unlink(missing_ok=True)
                    logger.warning("  %s отклонён: score=%.0f", raw.name, quality.score)
            except Exception as exc:
                logger.error("  %s ошибка: %s", raw.name, exc)

    elif reprocess_raw and ref_files:
        # Raw уже удалены после сбора — перечищаем существующие WAV
        logger.info(
            "Raw не найдены; перечистка %d существующих ref_*.wav (bandpass + denoise)...",
            len(ref_files),
        )
        await db.clear_voice_references(user_id)
        sample_rate = settings.reference_sample_rate

        for idx, ref in enumerate(ref_files):
            try:
                audio = load_audio_mono(ref, sample_rate)
                audio = apply_bandpass(audio, sample_rate)
                rms_pre = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
                if rms_pre > 1e-4:
                    denoised = apply_denoise(audio, sample_rate, prop_decrease=0.6)
                    rms_post = float(np.sqrt(np.mean(np.square(denoised)))) if denoised.size else 0.0
                    if rms_post >= rms_pre * 0.4:
                        audio = denoised
                audio = remove_long_pauses(audio, sample_rate)
                audio = normalize_loudness(audio)

                out_path = refs_dir / f"ref_{idx:03d}.wav"
                sf.write(out_path, audio, sample_rate, subtype="PCM_16")
                if out_path != ref and ref.exists():
                    ref.unlink(missing_ok=True)

                duration = len(audio) / sample_rate
                metrics = {
                    "duration_sec": round(duration, 2),
                    "speech_ratio": round(speech_ratio(audio, sample_rate), 3),
                    "clipping_ratio": round(detect_clipping(audio), 4),
                    "rms": round(float(np.sqrt(np.mean(np.square(audio)))), 5) if audio.size else 0.0,
                    "denoise_applied": True,
                }
                quality = evaluate_reference(metrics)
                if quality.accepted:
                    await db.add_voice_reference(
                        user_id, out_path.name, metrics["duration_sec"], quality.to_dict()
                    )
                    logger.info(
                        "  %s (%.1f сек, score=%.0f)",
                        out_path.name,
                        metrics["duration_sec"],
                        quality.score,
                    )
                else:
                    out_path.unlink(missing_ok=True)
                    logger.warning("  %s отклонён: score=%.0f", out_path.name, quality.score)
            except Exception as exc:
                logger.error("  %s ошибка: %s", ref.name, exc)
    else:
        if not ref_files and not raw_files:
            logger.error("Нет ни raw, ни ref_*.wav — нечего пересобирать")
            return
        logger.info("Пропуск переобработки (--no-reprocess)")

    logger.info("Пересборка профиля...")
    result = await profile_service.build_profile(user_id)
    logger.info("Профиль пересобран: %s", result)

    cond_cache = profile_service.conditioning_cache_path(user_id)
    if cond_cache.exists():
        cond_cache.unlink()
        logger.info("Кеш conditioning сброшен: %s", cond_cache)

    alo_cache = cache_dir / "alo.ogg"
    if alo_cache.exists():
        alo_cache.unlink()
        logger.info("Кеш alo.ogg сброшен")

    final_refs = sorted(refs_dir.glob("ref_*.wav"))
    total_dur = 0.0
    for ref in final_refs:
        data, sr = sf.read(ref)
        total_dur += len(data) / sr

    logger.info("Итог: %d референсов, %.1f сек суммарно", len(final_refs), total_dur)
    logger.info("Готово! При следующем запросе conditioning пересчитается автоматически.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Пересборка голосового профиля")
    parser.add_argument("--user-id", type=int, required=True, help="Telegram user ID")
    parser.add_argument(
        "--no-reprocess",
        action="store_true",
        help="Не переобрабатывать аудио, только пересобрать профиль",
    )
    args = parser.parse_args()
    asyncio.run(rebuild(args.user_id, reprocess_raw=not args.no_reprocess))


if __name__ == "__main__":
    main()
