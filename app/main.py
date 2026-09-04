"""Точка входа приложения."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# До импорта torch: на ноутбуках с iGPU выбираем дискретную NVIDIA
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import asyncio
import faulthandler
import logging
import sys

faulthandler.enable()

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.fsm.storage.memory import MemoryStorage

from app.bot.handlers import router
from app.bot.handlers_call import router as call_router
from app.bot.middleware import DependencyMiddleware
from app.config import Settings, get_settings
from app.database import Database
from app.services.call_feel import CallFeelService
from app.services.call_pipeline import CallOrchestrator
from app.services.call_session import CallSessionManager
from app.services.gigachat import GigaChatService
from app.services.synthesis import SynthesisService
from app.services.telegram_call import TelegramCallService
from app.services.transcription import TranscriptionService
from app.services.video_dub import VideoDubService
from app.services.voice_profile import VoiceProfileService
from app.text.accent import AccentService
from app.tts.cache import AudioCache
from app.tts.engine import TTSEngine
from app.tts.silero_engine import SileroEngine
from app.web.server import start_web_server


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def _build_silero(settings: Settings, *, required: bool) -> SileroEngine | None:
    if settings.silero_model_path.exists():
        return SileroEngine(
            model_path=settings.silero_model_path,
            speaker=settings.silero_speaker,
            sample_rate=settings.output_sample_rate,
            device=settings.silero_device,
            cpu_threads=settings.silero_cpu_threads,
        )
    if required:
        raise SystemExit(
            f"SILERO_MODEL_PATH не найден: {settings.silero_model_path}\n"
            "Скачайте: python scripts/download_silero.py"
        )
    return None


def build_engines(settings: Settings) -> tuple[TTSEngine, TTSEngine | None]:
    """Primary + optional fallback. XTTS/VoxCPM2 вместе не грузим (VRAM)."""
    if settings.tts_engine == "silero":
        silero = _build_silero(settings, required=True)
        assert silero is not None
        return silero, None

    silero = _build_silero(settings, required=False)

    if settings.tts_engine == "seamless_m4t":
        from app.tts.seamless_engine import SeamlessM4TEngine

        primary = SeamlessM4TEngine(
            model_id=settings.seamless_model_id,
            device=settings.device,
            speaker_id=settings.seamless_speaker_id,
        )
        return primary, silero

    if settings.tts_engine == "voxcpm2":
        from app.tts.voxcpm_engine import VoxCPMEngine

        primary = VoxCPMEngine(
            model_id=settings.voxcpm_model_id,
            device=settings.device,
            cfg_value=settings.voxcpm_cfg_value,
            inference_timesteps=settings.voxcpm_inference_timesteps,
            timeout_sec=settings.voxcpm_timeout_sec,
        )
        return primary, silero

    if settings.tts_engine == "openrouter_fish":
        from app.tts.openrouter_fish_engine import OpenRouterFishEngine

        primary = OpenRouterFishEngine(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_tts_model,
            sample_rate=int(settings.output_sample_rate or 24000),
            timeout_sec=settings.openrouter_tts_timeout_sec,
            response_format=settings.openrouter_tts_format,
            min_interval_sec=float(
                getattr(settings, "openrouter_tts_min_interval_sec", 3.1)
            ),
        )
        return primary, silero

    if settings.tts_engine == "mockingbird":
        from app.tts.mockingbird_engine import MockingBirdEngine

        primary = MockingBirdEngine(
            root=settings.mockingbird_root,
            python_exe=settings.mockingbird_python,
            encoder=settings.mockingbird_encoder,
            synthesizer=settings.mockingbird_synthesizer,
            vocoder=settings.mockingbird_vocoder,
            timeout_sec=settings.mockingbird_timeout_sec,
        )
        return primary, silero

    from app.tts.xtts_engine import XTTSEngine

    xtts = XTTSEngine(
        model_name=settings.tts_model_name,
        device=settings.device,
        use_fp16=settings.use_fp16,
        gpt_cond_len=settings.xtts_gpt_cond_len,
        gpt_cond_chunk_len=settings.xtts_gpt_cond_chunk_len,
        max_ref_len=settings.xtts_max_ref_len,
        sound_norm_refs=settings.xtts_sound_norm_refs,
        finetune_checkpoint=settings.xtts_finetune_checkpoint,
        finetune_config=settings.xtts_finetune_config,
    )
    return xtts, silero


def _telegram_http_session(settings: Settings) -> AiohttpSession:
    """Bot API: TELEGRAM_BOT_PROXY (туннель/SOCKS), иначе напрямую.

    TELEGRAM_PROXY оставлен для Telethon: с GPU он часто не достучится до SOCKS.
    """
    proxy = (settings.telegram_bot_proxy or "").strip() or None
    try:
        session = AiohttpSession(proxy=proxy, limit=20) if proxy else AiohttpSession(limit=20)
    except RuntimeError:
        logging.getLogger(__name__).warning(
            "TELEGRAM_BOT_PROXY задан, но aiohttp-socks не установлен — polling напрямую"
        )
        session = AiohttpSession(limit=20)
    session._connector_init["enable_cleanup_closed"] = True
    session._connector_init["ttl_dns_cache"] = 60
    session._should_reset_connector = True
    if proxy:
        logging.getLogger(__name__).info("Bot API HTTP session via TELEGRAM_BOT_PROXY")
    return session


async def main() -> None:
    import atexit
    import tempfile
    from pathlib import Path

    # Один poller на машину — иначе Telegram Conflict и «бот молчит».
    lock_path = Path(tempfile.gettempdir()) / "app_bot.lock"
    lock_fh = None
    try:
        lock_fh = open(lock_path, "a+b", buffering=0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                logging.basicConfig(level=logging.INFO)
                logging.error(
                    "Бот уже запущен (lock %s). Второй экземпляр остановлен.",
                    lock_path,
                )
                return
        else:
            import fcntl

            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                logging.basicConfig(level=logging.INFO)
                logging.error("Бот уже запущен. Второй экземпляр остановлен.")
                return

        def _unlock() -> None:
            try:
                if lock_fh is not None:
                    if sys.platform == "win32":
                        import msvcrt

                        lock_fh.seek(0)
                        msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                    lock_fh.close()
            except Exception:
                pass

        atexit.register(_unlock)
    except Exception:
        logging.getLogger(__name__).warning("Не удалось взять single-instance lock", exc_info=True)

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)


    if not settings.telegram_bot_token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN не задан в .env. "
            "Для сбора голосов из аккаунта без бота используйте:\n"
            "  python scripts/collect_account_voices.py --consent --build-profile"
        )

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "sessions").mkdir(parents=True, exist_ok=True)
    settings.users_dir.mkdir(parents=True, exist_ok=True)
    settings.audio_cache_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_path)
    await db.init()

    primary, fallback = build_engines(settings)
    if settings.device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(
                    "GPU: %s, VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                )
            else:
                logger.warning("DEVICE=cuda, но CUDA недоступна.")
        except Exception:
            pass

    profile_service = VoiceProfileService(settings, db)
    gigachat_service = GigaChatService(settings)
    accent_service = AccentService(settings)
    transcription_service = TranscriptionService(settings)
    if not gigachat_service.configured:
        logger.warning(
            "GIGACHAT_CREDENTIALS не задан: голосовые ответы на вопросы недоступны"
        )
    audio_cache = AudioCache(settings.audio_cache_dir, enabled=settings.enable_audio_cache)
    synthesis_service = SynthesisService(
        settings,
        db,
        primary,
        profile_service,
        fallback=fallback,
        audio_cache=audio_cache,
    )
    call_feel_service = CallFeelService(
        settings,
        synthesis_service,
        accent_service,
    )
    video_dub_service = VideoDubService(
        settings,
        transcription_service,
        gigachat_service,
        synthesis_service,
        accent_service,
    )
    from app.services.large_media import LargeMediaService

    large_media_service = LargeMediaService(settings)
    call_sessions = CallSessionManager(
        stop_phrases=settings.call_stop_phrase_list,
        topic_shift_phrases=settings.call_topic_shift_phrase_list,
        barge_in_enabled=settings.call_barge_in_enabled,
        vad_silence_ms=settings.call_vad_silence_ms,
    )
    telegram_call_service = TelegramCallService(settings)
    web_runner = None
    if settings.web_enabled:
        web_runner = await start_web_server(
            settings=settings,
            db=db,
            video_dub=video_dub_service,
        )

    call_orchestrator = CallOrchestrator(
        settings,
        call_sessions,
        telegram_call_service,
        transcription_service,
        gigachat_service,
        accent_service,
        synthesis_service,
    )
    if settings.call_feel_enabled:
        try:
            ring = call_feel_service.ensure_ringback()
            logger.info("Call-feel ringback готов: %s", ring)
        except Exception as exc:
            logger.warning("Не удалось подготовить ringback: %s", exc)
        warmup_uid = int(settings.call_feel_warmup_user_id or 0)
        if warmup_uid > 0:
            user = await db.get_user(warmup_uid)
            if user and user.has_voice_profile:
                try:
                    alo = await call_feel_service.ensure_alo_ogg(warmup_uid)
                    logger.info("Call-feel alo warmup: %s", alo)
                except Exception as exc:
                    logger.warning("Call-feel alo warmup не удался: %s", exc)
            else:
                logger.info(
                    "CALL_FEEL_WARMUP_USER_ID=%s без профиля — alo на первом запросе",
                    warmup_uid,
                )

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=_telegram_http_session(settings),
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.update.middleware(
        DependencyMiddleware(
            settings=settings,
            db=db,
            profile_service=profile_service,
            synthesis_service=synthesis_service,
            gigachat_service=gigachat_service,
            accent_service=accent_service,
            transcription_service=transcription_service,
            call_feel_service=call_feel_service,
            video_dub_service=video_dub_service,
            large_media_service=large_media_service,
            call_orchestrator=call_orchestrator,
            call_sessions=call_sessions,
        )
    )
    dp.include_router(call_router)
    dp.include_router(router)

    logger.info(
        "Бот запущен | TTS_ENGINE=%s primary=%s fallback=%s call_feel=%s live_call=%s web=%s",
        settings.tts_engine,
        primary.name,
        fallback.name if fallback else None,
        settings.call_feel_enabled,
        settings.call_enabled,
        f"{settings.web_host}:{settings.web_port}" if settings.web_enabled else "off",
    )

    async def _preload_tts() -> None:
        try:
            logger.info("Предзагрузка TTS (%s) в фоне...", primary.name)
            await asyncio.to_thread(primary.load)
            try:
                await asyncio.to_thread(primary.warmup)
            except Exception as exc:
                logger.warning("Warmup primary не удался: %s", exc)
            if fallback is not None:
                logger.info("Предзагрузка Silero fallback...")
                await asyncio.to_thread(fallback.load)
                try:
                    await asyncio.to_thread(fallback.warmup)
                    logger.info("Silero warmup OK (speaker=%s)", fallback.speaker)
                except Exception as exc:
                    logger.warning("Silero warmup не удался: %s", exc)
        except Exception:
            logger.exception("Предзагрузка TTS не удалась")

    if settings.tts_engine != "voxcpm2" and settings.tts_engine != "seamless_m4t":
        asyncio.create_task(_preload_tts())
    elif settings.tts_engine == "seamless_m4t":
        logger.info("SeamlessM4T загрузится при первом синтезе (экономия VRAM на старте)")
    else:
        logger.info("VoxCPM2 загрузится при первом синтезе (sentencepiece/transformers)")
    try:
        while True:
            try:
                me = await bot.get_me()
                logger.info("Telegram getMe OK @%s id=%s", me.username, me.id)
                break
            except TelegramNetworkError as exc:
                logger.error(
                    "api.telegram.org недоступен, повтор через 20с: %s", exc
                )
                await asyncio.sleep(20)
        await dp.start_polling(bot)
    finally:
        if web_runner is not None:
            await web_runner.cleanup()
        closer = getattr(primary, "close", None)
        if callable(closer):
            closer()
        await telegram_call_service.close()
        await gigachat_service.close()
        await bot.session.close()


if __name__ == "__main__":
    # Предпочтительно: python run_bot.py (см. scripts/detach_main.py).
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        multiprocessing.set_executable(sys.executable)
    except Exception:
        pass
    asyncio.run(main())
