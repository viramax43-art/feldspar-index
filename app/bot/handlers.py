"""Telegram bot handlers."""

from __future__ import annotations

import asyncio
import html
import logging
import shutil
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, Filter
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, FSInputFile, Message

from app.audio.preprocess import preprocess_telegram_voice
from app.audio.quality import evaluate_profile, evaluate_reference
from app.bot.keyboards import (
    CONSENT_TEXT,
    consent_keyboard,
    dub_language_keyboard,
    dub_loudness_keyboard,
    language_keyboard,
    settings_keyboard,
    voice_pick_keyboard,
)
from app.bot.pending_store import (
    PendingQuestion as _PendingQuestion,
    clear_pending,
    find_recoverable_video,
    load_pending,
    load_pending_file,
    load_recoverable_job,
    save_pending,
)
from app.bot.states import SpeakMode, VoiceCollection
from app.config import Settings, get_settings
from app.database import Database
from app.services.account_collector import AccountVoiceCollector
from app.services.call_feel import CallFeelService
from app.services.gigachat import GigaChatError, GigaChatService
from app.services.synthesis import ConsentRequiredError, ProfileRequiredError, SynthesisService
from app.services.large_media import LargeMediaError, LargeMediaService, extract_url
from app.services.transcription import TranscriptionError, TranscriptionService
from app.services.video_dub import (
    VideoDubService,
    format_cue_sheet,
    format_translate_pack,
    merge_translations,
    missing_translation_indices,
    parse_user_translation,
    split_plain_chunks,
)
from app.services import voice_pick
from app.services.voice_profile import VoiceProfileService
from app.text.accent import AccentService
from app.text.language import detect_transcript_language, resolve_language
from app.text.reply_lang import REPLY_LANGUAGES, normalize_reply_lang

logger = logging.getLogger(__name__)
router = Router()
_ANALYZE_HTML = "🎬 <b>Разбираю речь и тайминги…</b>"


class _StatusPulse:
    """Telegram не обновлялся, пока Whisper на CPU работал минутами."""

    def __init__(
        self,
        status: Message,
        title_html: str,
        *,
        hint: str = "Whisper на CPU — это не зависание",
        extra: Any = None,
    ) -> None:
        self.status = status
        self.title_html = title_html
        self.hint = hint
        self.extra = extra
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._started = time.monotonic()

    async def __aenter__(self) -> "_StatusPulse":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=8)
                return
            except asyncio.TimeoutError:
                elapsed = int(time.monotonic() - self._started)
                extra = ""
                if self.extra is not None:
                    try:
                        extra = str(self.extra() or "")
                    except Exception:
                        extra = ""
                extra_line = f"\n{extra}" if extra else ""
                hint = f" — {self.hint}" if self.hint else ""
                try:
                    await self.status.edit_text(
                        f"{self.title_html}\n⏳ {elapsed}с{hint}{extra_line}"
                    )
                except Exception:
                    pass


def _telethon_progress_line(progress: dict[str, int]) -> str:
    n = float(progress.get("n") or 0)
    total = float(progress.get("total") or 0)
    if total:
        pct = min(100.0, 100.0 * n / total)
        return (
            f"💾 {n / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} МБ ({pct:.0f}%)"
        )
    return f"💾 {n / (1024 * 1024):.1f} МБ"


def _humanize_handler_error(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()
    ename = type(exc).__name__.lower()
    if (
        "no space left" in low
        or "errno 28" in low
        or "not enough space" in low
        or getattr(exc, "errno", None) == 28
        or ("system error" in low and "sndfile" in ename)
    ):
        return (
            "На диске C: закончилось место. Освободите 3–5 ГБ "
            "(корзина, старые файлы в output и data/tmp) и пришлите видео снова."
        )
    if (
        "pytorch_model.bin" in low
        or "models--openbmb--voxcpm" in low
        or low.strip("'\"") == "state_dict"
    ):
        return (
            "Кэш VoxCPM2 на сервере не совпал с форматом весов. "
            "Попробуйте ещё раз — бот подхватит model.safetensors."
        )
    if "central directory" in low or "pytorchstreamreader" in low:
        return (
            "Файл модели TTS скачался не до конца и его пришлось перекачать. "
            "Нажмите язык дубляжа ещё раз."
        )
    return msg


async def _report_status_error(
    status: Message | None,
    destination: Message,
    text: str,
) -> None:
    body = html.escape(text)
    if status is not None:
        try:
            await status.edit_text(body)
            return
        except TelegramBadRequest:
            pass
        except Exception:
            logger.debug("Не удалось обновить статус ошибки", exc_info=True)
    try:
        await destination.answer(body)
    except Exception:
        logger.warning("Не удалось отправить ошибку пользователю: %s", text)


_pending_questions: dict[int, _PendingQuestion] = {}
_user_locks: dict[int, asyncio.Lock] = {}


def _data_dir() -> Path:
    return get_settings().data_dir


def _user_lock(user_id: int) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


def _get_pending(user_id: int) -> _PendingQuestion | None:
    pending = _pending_questions.get(user_id)
    if pending is not None:
        return pending
    loaded = load_recoverable_job(_data_dir(), user_id)
    if loaded is not None:
        _pending_questions[user_id] = loaded
    return loaded


def _put_pending(user_id: int, pending: _PendingQuestion) -> None:
    _pending_questions[user_id] = pending
    try:
        save_pending(_data_dir(), user_id, pending)
    except OSError:
        logger.warning("Не удалось сохранить pending %s", user_id, exc_info=True)


def _drop_pending_video(pending: _PendingQuestion | None) -> None:
    if pending is None or pending.video_path is None:
        return
    pending.video_path.unlink(missing_ok=True)
    workdir = pending.video_path.parent
    if workdir.name.startswith("vid_") and workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def _clear_user_outputs(user_id: int) -> None:
    """Wipe users/<id>/outputs after a finished dub (result is already in ./output)."""
    try:
        from app.audio import safe_user_path

        out = safe_user_path(get_settings().users_dir, user_id, "outputs")
        if out.is_dir():
            shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to clear user outputs for %s", user_id)


def _reset_after_dub(
    user_id: int,
    pending: _PendingQuestion | None,
    *,
    video_dub_service: VideoDubService,
    gigachat_service: GigaChatService | None = None,
) -> None:
    """Full session reset after the dubbed video is delivered — pending, tmp,
    user outputs, GigaChat memory, TTS RAM/disk cache."""
    import gc

    _drop_pending_video(pending)
    _pending_questions.pop(user_id, None)
    try:
        clear_pending(_data_dir(), user_id)
    except OSError:
        logger.warning("clear_pending failed for %s", user_id, exc_info=True)
    _clear_user_outputs(user_id)
    if gigachat_service is not None:
        try:
            gigachat_service.reset(user_id)
        except Exception:
            logger.exception("GigaChat reset after dub failed for %s", user_id)
    synth = getattr(video_dub_service, "synthesis", None)
    if synth is not None and hasattr(synth, "reset_after_dub"):
        try:
            synth.reset_after_dub(user_id)
        except Exception:
            logger.exception("TTS reset after dub failed for %s", user_id)
    gc.collect()
    logger.info("Dub session reset for user %s", user_id)


async def _ack_callback(
    callback: CallbackQuery, text: str = "", *, alert: bool = False
) -> None:
    try:
        await callback.answer(text, show_alert=alert)
    except TelegramBadRequest:
        logger.debug("callback.answer skipped")
    except Exception:
        logger.debug("callback.answer failed", exc_info=True)


async def _ensure_video_pending(
    *,
    user_id: int,
    callback: CallbackQuery,
    video_dub_service: VideoDubService,
) -> _PendingQuestion | None:
    pending = _get_pending(user_id)
    if (
        pending is not None
        and pending.kind == "video"
        and pending.segments
        and pending.video_path is not None
        and pending.video_path.exists()
    ):
        return pending
    if (
        pending is not None
        and pending.kind == "video"
        and pending.await_loudness
        and pending.video_path is not None
        and pending.video_path.exists()
    ):
        return pending
    src = find_recoverable_video(_data_dir(), user_id)
    if src is not None:
        sidecar = load_pending_file(src.parent / "job.json")
        if sidecar is not None:
            _put_pending(user_id, sidecar)
            return sidecar
    if src is None or callback.message is None:
        return None
    status = await callback.message.answer(
        "Бот перезапускался — заново снимаю тайминги с уже скачанного ролика."
    )
    try:
        quiet = bool(getattr(pending, "quiet_audio", False)) if pending else False
        async with _StatusPulse(status, "🎬 <b>Снова разбираю речь…</b>"):
            segments, duration_sec = await video_dub_service.analyze(
                src, quiet_audio=quiet
            )
        preview = " ".join(seg.text for seg in segments)
        pending = _PendingQuestion(
            kind="video",
            question=preview,
            video_path=src,
            segments=segments,
            duration_sec=duration_sec,
            await_translation=True,
            pasted=[""] * len(segments),
            quiet_audio=quiet,
        )
        _put_pending(user_id, pending)
        try:
            await status.delete()
        except Exception:
            pass
        return pending
    except Exception as exc:
        logger.exception("Recover analyze failed")
        await status.edit_text(
            f"Не удалось восстановить партитуру: {html.escape(_humanize_handler_error(exc))}\n"
            "Пришлите видео ещё раз."
        )
        return None


async def _prompt_video_loudness(
    message: Message,
    *,
    user_id: int,
    video_path: Path,
    status: Message | None = None,
) -> None:
    """After download: ask normal vs quiet/ASMR before STT."""
    # Новое видео — прежняя сессия выбора голоса больше не нужна.
    voice_pick.clear_session(_data_dir(), user_id)
    old = _get_pending(user_id)
    if (
        old is not None
        and old.video_path is not None
        and old.video_path != video_path
    ):
        _drop_pending_video(old)
        clear_pending(_data_dir(), user_id)
    pending = _PendingQuestion(
        kind="video",
        video_path=video_path,
        await_loudness=True,
    )
    _put_pending(user_id, pending)
    text = (
        "Видео скачано. Какая громкость речи?\n\n"
        "• <b>Обычная</b> — стандартное распознавание\n"
        "• <b>Тихая / ASMR</b> — усиливаю звук, чтобы разобрать шёпот"
    )
    markup = dub_loudness_keyboard()
    if status is not None:
        try:
            await status.edit_text(text, reply_markup=markup)
            return
        except Exception:
            logger.debug("loudness prompt edit failed", exc_info=True)
    await message.answer(text, reply_markup=markup)


async def _run_analyze_with_loudness(
    *,
    reply: Message,
    user_id: int,
    video_path: Path,
    quiet_audio: bool,
    db: Database,
    video_dub_service: VideoDubService,
    gigachat_service: GigaChatService | None,
    status: Message | None = None,
) -> None:
    analyze_html = (
        "🤫 <b>Усиливаю тихий звук и разбираю речь…</b>"
        if quiet_audio
        else _ANALYZE_HTML
    )
    if status is None:
        status = await reply.answer(analyze_html)
    else:
        try:
            await status.edit_text(analyze_html)
        except Exception:
            status = await reply.answer(analyze_html)
    try:
        async with _StatusPulse(status, analyze_html):
            segments, duration_sec = await video_dub_service.analyze(
                video_path, quiet_audio=quiet_audio
            )
        preview = " ".join(s.text for s in segments)
        try:
            await status.delete()
        except Exception:
            pass
        await _prompt_reply_language(
            reply,
            db,
            user_id,
            _PendingQuestion(
                kind="video",
                question=preview,
                video_path=video_path,
                segments=segments,
                duration_sec=duration_sec,
                quiet_audio=quiet_audio,
            ),
            gigachat_service=gigachat_service,
        )
    except (TranscriptionError, ValueError) as exc:
        await status.edit_text(f"Ошибка: {html.escape(str(exc))}")
    except Exception as exc:
        logger.exception("Ошибка разбора видео")
        await status.edit_text(
            f"Не удалось разобрать видео: {html.escape(_humanize_handler_error(exc))}"
        )


async def _prompt_reply_language(
    message: Message,
    db: Database,
    user_id: int,
    pending: _PendingQuestion,
    gigachat_service: GigaChatService | None = None,
) -> None:
    old = _pending_questions.get(user_id)
    if old is None:
        old = load_pending(_data_dir(), user_id)
    if (
        old is not None
        and old is not pending
        and old.video_path != pending.video_path
    ):
        _drop_pending_video(old)
        clear_pending(_data_dir(), user_id)
    # Fresh dub job: wipe pasted lines and chat memory so translations don't mix.
    if pending.kind == "video" and pending.segments:
        pending.pasted = [""] * len(pending.segments)
        pending.await_translation = True
        pending.await_loudness = False
    if gigachat_service is not None:
        try:
            gigachat_service.reset(user_id)
        except Exception:
            logger.exception("GigaChat reset failed for user %s", user_id)
    _put_pending(user_id, pending)
    user = await db.get_user(user_id)
    selected = normalize_reply_lang((user.settings or {}).get("reply_language"))
    preview = pending.question.strip()
    if len(preview) > 180:
        preview = preview[:177] + "…"
    if pending.kind == "video" and pending.segments:
        for chunk in format_cue_sheet(
            pending.segments,
            title="Партитура исходника (только просмотр)",
            media_duration=pending.duration_sec,
        ):
            await message.answer(chunk)
        pending.await_translation = True
        pending.pasted = [""] * len(pending.segments)
        _put_pending(user_id, pending)
        pack = format_translate_pack(pending.segments)
        src = detect_transcript_language(
            " ".join(s.text for s in pending.segments[:80]), default="en"
        )
        src_label = REPLY_LANGUAGES.get(src, src)
        await message.answer(
            f"Язык исходника: <b>{html.escape(src_label)}</b>.\n"
            "⬇️ Ниже — пакет <b>для DeepL</b> (не партитуру!).\n"
            "Скопируйте целиком → DeepL → пришлите перевод сюда "
            "(теги <code>&lt;c i=\"…\"&gt;</code> должны остаться).\n"
            "Тайминги/тон бот держит сам — в переводчик их не надо.\n"
            "Кнопка языка = автоперевод GigaChat и озвучка "
            "(<b>🇷🇺 Русский</b>, если ролик не на русском)."
        )
        for chunk in split_plain_chunks(pack):
            await message.answer(chunk, parse_mode=None)
        await message.answer_document(
            BufferedInputFile(pack.encode("utf-8"), filename="deepl_pack.xml"),
            caption=(
                "Тот же пакет файлом — удобно открыть и скопировать в DeepL целиком. "
                "Партитуру с таймкодами в переводчик НЕ вставляйте."
            ),
        )
        await message.answer(
            "Дальше: свой перевод из DeepL — можно <b>несколькими сообщениями</b> "
            "(с тегами <code>&lt;c i&gt;</code> или кусками подряд) — "
            "или кнопка языка для автоперевода.",
            reply_markup=dub_language_keyboard(selected),
        )
        return
    await message.answer(
        f"На каком языке ответить?\n\n<i>{html.escape(preview)}</i>",
        reply_markup=language_keyboard(selected),
    )


async def _saved_reply_lang(db: Database, user_id: int) -> str:
    user = await db.get_user(user_id)
    return normalize_reply_lang((user.settings or {}).get("reply_language"))


class WaitingDubPaste(Filter):
    async def __call__(self, message: Message) -> bool:
        user = message.from_user
        if user is None:
            return False
        pending = _get_pending(user.id)
        if not (
            pending
            and pending.kind == "video"
            and pending.await_translation
            and pending.segments
            and pending.video_path is not None
        ):
            return False
        if message.document:
            name = (message.document.file_name or "").lower()
            mime = (message.document.mime_type or "").lower()
            if mime.startswith("video/"):
                return False
            return name.endswith((".txt", ".srt")) or mime.startswith("text/")
        text = (message.text or "").strip()
        return bool(text) and not text.startswith("/")


async def _read_paste_text(message: Message, bot: Bot) -> str | None:
    if (message.text or "").strip():
        return message.text or ""
    doc = message.document
    if doc is None:
        return None
    if doc.file_size and doc.file_size > 2_000_000:
        return None
    buf = await bot.download(doc)
    if buf is None:
        return None
    data = buf.read()
    return data.decode("utf-8", errors="replace")


async def _deliver_video_dub(
    *,
    reply: Message,
    user_id: int,
    pending: _PendingQuestion,
    translated: list[str],
    lang: str,
    bot: Bot,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService | None = None,
) -> None:
    label = REPLY_LANGUAGES.get(lang, lang)
    for chunk in format_cue_sheet(
        pending.segments,
        translations=translated,
        title=f"Перевод → {label}",
        media_duration=pending.duration_sec,
    ):
        await reply.answer(chunk)
    status = await reply.answer("🎙 <b>Озвучиваю и накладываю на видео…</b>")

    async def _progress(done: int, total: int, preview: str) -> None:
        tip = html.escape(preview) if preview else "…"
        try:
            if done <= 0:
                await status.edit_text(f"🧬 <b>{tip}</b>")
            else:
                await status.edit_text(
                    f"🎙 <b>Озвучка {done}/{total}</b>\n<code>{tip}</code>"
                )
        except Exception:
            pass
        if done <= 1 or done == total or done % 3 == 0:
            await bot.send_chat_action(reply.chat.id, ChatAction.UPLOAD_VIDEO)

    assert pending.video_path is not None
    # Dump каждой реплики в wav — для ручного выбора эталонного голоса после.
    cue_dump_dir = pending.video_path.parent / "cues_dump"
    result = await video_dub_service.render(
        user_id,
        pending.video_path,
        pending.segments,
        translated,
        lang,
        pending.duration_sec,
        on_progress=_progress,
        cue_audio_dir=cue_dump_dir,
    )
    # Пакет для переозвучки выбранным голосом — ДО полного сброса сессии.
    voice_session = voice_pick.save_session(
        _data_dir(),
        user_id,
        source_video=pending.video_path,
        segments=result.segments,
        translated=result.translated,
        lang=lang,
        duration_sec=pending.duration_sec,
        cue_audio_dir=result.cue_audio_dir,
    )
    saved_paths = large_media_service.save_to_output(
        result.video_path,
        result.srt_path,
        label=label,
    )
    local_line = "Локально: <code>" + html.escape(str(saved_paths[0])) + "</code>"
    engine_name = getattr(getattr(video_dub_service, "synthesis", None), "primary", None)
    engine_name = getattr(engine_name, "name", "")
    if engine_name == "seamless_m4t":
        voice_line = "SeamlessM4T: речь→речь"
    elif settings.video_dub_mix_mode == "replace":
        voice_line = (
            f"Клон диктора: {result.clone_sec:.1f}с "
            f"({len(result.clone_refs)} клип.) · "
            f"фон 100% · голос диктора · целевой язык"
        )
    else:
        voice_line = (
            f"Клон из видео: {result.clone_sec:.1f}с "
            f"({len(result.clone_refs)} клип.) · RUAccent+интонация · "
            f"mix фон {settings.video_dub_bg_volume:.0%}"
        )
    caption = (
        f"Замена языка: <b>{html.escape(label)}</b>\n"
        f"{voice_line}"
    )
    out_size = result.video_path.stat().st_size if result.video_path.exists() else 0
    bot_upload_limit = int(settings.video_dub_bot_upload_mb * 1024 * 1024)
    if out_size > bot_upload_limit and large_media_service.telethon_ready():
        await status.edit_text(
            "💾 Сохранил в <code>output</code>\n"
            "📤 Файл крупный — отправляю в <b>Избранное</b> через User API…"
        )
        telegram_ok = True
        try:
            await large_media_service.send_file_to_user(
                user_id=user_id,
                path=result.video_path,
                caption=f"Дубляж ({label})",
            )
            if result.srt_path and result.srt_path.exists():
                await large_media_service.send_file_to_user(
                    user_id=user_id,
                    path=result.srt_path,
                    caption=f"Субтитры ({label})",
                )
        except Exception:
            telegram_ok = False
            logger.exception("Telethon upload of large dub failed")
        extra = (
            "Плюс копия в <b>Избранном</b> Telegram."
            if telegram_ok
            else "В Telegram не ушло — берите файл с диска."
        )
        await reply.answer(f"{caption}\n\n{local_line}\n{extra}")
    else:
        await reply.answer_video(
            FSInputFile(str(result.video_path)),
            caption=f"{caption}\n{local_line}",
        )
        if result.srt_path and result.srt_path.exists():
            await reply.answer_document(
                FSInputFile(str(result.srt_path)),
                caption=f"Субтитры ({html.escape(label)})",
            )
    try:
        await status.delete()
    except Exception:
        pass
    # Finished: wipe pending/tmp/outputs/TTS cache so the next video starts clean.
    _reset_after_dub(
        user_id,
        pending,
        video_dub_service=video_dub_service,
        gigachat_service=gigachat_service,
    )
    pickable = voice_pick.pickable_cues(voice_session) if voice_session else []
    if len(pickable) >= 2:
        await reply.answer(
            "✅ Готово. Сессия сброшена — можно присылать следующее видео.\n\n"
            "🎭 <b>Голос плавает между репликами?</b> Выберите реплику, чей голос "
            "нравится, — переозвучу всё видео именно этим голосом.\n"
            "⚡ — экспрессивные моменты: они останутся из текущей озвучки.",
            reply_markup=voice_pick_keyboard(pickable, page=0),
        )
    else:
        await reply.answer(
            "✅ Готово. Сессия сброшена — можно прислать следующее видео."
        )


async def _accept_user_translation(
    message: Message,
    *,
    user_id: int,
    bot: Bot,
    db: Database,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService | None = None,
) -> None:
    pending = _get_pending(user_id)
    if pending is None or pending.kind != "video" or not pending.segments:
        await message.answer("Нет активной партитуры. Пришлите видео заново.")
        return
    raw = await _read_paste_text(message, bot)
    if not raw or not raw.strip():
        await message.answer("Пустой текст. Пришлите перевод сообщением или .txt/.srt.")
        return
    if not pending.pasted or len(pending.pasted) != len(pending.segments):
        pending.pasted = [""] * len(pending.segments)
    parsed = parse_user_translation(
        raw,
        len(pending.segments),
        already_filled=pending.pasted,
    )
    if not parsed:
        missing = missing_translation_indices(pending.pasted, len(pending.segments))
        hint = ""
        if missing:
            sample = ", ".join(f"{n:02d}" for n in missing[:12])
            more = "…" if len(missing) > 12 else ""
            hint = f"\nЕщё пусто: <code>{sample}{more}</code>."
        await message.answer(
            f"Не распознал слоты. Нужно до <b>{len(pending.segments)}</b> реплик.\n"
            "Пришлите ответ DeepL с тегами <code>&lt;c i=\"…\"&gt;</code> "
            "(или старый формат <code>01. …</code> / куски подряд)."
            f"{hint}"
        )
        return
    pending.pasted = merge_translations(pending.pasted, parsed)[: len(pending.segments)]
    _put_pending(user_id, pending)
    filled = sum(1 for line in pending.pasted if line.strip())
    total = len(pending.segments)
    if filled < total:
        missing = missing_translation_indices(pending.pasted, total)
        sample = ", ".join(f"{n:02d}" for n in missing[:16])
        more = f" (+ещё {len(missing) - 16})" if len(missing) > 16 else ""
        await message.answer(
            f"Принял <b>{filled}/{total}</b> реплик. Можно прислать ещё "
            "сообщение(я) — с тегами недостающих <code>&lt;c i&gt;</code> "
            "или следующим куском.\n"
            f"Пусто: <code>{sample}{more}</code>."
        )
        return
    default_lang = await _saved_reply_lang(db, user_id)
    lang = resolve_language(" ".join(pending.pasted), default=default_lang)
    lang = normalize_reply_lang(lang)
    pending.await_translation = False
    _put_pending(user_id, pending)
    try:
        await _deliver_video_dub(
            reply=message,
            user_id=user_id,
            pending=pending,
            translated=list(pending.pasted),
            lang=lang,
            bot=bot,
            settings=settings,
            video_dub_service=video_dub_service,
            large_media_service=large_media_service,
            gigachat_service=gigachat_service,
        )
    except Exception as exc:
        pending.await_translation = True
        _put_pending(user_id, pending)
        logger.exception("Custom-translation dub failed")
        await message.answer(f"Не удалось собрать дубляж: {html.escape(str(exc))}")
        return


async def _run_chat_turn(
    message: Message,
    *,
    user_id: int,
    pending: _PendingQuestion,
    lang: str,
    bot: Bot,
    settings: Settings,
    gigachat_service: GigaChatService,
    call_feel_service: CallFeelService,
    play_cues: bool = True,
) -> None:
    _put_pending(user_id, pending)
    status = None
    if call_feel_service.enabled and play_cues:
        await call_feel_service.play_call_cues(message, user_id)
        status = await message.answer("📞 <b>На линии…</b>")
    else:
        status = await message.answer("🤔 Думаю над ответом…")
    await bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)
    if pending.use_stream:
        await call_feel_service.stream_and_speak(
            message,
            user_id,
            gigachat_service.stream_answer(
                user_id,
                pending.question,
                analysis_context=pending.analysis_context or None,
                language=lang,
            ),
            prepare_for_speech=GigaChatService.prepare_for_speech,
            max_text_length=settings.max_text_length,
            language=lang,
        )
    else:
        answer = await gigachat_service.answer(
            user_id, pending.question, language=lang
        )
        if status is not None:
            try:
                await status.edit_text("🎙 <b>Озвучиваю ответ…</b>")
            except Exception:
                pass
        await call_feel_service.speak_answer_parts(
            message, user_id, answer, language=lang
        )
    if status is not None:
        try:
            await status.delete()
        except Exception:
            pass
    await message.answer(
        f"Язык: <b>{html.escape(REPLY_LANGUAGES[lang])}</b> · повторить на другом:",
        reply_markup=language_keyboard(lang),
    )


def _welcome_text() -> str:
    return (
        "🎙 <b>Assistant</b> — голосовой ассистент с GigaChat.\n\n"
        "ℹ️ Бот создан на основе <b>бесплатно доступных моделей</b> "
        "(XTTS-v2, Silero, faster-whisper, RUAccent) и "
        "<b>собственных скриптов проекта</b>.\n"
        "Модели используются локально; их лицензии преимущественно разрешают "
        "некоммерческое использование. GigaChat работает по условиям и лимитам "
        "провайдера.\n\n"
        "<b>Возможности:</b>\n"
        "• задайте вопрос текстом или голосовым — получите голосовой ответ\n"
        "• распознавание голосовых сообщений через faster-whisper\n"
        "• сбор ваших голосовых из всех чатов аккаунта (/collectaccount)\n"
        "• XTTS-клонирование вашего голоса или быстрый Silero fallback\n"
        "• контекст диалога и команда /reset\n"
        "• настройка скорости и выразительности\n"
        "• <b>дубляж видео</b>: просто пришлите MP4 боту (и &gt;20 МБ — через User API)\n\n"
        "<b>Ограничения:</b>\n"
        "• собираются только <b>ваши</b> исходящие голосовые\n"
        "• качество зависит от чистоты записей и модели XTTS-v2\n"
        "• на CPU генерация очень медленная\n"
        "• запрещено клонировать чужой голос без согласия\n\n"
        "Начните с /consent. После этого просто задайте вопрос.\n"
        "Для ответа вашим голосом создайте профиль: /collectaccount или /addvoice."
    )


def _help_text() -> str:
    return (
        "<b>Команды:</b>\n"
        "/start — описание бота\n"
        "/consent — подтверждение прав на голос\n"
        "/collectaccount — собрать ваши голосовые из всех чатов аккаунта\n"
        "/addvoice — ручной сбор голосовых сообщений\n"
        "/finishvoice — завершить сбор и создать профиль\n"
        "/speak — режим озвучивания текста\n"
        "/call — живой Telegram-звонок (перебивания / смена темы)\n"
        "/hangup — завершить звонок\n"
        "/reset — очистить контекст разговора с GigaChat\n"
        "/settings — синтез и язык ответа\n"
        "/lang — сменить язык чата (без вопроса каждый раз)\n"
        "/dub — видео / ссылка / inbox\n"
        "/deletevoice — удалить все референсы\n"
        "/help — эта справка\n\n"
        "<b>Вопросы:</b>\n"
        "Текст или voice — сразу ответ на языке из /lang (кнопка «другой язык» под ответом).\n"
        "Видео — партитура + копируемый список реплик: переведите сами и пришлите текст боту, либо кнопка языка.\n\n"
        "<b>Автосбор из аккаунта:</b>\n"
        "Нужны TELEGRAM_API_ID и TELEGRAM_API_HASH в .env\n"
        "(https://my.telegram.org). Первый логин — через скрипт:\n"
        "<code>python scripts/collect_account_voices.py --consent</code>\n\n"
        "<b>Советы по записи:</b>\n"
        "• тихое помещение, без музыки\n"
        "• 5–15 секунд на сообщение\n"
        "• разная интонация: вопросы, числа, эмоции\n"
        "• общая чистая речь 30 с — 3 мин"
    )


async def _require_consent(db: Database, user_id: int) -> bool:
    return await db.has_consent(user_id)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(_welcome_text())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_help_text())


@router.message(Command("consent"))
async def cmd_consent(message: Message) -> None:
    await message.answer(
        f"Перед использованием подтвердите согласие:\n\n<i>{CONSENT_TEXT}</i>",
        reply_markup=consent_keyboard(),
    )


@router.callback_query(F.data == "consent:yes")
async def consent_yes(
    callback: CallbackQuery,
    db: Database,
) -> None:
    user_id = callback.from_user.id
    await db.set_consent(user_id, True)
    await callback.message.edit_text(
        "✅ Согласие сохранено.\n"
        "Дальше: /collectaccount (из всех чатов) или /addvoice (вручную)."
    )
    await callback.answer()


@router.callback_query(F.data == "consent:no")
async def consent_no(callback: CallbackQuery) -> None:
    await callback.message.edit_text(
        "Без подтверждения согласия загрузка голоса и синтез недоступны."
    )
    await callback.answer()


@router.message(Command("collectaccount"))
async def cmd_collectaccount(
    message: Message,
    db: Database,
    settings: Settings,
    profile_service: VoiceProfileService,
) -> None:
    """Сбор исходящих голосовых из всех чатов через Telethon User API."""
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        await message.answer(
            "Не настроен User API.\n\n"
            "1. Откройте https://my.telegram.org → API development tools\n"
            "2. Добавьте в .env: <code>TELEGRAM_API_ID</code> и <code>TELEGRAM_API_HASH</code>\n"
            "3. Один раз войдите в аккаунт:\n"
            "<code>python scripts/collect_account_voices.py --consent</code>\n"
            "4. Затем снова /collectaccount"
        )
        return

    session = (
        settings.data_dir / "sessions" / f"{settings.telegram_session_name}.session"
    )
    if not session.exists():
        await message.answer(
            "Сессия Telegram ещё не создана.\n"
            "Сначала выполните в терминале (потребуется код из Telegram):\n\n"
            "<code>python scripts/collect_account_voices.py --consent</code>\n\n"
            "После этого команда /collectaccount заработает без повторного логина."
        )
        return

    await message.answer(
        "⏳ Сканирую ваши исходящие голосовые по всем чатам...\n"
        "Это может занять несколько минут."
    )

    collector = AccountVoiceCollector(settings, db, profile_service)

    async def progress(text: str) -> None:
        # Не спамим каждым логом в чат — только ключевые
        if text.startswith("Авторизован") or text.startswith("Диалогов"):
            await message.answer(text)

    try:
        result = await collector.collect(profile_user_id=user_id, progress=progress)
        await message.answer(result.summary())
        if result.accepted > 0:
            await message.answer(
                "Когда наберётся достаточно речи — /finishvoice для создания профиля."
            )
    except PermissionError as exc:
        await message.answer(str(exc))
    except Exception as exc:
        logger.exception("Ошибка /collectaccount")
        await message.answer(
            f"Ошибка сбора: {exc}\n"
            "Проверьте сессию: python scripts/collect_account_voices.py --consent"
        )


@router.message(Command("addvoice"))
async def cmd_addvoice(
    message: Message,
    state: FSMContext,
    db: Database,
    settings: Settings,
) -> None:
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return

    count = await db.count_voice_references(user_id)
    if count >= settings.max_voice_messages:
        await message.answer(
            f"Уже загружено {count} сообщений (максимум {settings.max_voice_messages}). "
            "Используйте /finishvoice или /deletevoice."
        )
        return

    await state.set_state(VoiceCollection.collecting)
    await message.answer(
        "Отправьте от 3 до 10 голосовых сообщений.\n"
        "Говорите чётко, в тихом месте, с разной интонацией.\n"
        "Когда закончите — /finishvoice"
    )


@router.message(Command("finishvoice"))
async def cmd_finishvoice(
    message: Message,
    state: FSMContext,
    db: Database,
    settings: Settings,
    profile_service: VoiceProfileService,
    synthesis_service: SynthesisService,
) -> None:
    user_id = message.from_user.id
    await state.clear()

    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return

    refs = await db.list_voice_references(user_id)
    accepted = [r for r in refs if r.get("quality", {}).get("accepted")]
    profile_quality = evaluate_profile(
        refs,
        min_total_seconds=settings.min_reference_seconds,
        max_total_seconds=settings.max_reference_seconds,
        min_count=settings.min_voice_messages,
    )

    if not profile_quality.accepted:
        await message.answer(
            "❌ Профиль не создан.\n"
            + profile_quality.format_message(0)
            + "\n\nДобавьте записи через /addvoice."
        )
        return

    try:
        meta = await profile_service.build_profile(user_id)
        synthesis_service.invalidate_cache(user_id)
    except Exception as exc:
        logger.exception("Ошибка создания профиля")
        await message.answer(f"Ошибка создания профиля: {exc}")
        return

    await message.answer(
        "✅ Голосовой профиль создан!\n"
        f"Принято референсов: {meta['reference_count']}\n"
        f"Общая длительность: {meta['total_duration_sec']:.0f} с\n\n"
        "Теперь задавайте вопросы — ответы будут звучать вашим голосом.\n"
        "/speak по-прежнему озвучивает ваш текст без GigaChat."
    )


@router.message(Command("speak"))
async def cmd_speak(message: Message, state: FSMContext, db: Database) -> None:
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return
    user = await db.get_user(user_id)
    if not user.has_voice_profile:
        await message.answer("Сначала создайте профиль: /addvoice → /finishvoice")
        return

    await state.set_state(SpeakMode.waiting_text)
    await message.answer("Отправьте русский текст для озвучивания.")


@router.message(Command("settings"))
async def cmd_settings(message: Message, db: Database) -> None:
    user = await db.get_user(message.from_user.id)
    await message.answer("⚙️ Настройки синтеза:", reply_markup=settings_keyboard(user.settings))


@router.callback_query(F.data.startswith("settings:"))
async def settings_callbacks(callback: CallbackQuery, db: Database) -> None:
    user_id = callback.from_user.id
    user = await db.get_user(user_id)
    settings_data = user.settings or {}
    intonation_cycle = ["neutral", "calm", "expressive"]
    action = callback.data.split(":", 1)[1]

    if action == "lang":
        selected = normalize_reply_lang(settings_data.get("reply_language"))
        await callback.message.edit_text(
            "Язык ответов в чате:",
            reply_markup=language_keyboard(selected, prefix="setlang"),
        )
        await callback.answer()
        return
    if action == "intonation":
        current = settings_data.get("intonation", "neutral")
        idx = intonation_cycle.index(current) if current in intonation_cycle else 0
        settings_data["intonation"] = intonation_cycle[(idx + 1) % len(intonation_cycle)]
    elif action.startswith("speed:"):
        op = action.split(":")[1]
        speed = float(settings_data.get("speed", 1.0))
        if op == "+":
            settings_data["speed"] = min(1.3, round(speed + 0.05, 2))
        elif op == "-":
            settings_data["speed"] = max(0.7, round(speed - 0.05, 2))
    elif action.startswith("temp:"):
        op = action.split(":")[1]
        temp = float(settings_data.get("temperature", 0.75))
        if op == "+":
            settings_data["temperature"] = min(1.0, round(temp + 0.05, 2))
        elif op == "-":
            settings_data["temperature"] = max(0.3, round(temp - 0.05, 2))

    await db.update_settings(user_id, settings_data)
    await callback.message.edit_reply_markup(reply_markup=settings_keyboard(settings_data))
    await callback.answer("Настройки обновлены")


@router.callback_query(F.data.startswith("setlang:"))
async def on_set_chat_language(callback: CallbackQuery, db: Database) -> None:
    user_id = callback.from_user.id
    lang = normalize_reply_lang(callback.data.split(":", 1)[1])
    user = await db.get_user(user_id)
    settings_data = dict(user.settings or {})
    settings_data["reply_language"] = lang
    await db.update_settings(user_id, settings_data)
    await callback.message.edit_text(
        f"Язык чата: <b>{html.escape(REPLY_LANGUAGES[lang])}</b>",
        reply_markup=settings_keyboard(settings_data),
    )
    await callback.answer(REPLY_LANGUAGES[lang])


@router.message(Command("lang"))
async def cmd_lang(message: Message, db: Database) -> None:
    user = await db.get_user(message.from_user.id)
    selected = normalize_reply_lang((user.settings or {}).get("reply_language"))
    await message.answer(
        "Язык ответов в чате (видео по-прежнему спрашивает отдельно):",
        reply_markup=language_keyboard(selected, prefix="setlang"),
    )


@router.message(Command("dub"))
async def cmd_dub(
    message: Message,
    bot: Bot,
    db: Database,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService,
) -> None:
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return

    text = (message.text or "").strip()
    url = extract_url(text)
    parts = text.split(maxsplit=1)
    want_inbox = len(parts) > 1 and parts[1].strip().lower() in {
        "inbox",
        "папка",
        "folder",
    }

    if url or want_inbox:
        await _analyze_video_file(
            message,
            db,
            settings,
            video_dub_service,
            source="url" if url else "inbox",
            url=url,
            large_media_service=large_media_service,
            gigachat_service=gigachat_service,
        )
        return

    inbox = large_media_service.inbox_dir
    telethon_ok = large_media_service.telethon_ready()
    await message.answer(
        "🎬 <b>Дубляж без лимита 20 МБ</b>\n\n"
        "1) Просто пришлите MP4/видео боту — даже 100+ МБ "
        f"{'(скачает User API)' if telethon_ok else '(нужен TELEGRAM_API_ID/HASH)'}.\n"
        "2) Или ссылка: <code>/dub https://…/file.mp4</code>\n"
        f"3) Или положите файл в папку и: <code>/dub inbox</code>\n"
        f"<code>{html.escape(str(inbox))}</code>\n\n"
        f"Готовый MP4 всегда копируется в <code>{html.escape(str(large_media_service.output_dir))}</code>.\n"
        "Дальше: партитура → язык → замена речи, фон сохраняется → MP4 + SRT."
    )


@router.message(Command("dub_inbox"))
async def cmd_dub_inbox(
    message: Message,
    db: Database,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService,
) -> None:
    if not await _require_consent(db, message.from_user.id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return
    await _analyze_video_file(
        message,
        db,
        settings,
        video_dub_service,
        source="inbox",
        large_media_service=large_media_service,
        gigachat_service=gigachat_service,
    )


async def _analyze_video_file(
    message: Message,
    db: Database,
    settings: Settings,
    video_dub_service: VideoDubService,
    *,
    source: str,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService | None = None,
    url: str | None = None,
    incoming: Path | None = None,
    workdir: Path | None = None,
) -> None:
    user_id = message.from_user.id
    status = await message.answer("⬇️ <b>Готовлю видео…</b>")
    owned_workdir = workdir is None
    if workdir is None:
        workdir = settings.data_dir / "tmp" / f"vid_{user_id}_{uuid.uuid4().hex}"
        workdir.mkdir(parents=True, exist_ok=True)
    try:
        if incoming is None:
            if source == "url":
                assert url
                suffix = Path(urlparse_safe_suffix(url)).suffix or ".mp4"
                incoming = workdir / f"src{suffix}"
                await status.edit_text("⬇️ <b>Скачиваю по ссылке…</b>")
                await large_media_service.download_url(url, incoming)
            elif source == "inbox":
                incoming = workdir / "src.mp4"
                await status.edit_text("📂 <b>Беру файл из inbox…</b>")
                large_media_service.take_inbox_video(incoming)
            else:
                raise LargeMediaError(f"Неизвестный source={source}")
        await _prompt_video_loudness(
            message,
            user_id=user_id,
            video_path=incoming,
            status=status,
        )
    except (TranscriptionError, ValueError, LargeMediaError) as exc:
        if owned_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        await status.edit_text(f"Ошибка: {html.escape(str(exc))}")
    except Exception as exc:
        if owned_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        logger.exception("Ошибка разбора видео")
        await status.edit_text(f"Не удалось разобрать видео: {html.escape(str(exc))}")


def urlparse_safe_suffix(url: str) -> str:
    from urllib.parse import urlparse

    return Path(urlparse(url).path).name or "clip.mp4"


@router.message(Command("reset"))
async def cmd_reset(message: Message, gigachat_service: GigaChatService) -> None:
    gigachat_service.reset(message.from_user.id)
    await message.answer("✅ Контекст разговора очищен. Можете задать новый вопрос.")


@router.message(Command("deletevoice"))
async def cmd_deletevoice(
    message: Message,
    state: FSMContext,
    profile_service: VoiceProfileService,
    synthesis_service: SynthesisService,
) -> None:
    user_id = message.from_user.id
    await state.clear()
    await profile_service.delete_profile(user_id)
    synthesis_service.invalidate_cache(user_id)
    await message.answer("🗑 Все ваши голосовые референсы и профиль удалены.")


@router.message(VoiceCollection.collecting, F.voice)
async def collect_voice(
    message: Message,
    bot: Bot,
    db: Database,
    settings: Settings,
    profile_service: VoiceProfileService,
) -> None:
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return

    count = await db.count_voice_references(user_id)
    if count >= settings.max_voice_messages:
        await message.answer(f"Достигнут лимит ({settings.max_voice_messages}). Используйте /finishvoice.")
        return

    index = count + 1
    user_dir = profile_service.references_dir(user_id)
    raw_path = user_dir / f"raw_{index:03d}.oga"
    wav_path = profile_service.next_reference_path(user_id, index)

    file = await bot.get_file(message.voice.file_id)
    await bot.download_file(
        file.file_path,
        destination=raw_path,
        timeout=max(30, int(settings.telegram_download_timeout_sec)),
    )

    try:
        metrics = preprocess_telegram_voice(
            raw_path,
            wav_path,
            sample_rate=settings.reference_sample_rate,
            enable_denoise=settings.enable_denoise,
        )
        quality = evaluate_reference(metrics)
        await db.add_voice_reference(
            user_id,
            wav_path.name,
            metrics["duration_sec"],
            quality.to_dict(),
        )
        await message.answer(quality.format_message(index))
    except Exception as exc:
        logger.exception("Ошибка обработки голосового")
        await message.answer(f"Не удалось обработать запись: {exc}")
    finally:
        raw_path.unlink(missing_ok=True)


@router.message(VoiceCollection.collecting)
async def collect_voice_invalid(message: Message) -> None:
    await message.answer("В режиме сбора отправляйте только голосовые сообщения или /finishvoice.")


@router.message(F.voice)
async def answer_voice_question(
    message: Message,
    bot: Bot,
    db: Database,
    settings: Settings,
    transcription_service: TranscriptionService,
    gigachat_service: GigaChatService,
    accent_service: AccentService,
    synthesis_service: SynthesisService,
    call_feel_service: CallFeelService,
) -> None:
    """Voice в†' call-feel в†' chunked Whisper в†' GigaChat stream в†' cloned TTS."""
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return
    if not gigachat_service.configured:
        await message.answer(
            "GigaChat API не настроен. Добавьте "
            "<code>GIGACHAT_CREDENTIALS</code> в .env."
        )
        return
    if (
        message.voice.duration
        and message.voice.duration > settings.stt_max_voice_seconds
    ):
        await message.answer(
            f"Голосовое слишком длинное. Лимит: "
            f"{settings.stt_max_voice_seconds:.0f} секунд."
        )
        return

    status = None
    if not call_feel_service.enabled:
        status = await message.answer(
            "🎧 <b>Слушаю…</b>"
            + (
                "\nПервый запуск STT может занять время: загружается модель."
                if not transcription_service.loaded
                else ""
            )
        )
    else:
        await call_feel_service.play_call_cues(message, user_id)
        status = await message.answer("📞 <b>На линии…</b>")

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    incoming_dir = settings.data_dir / "tmp" / "telegram_voice"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    incoming_path = incoming_dir / f"{user_id}_{uuid.uuid4().hex}.ogg"
    analytics_tasks: list[asyncio.Task[str]] = []
    transcript = ""

    try:
        telegram_file = await bot.get_file(message.voice.file_id)
        await bot.download_file(
            telegram_file.file_path,
            destination=incoming_path,
            timeout=max(30, int(settings.telegram_download_timeout_sec)),
        )

        last_edit = 0.0
        async for update in transcription_service.transcribe_chunks(incoming_path):
            transcript = update.full_text
            if len(analytics_tasks) < settings.stt_max_analytics_chunks:
                analytics_tasks.append(
                    asyncio.create_task(
                        gigachat_service.analyze_chunk(update.part)
                    )
                )

            now = time.monotonic()
            if (
                status is not None
                and not call_feel_service.enabled
                and (
                    now - last_edit >= settings.telegram_edit_interval_sec
                    or update.chunk_index == update.chunk_count
                )
            ):
                try:
                    await status.edit_text(
                        f"🎧 <b>Слушаю…</b> "
                        f"{update.chunk_index}/{update.chunk_count}"
                    )
                    last_edit = now
                except Exception:
                    logger.debug("Telegram status edit пропущен", exc_info=True)

        transcript = transcript.strip()
        if not transcript:
            raise TranscriptionError("Не удалось распознать речь")
        if status is not None and not call_feel_service.enabled:
            await status.edit_text("💭 <b>Думаю над ответом…</b>")

        analysis_context: list[str] = []
        if analytics_tasks:
            done, pending = await asyncio.wait(analytics_tasks, timeout=3.0)
            for task in done:
                try:
                    analysis_context.append(task.result())
                except Exception:
                    logger.debug("Аналитика чанка не готова", exc_info=True)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if status is not None:
            try:
                await status.delete()
            except Exception:
                logger.debug("Не удалось удалить статус voice pipeline", exc_info=True)
        pending = _PendingQuestion(
            question=transcript,
            analysis_context=analysis_context,
            use_stream=True,
        )
        lang = await _saved_reply_lang(db, user_id)
        await _run_chat_turn(
            message,
            user_id=user_id,
            pending=pending,
            lang=lang,
            bot=bot,
            settings=settings,
            call_feel_service=call_feel_service,
            play_cues=False,
        )
    except (
        TranscriptionError,
        GigaChatError,
        ConsentRequiredError,
        ProfileRequiredError,
        ValueError,
    ) as exc:
        await _report_status_error(status, message, f"Ошибка: {_humanize_handler_error(exc)}")
    except Exception as exc:
        logger.exception("Ошибка обработки голосового вопроса")
        await _report_status_error(
            status,
            message,
            f"Не удалось обработать голосовое: {_humanize_handler_error(exc)}",
        )
    finally:
        incoming_path.unlink(missing_ok=True)
        for task in analytics_tasks:
            if not task.done():
                task.cancel()
        if analytics_tasks:
            await asyncio.gather(*analytics_tasks, return_exceptions=True)


def _is_video_message(message: Message) -> bool:
    if message.video or message.video_note:
        return True
    doc = message.document
    if doc and (doc.mime_type or "").startswith("video/"):
        return True
    return False


async def _download_bot_file(
    bot: Bot,
    file_id: str,
    dest: Path,
    *,
    timeout_sec: float,
) -> None:
    telegram_file = await bot.get_file(file_id)
    await bot.download_file(
        telegram_file.file_path,
        destination=dest,
        timeout=max(30, int(timeout_sec)),
    )


async def _download_chat_video(
    *,
    bot: Bot,
    media: Any,
    dest: Path,
    user_id: int,
    size: int,
    settings: Settings,
    large_media_service: LargeMediaService,
    status: Message,
    prefer_telethon: bool,
    chat_message_id: int | None = None,
    progress: dict[str, int] | None = None,
) -> None:
    timeout_sec = float(settings.telegram_download_timeout_sec)
    if prefer_telethon:
        if not large_media_service.telethon_ready():
            raise LargeMediaError(
                f"Файл {size / (1024 * 1024):.0f} МБ — Bot API не отдаст.\n"
                "Нужны TELEGRAM_API_ID/HASH и сессия (/collectaccount),\n"
                "либо <code>/dub https://…</code>, либо <code>/dub inbox</code>."
            )
        me = await bot.get_me()
        if not me.username:
            raise LargeMediaError("У бота нет @username — User API не найдёт чат")
        await large_media_service.download_from_bot_chat(
            user_id=user_id,
            bot_username=me.username,
            dest=dest,
            file_size=size,
            file_name=getattr(media, "file_name", None),
            chat_message_id=chat_message_id,
            progress=progress,
            resume_key=getattr(media, "file_unique_id", None),
        )
        return
    try:
        await _download_bot_file(
            bot, media.file_id, dest, timeout_sec=timeout_sec
        )
    except (TimeoutError, asyncio.TimeoutError, OSError) as exc:
        if not large_media_service.telethon_ready():
            raise LargeMediaError(
                "Telegram оборвал скачивание (таймаут Bot API). "
                "Пришлите файл ещё раз, ссылку <code>/dub https://…</code> "
                "или положите MP4 в inbox."
            ) from exc
        logger.warning("Bot API download failed (%s), falling back to User API", exc)
        await status.edit_text("⬇️ <b>Bot API завис — качаю через User API…</b>")
        me = await bot.get_me()
        if not me.username:
            raise LargeMediaError("У бота нет @username — User API не найдёт чат") from exc
        await large_media_service.download_from_bot_chat(
            user_id=user_id,
            bot_username=me.username,
            dest=dest,
            file_size=size,
            file_name=getattr(media, "file_name", None),
            chat_message_id=chat_message_id,
            progress=progress,
            resume_key=getattr(media, "file_unique_id", None),
        )


@router.message(WaitingDubPaste(), F.document)
@router.message(WaitingDubPaste(), F.text & ~F.text.startswith("/"))
async def on_custom_dub_translation(
    message: Message,
    bot: Bot,
    db: Database,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService,
) -> None:
    await _accept_user_translation(
        message,
        user_id=message.from_user.id,
        bot=bot,
        db=db,
        settings=settings,
        video_dub_service=video_dub_service,
        large_media_service=large_media_service,
        gigachat_service=gigachat_service,
    )


@router.callback_query(F.data == "dub:paste")
async def on_dub_paste(
    callback: CallbackQuery,
    video_dub_service: VideoDubService,
) -> None:
    user_id = callback.from_user.id
    await _ack_callback(callback, "Жду перевод")
    async with _user_lock(user_id):
        pending = await _ensure_video_pending(
            user_id=user_id,
            callback=callback,
            video_dub_service=video_dub_service,
        )
        if pending is None or pending.kind != "video" or not pending.segments:
            if callback.message:
                await callback.message.answer(
                    "Сессия сбросилась. Пришлите видео ещё раз — тогда кнопки заработают."
                )
            return
        pending.await_translation = True
        if len(pending.pasted) != len(pending.segments):
            pending.pasted = [""] * len(pending.segments)
        _put_pending(user_id, pending)
        if callback.message:
            await callback.message.answer(
                f"В DeepL копируйте пакет с тегами <code>&lt;c i&gt;</code> "
                f"(их {len(pending.segments)}) — не партитуру с таймкодами.\n"
                "Можно несколькими сообщениями: с тегами или кусками подряд."
            )


@router.message(F.video | F.video_note)
@router.message(F.document)
async def dub_video_question(
    message: Message,
    bot: Bot,
    db: Database,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
    gigachat_service: GigaChatService,
) -> None:
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return
    if message.document and not (message.document.mime_type or "").startswith("video/"):
        return
    if not _is_video_message(message):
        return

    media = message.video or message.video_note or message.document
    assert media is not None
    duration_limit = settings.video_dub_max_seconds
    duration = float(getattr(media, "duration", 0) or 0)
    if duration_limit > 0 and duration > duration_limit:
        await message.answer(
            f"Видео слишком длинное ({duration:.0f}с). "
            f"Лимит: {duration_limit:.0f}с"
        )
        return
    size = int(getattr(media, "file_size", 0) or 0)
    bot_limit = int(settings.video_dub_max_mb * 1024 * 1024)
    need_telethon = bool(size and size > bot_limit)

    status = await message.answer(
        "⬇️ <b>Скачиваю через User API…</b>"
        if need_telethon
        else "⬇️ <b>Скачиваю видео…</b>"
    )
    workdir = settings.data_dir / "tmp" / f"vid_{user_id}_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)
    suffix = Path(getattr(media, "file_name", "") or "clip.mp4").suffix or ".mp4"
    incoming = workdir / f"src{suffix}"
    progress: dict[str, int] = {"n": 0, "total": size}
    try:
        if need_telethon:
            async with _StatusPulse(
                status,
                "⬇️ <b>Скачиваю через User API…</b>",
                hint="это не зависание",
                extra=lambda: _telethon_progress_line(progress),
            ):
                await _download_chat_video(
                    bot=bot,
                    media=media,
                    dest=incoming,
                    user_id=user_id,
                    size=size,
                    settings=settings,
                    large_media_service=large_media_service,
                    status=status,
                    prefer_telethon=True,
                    chat_message_id=message.message_id,
                    progress=progress,
                )
        else:
            await _download_chat_video(
                bot=bot,
                media=media,
                dest=incoming,
                user_id=user_id,
                size=size,
                settings=settings,
                large_media_service=large_media_service,
                status=status,
                prefer_telethon=False,
                chat_message_id=message.message_id,
                progress=progress,
            )

        await _prompt_video_loudness(
            message,
            user_id=user_id,
            video_path=incoming,
            status=status,
        )
    except (TranscriptionError, ValueError, LargeMediaError) as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        await status.edit_text(f"Ошибка: {html.escape(str(exc))}")
    except (TimeoutError, asyncio.TimeoutError):
        shutil.rmtree(workdir, ignore_errors=True)
        await status.edit_text(
            "Telegram не успел отдать видео. Пришлите ролик ещё раз — "
            "скачивание теперь дольше, при срыве пойдёт через User API."
        )
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        logger.exception("Ошибка разбора видео")
        await status.edit_text(f"Не удалось разобрать видео: {html.escape(str(exc))}")


@router.callback_query(F.data.startswith("dubvol:"))
async def on_dub_loudness(
    callback: CallbackQuery,
    db: Database,
    video_dub_service: VideoDubService,
    gigachat_service: GigaChatService,
) -> None:
    user_id = callback.from_user.id
    mode = (callback.data or "").split(":", 1)[-1].strip().lower()
    quiet = mode == "quiet"
    await _ack_callback(
        callback,
        "Тихий режим" if quiet else "Обычная громкость",
    )
    async with _user_lock(user_id):
        pending = _get_pending(user_id)
        if (
            pending is None
            or pending.kind != "video"
            or pending.video_path is None
            or not pending.video_path.exists()
        ):
            src = find_recoverable_video(_data_dir(), user_id)
            if src is None:
                if callback.message:
                    await callback.message.answer(
                        "Сессия сбросилась. Пришлите видео ещё раз."
                    )
                return
            pending = _PendingQuestion(
                kind="video",
                video_path=src,
                await_loudness=True,
            )
            _put_pending(user_id, pending)
        if pending.segments and not pending.await_loudness:
            if callback.message:
                await callback.message.answer(
                    "Партитура уже снята. Выберите язык дубляжа или пришлите новый ролик."
                )
            return
        if callback.message is None:
            return
        await _run_analyze_with_loudness(
            reply=callback.message,
            user_id=user_id,
            video_path=pending.video_path,
            quiet_audio=quiet,
            db=db,
            video_dub_service=video_dub_service,
            gigachat_service=gigachat_service,
            status=callback.message,
        )


@router.callback_query(F.data.startswith("dubvoice:"))
async def on_voice_pick(
    callback: CallbackQuery,
    bot: Bot,
    settings: Settings,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
) -> None:
    """Выбор реплики-эталона: переозвучка всего видео её голосом."""
    user_id = callback.from_user.id
    parts = (callback.data or "").split(":")
    action = parts[1] if len(parts) > 1 else ""
    if action == "noop":
        await _ack_callback(callback)
        return
    if callback.message is None:
        await _ack_callback(callback)
        return
    session = voice_pick.load_session(_data_dir(), user_id)
    if session is None:
        await _ack_callback(callback, "Сессия озвучки уже очищена", alert=True)
        try:
            await callback.message.edit_text(
                "Сессия озвучки устарела. Пришлите видео заново."
            )
        except Exception:
            pass
        return
    pickable = voice_pick.pickable_cues(session)

    if action == "page":
        try:
            page = int(parts[2])
            chosen = int(parts[3]) if len(parts) > 3 else -1
        except (IndexError, ValueError):
            page, chosen = 0, -1
        await callback.message.edit_reply_markup(
            reply_markup=voice_pick_keyboard(
                pickable, page=page, chosen=None if chosen < 0 else chosen
            )
        )
        await _ack_callback(callback)
        return

    if action == "keep":
        voice_pick.clear_session(_data_dir(), user_id)
        await callback.message.edit_text(
            "✅ Оставили текущий вариант озвучки. Можно присылать следующее видео."
        )
        await _ack_callback(callback)
        return

    if action != "pick":
        await _ack_callback(callback)
        return

    try:
        idx = int(parts[2])
        page = int(parts[3]) if len(parts) > 3 else 0
    except (IndexError, ValueError):
        await _ack_callback(callback)
        return
    cue = next(
        (c for c in (session.get("cues") or []) if int(c.get("i", -1)) == idx),
        None,
    )
    # Voice from the ORIGINAL cue window — not the first-pass TTS wav
    # (Fish leaks leftover words if the ref is itself a previous synthesis).
    ref_path = voice_pick.extract_original_clone_ref(session, idx)
    if ref_path is None or not ref_path.exists():
        ref_path = (
            session["_dir"] / voice_pick.CUES_DIR / str(cue["wav"])
            if cue and cue.get("wav")
            else None
        )
    if ref_path is None or not ref_path.exists():
        await _ack_callback(
            callback, "У этой реплики нет звука — выберите другую", alert=True
        )
        return
    await _ack_callback(callback, f"Переозвучиваю голосом реплики {idx + 1}…")

    async with _user_lock(user_id):
        status = await callback.message.answer(
            f"🎭 <b>Переозвучиваю голосом реплики {idx + 1}</b>\n"
            "Экспрессивные моменты (⚡) останутся из первой озвучки."
        )

        async def _progress(done: int, total: int, preview: str) -> None:
            tip = html.escape(preview) if preview else "…"
            try:
                if done <= 0:
                    await status.edit_text(f"🧬 <b>{tip}</b>")
                else:
                    await status.edit_text(
                        f"🎙 <b>Переозвучка {done}/{total}</b>\n<code>{tip}</code>"
                    )
            except Exception:
                pass

        try:
            result = await video_dub_service.render(
                user_id,
                session["_source"],
                session["_segments"],
                [str(t) for t in session.get("translated") or []],
                str(session.get("lang") or "ru"),
                float(session.get("duration_sec") or 0.0),
                on_progress=_progress,
                clone_refs_override=[ref_path],
                reuse_clip_paths=voice_pick.expressive_reuse_paths(session),
                lock_to_speech=True,
            )
        except Exception as exc:
            logger.exception("Voice-pick re-dub failed for user %s", user_id)
            await status.edit_text(
                f"Ошибка переозвучки: {html.escape(str(exc))[:300]}"
            )
            return

        label = REPLY_LANGUAGES.get(str(session.get("lang") or "ru"), "ru")
        saved_paths = large_media_service.save_to_output(
            result.video_path,
            result.srt_path,
            label=label,
        )
        local_line = "Локально: <code>" + html.escape(str(saved_paths[0])) + "</code>"
        caption = (
            f"🎭 Голос реплики <b>{idx + 1}</b> · {html.escape(label)}\n"
            "⚡-моменты оставлены из первой озвучки"
        )
        out_size = (
            result.video_path.stat().st_size if result.video_path.exists() else 0
        )
        bot_upload_limit = int(settings.video_dub_bot_upload_mb * 1024 * 1024)
        if out_size > bot_upload_limit and large_media_service.telethon_ready():
            await status.edit_text(
                "💾 Сохранил в <code>output</code>\n"
                "📤 Файл крупный — отправляю в <b>Избранное</b> через User API…"
            )
            try:
                await large_media_service.send_file_to_user(
                    user_id=user_id,
                    path=result.video_path,
                    caption=f"Дубляж ({label}), голос реплики {idx + 1}",
                )
                if result.srt_path and result.srt_path.exists():
                    await large_media_service.send_file_to_user(
                        user_id=user_id,
                        path=result.srt_path,
                        caption=f"Субтитры ({label})",
                    )
                await callback.message.answer(
                    f"{caption}\n\n{local_line}\nПлюс копия в <b>Избранном</b> Telegram."
                )
            except Exception:
                logger.exception("Telethon upload of voice-pick dub failed")
                await callback.message.answer(
                    f"{caption}\n\n{local_line}\nВ Telegram не ушло — берите файл с диска."
                )
        else:
            await callback.message.answer_video(
                FSInputFile(str(result.video_path)),
                caption=f"{caption}\n{local_line}",
            )
            if result.srt_path and result.srt_path.exists():
                await callback.message.answer_document(
                    FSInputFile(str(result.srt_path)),
                    caption=f"Субтитры ({html.escape(label)})",
                )
        try:
            await status.delete()
        except Exception:
            pass
        # Сессия живёт: можно попробовать другую реплику-эталон.
        await callback.message.edit_text(
            "🎭 Можно выбрать другую реплику-эталон или оставить вариант:",
            reply_markup=voice_pick_keyboard(pickable, page=page, chosen=idx),
        )


@router.message(SpeakMode.waiting_text, F.text)
async def speak_text(
    message: Message,
    state: FSMContext,
    synthesis_service: SynthesisService,
    accent_service: AccentService,
    settings: Settings,
) -> None:
    if message.text and message.text.startswith("/"):
        return

    user_id = message.from_user.id
    await message.answer("⏳ Синтезирую речь, подождите...")

    try:
        accented_text = await accent_service.add_accents(message.text)
        # OGG из PCM stdin без обязательного WAV на диске
        _, ogg_path = await synthesis_service.synthesize(
            user_id, accented_text, save_wav=False
        )
        if ogg_path is None:
            raise RuntimeError("OGG не создан")
        voice_file = FSInputFile(str(ogg_path))
        await message.answer_voice(voice_file)
        if settings.enable_ai_audio_marker:
            await message.answer(
                f"ℹ️ Аудио синтезировано ИИ. {settings.ai_marker_text}"
            )
    except ConsentRequiredError as exc:
        await message.answer(str(exc))
    except ProfileRequiredError as exc:
        await message.answer(str(exc))
    except ValueError as exc:
        await message.answer(f"Ошибка: {exc}")
    except Exception as exc:
        logger.exception("Ошибка синтеза")
        await message.answer(f"Ошибка синтеза: {exc}")
    finally:
        await state.clear()


@router.message(F.text & ~F.text.startswith("/"))
async def answer_question(
    message: Message,
    bot: Bot,
    db: Database,
    settings: Settings,
    gigachat_service: GigaChatService,
    call_feel_service: CallFeelService,
) -> None:
    """Обычное сообщение: call-feel → GigaChat → hybrid TTS → Telegram voice."""
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return
    if not gigachat_service.configured:
        await message.answer(
            "GigaChat API не настроен. Добавьте "
            "<code>GIGACHAT_CREDENTIALS</code> в .env и перезапустите бота."
        )
        return

    pending = _PendingQuestion(question=message.text or "", use_stream=False)
    lang = await _saved_reply_lang(db, user_id)
    try:
        await _run_chat_turn(
            message,
            user_id=user_id,
            pending=pending,
            lang=lang,
            bot=bot,
            settings=settings,
            call_feel_service=call_feel_service,
        )
    except (GigaChatError, ConsentRequiredError, ProfileRequiredError, ValueError) as exc:
        await message.answer(f"Ошибка: {html.escape(str(exc))}")
    except Exception as exc:
        logger.exception("Ошибка голосового ответа GigaChat")
        await message.answer(
            f"Не удалось подготовить голосовой ответ: {html.escape(str(exc))}"
        )


@router.callback_query(F.data.startswith("lang:"))
async def on_reply_language(
    callback: CallbackQuery,
    bot: Bot,
    db: Database,
    settings: Settings,
    gigachat_service: GigaChatService,
    call_feel_service: CallFeelService,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
) -> None:
    user_id = callback.from_user.id
    data = callback.data or "lang:ru"
    lang = normalize_reply_lang(data.split(":", 1)[1])
    label = REPLY_LANGUAGES[lang]
    await _ack_callback(callback, label)
    async with _user_lock(user_id):
        await _run_reply_language(
            callback,
            user_id=user_id,
            lang=lang,
            label=label,
            bot=bot,
            db=db,
            settings=settings,
            gigachat_service=gigachat_service,
            call_feel_service=call_feel_service,
            video_dub_service=video_dub_service,
            large_media_service=large_media_service,
        )


async def _run_reply_language(
    callback: CallbackQuery,
    *,
    user_id: int,
    lang: str,
    label: str,
    bot: Bot,
    db: Database,
    settings: Settings,
    gigachat_service: GigaChatService,
    call_feel_service: CallFeelService,
    video_dub_service: VideoDubService,
    large_media_service: LargeMediaService,
) -> None:
    pending = _get_pending(user_id)
    if pending is not None and pending.kind == "video" and (
        not pending.segments
        or pending.video_path is None
        or not pending.video_path.exists()
    ):
        pending = None
    if pending is None or (
        pending.kind != "video" and not pending.question.strip()
    ):
        pending = await _ensure_video_pending(
            user_id=user_id,
            callback=callback,
            video_dub_service=video_dub_service,
        )
    if pending is None or (
        pending.kind != "video" and not pending.question.strip()
    ):
        if callback.message:
            await callback.message.answer(
                "Сессия сбросилась. Пришлите вопрос или видео ещё раз."
            )
        return

    user = await db.get_user(user_id)
    user_settings = dict(user.settings or {})
    user_settings["reply_language"] = lang
    await db.update_settings(user_id, user_settings)

    if callback.message:
        try:
            markup = (
                dub_language_keyboard(lang)
                if pending.kind == "video"
                else language_keyboard(lang)
            )
            await callback.message.edit_text(
                f"Язык: <b>{html.escape(label)}</b>",
                reply_markup=markup,
            )
        except Exception:
            pass

    status = None
    try:
        if pending.kind == "video":
            if callback.message is None:
                return
            if pending.video_path is None or not pending.segments:
                await callback.message.answer("Видео уже недоступно, пришлите ещё раз.")
                return
            pending.await_translation = False
            # Drop any leftover pasted lines from a previous video / paste attempt.
            pending.pasted = [""] * len(pending.segments)
            _put_pending(user_id, pending)
            status = await callback.message.answer("🌐 <b>Перевожу реплики в тайминг…</b>")
            await bot.send_chat_action(callback.message.chat.id, ChatAction.UPLOAD_VIDEO)

            async def _tr_progress(done: int, total: int, preview: str) -> None:
                try:
                    await status.edit_text(
                        f"🌐 <b>Перевод {html.escape(preview)}</b>"
                    )
                except Exception:
                    pass

            translated = await video_dub_service.translate_segments(
                pending.segments,
                lang,
                media_duration=pending.duration_sec,
                on_progress=_tr_progress,
                user_id=user_id,
            )
            if status is not None:
                try:
                    await status.delete()
                except Exception:
                    pass
            await _deliver_video_dub(
                reply=callback.message,
                user_id=user_id,
                pending=pending,
                translated=translated,
                lang=lang,
                bot=bot,
                settings=settings,
                video_dub_service=video_dub_service,
                large_media_service=large_media_service,
                gigachat_service=gigachat_service,
            )
            return

        if callback.message is None:
            return
        if call_feel_service.enabled:
            await call_feel_service.play_call_cues(callback.message, user_id)
            status = await callback.message.answer("📞 <b>На линии…</b>")
        else:
            status = await callback.message.answer("🤔 Думаю над ответом…")
        await bot.send_chat_action(callback.message.chat.id, ChatAction.RECORD_VOICE)
        if pending.use_stream:
            await call_feel_service.stream_and_speak(
                callback.message,
                user_id,
                gigachat_service.stream_answer(
                    user_id,
                    pending.question,
                    analysis_context=pending.analysis_context or None,
                    language=lang,
                ),
                prepare_for_speech=GigaChatService.prepare_for_speech,
                max_text_length=settings.max_text_length,
                language=lang,
            )
        else:
            answer = await gigachat_service.answer(
                user_id, pending.question, language=lang
            )
            if status is not None:
                try:
                    await status.edit_text("🎙 <b>Озвучиваю ответ…</b>")
                except Exception:
                    pass
            await call_feel_service.speak_answer_parts(
                callback.message, user_id, answer, language=lang
            )
        if settings.enable_ai_audio_marker:
            await callback.message.answer(
                f"ℹ️ Ответ создан GigaChat и синтезирован ИИ. {settings.ai_marker_text}"
            )
        if status is not None:
            try:
                await status.delete()
            except Exception:
                pass
    except (GigaChatError, ConsentRequiredError, ProfileRequiredError, ValueError) as exc:
        if callback.message:
            await _report_status_error(
                status, callback.message, f"Ошибка: {_humanize_handler_error(exc)}"
            )
    except Exception as exc:
        logger.exception("Ошибка голосового ответа GigaChat")
        if callback.message:
            await _report_status_error(
                status,
                callback.message,
                f"Не удалось подготовить голосовой ответ: {_humanize_handler_error(exc)}",
            )
