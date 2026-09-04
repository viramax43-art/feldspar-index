"""Имитация живого обзвона: гудок → пауза → «алло» → ответ."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from aiogram.types import FSInputFile, Message

from app.audio.call_cues import generate_ringback_ogg, split_first_utterance
from app.config import Settings
from app.services.synthesis import SynthesisService
from app.text.accent import AccentService

logger = logging.getLogger(__name__)

# реэкспорт для удобства
__all__ = ["CallFeelService", "generate_ringback_ogg", "split_first_utterance"]


class CallFeelService:
    """Отправка cue обзвона и ранней первой фразы ответа."""

    def __init__(
        self,
        settings: Settings,
        synthesis_service: SynthesisService,
        accent_service: AccentService,
    ) -> None:
        self.settings = settings
        self.synthesis = synthesis_service
        self.accent = accent_service
        self._alo_locks: dict[int, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.settings.call_feel_enabled)

    def ensure_ringback(self) -> Path:
        return generate_ringback_ogg(self.settings.call_feel_ringback_path)

    def alo_path_for_user(self, user_id: int) -> Path:
        cache = self.synthesis.profile_service.cache_dir(user_id)
        return cache / "alo.ogg"

    async def ensure_alo_ogg(self, user_id: int) -> Path | None:
        """Синтезировать и закэшировать «алло» голосом пользователя (или Silero)."""
        if user_id <= 0:
            return None
        path = self.alo_path_for_user(user_id)
        if path.exists() and path.stat().st_size > 0:
            return path

        lock = self._alo_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            if path.exists() and path.stat().st_size > 0:
                return path
            text = self.settings.call_feel_alo_text.strip()
            if not text:
                return None
            try:
                accented = await self.accent.add_accents(text)
                _, ogg = await self.synthesis.synthesize(
                    user_id,
                    accented,
                    save_wav=False,
                )
                if ogg is None or not ogg.exists():
                    return None
                path.parent.mkdir(parents=True, exist_ok=True)
                if ogg.resolve() != path.resolve():
                    path.write_bytes(ogg.read_bytes())
                logger.info("Закэширован alo.ogg для user_id=%s", user_id)
                return path
            except Exception as exc:
                logger.warning("Не удалось подготовить alo.ogg: %s", exc)
                return None

    async def play_call_cues(self, message: Message, user_id: int) -> None:
        """Гудок → пауза → «алло». Ошибки cue не ломают основной ответ."""
        if not self.enabled:
            return
        try:
            alo_task = asyncio.create_task(self.ensure_alo_ogg(user_id))
            ring = self.ensure_ringback()
            await message.answer_voice(FSInputFile(str(ring)))
            await asyncio.sleep(max(0.0, self.settings.call_feel_ring_delay_sec))
            await asyncio.sleep(max(0.0, self.settings.call_feel_pickup_delay_sec))
            try:
                alo = await alo_task
            except Exception as exc:
                logger.warning("alo.ogg не готов: %s", exc)
                alo = None
            if alo is not None:
                await message.answer_voice(FSInputFile(str(alo)))
        except Exception:
            logger.exception("Call-feel cue не отправлен")

    async def speak_answer_parts(
        self,
        message: Message,
        user_id: int,
        answer_text: str,
        language: str | None = None,
    ) -> None:
        """Озвучить ответ целиком или первой фразой + хвостом."""
        if self.enabled and self.settings.call_feel_early_first_phrase:
            first, rest = split_first_utterance(
                answer_text,
                min_first_chars=self.settings.call_feel_min_first_chars,
                min_rest_chars=self.settings.call_feel_min_rest_chars,
            )
        else:
            first, rest = answer_text, ""

        parts = [p for p in (first, rest) if p]
        if not parts:
            parts = [answer_text]

        for part in parts:
            accented = await self.accent.add_accents(part)
            _, ogg_path = await self.synthesis.synthesize(
                user_id,
                accented,
                save_wav=False,
                language=language,
            )
            if ogg_path is None:
                raise RuntimeError("OGG не создан")
            await message.answer_voice(FSInputFile(str(ogg_path)))

    async def stream_and_speak(
        self,
        message: Message,
        user_id: int,
        token_stream: AsyncIterator[str],
        *,
        prepare_for_speech,
        max_text_length: int,
        language: str | None = None,
    ) -> str:
        """
        Стрим GigaChat: первая фраза уходит в TTS сразу после конца предложения,
        хвост — вторым voice. Возвращает полный сырой ответ.
        """
        early = self.enabled and self.settings.call_feel_early_first_phrase
        raw = ""
        first_sent = False

        async for token in token_stream:
            raw += token
            if not early or first_sent:
                continue
            prepared = prepare_for_speech(raw, max_text_length)
            first, rest = split_first_utterance(
                prepared,
                min_first_chars=self.settings.call_feel_min_first_chars,
                min_rest_chars=self.settings.call_feel_min_rest_chars,
            )
            if not rest:
                continue
            accented = await self.accent.add_accents(first)
            _, ogg_path = await self.synthesis.synthesize(
                user_id,
                accented,
                save_wav=False,
                language=language,
            )
            if ogg_path is None:
                raise RuntimeError("OGG не создан")
            await message.answer_voice(FSInputFile(str(ogg_path)))
            first_sent = True

        prepared = prepare_for_speech(raw, max_text_length)
        if not first_sent:
            await self.speak_answer_parts(message, user_id, prepared, language=language)
            return raw

        _first, rest = split_first_utterance(
            prepared,
            min_first_chars=self.settings.call_feel_min_first_chars,
            min_rest_chars=self.settings.call_feel_min_rest_chars,
        )
        if rest.strip():
            accented = await self.accent.add_accents(rest)
            _, ogg_path = await self.synthesis.synthesize(
                user_id,
                accented,
                save_wav=False,
                language=language,
            )
            if ogg_path is None:
                raise RuntimeError("OGG не создан")
            await message.answer_voice(FSInputFile(str(ogg_path)))
        return raw
