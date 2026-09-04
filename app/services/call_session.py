"""Конечный автомат живого звонка: barge-in, стоп-фразы, смена темы."""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CallState(str, enum.Enum):
    IDLE = "idle"
    RINGING = "ringing"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    BARGE_IN = "barge_in"
    TOPIC_SHIFT = "topic_shift"


@dataclass
class CallSession:
    """Состояние одного пользователя в звонке."""

    user_id: int
    state: CallState = CallState.IDLE
    turn_id: int = 0
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    playback_stop: asyncio.Event = field(default_factory=asyncio.Event)
    last_transcript: str = ""
    pending_transcript: str = ""
    topic_shift_pending: bool = False
    barge_in_pending: bool = False
    started_at: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def new_turn(self) -> int:
        self.turn_id += 1
        self.cancel_event = asyncio.Event()
        self.playback_stop = asyncio.Event()
        return self.turn_id

    def cancel_generation(self) -> None:
        self.cancel_event.set()
        self.playback_stop.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    async def transition(self, new_state: CallState) -> None:
        async with self._lock:
            old = self.state
            self.state = new_state
            logger.debug(
                "CallSession user=%s %s -> %s turn=%s",
                self.user_id,
                old.value,
                new_state.value,
                self.turn_id,
            )


class CallSessionManager:
    """Пул сессий + эвристики стоп/смена темы."""

    def __init__(
        self,
        *,
        stop_phrases: list[str] | None = None,
        topic_shift_phrases: list[str] | None = None,
        barge_in_enabled: bool = True,
        vad_silence_ms: int = 750,
    ) -> None:
        self._sessions: dict[int, CallSession] = {}
        self.stop_phrases = [p.lower() for p in (stop_phrases or [])]
        self.topic_shift_phrases = [p.lower() for p in (topic_shift_phrases or [])]
        self.barge_in_enabled = barge_in_enabled
        self.vad_silence_ms = vad_silence_ms
        self._lock = asyncio.Lock()

    async def get(self, user_id: int) -> CallSession:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = CallSession(user_id=user_id)
                self._sessions[user_id] = session
            return session

    async def drop(self, user_id: int) -> None:
        async with self._lock:
            session = self._sessions.pop(user_id, None)
            if session is not None:
                session.cancel_generation()
                await session.transition(CallState.IDLE)

    def is_stop_phrase(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(p in normalized for p in self.stop_phrases)

    def is_topic_shift_phrase(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(p in normalized for p in self.topic_shift_phrases)

    async def start_call(self, user_id: int) -> CallSession:
        session = await self.get(user_id)
        session.cancel_generation()
        session.new_turn()
        session.topic_shift_pending = False
        session.barge_in_pending = False
        session.last_transcript = ""
        session.pending_transcript = ""
        await session.transition(CallState.RINGING)
        return session

    async def on_answered(self, user_id: int) -> CallSession:
        session = await self.get(user_id)
        await session.transition(CallState.LISTENING)
        return session

    async def on_end_of_utterance(self, user_id: int, transcript: str) -> CallSession:
        session = await self.get(user_id)
        text = (transcript or "").strip()
        session.pending_transcript = text
        session.last_transcript = text

        if self.is_stop_phrase(text):
            session.cancel_generation()
            await session.transition(CallState.LISTENING)
            session.pending_transcript = ""
            return session

        if self.is_topic_shift_phrase(text):
            session.cancel_generation()
            session.topic_shift_pending = True
            session.new_turn()
            await session.transition(CallState.TOPIC_SHIFT)
            await session.transition(CallState.THINKING)
            return session

        session.new_turn()
        await session.transition(CallState.THINKING)
        return session

    async def on_answer_ready(self, user_id: int, turn_id: int) -> bool:
        """True если можно начать Speaking для этого turn."""
        session = await self.get(user_id)
        if session.turn_id != turn_id or session.is_cancelled():
            return False
        if session.state not in {CallState.THINKING, CallState.TOPIC_SHIFT, CallState.BARGE_IN}:
            return False
        await session.transition(CallState.SPEAKING)
        return True

    async def on_playback_done(self, user_id: int, turn_id: int) -> None:
        session = await self.get(user_id)
        if session.turn_id != turn_id:
            return
        if session.state == CallState.SPEAKING:
            await session.transition(CallState.LISTENING)

    async def on_user_speech_while_speaking(self, user_id: int) -> CallSession | None:
        """Barge-in: речь клиента во время ответа ИИ."""
        if not self.barge_in_enabled:
            return None
        session = await self.get(user_id)
        if session.state != CallState.SPEAKING:
            return None
        session.cancel_generation()
        session.barge_in_pending = True
        await session.transition(CallState.BARGE_IN)
        return session

    async def hangup(self, user_id: int) -> None:
        await self.drop(user_id)
