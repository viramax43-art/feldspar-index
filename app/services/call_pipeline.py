"""Оркестратор живого звонка: VAD → STT → GigaChat → TTS с barge-in."""

from __future__ import annotations

import asyncio
import logging
import struct
import tempfile
from pathlib import Path

from app.config import Settings
from app.services.call_session import CallSessionManager, CallState
from app.services.gigachat import GigaChatService
from app.services.synthesis import SynthesisService
from app.services.telegram_call import CallAudioBridge, TelegramCallService
from app.services.transcription import TranscriptionService
from app.text.accent import AccentService

logger = logging.getLogger(__name__)


def _pcm16_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    if not samples:
        return 0.0
    acc = sum(s * s for s in samples) / len(samples)
    return acc ** 0.5


class CallOrchestrator:
    """Связывает CallSession + TelegramCallService + STT/LLM/TTS."""

    def __init__(
        self,
        settings: Settings,
        sessions: CallSessionManager,
        call_transport: TelegramCallService,
        transcription: TranscriptionService,
        gigachat: GigaChatService,
        accents: AccentService,
        synthesis: SynthesisService,
    ) -> None:
        self.settings = settings
        self.sessions = sessions
        self.transport = call_transport
        self.transcription = transcription
        self.gigachat = gigachat
        self.accents = accents
        self.synthesis = synthesis
        self._tasks: dict[int, asyncio.Task[None]] = {}

    async def start_call_for_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
    ) -> None:
        await self.stop_call_for_user(user_id)
        await self.sessions.start_call(user_id)
        bridge = await self.transport.start_outgoing_call(
            user_id,
            username=username,
        )
        await self.sessions.on_answered(user_id)
        task = asyncio.create_task(
            self._run_loop(user_id, bridge),
            name=f"call-loop-{user_id}",
        )
        self._tasks[user_id] = task

    async def stop_call_for_user(self, user_id: int) -> None:
        task = self._tasks.pop(user_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.transport.hangup(user_id)
        await self.sessions.hangup(user_id)

    async def _run_loop(self, user_id: int, bridge: CallAudioBridge) -> None:
        """Простой energy-VAD endpointing + обработка реплик."""
        sample_rate = bridge.sample_rate
        silence_needed = max(
            1,
            int(self.settings.call_vad_silence_ms / 20),
        )  # ~20ms frames
        speech_needed = max(1, int(self.settings.call_vad_speech_ms / 20))
        frame_bytes = int(sample_rate * 0.02) * 2  # 20ms s16le mono
        speech_frames = 0
        silence_frames = 0
        capturing = False
        buffer = bytearray()
        threshold = 500.0

        try:
            # Приветствие «алло»
            await self._speak_text(user_id, bridge, self.settings.call_feel_alo_text)

            while not bridge.closed:
                session = await self.sessions.get(user_id)
                if session.state == CallState.IDLE:
                    break

                pcm = await bridge.receive_incoming(timeout=0.05)
                if pcm is None:
                    # тишина: если слушаем и был захват — считаем endpoint
                    if capturing and session.state in {
                        CallState.LISTENING,
                        CallState.BARGE_IN,
                        CallState.SPEAKING,
                    }:
                        silence_frames += 1
                        if silence_frames >= silence_needed and buffer:
                            await self._handle_utterance(
                                user_id, bridge, bytes(buffer)
                            )
                            buffer.clear()
                            capturing = False
                            speech_frames = 0
                            silence_frames = 0
                    continue

                # Нарезка на кадры
                for offset in range(0, len(pcm), frame_bytes):
                    frame = pcm[offset : offset + frame_bytes]
                    if len(frame) < frame_bytes:
                        break
                    rms = _pcm16_rms(frame)
                    is_speech = rms >= threshold

                    if session.state == CallState.SPEAKING and is_speech:
                        barged = await self.sessions.on_user_speech_while_speaking(
                            user_id
                        )
                        if barged is not None:
                            await bridge.clear_outgoing()
                            capturing = True
                            buffer.clear()
                            speech_frames = 1
                            silence_frames = 0
                            buffer.extend(frame)
                            continue

                    if session.state not in {
                        CallState.LISTENING,
                        CallState.BARGE_IN,
                    }:
                        continue

                    if is_speech:
                        speech_frames += 1
                        silence_frames = 0
                        if speech_frames >= speech_needed:
                            capturing = True
                        if capturing:
                            buffer.extend(frame)
                    else:
                        if capturing:
                            buffer.extend(frame)
                            silence_frames += 1
                            if silence_frames >= silence_needed and buffer:
                                await self._handle_utterance(
                                    user_id, bridge, bytes(buffer)
                                )
                                buffer.clear()
                                capturing = False
                                speech_frames = 0
                                silence_frames = 0
                        else:
                            speech_frames = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Call loop crashed user=%s", user_id)
        finally:
            await self.transport.hangup(user_id)
            await self.sessions.hangup(user_id)
            self._tasks.pop(user_id, None)

    async def _handle_utterance(
        self,
        user_id: int,
        bridge: CallAudioBridge,
        pcm: bytes,
    ) -> None:
        if len(pcm) < self.settings.call_pcm_sample_rate:  # <1s
            return

        transcript = await self._transcribe_pcm(pcm, bridge.sample_rate)
        if not transcript.strip():
            session = await self.sessions.get(user_id)
            if session.state == CallState.BARGE_IN:
                await session.transition(CallState.LISTENING)
            return

        session = await self.sessions.on_end_of_utterance(user_id, transcript)
        if not session.pending_transcript:
            # stop-фраза
            return

        turn_id = session.turn_id
        analysis = []
        if session.topic_shift_pending or session.barge_in_pending:
            analysis = [self.settings.call_interrupt_system_hint]
            session.topic_shift_pending = False
            session.barge_in_pending = False

        try:
            raw = ""
            user = await self.synthesis.db.get_user(user_id)
            lang = (user.settings or {}).get("reply_language") or "ru"
            async for token in self.gigachat.stream_answer(
                user_id,
                transcript,
                analysis_context=analysis,
                language=lang,
            ):
                if session.is_cancelled() or session.turn_id != turn_id:
                    return
                raw += token
            answer = GigaChatService.prepare_for_speech(
                raw, self.settings.max_text_length
            )
            if not await self.sessions.on_answer_ready(user_id, turn_id):
                return
            await self._speak_text(user_id, bridge, answer, turn_id=turn_id)
            await self.sessions.on_playback_done(user_id, turn_id)
        except Exception:
            logger.exception("Call turn failed user=%s", user_id)
            await self.sessions.on_answered(user_id)

    async def _transcribe_pcm(self, pcm: bytes, sample_rate: int) -> str:
        from app.audio import convert_pcm16_to_ogg_opus

        with tempfile.TemporaryDirectory(prefix="call_stt_") as tmp:
            ogg = Path(tmp) / "utt.ogg"
            await asyncio.to_thread(
                convert_pcm16_to_ogg_opus, pcm, sample_rate, ogg
            )
            text = ""
            async for update in self.transcription.transcribe_chunks(ogg):
                text = update.full_text
            return text.strip()

    async def _speak_text(
        self,
        user_id: int,
        bridge: CallAudioBridge,
        text: str,
        turn_id: int | None = None,
    ) -> None:
        session = await self.sessions.get(user_id)
        if turn_id is None:
            turn_id = session.turn_id
        if session.is_cancelled() and turn_id == session.turn_id:
            # новый turn мог сбросить cancel — проверяем turn
            pass
        accented = await self.accents.add_accents(text)
        # Синтез в OGG, затем декод в PCM для bridge
        _, ogg_path = await self.synthesis.synthesize(
            user_id, accented, save_wav=False
        )
        if ogg_path is None:
            return
        if session.turn_id != turn_id or session.playback_stop.is_set():
            return
        pcm = await asyncio.to_thread(
            self._ogg_to_pcm16, ogg_path, bridge.sample_rate
        )
        # чанками, чтобы barge-in мог очистить очередь
        chunk = bridge.sample_rate * 2 // 5  # 200ms
        for i in range(0, len(pcm), chunk):
            if session.turn_id != turn_id or session.playback_stop.is_set():
                await bridge.clear_outgoing()
                return
            await bridge.push_outgoing(pcm[i : i + chunk])
            await asyncio.sleep(0.05)

    @staticmethod
    def _ogg_to_pcm16(ogg_path: Path, sample_rate: int) -> bytes:
        import subprocess

        from app.audio import require_ffmpeg

        ffmpeg = require_ffmpeg()
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(ogg_path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg decode failed: {err}")
        return result.stdout
