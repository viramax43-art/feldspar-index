"""Тесты FSM живого звонка (barge-in / стоп / смена темы)."""

from __future__ import annotations

import asyncio

from app.services.call_session import CallSessionManager, CallState
from app.services.telegram_call import CallAudioBridge


def test_call_session_stop_phrase() -> None:
    mgr = CallSessionManager(
        stop_phrases=["стоп", "подожди"],
        topic_shift_phrases=["сменим тему"],
    )

    async def run() -> None:
        await mgr.start_call(1)
        await mgr.on_answered(1)
        session = await mgr.on_end_of_utterance(1, "Подожди секунду")
        assert session.state == CallState.LISTENING
        assert session.pending_transcript == ""

    asyncio.run(run())


def test_call_session_topic_shift() -> None:
    mgr = CallSessionManager(
        stop_phrases=["стоп"],
        topic_shift_phrases=["сменим тему", "другой вопрос"],
    )

    async def run() -> None:
        await mgr.start_call(2)
        await mgr.on_answered(2)
        session = await mgr.on_end_of_utterance(2, "Сменим тему про договор")
        assert session.state == CallState.THINKING
        assert session.topic_shift_pending is True
        assert "договор" in session.pending_transcript.lower()

    asyncio.run(run())


def test_call_session_barge_in_cancels_speaking() -> None:
    mgr = CallSessionManager(barge_in_enabled=True)

    async def run() -> None:
        await mgr.start_call(3)
        await mgr.on_answered(3)
        session = await mgr.on_end_of_utterance(3, "Расскажи длинную историю")
        turn = session.turn_id
        assert await mgr.on_answer_ready(3, turn)
        assert session.state == CallState.SPEAKING
        barged = await mgr.on_user_speech_while_speaking(3)
        assert barged is not None
        assert barged.state == CallState.BARGE_IN
        assert barged.is_cancelled()
        assert barged.playback_stop.is_set()

    asyncio.run(run())


def test_call_audio_bridge_clear_outgoing() -> None:
    async def run() -> None:
        bridge = CallAudioBridge(sample_rate=8000)
        await bridge.push_outgoing(b"\x01\x02" * 100)
        await bridge.clear_outgoing()
        silence = await bridge.pop_outgoing(20)
        assert silence == b"\x00" * 20

    asyncio.run(run())


def test_ruaccent_priority_and_pomoshchnik() -> None:
    from app.text.accent import apply_custom_accents, merge_gigachat_accents

    merged = merge_gigachat_accents("Это пом+ощник.", "Это п+омощник.")
    assert "пом+ощник" in merged
    custom = apply_custom_accents(
        "Нужен помощник.",
        {"помощник": "пом+ощник"},
    )
    assert custom == "Нужен пом+ощник."
