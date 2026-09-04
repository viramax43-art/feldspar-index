"""Смоук FSM звонка без сети/Telegram."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.call_session import CallSessionManager, CallState


async def main() -> None:
    mgr = CallSessionManager(
        stop_phrases=["стоп", "подожди"],
        topic_shift_phrases=["сменим тему"],
        barge_in_enabled=True,
    )
    await mgr.start_call(42)
    await mgr.on_answered(42)
    assert (await mgr.get(42)).state == CallState.LISTENING

    s = await mgr.on_end_of_utterance(42, "Привет, кто ты?")
    assert s.state == CallState.THINKING
    turn = s.turn_id
    assert await mgr.on_answer_ready(42, turn)
    assert s.state == CallState.SPEAKING

    barged = await mgr.on_user_speech_while_speaking(42)
    assert barged and barged.state == CallState.BARGE_IN

    s2 = await mgr.on_end_of_utterance(42, "Сменим тему: какой сегодня день?")
    assert s2.state == CallState.THINKING
    assert s2.topic_shift_pending

    await mgr.on_end_of_utterance(42, "Стоп")
    assert (await mgr.get(42)).state == CallState.LISTENING

    await mgr.hangup(42)
    assert (await mgr.get(42)).state == CallState.IDLE
    print("SMOKE CALL FSM OK")


if __name__ == "__main__":
    asyncio.run(main())
