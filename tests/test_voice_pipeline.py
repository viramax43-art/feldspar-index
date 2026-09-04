"""Тесты ударений, chunked STT и streaming GigaChat без внешних моделей/API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import numpy as np
import soundfile as sf

from app.config import Settings
from app.services.gigachat import GigaChatService
from app.services.transcription import TranscriptionService
from app.text.accent import (
    AccentService,
    apply_custom_accents,
    merge_gigachat_accents,
    strip_stress_plus,
)


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "TELEGRAM_BOT_TOKEN": "test",
        "GIGACHAT_CREDENTIALS": "test",
        "ENABLE_RUACCENT": False,
        "MAX_TEXT_LENGTH": 500,
        "STT_CHUNK_SECONDS": 1,
        "STT_MAX_VOICE_SECONDS": 10,
    }
    values.update(overrides)
    return Settings(**values)


def test_custom_accents_override_existing_stress() -> None:
    accents = {"договоры": "догов+оры", "войсер": "в+ойсер"}
    text = apply_custom_accents("Дог+оворы и Войсер", accents)
    assert text == "Догов+оры и В+ойсер"
    assert strip_stress_plus(text) == "Договоры и Войсер"


def test_accent_service_fallback_dictionary() -> None:
    service = AccentService(_settings())

    async def run() -> None:
        result = await service.add_accents("Наш каталог и договоры готовы.")
        assert "катал+ог" in result
        assert "догов+оры" in result

    asyncio.run(run())


def test_gigachat_accents_preserved_for_every_word() -> None:
    source = "Н+аш катал+ог +и н+овый догов+ор."
    # RUAccent может оставить служебные/односложные слова без разметки.
    ruaccent_result = "Наш катал+ог и н+овый догов+ор."
    merged = merge_gigachat_accents(ruaccent_result, source)
    assert merged == source


def test_ruaccent_keeps_priority_over_wrong_gigachat() -> None:
    # RUAccent уже поставил з+амок; неверный GigaChat зам+ок не должен победить.
    merged = merge_gigachat_accents("Открой з+амок.", "Открой зам+ок.")
    assert "з+амок" in merged


def test_custom_accent_pomoshchnik() -> None:
    accents = {"помощник": "пом+ощник"}
    assert apply_custom_accents("Нужен помощник.", accents) == "Нужен пом+ощник."


def test_gigachat_stream_answer_saves_history() -> None:
    service = GigaChatService(_settings(GIGACHAT_HISTORY_TURNS=2))
    captured: list[list[dict[str, str]]] = []

    async def fake_stream(
        messages: list[dict[str, str]],
        retry_auth: bool = True,
    ) -> AsyncIterator[str]:
        del retry_auth
        captured.append(messages)
        yield "Короткий "
        yield "ответ."

    service._stream_completion = fake_stream  # type: ignore[method-assign]

    async def run() -> None:
        answer = ""
        async for token in service.stream_answer(
            7,
            "Вопрос?",
            analysis_context=["интент: вопрос"],
        ):
            answer += token
        assert answer == "Короткий ответ."
        # GigaChat требует единственный system строго первым; аналитика внутри него.
        assert [m["role"] for m in captured[0]] == ["system", "user"]
        assert "интент: вопрос" in captured[0][0]["content"]
        assert len(service._history[7]) == 2

    asyncio.run(run())


def test_chunked_transcription_yields_progress(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    settings = _settings(DATA_DIR=tmp_path)
    service = TranscriptionService(settings)

    class FakeModel:
        def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
            del kwargs
            samples, _ = sf.read(path)
            text = f"фрагмент {len(samples)}"
            return [SimpleNamespace(text=text)], None

    service._model = FakeModel()
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(35_200, dtype=np.float32), 16_000)

    # В unit-тесте вход уже WAV — копируем вместо запуска ffmpeg.
    def fake_convert(
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        mono: bool,
    ) -> None:
        del sample_rate, mono
        data, sr = sf.read(input_path, dtype="float32")
        sf.write(output_path, data, sr)

    monkeypatch.setattr(
        "app.services.transcription.convert_to_wav",
        fake_convert,
    )

    async def run() -> None:
        updates = [update async for update in service.transcribe_chunks(source)]
        assert len(updates) == 3
        assert updates[-1].chunk_index == 3
        assert updates[-1].chunk_count == 3
        assert updates[-1].full_text.count("фрагмент") == 3

    asyncio.run(run())


def test_normalize_stt_language_auto_is_none(tmp_path: Path) -> None:
    service = TranscriptionService(_settings(DATA_DIR=tmp_path))
    assert service._normalize_stt_language("auto") is None
    assert service._normalize_stt_language("EN") == "en"
    assert service._normalize_stt_language("ru") == "ru"
