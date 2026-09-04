"""Тесты call-feel: разбиение фраз и генерация гудка."""

from __future__ import annotations

from pathlib import Path

from app.audio.call_cues import generate_ringback_ogg, split_first_utterance


def test_split_first_utterance_splits_on_sentence() -> None:
    first, rest = split_first_utterance(
        "Привет, это тестовый ответ. А вот продолжение длиннее двадцати четырёх.",
        min_first_chars=12,
        min_rest_chars=24,
    )
    assert first.startswith("Привет")
    assert "продолж" in rest


def test_split_first_utterance_keeps_short_answers() -> None:
    text = "Короткий ответ без хвоста."
    first, rest = split_first_utterance(text)
    assert first == text
    assert rest == ""


def test_generate_ringback_ogg(tmp_path: Path) -> None:
    path = tmp_path / "ringback.ogg"
    out = generate_ringback_ogg(path, duration_sec=0.3)
    assert out.exists()
    assert out.stat().st_size > 100
    size = out.stat().st_size
    generate_ringback_ogg(path, duration_sec=0.3)
    assert out.stat().st_size == size
