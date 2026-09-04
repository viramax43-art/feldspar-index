"""Тесты GigaChat-сервиса без сетевых запросов."""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings
from app.services.gigachat import GigaChatService


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "TELEGRAM_BOT_TOKEN": "test",
        "GIGACHAT_CREDENTIALS": "test-credentials",
        "GIGACHAT_HISTORY_TURNS": 2,
        "MAX_TEXT_LENGTH": 200,
    }
    values.update(overrides)
    return Settings(**values)


def test_prepare_for_speech_removes_markdown() -> None:
    text = (
        "## Ответ\n"
        "1. **Первый** пункт.\n"
        "2. [Второй](https://example.com) пункт с `кодом`."
    )
    result = GigaChatService.prepare_for_speech(text, max_chars=500)
    assert "#" not in result
    assert "**" not in result
    assert "https://" not in result
    assert "`" not in result
    assert "Первый пункт." in result


def test_prepare_for_speech_limits_at_sentence() -> None:
    text = "Первая короткая фраза. " + "Очень длинное продолжение " * 20
    result = GigaChatService.prepare_for_speech(text, max_chars=80)
    assert len(result) <= 81
    assert result.endswith(".")


def test_answer_keeps_bounded_user_history() -> None:
    service = GigaChatService(_settings())
    seen_messages: list[list[dict[str, str]]] = []

    async def fake_completion(messages: list[dict[str, str]]) -> str:
        seen_messages.append(messages)
        return "**Краткий** голосовой ответ."

    service._request_completion = fake_completion  # type: ignore[method-assign]

    async def run() -> None:
        first = await service.answer(42, "Первый вопрос?")
        second = await service.answer(42, "Второй вопрос?")
        third = await service.answer(42, "Третий вопрос?")

        assert first == "Краткий голосовой ответ."
        assert second == first == third
        assert len(service._history[42]) == 4
        # Во второй запрос попала первая пара user/assistant.
        assert [m["role"] for m in seen_messages[1]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

        service.reset(42)
        assert 42 not in service._history

    asyncio.run(run())


def test_not_configured_without_credentials() -> None:
    service = GigaChatService(_settings(GIGACHAT_CREDENTIALS=""))
    assert service.configured is False
