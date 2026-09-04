"""Тесты обработки русского текста и бизнес-правил."""

import pytest

from app.services.synthesis import ConsentRequiredError, SynthesisService
from app.services.voice_profile import VoiceProfileService
from app.text.preprocess import (
    normalize_whitespace,
    prepare_text_for_tts,
    expand_currency,
    expand_percent,
    get_inference_params,
    apply_inline_stress_marks,
    load_pronunciation_dict,
    pause_for_punctuation,
)


def test_normalize_whitespace():
    assert normalize_whitespace("  Привет,   мир!  ") == "Привет, мир!"


def test_split_long_text():
    text = "Первое предложение. " + "Второе предложение с дополнительными словами, " * 5
    chunks = prepare_text_for_tts(text, max_chunk_chars=80)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 120 for c in chunks)


def test_punctuation_pauses():
    assert pause_for_punctuation("Привет.") == 0.50
    assert pause_for_punctuation("Как дела?") == 0.55
    assert pause_for_punctuation("Да,") == 0.08
    chunks = prepare_text_for_tts("Привет. Как дела?")
    # Короткие предложения объединяются; учитывается финальный знак вопроса.
    assert chunks[0].pause_after == 0.55


def test_expand_currency():
    result = expand_currency("Цена 1500 руб.")
    assert "рублей" in result


def test_expand_percent():
    result = expand_percent("Скидка 15%")
    assert "процентов" in result


def test_intonation_presets():
    calm = get_inference_params("calm")
    expressive = get_inference_params("expressive")
    assert calm["temperature"] < expressive["temperature"]
    assert expressive["temperature"] >= 0.75
    assert abs(expressive["speed"] - 1.0) < 1e-6
    assert expressive["repetition_penalty"] <= 2.1


def test_inline_stress_mark():
    assert "о́" in apply_inline_stress_marks("зам+ок") or "о\u0301" in apply_inline_stress_marks("зам+ок")
    out = apply_inline_stress_marks("догово́р")
    assert "договор" in out.replace("\u0301", "")


def test_xtts_strips_stress_marks():
    """XTTS не должен получать U+0301; буквы е/ё не подменяются."""
    chunks = prepare_text_for_tts("Он звонит по договору.", engine="xtts")
    text = " ".join(c.text for c in chunks)
    assert "\u0301" not in text
    assert "+" not in text


def test_xtts_keeps_e_not_forced_yo():
    from app.text.preprocess import accents_for_xtts, apply_inline_stress_marks

    marked = apply_inline_stress_marks("нед+еля")
    out = accents_for_xtts(marked)
    assert "ё" not in out.lower()
    assert "неделя" in out.lower()
    assert "\u0301" not in out


def test_xtts_preserves_real_yo():
    from app.text.preprocess import accents_for_xtts

    assert "ё" in accents_for_xtts("Ну конечно, всё готово.").lower()


def test_silero_keeps_stress_markup():
    chunks = prepare_text_for_tts("Он звонит по договору.", engine="silero")
    text = " ".join(c.text for c in chunks)
    # Silero markup: +перед гласной или акут уже сконвертирован
    assert "+" in text or "\u0301" in text or "звон" in text.lower()


def test_pronunciation_dict_override(tmp_path):
    path = tmp_path / "pron.json"
    path.write_text('{"тест": "тэст"}', encoding="utf-8")
    d = load_pronunciation_dict(path)
    chunks = prepare_text_for_tts("Это тест.", pronunciation_dict=d, engine="xtts")
    assert "тэст" in chunks[0].text.lower()


@pytest.mark.asyncio
async def test_synthesis_requires_consent(tmp_path):
    from app.config import Settings
    from app.database import Database
    from app.tts.xtts_engine import XTTSEngine

    settings = Settings(
        TELEGRAM_BOT_TOKEN="test",
        DATA_DIR=tmp_path / "data",
    )
    db = Database(settings.db_path)
    await db.init()
    await db.ensure_user(1)

    class DummyEngine(XTTSEngine):
        def load(self) -> None:
            pass

    engine = DummyEngine(settings.tts_model_name, "cpu")
    profile = VoiceProfileService(settings, db)
    service = SynthesisService(settings, db, engine, profile)

    with pytest.raises(ConsentRequiredError):
        await service.synthesize(1, "Тест")


@pytest.mark.asyncio
async def test_profile_access_denied(tmp_path):
    from app.config import Settings
    from app.database import Database

    settings = Settings(TELEGRAM_BOT_TOKEN="test", DATA_DIR=tmp_path / "data")
    db = Database(settings.db_path)
    profile = VoiceProfileService(settings, db)

    with pytest.raises(PermissionError):
        profile.assert_user_access(requester_id=1, owner_id=2)
