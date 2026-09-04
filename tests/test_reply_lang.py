from app.bot.keyboards import language_keyboard
from app.text.reply_lang import (
    REPLY_LANGUAGES,
    normalize_reply_lang,
    system_prompt_for_language,
    xtts_language_code,
)


def test_normalize_and_xtts_codes():
    assert normalize_reply_lang("JA") == "ja"
    assert normalize_reply_lang("xx") == "ru"
    assert xtts_language_code("ko") == "ko"


def test_system_prompt_switches_language():
    base = "Ты голосовой ассистент. Отвечай на русском."
    ru = system_prompt_for_language(base, "ru")
    de = system_prompt_for_language(base, "de")
    assert "только на русском" in ru.lower() or "русском" in ru
    assert "Deutsch" in de
    assert "ударен" not in de.lower()


def test_language_keyboard_callback_data():
    kb = language_keyboard("fr")
    codes = {
        btn.callback_data
        for row in kb.inline_keyboard
        for btn in row
    }
    assert codes == {f"lang:{code}" for code in REPLY_LANGUAGES}
    assert any("✓" in btn.text and "Français" in btn.text for row in kb.inline_keyboard for btn in row)
    settings_kb = language_keyboard("ja", prefix="setlang")
    assert {btn.callback_data for row in settings_kb.inline_keyboard for btn in row} == {
        f"setlang:{code}" for code in REPLY_LANGUAGES
    }
