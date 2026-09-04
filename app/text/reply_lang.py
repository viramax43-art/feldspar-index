"""Язык ответа бота: подписи кнопок, промпт GigaChat, код XTTS."""

from __future__ import annotations

REPLY_LANGUAGES: dict[str, str] = {
    "ru": "🇷🇺 Русский",
    "de": "🇩🇪 Deutsch",
    "fr": "🇫🇷 Français",
    "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어",
    "en": "🇬🇧 English",
}

_XTTS_CODES = {"ru", "de", "fr", "ja", "ko", "en"}

# Лимиты XTTS tokenizer (с запасом, иначе audio обрезается)
XTTS_CHAR_LIMITS = {
    "en": 230,
    "de": 230,
    "fr": 240,
    "ru": 165,
    "ja": 62,
    "ko": 82,
}

LANG_NAMES = {
    "ru": "русский",
    "de": "немецкий",
    "fr": "французский",
    "ja": "японский",
    "ko": "корейский",
    "en": "английский",
}

# Fish Audio S2 language prompts (English labels — more stable than native script
# on the free OpenRouter route; avoid tagging the *reference* as source language).
FISH_LANG_TAGS = {
    "ru": "russian",
    "en": "english",
    "de": "german",
    "fr": "french",
    "ja": "japanese",
    "ko": "korean",
}


def fish_language_tag(code: str | None, default: str = "ru") -> str:
    lang = normalize_reply_lang(code, default)
    return FISH_LANG_TAGS.get(lang, FISH_LANG_TAGS["en"])


def apply_fish_language_tag(text: str, language: str | None) -> str:
    """Prepend Fish language control tag (keeps existing emotion tags after it)."""
    spoken = (text or "").strip()
    if not spoken:
        return spoken
    tag = fish_language_tag(language)
    prefix = f"[{tag}]"
    if spoken.startswith(prefix):
        return spoken
    return f"{prefix} {spoken}"


_VOICE_PROMPT = (
    "You are a voice phone assistant. Write only what will be spoken aloud. "
    "No Markdown, tables, emoji, or bullet lists. "
    "Sound natural and conversational. Use punctuation for intonation: "
    "questions, exclamations, ellipses for pauses."
)

_INSTRUCTIONS = {
    "ru": "Отвечай только на русском языке.",
    "de": "Antworte ausschließlich auf Deutsch. Keine russischen Wörter.",
    "fr": "Réponds uniquement en français. Pas de mots russes.",
    "ja": "日本語だけで答えてください。ロシア語は使わないでください。",
    "ko": "한국어로만 대답하세요. 러시아어를 쓰지 마세요.",
    "en": "Reply only in English. Do not use Russian.",
}


def normalize_reply_lang(code: str | None, default: str = "ru") -> str:
    code = (code or default).lower().strip()
    if code in REPLY_LANGUAGES:
        return code
    return default if default in REPLY_LANGUAGES else "ru"


def xtts_language_code(code: str | None, default: str = "ru") -> str:
    lang = normalize_reply_lang(code, default)
    return lang if lang in _XTTS_CODES else "en"


def xtts_chunk_limit(language: str | None, default_chars: int = 165) -> int:
    lang = xtts_language_code(language)
    return min(default_chars, XTTS_CHAR_LIMITS.get(lang, default_chars))


def system_prompt_for_language(base_prompt: str, language: str) -> str:
    lang = normalize_reply_lang(language)
    instruction = _INSTRUCTIONS[lang]
    if lang == "ru":
        return f"{base_prompt.rstrip()}\n\n{instruction}"
    return f"{_VOICE_PROMPT}\n\n{instruction}"
