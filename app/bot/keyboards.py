"""Telegram bot keyboards."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton

from app.text.reply_lang import REPLY_LANGUAGES, normalize_reply_lang

CONSENT_TEXT = (
    "Я подтверждаю, что загружаю свой голос или имею явное разрешение "
    "владельца голоса на его клонирование."
)


def consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Подтверждаю",
                    callback_data="consent:yes",
                )
            ],
            [
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="consent:no",
                )
            ],
        ]
    )


def settings_keyboard(current: dict) -> InlineKeyboardMarkup:
    intonation = current.get("intonation", "neutral")
    speed = current.get("speed", 1.0)
    lang = normalize_reply_lang(current.get("reply_language"))
    lang_label = REPLY_LANGUAGES.get(lang, lang)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Язык ответа: {lang_label}",
                    callback_data="settings:lang",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"Интонация: {intonation}",
                    callback_data="settings:intonation",
                )
            ],
            [
                InlineKeyboardButton(text="Скорость −", callback_data="settings:speed:-"),
                InlineKeyboardButton(text=f"{speed:.2f}x", callback_data="settings:speed:show"),
                InlineKeyboardButton(text="Скорость +", callback_data="settings:speed:+"),
            ],
            [
                InlineKeyboardButton(text="Температура −", callback_data="settings:temp:-"),
                InlineKeyboardButton(
                    text=f"T={current.get('temperature', 0.75):.2f}",
                    callback_data="settings:temp:show",
                ),
                InlineKeyboardButton(text="Температура +", callback_data="settings:temp:+"),
            ],
        ]
    )


def language_keyboard(
    selected: str | None = None,
    *,
    prefix: str = "lang",
) -> InlineKeyboardMarkup:
    current = normalize_reply_lang(selected)
    row1: list[InlineKeyboardButton] = []
    row2: list[InlineKeyboardButton] = []
    for i, (code, label) in enumerate(REPLY_LANGUAGES.items()):
        mark = "✓ " if code == current else ""
        btn = InlineKeyboardButton(
            text=f"{mark}{label}".strip(),
            callback_data=f"{prefix}:{code}",
        )
        (row1 if i < 3 else row2).append(btn)
    return InlineKeyboardMarkup(inline_keyboard=[row1, row2])


def dub_language_keyboard(selected: str | None = None) -> InlineKeyboardMarkup:
    base = language_keyboard(selected)
    rows = list(base.inline_keyboard)
    rows.append(
        [
            InlineKeyboardButton(
                text="📋 Свой перевод — жду текст",
                callback_data="dub:paste",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def dub_loudness_keyboard() -> InlineKeyboardMarkup:
    """Choose STT gain before analyze (normal vs quiet/ASMR)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔊 Обычная громкость",
                    callback_data="dubvol:normal",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🤫 Тихая / ASMR",
                    callback_data="dubvol:quiet",
                )
            ],
        ]
    )


VOICE_PICK_PER_PAGE = 8


def voice_pick_keyboard(
    cues: list[dict],
    *,
    page: int = 0,
    chosen: int | None = None,
) -> InlineKeyboardMarkup:
    """Выбор реплики-эталона голоса для переозвучки.

    cues: элементы voice_pick.pickable_cues() — {i, preview, expressive, wav}.
    ⚡ — экспрессивный момент (останется из первой озвучки).
    """
    per = VOICE_PICK_PER_PAGE
    total_pages = max(1, (len(cues) + per - 1) // per)
    page = max(0, min(page, total_pages - 1))
    chunk = cues[page * per : (page + 1) * per]
    rows: list[list[InlineKeyboardButton]] = []
    for cue in chunk:
        idx = int(cue["i"])
        preview = str(cue.get("preview") or "…")
        if len(preview) > 30:
            preview = preview[:29] + "…"
        marks = ""
        if cue.get("expressive"):
            marks += " ⚡"
        if chosen is not None and chosen == idx:
            marks = " ✅" + marks
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{idx + 1}. {preview}{marks}",
                    callback_data=f"dubvoice:pick:{idx}:{page}",
                )
            ]
        )
    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀️", callback_data=f"dubvoice:page:{page - 1}:{chosen if chosen is not None else -1}"
                )
            )
        nav.append(
            InlineKeyboardButton(
                text=f"{page + 1}/{total_pages}",
                callback_data="dubvoice:noop",
            )
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="▶️", callback_data=f"dubvoice:page:{page + 1}:{chosen if chosen is not None else -1}"
                )
            )
        rows.append(nav)
    rows.append(
        [
            InlineKeyboardButton(
                text="✅ Оставить текущий вариант",
                callback_data="dubvoice:keep",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def remove_keyboard() -> ReplyKeyboardMarkup:    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Отмена")]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
