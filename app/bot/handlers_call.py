"""Команды живого Telegram-звонка: /call, /hangup."""

from __future__ import annotations

import html
import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.config import Settings
from app.database import Database
from app.services.call_pipeline import CallOrchestrator
from app.services.telegram_call import CallTransportError

logger = logging.getLogger(__name__)
router = Router(name="call")


async def _require_consent(db: Database, user_id: int) -> bool:
    return await db.has_consent(user_id)


@router.message(Command("call"))
async def cmd_call(
    message: Message,
    db: Database,
    settings: Settings,
    call_orchestrator: CallOrchestrator,
) -> None:
    if not settings.call_enabled:
        await message.answer("Живые звонки отключены (CALL_ENABLED=false).")
        return
    user_id = message.from_user.id
    if not await _require_consent(db, user_id):
        await message.answer("Сначала подтвердите согласие: /consent")
        return
    if not call_orchestrator.gigachat.configured:
        await message.answer("GigaChat не настроен — звонок без ответов невозможен.")
        return
    if not call_orchestrator.transport.configured:
        await message.answer(
            "Telethon не настроен. Нужны TELEGRAM_API_ID, TELEGRAM_API_HASH "
            "и авторизованная сессия:\n"
            "<code>python scripts/collect_account_voices.py --consent</code>"
        )
        return

    status = await message.answer(
        "📞 Сейчас поступит <b>входящий звонок</b> с user-аккаунта бота.\n"
        "Примите его. Можно перебивать речь ассистента.\n"
        "Завершить: /hangup"
    )
    username = message.from_user.username
    try:
        await call_orchestrator.start_call_for_user(
            user_id,
            username=username,
        )
        await status.edit_text(
            "✅ Звонок инициирован. Примите вызов в Telegram.\n"
            "Перебивание и смена темы поддерживаются.\n"
            "/hangup — сбросить."
        )
    except CallTransportError as exc:
        logger.warning("Call transport error: %s", exc)
        phone = (settings.telegram_phone or "").strip()
        extra = (
            f"\n\nЕсли аккаунт ещё не в контактах — напишите ему "
            f"(<code>{html.escape(phone)}</code>) любое сообщение и повторите /call."
            if phone
            else ""
        )
        await status.edit_text(
            f"Не удалось установить звонок:\n{html.escape(str(exc))}"
            f"{extra}\n"
            "Можно продолжить обычными voice-сообщениями."
        )
    except Exception as exc:
        logger.exception("Call start failed")
        await status.edit_text(
            f"Ошибка звонка: {html.escape(str(exc))}"
        )


@router.message(Command("hangup"))
async def cmd_hangup(
    message: Message,
    call_orchestrator: CallOrchestrator,
) -> None:
    user_id = message.from_user.id
    await call_orchestrator.stop_call_for_user(user_id)
    await message.answer("📴 Звонок завершён.")
