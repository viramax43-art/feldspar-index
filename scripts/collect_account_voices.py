#!/usr/bin/env python3
"""
Сбор собственных голосовых сообщений из всех чатов Telegram-аккаунта.

Требует TELEGRAM_API_ID и TELEGRAM_API_HASH (https://my.telegram.org).
При первом запуске попросит код из Telegram / пароль 2FA.

Примеры:
  # Полный сбор (сотни файлов / до часа речи в пуле)
  python scripts/collect_account_voices.py --consent --build-profile

  # Продолжить сбор с большими лимитами
  python scripts/collect_account_voices.py --limit 2000 --max-seconds 7200 --build-profile

  # Только пересобрать профиль из уже скачанных лучших референсов
  python scripts/collect_account_voices.py --build-profile-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio.quality import evaluate_profile
from app.config import get_settings
from app.database import Database
from app.services.account_collector import AccountVoiceCollector
from app.services.voice_profile import VoiceProfileService


def _safe_print(text: str) -> None:
    """Windows cp1251 консоль ломается на emoji — печатаем безопасно."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.users_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_path)
    await db.init()
    profile_service = VoiceProfileService(settings, db)
    collector = AccountVoiceCollector(settings, db, profile_service)

    try:
        from telethon import TelegramClient

        if not settings.telegram_api_id or not settings.telegram_api_hash:
            print(
                "Укажите TELEGRAM_API_ID и TELEGRAM_API_HASH в .env\n"
                "Получить: https://my.telegram.org → API development tools"
            )
            return 1

        session = collector.session_path()
        session.parent.mkdir(parents=True, exist_ok=True)
        client = TelegramClient(
            str(session),
            settings.telegram_api_id,
            settings.telegram_api_hash,
        )
        await client.start(phone=settings.telegram_phone or None)
        me = await client.get_me()
        user_id = me.id
        await client.disconnect()

        if args.consent:
            await db.set_consent(user_id, True)
            _safe_print(f"OK consent saved for user_id={user_id}")

        if not await db.has_consent(user_id):
            _safe_print("No consent. Run with --consent or /consent in the bot.")
            return 1

        if args.clear:
            await profile_service.delete_profile(user_id)
            await db.set_consent(user_id, True)
            _safe_print("Old references deleted, consent kept.")

        if not args.build_profile_only:
            _safe_print(
                "Starting full collect of outgoing voice messages...\n"
                "This can take a long time."
            )
            result = await collector.collect(
                profile_user_id=user_id,
                limit=args.limit,
                max_seconds=args.max_seconds,
                per_dialog=args.per_dialog,
                messages_per_dialog=args.messages_per_dialog,
                workers=args.workers,
            )
            _safe_print(result.summary())

        if args.build_profile or args.build_profile_only:
            refs = await db.list_voice_references(user_id)
            quality = evaluate_profile(
                refs,
                min_total_seconds=settings.min_reference_seconds,
                max_total_seconds=max(
                    settings.max_reference_seconds,
                    settings.profile_max_seconds,
                ),
                min_count=settings.min_voice_messages,
            )
            if not quality.accepted:
                _safe_print("Profile not created:\n" + quality.format_message(0))
                return 1
            meta = await profile_service.build_profile(user_id)
            _safe_print(
                f"Profile built: {meta['reference_count']} refs "
                f"({meta['total_duration_sec']:.0f} s) "
                f"from pool {meta['pool_accepted_count']} "
                f"({meta['pool_accepted_sec']:.0f} s)"
            )
        elif not args.build_profile_only:
            _safe_print("\nNext: python scripts/collect_account_voices.py --build-profile-only")

        return 0
    except PermissionError as exc:
        _safe_print(f"Permission error: {exc}")
        return 1
    except Exception as exc:
        logging.exception("Collect failed")
        _safe_print(f"Error: {exc}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сбор своих голосовых из всех чатов Telegram-аккаунта"
    )
    parser.add_argument(
        "--consent",
        action="store_true",
        help="Подтвердить согласие на использование своего голоса",
    )
    parser.add_argument(
        "--build-profile",
        action="store_true",
        help="После сбора создать профиль из лучших референсов",
    )
    parser.add_argument(
        "--build-profile-only",
        action="store_true",
        help="Только пересобрать профиль без нового сканирования",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Удалить старые референсы перед сбором",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Макс. число принятых файлов (по умолчанию из .env, сейчас 500)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Макс. суммарная длительность принятой речи в секундах (по умолчанию 3600)",
    )
    parser.add_argument(
        "--per-dialog",
        type=int,
        default=None,
        help="Макс. принятых голосовых на один чат (по умолчанию 200)",
    )
    parser.add_argument(
        "--messages-per-dialog",
        type=int,
        default=None,
        help="Сколько последних сообщений смотреть в чате (0 = вся история)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Число параллельных воркеров загрузки/обработки (по умолчанию 8)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
