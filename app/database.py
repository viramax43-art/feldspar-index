"""SQLite-хранилище пользователей, согласий и настроек."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UserRecord:
    user_id: int
    has_consent: bool
    consent_at: str | None
    has_voice_profile: bool
    profile_created_at: str | None
    settings: dict[str, Any]


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    has_consent INTEGER NOT NULL DEFAULT 0,
                    consent_at TEXT,
                    has_voice_profile INTEGER NOT NULL DEFAULT 0,
                    profile_created_at TEXT,
                    settings_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    quality_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_voice_refs_user
                ON voice_references(user_id)
                """
            )
            await db.commit()

    async def ensure_user(self, user_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
                (user_id,),
            )
            await db.commit()

    async def set_consent(self, user_id: int, granted: bool) -> None:
        await self.ensure_user(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE users
                SET has_consent = ?, consent_at = ?
                WHERE user_id = ?
                """,
                (1 if granted else 0, _utc_now() if granted else None, user_id),
            )
            await db.commit()

    async def has_consent(self, user_id: int) -> bool:
        await self.ensure_user(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT has_consent FROM users WHERE user_id = ?",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return bool(row and row[0])

    async def get_user(self, user_id: int) -> UserRecord:
        await self.ensure_user(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT user_id, has_consent, consent_at, has_voice_profile,
                       profile_created_at, settings_json
                FROM users WHERE user_id = ?
                """,
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        assert row is not None
        return UserRecord(
            user_id=row[0],
            has_consent=bool(row[1]),
            consent_at=row[2],
            has_voice_profile=bool(row[3]),
            profile_created_at=row[4],
            settings=json.loads(row[5] or "{}"),
        )

    async def update_settings(self, user_id: int, settings: dict[str, Any]) -> None:
        await self.ensure_user(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET settings_json = ? WHERE user_id = ?",
                (json.dumps(settings, ensure_ascii=False), user_id),
            )
            await db.commit()

    async def add_voice_reference(
        self,
        user_id: int,
        file_name: str,
        duration_sec: float,
        quality: dict[str, Any],
    ) -> int:
        await self.ensure_user(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO voice_references
                (user_id, file_name, duration_sec, quality_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    file_name,
                    duration_sec,
                    json.dumps(quality, ensure_ascii=False),
                    _utc_now(),
                ),
            )
            await db.commit()
            return cursor.lastrowid or 0

    async def list_voice_references(self, user_id: int) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, file_name, duration_sec, quality_json, created_at
                FROM voice_references
                WHERE user_id = ?
                ORDER BY id
                """,
                (user_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "file_name": row[1],
                "duration_sec": row[2],
                "quality": json.loads(row[3] or "{}"),
                "created_at": row[4],
            }
            for row in rows
        ]

    async def count_voice_references(self, user_id: int) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM voice_references WHERE user_id = ?",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def clear_voice_references(self, user_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM voice_references WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()

    async def set_voice_profile(self, user_id: int, active: bool) -> None:
        await self.ensure_user(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE users
                SET has_voice_profile = ?, profile_created_at = ?
                WHERE user_id = ?
                """,
                (1 if active else 0, _utc_now() if active else None, user_id),
            )
            await db.commit()

    async def delete_user_data(self, user_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM voice_references WHERE user_id = ?", (user_id,))
            await db.execute(
                """
                UPDATE users
                SET has_voice_profile = 0, profile_created_at = NULL
                WHERE user_id = ?
                """,
                (user_id,),
            )
            await db.commit()
