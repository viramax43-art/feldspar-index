"""Middleware для внедрения зависимостей в обработчики."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject


class DependencyMiddleware(BaseMiddleware):
    def __init__(self, **dependencies: Any) -> None:
        self.dependencies = dependencies

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data.update(self.dependencies)
        return await handler(event, data)
