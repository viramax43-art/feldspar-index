"""Асинхронный GigaChat API клиент с OAuth и историей диалога."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp

from app.config import Settings

logger = logging.getLogger(__name__)


class GigaChatError(RuntimeError):
    """Ошибка настройки или запроса GigaChat."""


_SSL_HINT = (
    "Похоже на проблему с российским корневым сертификатом Минцифры. "
    "Соберите CA-bundle: python scripts/setup_gigachat_certs.py, затем укажите "
    "GIGACHAT_CA_BUNDLE_FILE=./assets/certs/russian_trusted_ca.pem в .env. "
    "Быстрый (небезопасный) вариант: GIGACHAT_VERIFY_SSL=false."
)


def _is_ssl_error(exc: BaseException) -> bool:
    import ssl as _ssl

    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, _ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(seen):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


class GigaChatService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._user_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        max_messages = max(0, settings.gigachat_history_turns) * 2
        self._history: defaultdict[int, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_messages or None)
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.gigachat_credentials.strip())

    def _ssl_context(self) -> ssl.SSLContext | bool:
        if not self.settings.gigachat_verify_ssl:
            logger.warning(
                "Проверка TLS GigaChat отключена. Используйте только для локальной диагностики."
            )
            return False
        ca_file = self.settings.gigachat_ca_bundle_file
        if ca_file:
            path = Path(ca_file)
            if not path.exists():
                raise GigaChatError(f"CA bundle GigaChat не найден: {path}")
            return ssl.create_default_context(cafile=str(path))
        return ssl.create_default_context()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.settings.gigachat_timeout_sec,
                connect=min(15.0, self.settings.gigachat_timeout_sec),
            )
            connector = aiohttp.TCPConnector(ssl=self._ssl_context())
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get_access_token(self, force_refresh: bool = False) -> str:
        if not self.configured:
            raise GigaChatError(
                "GigaChat не настроен: добавьте GIGACHAT_CREDENTIALS в .env"
            )

        # Оставляем минутный запас до истечения токена.
        if (
            not force_refresh
            and self._access_token
            and time.time() < self._token_expires_at - 60
        ):
            return self._access_token

        async with self._token_lock:
            if (
                not force_refresh
                and self._access_token
                and time.time() < self._token_expires_at - 60
            ):
                return self._access_token

            session = await self._get_session()
            headers = {
                "Authorization": f"Basic {self.settings.gigachat_credentials.strip()}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            try:
                async with session.post(
                    self.settings.gigachat_auth_url,
                    headers=headers,
                    data={"scope": self.settings.gigachat_scope},
                ) as response:
                    payload = await self._read_json(response)
                    if response.status != 200:
                        raise GigaChatError(
                            f"Ошибка авторизации GigaChat ({response.status}): "
                            f"{self._error_text(payload)}"
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if _is_ssl_error(exc):
                    raise GigaChatError(f"GigaChat OAuth: TLS. {_SSL_HINT}") from exc
                raise GigaChatError(f"GigaChat OAuth недоступен: {exc}") from exc

            token = str(payload.get("access_token", "")).strip()
            if not token:
                raise GigaChatError("GigaChat OAuth не вернул access_token")
            self._access_token = token

            # API возвращает expires_at в миллисекундах Unix.
            expires_at = float(payload.get("expires_at", 0) or 0)
            self._token_expires_at = (
                expires_at / 1000.0 if expires_at > 10_000_000_000 else expires_at
            )
            if self._token_expires_at <= time.time():
                self._token_expires_at = time.time() + 1_700
            return token

    @staticmethod
    async def _read_json(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            data = await response.json(content_type=None)
        except (ValueError, aiohttp.ContentTypeError):
            text = await response.text()
            return {"message": text[:1000]}
        return data if isinstance(data, dict) else {"message": str(data)}

    @staticmethod
    def _error_text(payload: dict[str, Any]) -> str:
        return str(
            payload.get("message")
            or payload.get("error_description")
            or payload.get("error")
            or payload
        )

    async def _request_completion(
        self,
        messages: list[dict[str, str]],
        retry_auth: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        token = await self._get_access_token()
        session = await self._get_session()
        body = {
            "model": self.settings.gigachat_model,
            "messages": messages,
            "temperature": (
                self.settings.gigachat_temperature
                if temperature is None
                else temperature
            ),
            "max_tokens": (
                self.settings.gigachat_max_tokens
                if max_tokens is None
                else max_tokens
            ),
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with session.post(
                f"{self.settings.gigachat_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            ) as response:
                payload = await self._read_json(response)
                if response.status == 401 and retry_auth:
                    await self._get_access_token(force_refresh=True)
                    return await self._request_completion(
                        messages,
                        retry_auth=False,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                if response.status != 200:
                    raise GigaChatError(
                        f"Ошибка GigaChat ({response.status}): "
                        f"{self._error_text(payload)}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if _is_ssl_error(exc):
                raise GigaChatError(f"GigaChat API: TLS. {_SSL_HINT}") from exc
            raise GigaChatError(f"GigaChat API недоступен: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GigaChatError("GigaChat вернул ответ неизвестного формата") from exc
        if not isinstance(content, str) or not content.strip():
            raise GigaChatError("GigaChat вернул пустой ответ")
        return content.strip()

    async def complete_stateless(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 120,
    ) -> str:
        """Короткий запрос без пользовательской истории (аналитика чанков)."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._request_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def analyze_chunk(self, text: str) -> str:
        """Интент и сущности для предварительного контекста финального ответа."""
        return await self.complete_stateless(
            (
                "Проанализируй фрагмент речи клиента. Верни одной короткой строкой: "
                "интент (вопрос/жалоба/заказ/просьба/другое), тональность и ключевые "
                f"сущности. Без пояснений. Фрагмент: {text}"
            ),
            system_prompt="Ты быстрый аналитический классификатор русской речи.",
            temperature=0.1,
            max_tokens=100,
        )

    async def _stream_completion(
        self,
        messages: list[dict[str, str]],
        retry_auth: bool = True,
    ) -> AsyncIterator[str]:
        token = await self._get_access_token()
        session = await self._get_session()
        body = {
            "model": self.settings.gigachat_model,
            "messages": messages,
            "temperature": self.settings.gigachat_temperature,
            "max_tokens": self.settings.gigachat_max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            async with session.post(
                f"{self.settings.gigachat_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            ) as response:
                if response.status == 401 and retry_auth:
                    await response.read()
                    await self._get_access_token(force_refresh=True)
                    async for part in self._stream_completion(
                        messages,
                        retry_auth=False,
                    ):
                        yield part
                    return
                if response.status != 200:
                    payload = await self._read_json(response)
                    raise GigaChatError(
                        f"Ошибка GigaChat stream ({response.status}): "
                        f"{self._error_text(payload)}"
                    )

                async for raw_line in response.content:
                    for line in raw_line.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = json.loads(data)
                            delta = payload["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if isinstance(content, str) and content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            logger.debug("Неизвестный SSE chunk GigaChat: %s", data[:300])
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if _is_ssl_error(exc):
                raise GigaChatError(f"GigaChat stream: TLS. {_SSL_HINT}") from exc
            raise GigaChatError(f"GigaChat stream недоступен: {exc}") from exc

    async def stream_answer(
        self,
        user_id: int,
        question: str,
        *,
        analysis_context: list[str] | None = None,
        language: str | None = None,
    ) -> AsyncIterator[str]:
        """Стрим ответа с сохранением истории после успешного завершения."""
        from app.text.reply_lang import system_prompt_for_language

        question = question.strip()
        if not question:
            raise ValueError("Пустой вопрос")

        async with self._user_locks[user_id]:
            # GigaChat требует единственный system-message строго первым,
            # поэтому аналитику вплетаем в него, а не отдельным сообщением.
            system_content = system_prompt_for_language(
                self.settings.gigachat_system_prompt, language or "ru"
            )
            if analysis_context:
                system_content += (
                    "\n\nПредварительный анализ голосового запроса (используй "
                    "только как контекст, не упоминай его): "
                    + " | ".join(analysis_context)
                )
            messages = [
                {"role": "system", "content": system_content},
                *list(self._history[user_id]),
                {"role": "user", "content": question},
            ]

            parts: list[str] = []
            async for part in self._stream_completion(messages):
                parts.append(part)
                yield part

            raw_answer = "".join(parts).strip()
            if not raw_answer:
                raise GigaChatError("GigaChat вернул пустой streaming-ответ")
            if self.settings.gigachat_history_turns > 0:
                self._history[user_id].append({"role": "user", "content": question})
                self._history[user_id].append(
                    {"role": "assistant", "content": raw_answer}
                )

    async def answer(self, user_id: int, question: str, language: str | None = None) -> str:
        from app.text.reply_lang import system_prompt_for_language

        question = question.strip()
        if not question:
            raise ValueError("Пустой вопрос")

        async with self._user_locks[user_id]:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_for_language(
                        self.settings.gigachat_system_prompt, language or "ru"
                    ),
                },
                *list(self._history[user_id]),
                {"role": "user", "content": question},
            ]
            raw_answer = await self._request_completion(messages)
            answer = self.prepare_for_speech(
                raw_answer,
                max_chars=self.settings.max_text_length,
            )
            if not answer:
                raise GigaChatError("После подготовки к озвучиванию ответ пуст")

            if self.settings.gigachat_history_turns > 0:
                self._history[user_id].append({"role": "user", "content": question})
                self._history[user_id].append(
                    {"role": "assistant", "content": raw_answer}
                )
            return answer

    def reset(self, user_id: int) -> None:
        self._history.pop(user_id, None)

    @staticmethod
    def prepare_for_speech(text: str, max_chars: int) -> str:
        """Убирает Markdown и безопасно ограничивает текст перед TTS."""
        if max_chars <= 0:
            return ""
        text = re.sub(r"```(?:\w+)?\s*(.*?)```", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
        text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
        text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
        text = text.replace("**", "").replace("__", "").replace("~~", "")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text

        clipped = text[:max_chars]
        # Предпочитаем закончить на последней полной фразе.
        boundary = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if boundary >= max_chars // 2:
            return clipped[: boundary + 1].strip()
        return clipped[: max(0, max_chars - 1)].rstrip(" ,;:-") + "."
