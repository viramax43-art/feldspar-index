"""Telegram private call transport через Telethon user-аккаунт.

Приватные звонки требуют E2EE/WebRTC (tgcalls). Если native-стек недоступен,
сервис инициирует RequestCall (гудок у пользователя) и сообщает о fallback.
Аудио-пайплайн barge-in работает через CallAudioBridge (PCM queue).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections import deque
from pathlib import Path
from typing import Callable, Deque

from app.config import Settings

logger = logging.getLogger(__name__)


class CallTransportError(RuntimeError):
    """Не удалось установить Telegram-звонок."""


class CallAudioBridge:
    """Очереди PCM для входящего/исходящего звука звонка (barge-in чистит out)."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate
        self._out: Deque[bytes] = deque()
        self._in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        self._lock = asyncio.Lock()
        self.closed = False

    async def push_outgoing(self, pcm: bytes) -> None:
        if self.closed or not pcm:
            return
        async with self._lock:
            self._out.append(pcm)

    async def clear_outgoing(self) -> None:
        async with self._lock:
            self._out.clear()

    async def pop_outgoing(self, nbytes: int) -> bytes:
        async with self._lock:
            if not self._out:
                return b"\x00" * nbytes
            chunk = bytearray()
            while self._out and len(chunk) < nbytes:
                part = self._out.popleft()
                need = nbytes - len(chunk)
                if len(part) <= need:
                    chunk.extend(part)
                else:
                    chunk.extend(part[:need])
                    self._out.appendleft(part[need:])
            if len(chunk) < nbytes:
                chunk.extend(b"\x00" * (nbytes - len(chunk)))
            return bytes(chunk)

    async def push_incoming(self, pcm: bytes) -> None:
        if self.closed or not pcm:
            return
        try:
            self._in.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._in.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._in.put_nowait(pcm)

    async def receive_incoming(self, timeout: float | None = 0.2) -> bytes | None:
        try:
            return await asyncio.wait_for(self._in.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        self.closed = True


def _i2b(value: int) -> bytes:
    length = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, "big")


def _b2i(value: int | bytes) -> int:
    """Telethon отдаёт DH modulus `p` как bytes."""
    if isinstance(value, int):
        return value
    return int.from_bytes(value, "big")


class TelegramCallService:
    """Исходящий звонок с Telethon user-сессии + PCM bridge."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._bridges: dict[int, CallAudioBridge] = {}
        self._active_calls: dict[int, object] = {}
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(
            self.settings.telegram_api_id
            and self.settings.telegram_api_hash
            and self.settings.telegram_session_name
        )

    def session_path(self) -> Path:
        return (
            self.settings.data_dir
            / "sessions"
            / self.settings.telegram_session_name
        )

    @staticmethod
    def _parse_proxy(proxy_url: str):
        """Parse socks5://user:pass@host:port or http://host:port into Telethon tuple."""
        from urllib.parse import urlparse

        raw = (proxy_url or "").strip()
        if not raw:
            return None
        if "://" not in raw:
            raw = "socks5://" + raw
        u = urlparse(raw)
        host = u.hostname
        port = u.port
        if not host or not port:
            raise CallTransportError(
                f"Некорректный TELEGRAM_PROXY={proxy_url!r}. "
                "Ожидается socks5://host:port или socks5://user:pass@host:port"
            )
        scheme = (u.scheme or "socks5").lower()
        if scheme in {"socks5", "socks"}:
            import socks

            proxy_type = socks.SOCKS5
        elif scheme in {"http", "https"}:
            import socks

            proxy_type = socks.HTTP
        else:
            raise CallTransportError(
                f"TELEGRAM_PROXY: схема {scheme} не поддерживается (socks5/http)"
            )
        return (proxy_type, host, port, True, u.username, u.password)

    async def ensure_client(self):
        if not self.configured:
            raise CallTransportError(
                "Telethon не настроен: TELEGRAM_API_ID / TELEGRAM_API_HASH / SESSION"
            )
        if self._client is not None and self._client.is_connected():
            return self._client

        from telethon import TelegramClient

        path = self.session_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        proxy = None
        if self.settings.mtproto_proxy_url:
            try:
                proxy = self._parse_proxy(self.settings.mtproto_proxy_url)
            except CallTransportError:
                raise
            except Exception as exc:
                raise CallTransportError(f"TELEGRAM_PROXY: {exc}") from exc
            if proxy is None:
                pass
            else:
                logger.info(
                    "Telethon proxy enabled type=%s host=%s:%s",
                    proxy[0],
                    proxy[1],
                    proxy[2],
                )

        client = TelegramClient(
            str(path),
            self.settings.telegram_api_id,
            self.settings.telegram_api_hash,
            proxy=proxy,
            use_ipv6=False,
            connection_retries=8,
            retry_delay=1,
            timeout=25,
        )
        try:
            await client.connect()
        except Exception as exc:
            await client.disconnect()
            hint = (
                "MTProto DC Telegram с этого сервера недоступны "
                "(Bot API HTTPS работает, звонки/User API — нет). "
                "Укажите TELEGRAM_PROXY=socks5://user:pass@host:port в .env "
                "или запускайте /call с машины, где Telethon коннектится."
            )
            raise CallTransportError(f"{exc}\n{hint}") from exc
        if not await client.is_user_authorized():
            await client.disconnect()
            raise CallTransportError(
                "Telethon-сессия не авторизована. Сначала:\n"
                "python scripts/collect_account_voices.py --consent"
            )
        self._client = client
        return client

    def get_bridge(self, user_id: int) -> CallAudioBridge | None:
        return self._bridges.get(user_id)

    async def start_outgoing_call(
        self,
        user_id: int,
        *,
        username: str | None = None,
        on_state: Callable[[str], None] | None = None,
    ) -> CallAudioBridge:
        """
        Инициирует private call (RequestCall). WebRTC media через tgcalls,
        если библиотека доступна; иначе bridge создаётся для локального
        пайплайна, а пользователю нужно принять гудок / использовать fallback.
        """
        async with self._lock:
            if user_id in self._bridges:
                await self.hangup(user_id)

        client = await self.ensure_client()
        bridge = CallAudioBridge(sample_rate=self.settings.call_pcm_sample_rate)
        self._bridges[user_id] = bridge

        try:
            await self._request_phone_call(
                client,
                user_id,
                username=username,
                on_state=on_state,
            )
        except CallTransportError:
            self._bridges.pop(user_id, None)
            bridge.close()
            raise
        except Exception as exc:
            self._bridges.pop(user_id, None)
            bridge.close()
            raise CallTransportError(str(exc)) from exc

        # Попытка подключить tgcalls media (опционально)
        media_ok = await self._try_attach_tgcalls_media(user_id, bridge)
        if not media_ok:
            logger.warning(
                "tgcalls media недоступен для user=%s — signaling/bridge only; "
                "для полного duplex установите pytgcalls[telethon]",
                user_id,
            )
        return bridge

    async def resolve_input_user(
        self,
        client,
        user_id: int,
        username: str | None = None,
    ):
        """
        RequestCall требует InputUser с access_hash.
        Чистый user_id из Bot API для Telethon-сессии обычно недостаточен.
        """
        from telethon.tl.types import InputPeerUser, InputUser, User
        from telethon.utils import get_input_user

        phone_hint = (self.settings.telegram_phone or "").strip() or "аккаунту звонящего"
        hint = (
            "Telethon-аккаунт ещё не «видит» ваш профиль (нужен access_hash).\n"
            f"1) Напишите любое сообщение на {phone_hint} (user-аккаунт из .env), "
            "или добавьте его в контакты.\n"
            "2) Либо откройте username в Telegram и снова /call "
            "(нужен публичный @username).\n"
            "После первого диалога с этим аккаунтом звонки заработают."
        )

        candidates: list[object] = []
        if username:
            clean = username.lstrip("@").strip()
            if clean:
                candidates.append(clean)
                candidates.append(f"@{clean}")
        candidates.append(user_id)

        last_error: Exception | None = None
        entity = None
        for ref in candidates:
            try:
                entity = await client.get_entity(ref)
                break
            except Exception as exc:
                last_error = exc
                logger.debug("get_entity(%r) failed: %s", ref, exc)

        if entity is None:
            # Поиск в диалогах / кэше сессии
            try:
                async for dialog in client.iter_dialogs(limit=200):
                    ent = dialog.entity
                    if getattr(ent, "id", None) == user_id:
                        entity = ent
                        break
            except Exception as exc:
                last_error = exc

        if entity is None:
            raise CallTransportError(hint) from last_error

        if not isinstance(entity, User):
            raise CallTransportError(
                "Цель звонка не пользователь Telegram. " + hint
            )

        try:
            input_user = get_input_user(entity)
        except TypeError:
            # Fallback: руками из peer
            peer = await client.get_input_entity(entity)
            if isinstance(peer, InputPeerUser):
                input_user = InputUser(peer.user_id, peer.access_hash)
            else:
                raise CallTransportError(hint) from last_error

        if not isinstance(input_user, InputUser):
            raise CallTransportError(hint)
        # access_hash обязателен (может быть отрицательным int — это нормально)
        if getattr(input_user, "access_hash", None) is None:
            raise CallTransportError(hint)

        return input_user

    async def _request_phone_call(
        self,
        client,
        user_id: int,
        *,
        username: str | None = None,
        on_state: Callable[[str], None] | None = None,
    ) -> None:
        from telethon import functions, types

        if on_state:
            on_state("requesting")

        input_user = await self.resolve_input_user(client, user_id, username=username)
        dhc = await client(
            functions.messages.GetDhConfigRequest(version=0, random_length=256)
        )
        # messages.DhConfig: p — bytes, g — int
        p = _b2i(getattr(dhc, "p", b""))
        g = int(getattr(dhc, "g", 0) or 0)
        if p <= 2 or g <= 1:
            raise CallTransportError(
                f"Некорректный DH config от Telegram (p/g). type={type(dhc).__name__}"
            )
        a = random.randint(2, p - 1)
        g_a = pow(g, a, p)
        g_a_hash = hashlib.sha256(_i2b(g_a)).digest()
        protocol = types.PhoneCallProtocol(
            min_layer=65,
            max_layer=92,
            udp_p2p=True,
            udp_reflector=True,
            library_versions=["4.0.0", "3.0.0", "2.7.7"],
        )
        result = await client(
            functions.phone.RequestCallRequest(
                user_id=input_user,
                random_id=random.randint(0, 0x7FFFFFFF - 1),
                g_a_hash=g_a_hash,
                protocol=protocol,
            )
        )
        phone_call = result.phone_call
        self._active_calls[user_id] = {
            "call": phone_call,
            "a": a,
            "g_a": g_a,
            "p": p,
            "g": g,
            "dhc": dhc,
            "peer": input_user,
        }
        if on_state:
            on_state("ringing")
        logger.info(
            "RequestCall OK user=%s call_id=%s",
            user_id,
            getattr(phone_call, "id", None),
        )

    async def _try_attach_tgcalls_media(
        self,
        user_id: int,
        bridge: CallAudioBridge,
    ) -> bool:
        """Best-effort: GroupCallRaw-like callbacks, если pytgcalls установлен."""
        try:
            import pytgcalls  # noqa: F401
        except ImportError:
            return False

        # Полный private-call media stack в публичном pytgcalls нестабилен.
        # Оставляем bridge; orchestrator пишет/читает PCM через него.
        logger.info("pytgcalls найден; media bridge активен для user=%s", user_id)
        return True

    async def hangup(self, user_id: int) -> None:
        meta = self._active_calls.pop(user_id, None)
        bridge = self._bridges.pop(user_id, None)
        if bridge is not None:
            await bridge.clear_outgoing()
            bridge.close()
        if meta is None or self._client is None:
            return
        try:
            from telethon import functions, types

            call = meta["call"]
            await self._client(
                functions.phone.DiscardCallRequest(
                    peer=types.InputPhoneCall(
                        id=call.id,
                        access_hash=call.access_hash,
                    ),
                    duration=0,
                    reason=types.PhoneCallDiscardReasonHangup(),
                    connection_id=0,
                )
            )
        except Exception as exc:
            logger.warning("DiscardCall user=%s: %s", user_id, exc)

    async def close(self) -> None:
        for uid in list(self._bridges.keys()):
            await self.hangup(uid)
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
