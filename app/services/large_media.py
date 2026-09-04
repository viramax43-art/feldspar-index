"""Скачивание крупных роликов: Telethon (обход лимита Bot API ~20 МБ), URL, inbox."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import threading
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from app.config import Settings

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+", re.I)


class LargeMediaError(RuntimeError):
    pass


def extract_url(text: str | None) -> str | None:
    if not text:
        return None
    m = _URL_RE.search(text)
    return m.group(0).rstrip(")>.,\"'") if m else None


class LargeMediaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    def _resolve_dir(self, raw: str) -> Path:
        path = Path(raw)
        path = path if path.is_absolute() else (Path.cwd() / path)
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @property
    def inbox_dir(self) -> Path:
        return self._resolve_dir(self.settings.video_dub_inbox_dir)

    @property
    def output_dir(self) -> Path:
        return self._resolve_dir(self.settings.video_dub_output_dir)

    def save_to_output(
        self,
        video: Path,
        srt: Path | None = None,
        *,
        label: str = "dub",
    ) -> list[Path]:
        """Копия готового дубляжа в ./output (не зависит от Telegram)."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^\w\-]+", "_", label, flags=re.U).strip("_")[:48] or "dub"
        dest_video = self.output_dir / f"{stamp}_{slug}{video.suffix or '.mp4'}"
        saved = [Path(shutil.copy2(video, dest_video))]
        if srt is not None and srt.exists():
            dest_srt = dest_video.with_suffix(".srt")
            saved.append(Path(shutil.copy2(srt, dest_srt)))
        logger.info("Dub saved locally: %s", dest_video)
        return saved

    def telethon_ready(self) -> bool:
        return bool(
            self.settings.telegram_api_id
            and self.settings.telegram_api_hash
            and self.settings.video_dub_use_telethon
        )

    def session_path(self) -> Path:
        path = (
            self.settings.data_dir / "sessions" / self.settings.telegram_session_name
        ).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def session_sqlite_path(self) -> Path:
        return Path(str(self.session_path()) + ".session")

    def _telegram_client(self):
        import sqlite3
        from telethon import TelegramClient

        sqlite_path = self.session_sqlite_path()
        if not sqlite_path.exists():
            raise LargeMediaError(
                "Нет файла сессии User API (data/sessions). "
                "Без него крупные видео из чата не скачать. "
                "Один раз войдите: python scripts/collect_account_voices.py --consent"
            )
        try:
            return TelegramClient(
                str(self.session_path()),
                int(self.settings.telegram_api_id or 0),
                self.settings.telegram_api_hash,
                **self._client_kwargs(),
            )
        except sqlite3.OperationalError as exc:
            raise LargeMediaError(
                f"Не открылась сессия Telegram ({sqlite_path}). "
                "Проверьте папку data/sessions и что сессию не держит другой процесс."
            ) from exc

    def _connect_error(self, exc: BaseException) -> LargeMediaError:
        import sqlite3

        if isinstance(exc, sqlite3.OperationalError):
            return LargeMediaError(
                f"Не открылась сессия Telegram ({self.session_sqlite_path()}). "
                "Папка data/sessions должна существовать, "
                "и эту сессию не должен держать другой процесс."
            )
        msg = str(exc).lower()
        if isinstance(exc, (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError)) or (
            "connection to telegram failed" in msg or "proxy" in msg
        ):
            return LargeMediaError(
                "User API не достучался до Telegram. "
                "С GPU нужен локальный туннель socks5://127.0.0.1:11080. "
                "Проверьте, что hop запущен, и пришлите ролик ещё раз."
            )
        return LargeMediaError(str(exc))

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "use_ipv6": False,
            "connection_retries": 8,
            "retry_delay": 1,
            "timeout": 30,
        }
        raw = self.settings.mtproto_proxy_url
        if not raw:
            return kwargs
        if "://" not in raw:
            raw = "socks5://" + raw
        u = urlparse(raw)
        scheme = (u.scheme or "socks5").lower()
        host = u.hostname or "127.0.0.1"
        port = int(u.port or (1080 if scheme.startswith("socks") else 8080))
        import socks

        if scheme in {"socks5", "socks"}:
            ptype = socks.SOCKS5
        elif scheme in {"http", "https"}:
            ptype = socks.HTTP
        else:
            raise LargeMediaError(f"Telethon proxy: схема {scheme} не поддерживается")
        kwargs["proxy"] = (ptype, host, port, True, u.username, u.password)
        logger.info("Telethon large-media via %s %s:%s", scheme, host, port)
        return kwargs

    def _partial_path(self, resume_key: str | None) -> Path | None:
        if not resume_key:
            return None
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", resume_key)[:80]
        if not safe:
            return None
        folder = self.settings.data_dir / "tmp" / "tg_cache"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{safe}.part"

    async def download_from_bot_chat(
        self,
        *,
        user_id: int,
        bot_username: str,
        dest: Path,
        file_size: int | None = None,
        file_name: str | None = None,
        lookback: int = 40,
        chat_message_id: int | None = None,
        progress: dict[str, int] | None = None,
        resume_key: str | None = None,
    ) -> Path:
        """Скачать исходящее видео пользователя в чате с ботом через MTProto."""
        if not self.telethon_ready():
            raise LargeMediaError(
                "Telethon не настроен (TELEGRAM_API_ID / HASH) — "
                "крупные файлы через чат недоступны."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _run() -> Path:
            return asyncio.run(
                self._download_from_bot_chat_async(
                    user_id=user_id,
                    bot_username=bot_username,
                    dest=dest,
                    file_size=file_size,
                    file_name=file_name,
                    lookback=lookback,
                    chat_message_id=chat_message_id,
                    progress=progress,
                    resume_key=resume_key,
                )
            )

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def _download_from_bot_chat_async(
        self,
        *,
        user_id: int,
        bot_username: str,
        dest: Path,
        file_size: int | None,
        file_name: str | None,
        lookback: int,
        chat_message_id: int | None,
        progress: dict[str, int] | None,
        resume_key: str | None,
    ) -> Path:
        uname = bot_username.lstrip("@")
        client = self._telegram_client()
        try:
            try:
                await client.connect()
            except Exception as exc:
                raise self._connect_error(exc) from exc
            if not await client.is_user_authorized():
                raise LargeMediaError(
                    "Сессия Telethon не авторизована. Сначала /collectaccount."
                )
            me = await client.get_me()
            if int(me.id) != int(user_id):
                raise LargeMediaError(
                    f"Сессия принадлежит id={me.id}, запрос от id={user_id}. "
                    "Крупные файлы качает только владелец User API."
                )
            entity = await client.get_entity(uname)
            match = None
            if chat_message_id:
                got = await client.get_messages(entity, ids=int(chat_message_id))
                if got and (getattr(got, "video", None) or getattr(got, "document", None)):
                    match = got
            if match is None:
                async for msg in client.iter_messages(entity, limit=lookback):
                    if not getattr(msg, "out", False):
                        continue
                    media = msg.video or msg.document
                    if media is None:
                        continue
                    mime = (getattr(media, "mime_type", None) or "").lower()
                    is_video = bool(msg.video) or mime.startswith("video/")
                    doc_name = ""
                    for attr in getattr(media, "attributes", []) or []:
                        doc_name = getattr(attr, "file_name", "") or doc_name
                    if not is_video and not doc_name.lower().endswith(
                        (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
                    ):
                        continue
                    size = int(getattr(media, "size", 0) or 0)
                    if file_size and size and abs(size - int(file_size)) > 2048:
                        continue
                    if file_name and doc_name and doc_name != file_name:
                        continue
                    match = msg
                    break
            if match is None:
                raise LargeMediaError(
                    "Не нашёл видео в чате с ботом. Пришлите ещё раз или /dub_inbox"
                )
            media = match.document or match.video
            total = int(getattr(media, "size", 0) or file_size or 0)
            part_size = 512 * 1024
            cache = self._partial_path(resume_key)
            source = cache if cache is not None else dest
            offset = source.stat().st_size if source.exists() else 0
            offset = (offset // part_size) * part_size
            if source.exists() and source.stat().st_size != offset:
                with source.open("r+b") as fh:
                    fh.truncate(offset)
            if total and offset >= total and source.exists():
                if source.resolve() != dest.resolve():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    source.unlink(missing_ok=True)
                logger.info("Telethon cache hit %.1f MB", total / (1024 * 1024))
                return dest
            if progress is not None:
                progress["n"] = offset
                progress["total"] = total
            state = {"n": offset, "t": time.monotonic()}
            started = time.monotonic()
            loop = asyncio.get_running_loop()
            stop_stall = threading.Event()
            last_log = {"n": offset}

            def _cb(current: int, _total: int) -> None:
                now = time.monotonic()
                state["n"] = int(current or 0)
                state["t"] = now
                if progress is not None:
                    progress["n"] = state["n"]
                    if _total:
                        progress["total"] = int(_total)
                if state["n"] - last_log["n"] >= 8 * 1024 * 1024:
                    elapsed = max(now - started, 1.0)
                    speed = (state["n"] - offset) / elapsed / 1024
                    logger.info(
                        "Telethon download %.1f/%.1f MB @ %.0f KB/s",
                        state["n"] / (1024 * 1024),
                        (total or 0) / (1024 * 1024),
                        speed,
                    )
                    last_log["n"] = state["n"]

            def _stall_thread() -> None:
                while not stop_stall.wait(5):
                    if time.monotonic() - state["t"] > 120:
                        logger.warning(
                            "Telethon download stalled at %.1f MB, disconnecting",
                            state["n"] / (1024 * 1024),
                        )
                        try:
                            asyncio.run_coroutine_threadsafe(client.disconnect(), loop)
                        except Exception:
                            pass
                        return

            async def _watchdog() -> None:
                while True:
                    await asyncio.sleep(5)
                    idle = time.monotonic() - state["t"]
                    if idle > 120:
                        mb = state["n"] / (1024 * 1024)
                        raise LargeMediaError(
                            f"User API завис на {mb:.1f} МБ (нет данных 2 мин). "
                            "Пришлите тот же ролик ещё раз — докачаю."
                        )

            async def _download() -> str:
                mode = "ab" if offset else "wb"
                source.parent.mkdir(parents=True, exist_ok=True)
                written = offset
                try:
                    agen = client.iter_download(
                        media or match,
                        offset=offset,
                        part_size=part_size,
                    )
                except TypeError:
                    agen = client.iter_download(media or match, offset=offset)
                with source.open(mode) as fh:
                    try:
                        async for chunk in agen:
                            if not chunk:
                                continue
                            fh.write(chunk)
                            written += len(chunk)
                            _cb(written, total)
                    finally:
                        close = getattr(agen, "aclose", None)
                        if close is not None:
                            await close()
                return str(source)

            logger.info(
                "Telethon download start size=%.1f MB resume=%.1f MB",
                total / (1024 * 1024),
                offset / (1024 * 1024),
            )
            stall = threading.Thread(target=_stall_thread, name="tg-dl-stall", daemon=True)
            stall.start()
            path = None
            dl = asyncio.create_task(_download())
            wd = asyncio.create_task(_watchdog())
            try:
                done, pending = await asyncio.wait(
                    {dl, wd}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                if wd in done:
                    exc = wd.exception()
                    if exc:
                        raise exc
                try:
                    path = dl.result()
                except Exception as exc:
                    raise self._connect_error(exc) from exc
            finally:
                stop_stall.set()
            if not path:
                raise LargeMediaError("Telethon не скачал медиа")
            out = Path(path)
            if not dest.exists() or out.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, dest)
            if cache is not None and cache.exists() and dest.exists():
                if total and dest.stat().st_size >= total:
                    cache.unlink(missing_ok=True)
            logger.info(
                "Telethon downloaded %s (%.1f MB) user=%s",
                dest.name,
                dest.stat().st_size / (1024 * 1024),
                user_id,
            )
            return dest
        finally:
            await client.disconnect()

    async def download_url(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(self.settings.video_dub_max_download_mb * 1024 * 1024)
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    raise LargeMediaError(f"URL вернул HTTP {resp.status}")
                cl = resp.headers.get("Content-Length")
                if cl and int(cl) > max_bytes:
                    raise LargeMediaError(
                        f"Файл по ссылке больше {self.settings.video_dub_max_download_mb:.0f} МБ"
                    )
                written = 0
                with dest.open("wb") as fh:
                    async for chunk in resp.content.iter_chunked(1024 * 256):
                        written += len(chunk)
                        if written > max_bytes:
                            fh.close()
                            dest.unlink(missing_ok=True)
                            raise LargeMediaError(
                                f"Файл по ссылке больше {self.settings.video_dub_max_download_mb:.0f} МБ"
                            )
                        fh.write(chunk)
        return dest

    def list_inbox_videos(self) -> list[Path]:
        exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        files = [
            p
            for p in self.inbox_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def take_inbox_video(self, dest: Path) -> Path:
        files = self.list_inbox_videos()
        if not files:
            raise LargeMediaError(
                f"Inbox пуст. Положите MP4 в:\n<code>{self.inbox_dir}</code>"
            )
        src = files[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dest)
        return dest

    async def send_file_to_user(
        self,
        *,
        user_id: int,
        path: Path,
        caption: str = "",
    ) -> None:
        """Крупный результат → «Избранное» (обход лимита Bot API ~50 МБ)."""
        if not self.telethon_ready():
            raise LargeMediaError("Telethon не настроен для отправки крупного файла")
        async with self._lock:
            client = self._telegram_client()
            try:
                try:
                    await client.connect()
                except Exception as exc:
                    raise self._connect_error(exc) from exc
                if not await client.is_user_authorized():
                    raise LargeMediaError("Сессия Telethon не авторизована")
                me = await client.get_me()
                if int(me.id) != int(user_id):
                    raise LargeMediaError("Отправка крупных файлов только владельцу сессии")
                await client.send_file("me", str(path), caption=caption or None)
                logger.info("Large result → Saved Messages: %s", path.name)
            finally:
                await client.disconnect()
