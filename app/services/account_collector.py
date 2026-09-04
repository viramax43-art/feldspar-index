"""Сбор собственных голосовых сообщений из всех чатов аккаунта (Telethon User API)."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.audio.preprocess import preprocess_telegram_voice, warm_up_vad
from app.audio.quality import evaluate_reference
from app.config import Settings
from app.database import Database
from app.services.voice_profile import VoiceProfileService

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], Awaitable[None]]]


@dataclass
class CollectionResult:
    scanned_dialogs: int = 0
    found_voices: int = 0
    skipped_duplicates: int = 0
    skipped_duration: int = 0
    downloaded: int = 0
    accepted: int = 0
    rejected: int = 0
    total_accepted_sec: float = 0.0
    workers: int = 1
    errors: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "📥 Сбор голосовых из аккаунта завершён.",
            f"Воркеров: {self.workers}",
            f"Чатов просмотрено: {self.scanned_dialogs}",
            f"Голосовых найдено (ваши): {self.found_voices}",
            f"Пропущено (уже были): {self.skipped_duplicates}",
            f"Пропущено (длина): {self.skipped_duration}",
            f"Скачано: {self.downloaded}",
            f"Принято: {self.accepted}",
            f"Отклонено: {self.rejected}",
            f"Суммарная чистая речь: {self.total_accepted_sec:.0f} с",
        ]
        if self.reports:
            lines.append("")
            lines.extend(self.reports[:20])
            if len(self.reports) > 20:
                lines.append(f"... и ещё {len(self.reports) - 20} отчётов")
        if self.errors:
            lines.append("")
            lines.append("Ошибки:")
            lines.extend(f"• {e}" for e in self.errors[:8])
        return "\n".join(lines)


@dataclass
class _VoiceCandidate:
    message: Any
    dialog_id: Any
    dialog_name: str
    key: str


class AccountVoiceCollector:
    """
    Собирает ТОЛЬКО исходящие голосовые сообщения владельца сессии.

    Параллельность: asyncio + пул воркеров (скачивание) и ThreadPoolExecutor (VAD/preprocess).
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        profile_service: VoiceProfileService,
    ) -> None:
        self.settings = settings
        self.db = db
        self.profile_service = profile_service

    def _require_api_credentials(self) -> None:
        if not self.settings.telegram_api_id or not self.settings.telegram_api_hash:
            raise RuntimeError(
                "Для сбора голоса из аккаунта укажите TELEGRAM_API_ID и TELEGRAM_API_HASH "
                "в .env (получить на https://my.telegram.org)."
            )

    def session_path(self) -> Path:
        return self.settings.data_dir / "sessions" / self.settings.telegram_session_name

    async def _known_message_keys(self, user_id: int) -> set[str]:
        refs = await self.db.list_voice_references(user_id)
        keys: set[str] = set()
        for ref in refs:
            quality = ref.get("quality") or {}
            dialog = quality.get("dialog_id")
            msg_id = quality.get("message_id")
            if dialog is not None and msg_id is not None:
                keys.add(f"{dialog}:{msg_id}")
        return keys

    @staticmethod
    def _is_outgoing_voice(message: object, owner_id: int | None = None) -> bool:
        """Только исходящие голосовые владельца сессии — чужие не берём."""
        if not getattr(message, "out", False):
            return False
        if owner_id is not None:
            sender = getattr(message, "sender_id", None)
            if sender is None:
                from_id = getattr(message, "from_id", None)
                sender = getattr(from_id, "user_id", None) if from_id is not None else None
            if sender is not None and int(sender) != int(owner_id):
                return False
        if getattr(message, "voice", None) is not None:
            return True
        document = getattr(message, "document", None)
        if document is None:
            return False
        for attr in getattr(document, "attributes", None) or []:
            if getattr(attr, "voice", False):
                return True
        return False

    @staticmethod
    def _voice_duration(message: object) -> float | None:
        voice = getattr(message, "voice", None)
        if voice is not None and getattr(voice, "duration", None) is not None:
            return float(voice.duration)
        document = getattr(message, "document", None)
        if document is None:
            return None
        for attr in getattr(document, "attributes", None) or []:
            if getattr(attr, "voice", False) and getattr(attr, "duration", None) is not None:
                return float(attr.duration)
        return None

    async def collect(
        self,
        profile_user_id: int | None = None,
        progress: ProgressCallback | None = None,
        limit: int | None = None,
        max_seconds: float | None = None,
        per_dialog: int | None = None,
        messages_per_dialog: int | None = None,
        workers: int | None = None,
    ) -> CollectionResult:
        self._require_api_credentials()
        result = CollectionResult()

        async def notify(text: str) -> None:
            logger.info(text)
            if progress:
                maybe = progress(text)
                if maybe is not None:
                    await maybe

        from telethon import TelegramClient

        session_file = self.session_path()
        session_file.parent.mkdir(parents=True, exist_ok=True)

        client = TelegramClient(
            str(session_file),
            self.settings.telegram_api_id,
            self.settings.telegram_api_hash,
            use_ipv6=False,
            connection_retries=8,
            retry_delay=1,
            timeout=25,
        )

        await client.start(phone=self.settings.telegram_phone or None)
        me = await client.get_me()
        owner_id = me.id
        target_user_id = profile_user_id or owner_id

        if profile_user_id is not None and profile_user_id != owner_id:
            await client.disconnect()
            raise PermissionError(
                f"Сессия Telegram принадлежит user_id={owner_id}, "
                f"а запрос от user_id={profile_user_id}. "
                "Автосбор разрешён только для владельца аккаунта."
            )

        await notify(
            f"Авторизован как {me.first_name} (id={owner_id}). "
            "Ищем только ваши исходящие голосовые..."
        )

        if not await self.db.has_consent(target_user_id):
            await client.disconnect()
            raise PermissionError(
                "Сначала подтвердите согласие (/consent в боте или "
                "--consent в скрипте). Без согласия сбор не выполняется."
            )

        max_total = float(
            max_seconds
            if max_seconds is not None
            else self.settings.account_collect_max_seconds
        )
        max_files = int(
            limit if limit is not None else self.settings.account_collect_limit
        )
        per_dialog_limit = int(
            per_dialog
            if per_dialog is not None
            else self.settings.account_collect_per_dialog
        )
        msg_limit = (
            messages_per_dialog
            if messages_per_dialog is not None
            else self.settings.account_collect_messages_per_dialog
        )
        iter_limit = None if msg_limit <= 0 else msg_limit
        worker_count = max(
            1,
            int(workers if workers is not None else self.settings.account_collect_workers),
        )
        scan_workers = max(1, min(worker_count, self.settings.account_collect_scan_workers))
        result.workers = worker_count

        min_duration = self.settings.account_collect_min_duration
        max_duration = self.settings.account_collect_max_duration

        known = await self._known_message_keys(target_user_id)
        existing = await self.db.count_voice_references(target_user_id)

        existing_refs = await self.db.list_voice_references(target_user_id)
        already_sec = sum(
            float(r["duration_sec"])
            for r in existing_refs
            if (r.get("quality") or {}).get("accepted")
        )
        result.total_accepted_sec = already_sec
        result.accepted = sum(
            1 for r in existing_refs if (r.get("quality") or {}).get("accepted")
        )

        await notify(
            f"Уже в базе: {existing} файлов, {already_sec:.0f} с принятой речи. "
            f"Цель: до {max_files} файлов / {max_total:.0f} с "
            f"(минимум для обучения: {self.settings.profile_min_files}) | "
            f"воркеров загрузки: {worker_count}, сканирования: {scan_workers}"
        )

        min_for_train = self.settings.profile_min_files
        # Секундный лимит не блокирует добор до минимума файлов (500)
        pool_full = result.accepted >= max_files or (
            result.total_accepted_sec >= max_total and result.accepted >= min_for_train
        )
        if pool_full:
            await client.disconnect()
            await notify(
                "Лимит уже достигнут. Увеличьте ACCOUNT_COLLECT_LIMIT / "
                "ACCOUNT_COLLECT_MAX_SECONDS или передайте --limit / --max-seconds."
            )
            return result

        dialogs = await client.get_dialogs()
        result.scanned_dialogs = len(dialogs)
        await notify(f"Диалогов: {len(dialogs)}. Параллельное сканирование...")

        # Запас на отбраковку по качеству
        candidate_cap = max(max_files * 2, max_files + 50)
        candidates: list[_VoiceCandidate] = []
        scan_lock = asyncio.Lock()
        scan_sem = asyncio.Semaphore(scan_workers)

        async def scan_dialog(dialog_idx: int, dialog: Any) -> None:
            nonlocal candidates
            async with scan_sem:
                async with scan_lock:
                    if len(candidates) >= candidate_cap:
                        return

                dialog_id = getattr(dialog, "id", None)
                dialog_name = getattr(dialog, "name", "") or str(dialog_id)
                dialog_taken = 0
                local: list[_VoiceCandidate] = []

                try:
                    async for message in client.iter_messages(
                        dialog.entity,
                        limit=iter_limit,
                    ):
                        async with scan_lock:
                            if len(candidates) + len(local) >= candidate_cap:
                                break
                        if dialog_taken >= per_dialog_limit:
                            break
                        if not self._is_outgoing_voice(message, owner_id=owner_id):
                            continue

                        async with scan_lock:
                            result.found_voices += 1

                        key = f"{dialog_id}:{message.id}"
                        if key in known:
                            async with scan_lock:
                                result.skipped_duplicates += 1
                            continue

                        duration = self._voice_duration(message)
                        if duration is not None and (
                            duration < min_duration or duration > max_duration
                        ):
                            async with scan_lock:
                                result.skipped_duration += 1
                            continue

                        local.append(
                            _VoiceCandidate(
                                message=message,
                                dialog_id=dialog_id,
                                dialog_name=dialog_name,
                                key=key,
                            )
                        )
                        dialog_taken += 1
                except Exception as exc:
                    async with scan_lock:
                        result.errors.append(f"{dialog_name}: {exc}")
                    return

                async with scan_lock:
                    room = candidate_cap - len(candidates)
                    if room > 0:
                        candidates.extend(local[:room])
                    if dialog_idx == 1 or dialog_idx % 10 == 0:
                        await notify(
                            f"[скан {dialog_idx}/{len(dialogs)}] {dialog_name} | "
                            f"найдено {result.found_voices}, "
                            f"в очереди {len(candidates)}"
                        )

        # При scan_workers=1 идём по чатам последовательно — меньше FloodWait
        if scan_workers <= 1:
            for i, d in enumerate(dialogs, start=1):
                try:
                    await scan_dialog(i, d)
                except asyncio.CancelledError:
                    await notify("Сканирование прервано — обрабатываю уже найденное...")
                    break
                async with scan_lock:
                    if len(candidates) >= candidate_cap:
                        break
        else:
            try:
                await asyncio.gather(
                    *[scan_dialog(i, d) for i, d in enumerate(dialogs, start=1)]
                )
            except asyncio.CancelledError:
                await notify("Сканирование прервано — обрабатываю уже найденное...")

        await notify(
            f"Скан завершён: найдено {result.found_voices}, "
            f"к загрузке {len(candidates)}. Параллельная обработка ({worker_count})..."
        )

        if not candidates:
            await client.disconnect()
            await notify(result.summary())
            return result

        state_lock = asyncio.Lock()
        download_sem = asyncio.Semaphore(worker_count)
        index = existing
        stop = False
        processed = 0
        loop = asyncio.get_running_loop()
        # Не больше 4 потоков на VAD — у каждой своя thread-local модель
        vad_workers = max(1, min(worker_count, 4))
        executor = ThreadPoolExecutor(
            max_workers=vad_workers,
            initializer=warm_up_vad,
        )

        def _preprocess_sync(raw: Path, wav: Path) -> dict:
            return preprocess_telegram_voice(
                raw,
                wav,
                sample_rate=self.settings.reference_sample_rate,
                enable_denoise=self.settings.enable_denoise,
            )

        async def process_candidate(cand: _VoiceCandidate) -> None:
            nonlocal index, stop, processed
            async with download_sem:
                async with state_lock:
                    if stop or result.accepted >= max_files or (
                        result.total_accepted_sec >= max_total
                        and result.accepted >= self.settings.profile_min_files
                    ):
                        stop = True
                        return
                    index += 1
                    file_index = index

                refs_dir = self.profile_service.references_dir(target_user_id)
                raw_path = refs_dir / f"account_raw_{file_index:04d}.oga"
                wav_path = self.profile_service.next_reference_path(
                    target_user_id, file_index
                )

                try:
                    await client.download_media(cand.message, file=str(raw_path))
                    async with state_lock:
                        result.downloaded += 1

                    metrics = await loop.run_in_executor(
                        executor,
                        _preprocess_sync,
                        raw_path,
                        wav_path,
                    )
                    quality = evaluate_reference(
                        metrics,
                        min_duration=min_duration,
                        max_duration=max_duration,
                    )
                    await self.db.add_voice_reference(
                        target_user_id,
                        wav_path.name,
                        metrics["duration_sec"],
                        {
                            **quality.to_dict(),
                            "source": "account",
                            "dialog": cand.dialog_name,
                            "dialog_id": cand.dialog_id,
                            "message_id": cand.message.id,
                        },
                    )

                    async with state_lock:
                        known.add(cand.key)
                        result.reports.append(quality.format_message(file_index))
                        if quality.accepted:
                            result.accepted += 1
                            result.total_accepted_sec += float(metrics["duration_sec"])
                            if result.accepted >= max_files or (
                                result.total_accepted_sec >= max_total
                                and result.accepted >= self.settings.profile_min_files
                            ):
                                stop = True
                        else:
                            result.rejected += 1
                            wav_path.unlink(missing_ok=True)
                        processed += 1
                        if processed == 1 or processed % 25 == 0:
                            await notify(
                                f"[обработка {processed}/{len(candidates)}] "
                                f"принято {result.accepted}, "
                                f"{result.total_accepted_sec:.0f} с"
                            )
                except Exception as exc:
                    logger.exception("Ошибка обработки голосового из аккаунта")
                    async with state_lock:
                        result.errors.append(str(exc))
                        result.rejected += 1
                        processed += 1
                finally:
                    raw_path.unlink(missing_ok=True)

        try:
            await asyncio.gather(*(process_candidate(c) for c in candidates))
        except asyncio.CancelledError:
            await notify("Обработка прервана — уже скачанное сохранено в базе.")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            await client.disconnect()

        await notify(result.summary())
        return result
