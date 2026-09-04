"""Управление голосовыми профилями пользователей."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from app.audio import safe_user_path
from app.audio.preprocess import merge_references
from app.config import Settings
from app.database import Database

logger = logging.getLogger(__name__)


class VoiceProfileService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    def user_dir(self, user_id: int) -> Path:
        path = safe_user_path(self.settings.users_dir, user_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def references_dir(self, user_id: int) -> Path:
        path = self.user_dir(user_id) / "references"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_dir(self, user_id: int) -> Path:
        path = self.user_dir(user_id) / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def next_reference_path(self, user_id: int, index: int) -> Path:
        return self.references_dir(user_id) / f"ref_{index:03d}.wav"

    def merged_reference_path(self, user_id: int) -> Path:
        return self.user_dir(user_id) / "merged_reference.wav"

    def conditioning_cache_path(self, user_id: int) -> Path:
        return self.cache_dir(user_id) / "conditioning.pt"

    async def get_reference_paths(self, user_id: int, only_accepted: bool = True) -> list[Path]:
        """Пути для XTTS conditioning — ограниченный топ клипов (быстро и стабильно)."""
        if only_accepted:
            refs = await self.select_conditioning_references(user_id)
        else:
            refs = await self.db.list_voice_references(user_id)
        paths: list[Path] = []
        for ref in refs:
            quality = ref.get("quality", {})
            if only_accepted and not quality.get("accepted"):
                continue
            path = self.references_dir(user_id) / ref["file_name"]
            if path.exists():
                paths.append(path)
        return paths

    async def select_conditioning_references(self, user_id: int) -> list[dict]:
        """Подмножество профиля для live-клонирования XTTS."""
        selected = await self.select_profile_references(user_id)
        max_files = self.settings.xtts_conditioning_max_files
        max_sec = self.settings.xtts_conditioning_max_seconds
        out: list[dict] = []
        total = 0.0
        for ref in selected:
            if len(out) >= max_files:
                break
            duration = float(ref.get("duration_sec", 0))
            if out and total + duration > max_sec:
                continue
            out.append(ref)
            total += duration
            if total >= max_sec and len(out) >= min(20, max_files):
                break
        return out

    async def select_profile_references(self, user_id: int) -> list[dict]:
        """Полный пул лучших референсов для профиля / fine-tune (до 500 файлов).

        XTTS лучше клонирует голос на разнообразии клипов 4–25 с,
        чем на 3–4 длинных файлах.
        """
        refs = await self.db.list_voice_references(user_id)
        accepted = [r for r in refs if (r.get("quality") or {}).get("accepted")]

        def _rank(r: dict) -> tuple[float, float]:
            quality = r.get("quality") or {}
            score = float(quality.get("score", 0))
            dur = float(r.get("duration_sec", 0))
            if 4.0 <= dur <= 25.0:
                length_bonus = 25.0
            elif 3.0 <= dur <= 40.0:
                length_bonus = 10.0
            elif dur > 60.0:
                length_bonus = -15.0
            else:
                length_bonus = 0.0
            return (score + length_bonus, -abs(dur - 12.0))

        accepted.sort(key=_rank, reverse=True)

        selected: list[dict] = []
        total = 0.0
        max_sec = self.settings.profile_max_seconds
        max_files = self.settings.profile_max_files
        for ref in accepted:
            path = self.references_dir(user_id) / ref["file_name"]
            if not path.exists():
                continue
            if len(selected) >= max_files:
                break
            duration = float(ref.get("duration_sec", 0))
            if duration <= 0:
                continue
            if selected and total + duration > max_sec:
                continue
            selected.append(ref)
            total += duration
            if len(selected) >= max_files:
                break
            if total >= max_sec and len(selected) >= min(50, max_files):
                break
        return selected

    async def build_profile(self, user_id: int) -> dict[str, Any]:
        selected = await self.select_profile_references(user_id)
        paths = [
            self.references_dir(user_id) / r["file_name"]
            for r in selected
            if (self.references_dir(user_id) / r["file_name"]).exists()
        ]
        if not paths:
            raise ValueError("Нет принятых референсов для создания профиля")

        all_accepted = [
            r
            for r in await self.db.list_voice_references(user_id)
            if (r.get("quality") or {}).get("accepted")
        ]
        min_files = self.settings.profile_min_files
        if len(all_accepted) < min_files:
            logger.warning(
                "В пуле только %s принятых голосовых (минимум для обучения: %s). "
                "Запустите: python scripts/collect_account_voices.py --build-profile",
                len(all_accepted),
                min_files,
            )

        merged = self.merged_reference_path(user_id)
        merge_references(
            paths,
            merged,
            sample_rate=self.settings.reference_sample_rate,
        )

        total_duration = sum(float(r["duration_sec"]) for r in selected)
        conditioning = await self.select_conditioning_references(user_id)
        meta = {
            "reference_count": len(paths),
            "total_duration_sec": total_duration,
            "conditioning_count": len(conditioning),
            "conditioning_sec": sum(float(r["duration_sec"]) for r in conditioning),
            "pool_accepted_count": len(all_accepted),
            "pool_accepted_sec": sum(float(r["duration_sec"]) for r in all_accepted),
            "merged_reference": merged.name,
            "selected_files": [r["file_name"] for r in selected],
            "conditioning_files": [r["file_name"] for r in conditioning],
            "profile_min_files": min_files,
            "ready_for_finetune": len(all_accepted) >= min_files,
        }
        meta_path = self.user_dir(user_id) / "profile.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        # Старые latents XTTS больше не соответствуют новым референсам
        self.conditioning_cache_path(user_id).unlink(missing_ok=True)
        await self.db.set_voice_profile(user_id, True)
        logger.info(
            "Профиль пользователя %s создан (%s из %s референсов, %.0f с)",
            user_id,
            len(paths),
            len(all_accepted),
            total_duration,
        )
        return meta

    async def delete_profile(self, user_id: int) -> None:
        await self.db.delete_user_data(user_id)
        user_path = self.user_dir(user_id)
        if user_path.exists():
            shutil.rmtree(user_path, ignore_errors=True)
        logger.info("Профиль пользователя %s удалён", user_id)

    def assert_user_access(self, requester_id: int, owner_id: int) -> None:
        if requester_id != owner_id:
            raise PermissionError("Доступ к профилю другого пользователя запрещён")
