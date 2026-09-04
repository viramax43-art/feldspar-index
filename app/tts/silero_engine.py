"""Offline Silero TTS engine (fixed speakers, no voice cloning)."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)


class SileroEngine(TTSEngine):
    """
    Быстрый локальный fallback на фиксированных голосах Silero v5_5_ru.
    Не клонирует голос пользователя — speaker_wavs игнорируются.
    """

    def __init__(
        self,
        model_path: Path,
        speaker: str = "xenia",
        sample_rate: int = 24000,
        device: str = "cpu",
        cpu_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.speaker = speaker
        self._sample_rate = sample_rate
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.cpu_threads = cpu_threads
        self._model = None
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "silero"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Silero модель не найдена: {self.model_path}. "
                "Скачайте: python scripts/download_silero.py"
            )
        if self.device == "cpu":
            torch.set_num_threads(max(1, self.cpu_threads))

        logger.info("Загрузка Silero TTS из %s на %s", self.model_path, self.device)
        importer = torch.package.PackageImporter(str(self.model_path))
        self._model = importer.load_pickle("tts_models", "model")
        if hasattr(self._model, "to"):
            self._model.to(torch.device(self.device))
        if hasattr(self._model, "eval"):
            self._model.eval()

        speakers = list(getattr(self._model, "speakers", []) or [])
        if not speakers and hasattr(self._model, "speakers"):
            try:
                speakers = list(self._model.speakers)
            except Exception:
                speakers = []
        if speakers and self.speaker not in speakers:
            raise ValueError(
                f"Голос {self.speaker!r} недоступен. Доступные: {speakers}"
            )
        logger.info(
            "Silero TTS загружен (speaker=%s, speakers=%s)",
            self.speaker,
            speakers[:8] if speakers else "?",
        )

    def warmup(self) -> None:
        self.synthesize_chunk(
            "Здравствуйте.",
            [],
            "ru",
            {},
            None,
        )

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        del speaker_wavs, language, params  # fixed-speaker engine
        if self._model is None:
            self.load()
        assert self._model is not None
        text = (text or "").strip()
        if not text:
            return np.empty(0, dtype=np.float32), conditioning_cache

        kwargs: dict[str, Any] = {
            "text": text,
            "speaker": self.speaker,
            "sample_rate": self._sample_rate,
            "put_accent": True,
            "put_yo": True,
        }
        # Флаги омографов есть в v5.5; на старых пакетах — мягкий fallback
        for flag in ("put_stress_homo", "put_yo_homo"):
            kwargs[flag] = True

        with self._thread_lock, torch.inference_mode():
            try:
                audio = self._model.apply_tts(**kwargs)
            except TypeError:
                kwargs.pop("put_stress_homo", None)
                kwargs.pop("put_yo_homo", None)
                audio = self._model.apply_tts(**kwargs)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        return wav, conditioning_cache

    async def synthesize_chunk_async(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.synthesize_chunk(
                    text,
                    speaker_wavs,
                    language,
                    params,
                    conditioning_cache,
                ),
            )

    def clear_gpu_cache(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
