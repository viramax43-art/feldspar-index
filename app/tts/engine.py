"""TTS engine abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TTSEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    def warmup(self) -> None:
        """Короткий прогрев после load(); по умолчанию — no-op."""

    @abstractmethod
    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[Any, tuple[Any, Any] | None]:
        ...

    @abstractmethod
    def clear_gpu_cache(self) -> None:
        ...
