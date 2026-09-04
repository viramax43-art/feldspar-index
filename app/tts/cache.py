"""RAM + disk PCM cache for TTS phrases."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NORMALIZER_VERSION = "fish-no-cross-video-v3"


class AudioCache:
    def __init__(
        self,
        directory: Path,
        enabled: bool = True,
        ram_max_entries: int = 128,
    ) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.ram_max_entries = ram_max_entries
        self._ram: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_key(
        text: str,
        engine: str,
        speaker_or_profile: str,
        sample_rate: int,
        normalizer_version: str = NORMALIZER_VERSION,
        extra: str = "",
    ) -> str:
        payload = {
            "text": text,
            "engine": engine,
            "speaker": speaker_or_profile,
            "sample_rate": sample_rate,
            "normalizer_version": normalizer_version,
            "extra": extra,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        with self._lock:
            if key in self._ram:
                self._ram.move_to_end(key)
                self.hits += 1
                return self._ram[key].copy()
            path = self.directory / f"{key}.npy"
            if path.exists():
                try:
                    audio = np.load(path).astype(np.float32)
                    self._ram[key] = audio
                    self._ram.move_to_end(key)
                    while len(self._ram) > self.ram_max_entries:
                        self._ram.popitem(last=False)
                    self.hits += 1
                    return audio.copy()
                except Exception as exc:
                    logger.warning("Ошибка чтения кэша %s: %s", path, exc)
            self.misses += 1
            return None

    def put(self, key: str, audio: np.ndarray) -> None:
        if not self.enabled or audio.size == 0:
            return
        arr = np.asarray(audio, dtype=np.float32)
        with self._lock:
            self._ram[key] = arr
            self._ram.move_to_end(key)
            while len(self._ram) > self.ram_max_entries:
                self._ram.popitem(last=False)
            path = self.directory / f"{key}.npy"
            tmp = path.with_suffix(".tmp.npy")
            try:
                np.save(tmp, arr)
                tmp.replace(path)
            except Exception as exc:
                logger.warning("Ошибка записи кэша %s: %s", path, exc)
                tmp.unlink(missing_ok=True)

    def clear(self) -> None:
        """Drop RAM + disk entries (call after each finished video dub)."""
        with self._lock:
            self._ram.clear()
            self.hits = 0
            self.misses = 0
            if not self.enabled:
                return
            try:
                for path in self.directory.glob("*.npy"):
                    path.unlink(missing_ok=True)
                for path in self.directory.glob("*.tmp.npy"):
                    path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Ошибка очистки TTS-кэша %s: %s", self.directory, exc)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "ram_entries": len(self._ram)}
