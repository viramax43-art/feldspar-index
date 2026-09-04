"""VoxCPM2 TTS: multilingual clone, ~8 GB VRAM, 48 kHz."""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)

# Одна реплика: склейка нескольких клипов путает тембр. Длинный английский
# референс часто «просачивает» исходный язык в новые фразы.
_MAX_REF_SEC = 8.0
_STYLE_PARENS = re.compile(r"[\(\（][^)\）]{0,120}[\)\）]")
_SPEAK_HINT = {
    "ru": "speak Russian",
    "en": "speak English",
    "de": "speak German",
    "fr": "speak French",
    "ja": "speak Japanese",
    "ko": "speak Korean",
}


def wrap_voxcpm_text(
    text: str, language: str | None, extra: str | None = None
) -> str:
    """Скобки VoxCPM2 читает как стиль; чужие (...) меняют голос/язык."""
    cleaned = _STYLE_PARENS.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    parts: list[str] = []
    lang = _SPEAK_HINT.get((language or "").lower().strip())
    if lang:
        parts.append(lang)
    if extra:
        parts.extend(bit.strip() for bit in extra.split(",") if bit.strip())
    # unique
    seen: set[str] = set()
    ordered: list[str] = []
    for bit in parts:
        key = bit.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(bit)
    if ordered:
        return "(" + ", ".join(ordered) + ")" + cleaned
    return cleaned


class VoxCPMEngine(TTSEngine):
    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        device: str = "cuda",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        timeout_sec: float = 180.0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.timeout_sec = timeout_sec
        self._model = None
        self._sample_rate = 48000
        self._lock = asyncio.Lock()
        self._load_lock = threading.Lock()
        self._ref_cache: dict[str, Path] = {}

    @property
    def name(self) -> str:
        return "voxcpm2"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @staticmethod
    def _patch_missing_bin_to_safetensors() -> None:
        """Hub snapshot of VoxCPM2 ships model.safetensors; old voxcpm still opens pytorch_model.bin."""
        if getattr(torch.load, "_voxcpm_st_patch", False):
            return
        orig = torch.load

        def _load(f, *args, **kwargs):  # type: ignore[no-untyped-def]
            path: str | None = f if isinstance(f, str) else None
            if path is None and isinstance(f, Path):
                path = str(f)
            if (
                isinstance(path, str)
                and path.endswith("pytorch_model.bin")
                and not Path(path).is_file()
            ):
                alt = Path(path).with_name("model.safetensors")
                if alt.is_file():
                    from safetensors.torch import load_file

                    logger.info("VoxCPM weights via %s", alt.name)
                    weights = load_file(str(alt))
                    if isinstance(weights, dict) and "state_dict" in weights:
                        return weights
                    return {"state_dict": weights}
            return orig(f, *args, **kwargs)

        _load._voxcpm_st_patch = True  # type: ignore[attr-defined]
        torch.load = _load  # type: ignore[assignment]

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            from voxcpm import VoxCPM

            self._patch_missing_bin_to_safetensors()
            kwargs: dict[str, Any] = {"load_denoiser": False, "optimize": False}
            try:
                self._model = VoxCPM.from_pretrained(self.model_id, **kwargs)
            except TypeError:
                kwargs.pop("optimize", None)
                try:
                    self._model = VoxCPM.from_pretrained(self.model_id, **kwargs)
                except TypeError:
                    self._model = VoxCPM.from_pretrained(self.model_id)
            sr = getattr(getattr(self._model, "tts_model", None), "sample_rate", None)
            if sr:
                self._sample_rate = int(sr)
            logger.info(
                "VoxCPM2 loaded (%s, sr=%d). Denoiser off to fit 8 GB VRAM.",
                self.model_id,
                self._sample_rate,
            )

    def warmup(self) -> None:
        if self._model is None:
            return
        try:
            _ = self._model.generate(
                text="Hello.",
                cfg_value=self.cfg_value,
                inference_timesteps=max(4, self.inference_timesteps // 2),
            )
            logger.info("VoxCPM2 warmup OK")
        except Exception as exc:
            logger.warning("VoxCPM2 warmup skipped: %s", exc)

    def _concat_refs(self, speaker_wavs: list[Path]) -> Path | None:
        existing = [p for p in speaker_wavs if p.exists()]
        if not existing:
            return None
        best = max(existing, key=lambda p: p.stat().st_size)
        key = f"{best.resolve()}:{best.stat().st_mtime_ns}:{best.stat().st_size}:one8"
        cached = self._ref_cache.get(key)
        if cached is not None and cached.exists():
            return cached
        audio, sr = sf.read(str(best), always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        sr_out = 16000
        if int(sr) != sr_out:
            import librosa

            audio = librosa.resample(audio, orig_sr=int(sr), target_sr=sr_out)
        max_n = int(_MAX_REF_SEC * sr_out)
        if audio.size > max_n:
            audio = audio[:max_n]
        tmp = Path(tempfile.gettempdir()) / "voxcpm_clone_ref.wav"
        sf.write(str(tmp), audio, sr_out, subtype="PCM_16")
        self._ref_cache[key] = tmp
        return tmp

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        extra = None
        if isinstance(params, dict):
            extra = params.get("vox_style") or None
        spoken = wrap_voxcpm_text(text, language, extra=extra)
        if not spoken:
            return np.zeros(1, dtype=np.float32), conditioning_cache
        if self._model is None:
            self.load()
        assert self._model is not None
        ref = self._concat_refs(speaker_wavs)
        kwargs: dict[str, Any] = {
            "text": spoken,
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
        }
        if ref is not None:
            kwargs["reference_wav_path"] = str(ref)
        with torch.inference_mode():
            wav = self._model.generate(**kwargs)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return audio, conditioning_cache

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
