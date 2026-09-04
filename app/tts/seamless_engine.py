"""SeamlessM4T Medium — speech-to-speech / text-to-speech for <=4GB GPUs."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import numpy as np
import torch

from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)

# Bot reply_lang -> Seamless vocoder / tgt_lang codes
_SEAMLESS_LANG = {
    "ru": "rus",
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "ja": "jpn",
    "ko": "kor",
    "es": "spa",
    "pt": "por",
    "it": "ita",
    "tr": "tur",
    "uk": "ukr",
    "ar": "arb",
    "hi": "hin",
    "zh": "cmn",
}


def seamless_tgt_lang(code: str | None, default: str = "rus") -> str:
    key = (code or "").lower().strip()
    if key in _SEAMLESS_LANG:
        return _SEAMLESS_LANG[key]
    if key in _SEAMLESS_LANG.values():
        return key
    return default


class SeamlessM4TEngine(TTSEngine):
    """Meta SeamlessM4T medium via 🤗 Transformers (fp16 on CUDA)."""

    def __init__(
        self,
        model_id: str = "facebook/hf-seamless-m4t-medium",
        device: str = "cuda",
        speaker_id: int = 0,
    ) -> None:
        self.model_id = model_id
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.speaker_id = int(speaker_id)
        self._sample_rate = 16000
        self._processor = None
        self._model = None
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "seamless_m4t"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, SeamlessM4TModel

        logger.info("Загрузка SeamlessM4T %s на %s (fp16)", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self._model = SeamlessM4TModel.from_pretrained(
                self.model_id,
                dtype=dtype,
            )
        except TypeError:
            self._model = SeamlessM4TModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
        self._model.to(self.device)
        self._model.eval()
        # vocoder sample rate
        cfg = getattr(self._model, "config", None)
        sr = getattr(cfg, "sampling_rate", None) if cfg is not None else None
        if sr:
            self._sample_rate = int(sr)
        logger.info("SeamlessM4T готов (sr=%s)", self._sample_rate)

    def warmup(self) -> None:
        self.load()
        try:
            self.synthesize_chunk(
                "Hello.",
                [],
                "en",
                {"temperature": 0.8},
                None,
            )
        except Exception as exc:
            logger.warning("Seamless warmup: %s", exc)

    def clear_gpu_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_from_text(self, text: str, tgt_lang: str, speaker_id: int) -> np.ndarray:
        assert self._model is not None and self._processor is not None
        text = (text or "").strip()
        if not text:
            return np.zeros(self._sample_rate // 10, dtype=np.float32)
        inputs = self._processor(text=text, src_lang=tgt_lang, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                spkr_id=speaker_id,
            )
        wav = out[0].detach().float().cpu().numpy().squeeze()
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def _generate_from_audio(
        self, audio: np.ndarray, sample_rate: int, tgt_lang: str, speaker_id: int
    ) -> np.ndarray:
        assert self._model is not None and self._processor is not None
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size < sample_rate // 20:
            return np.zeros(self._sample_rate // 10, dtype=np.float32)
        if int(sample_rate) != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=16000)
            sample_rate = 16000
        peak = float(np.max(np.abs(wav)) or 1.0)
        if peak > 1.0:
            wav = wav / peak
        inputs = self._processor(
            audio=wav,
            sampling_rate=16000,
            return_tensors="pt",
        )
        # Speech-encoder LayerNorm stays fp32 in HF Seamless; fp16 activations break it.
        # Temporarily run the whole model in fp32 for S2ST (T2S stays fp16).
        inputs = {
            k: (
                v.to(device=self.device, dtype=torch.float32)
                if torch.is_tensor(v) and v.is_floating_point()
                else v.to(self.device) if torch.is_tensor(v) else v
            )
            for k, v in inputs.items()
        }
        was_half = next(self._model.parameters()).dtype == torch.float16
        if was_half:
            self._model.float()
        try:
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    tgt_lang=tgt_lang,
                    generate_speech=True,
                    spkr_id=int(speaker_id),
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning("Seamless S2ST OOM on GPU — fallback CPU")
            self.clear_gpu_cache()
            cpu_inputs = {
                k: v.to("cpu") if torch.is_tensor(v) else v for k, v in inputs.items()
            }
            self._model.to("cpu")
            try:
                with torch.inference_mode():
                    out = self._model.generate(
                        **cpu_inputs,
                        tgt_lang=tgt_lang,
                        generate_speech=True,
                        spkr_id=int(speaker_id),
                    )
            finally:
                self._model.to(self.device)
        finally:
            if was_half:
                self._model.half()
            self.clear_gpu_cache()
        speech = out[0].detach().float().cpu().numpy().squeeze()
        return np.asarray(speech, dtype=np.float32).reshape(-1)

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list,
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        del speaker_wavs, conditioning_cache
        self.load()
        tgt = seamless_tgt_lang(language)
        spk = int(params.get("speaker_id", self.speaker_id))
        with self._thread_lock:
            wav = self._generate_from_text(text, tgt, spk)
        return wav, None

    def synthesize_s2st(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        speaker_id: int | None = None,
    ) -> np.ndarray:
        """Прямой speech-to-speech перевод сегмента."""
        self.load()
        tgt = seamless_tgt_lang(language)
        spk = self.speaker_id if speaker_id is None else int(speaker_id)
        with self._thread_lock:
            return self._generate_from_audio(audio, sample_rate, tgt, spk)

    async def synthesize_chunk_async(
        self,
        text: str,
        speaker_wavs: list,
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.synthesize_chunk(
                    text, speaker_wavs, language, params, conditioning_cache
                ),
            )
