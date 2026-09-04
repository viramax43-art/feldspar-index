"""XTTS-v2 engine with GPU queue and latent caching."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.tts.engine import TTSEngine

logger = logging.getLogger(__name__)


def prefer_nvidia_gpu() -> None:
    """На системах с iGPU + NVIDIA выбираем дискретную карту."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class XTTSEngine(TTSEngine):
    def __init__(
        self,
        model_name: str,
        device: str,
        use_fp16: bool = False,
        gpt_cond_len: int = 30,
        gpt_cond_chunk_len: int = 6,
        max_ref_len: int = 30,
        sound_norm_refs: bool = True,
        finetune_checkpoint: Path | None = None,
        finetune_config: Path | None = None,
    ) -> None:
        prefer_nvidia_gpu()
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        self.gpt_cond_len = gpt_cond_len
        self.gpt_cond_chunk_len = gpt_cond_chunk_len
        self.max_ref_len = max_ref_len
        self.sound_norm_refs = sound_norm_refs
        self.finetune_checkpoint = Path(finetune_checkpoint) if finetune_checkpoint else None
        self.finetune_config = Path(finetune_config) if finetune_config else None
        self._model = None
        self._config = None
        self._api = None
        self._lock = asyncio.Lock()
        self._sample_rate = 24000

    @property
    def name(self) -> str:
        return "xtts"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _resolve_device(self) -> str:
        prefer_nvidia_gpu()
        if self.device == "cuda":
            if "+cpu" in torch.__version__ or torch.version.cuda is None:
                logger.error(
                    "Установлен CPU-only PyTorch (%s). "
                    "Переустановите: pip install torch torchaudio "
                    "--index-url https://download.pytorch.org/whl/cu124",
                    torch.__version__,
                )
                return "cpu"
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info("Используется NVIDIA GPU: %s (cuda:0)", name)
                return "cuda"
            logger.warning(
                "DEVICE=cuda, но torch.cuda.is_available()=False. "
                "Проверьте драйвер NVIDIA и CUDA-сборку PyTorch."
            )
        return "cpu"

    def load(self) -> None:
        from TTS.api import TTS

        resolved = self._resolve_device()
        self.device = resolved
        logger.info("Загрузка TTS-модели %s на %s", self.model_name, resolved)
        self._api = TTS(self.model_name)
        self._api.to(resolved)

        self._model = self._api.synthesizer.tts_model
        self._config = self._api.synthesizer.tts_config
        self._model.eval()

        if self.finetune_checkpoint and self.finetune_checkpoint.exists():
            self._load_finetune_checkpoint(resolved)

        # ВНИМАНИЕ: model.half() ломает XTTS get_conditioning_latents
        # (аудио-энкодер получает float32-вход при half-весах →
        # "Input type FloatTensor and weight type HalfTensor").
        # Поэтому веса держим в float32, а FP16 применяем только на инференсе
        # через autocast — так клонирование голоса не ломается.
        if self.use_fp16 and resolved == "cuda":
            logger.info("FP16 включён через autocast (веса остаются float32)")

        logger.info("TTS-модель загружена")

    def _load_finetune_checkpoint(self, device: str) -> None:
        """Подгружает GPT-веса после fine-tune поверх базовой XTTS-v2."""
        assert self._model is not None
        ckpt = self.finetune_checkpoint
        assert ckpt is not None
        vocab = None
        # vocab лежит рядом с базовой моделью TTS
        try:
            from TTS.utils.manage import ModelManager

            model_path, _, _ = ModelManager().download_model(self.model_name)
            model_dir = Path(model_path).parent if Path(model_path).is_file() else Path(model_path)
            candidate = model_dir / "vocab.json"
            if candidate.exists():
                vocab = str(candidate)
        except Exception as exc:
            logger.warning("Не удалось найти vocab.json для fine-tune: %s", exc)

        logger.info("Загрузка fine-tune checkpoint: %s", ckpt)
        try:
            self._model.load_checkpoint(
                self._config,
                checkpoint_path=str(ckpt),
                vocab_path=vocab,
                use_deepspeed=False,
            )
            self._model.to(device)
            self._model.eval()
            logger.info("Fine-tuned GPT weights загружены")
        except Exception as exc:
            logger.error("Не удалось загрузить fine-tune checkpoint: %s", exc)
            raise

    def warmup(self) -> None:
        """Прогрев CUDA/cuDNN коротким inference (без референса — skip)."""
        logger.info("XTTS warmup пропущен (нужен speaker_wav); первый запрос прогреет GPU")

    def warmup_with_reference(self, speaker_wavs: list[Path]) -> None:
        if not speaker_wavs:
            return
        try:
            self.synthesize_chunk(
                "Привет.",
                speaker_wavs,
                "ru",
                {"temperature": 0.75, "speed": 1.0, "repetition_penalty": 2.0, "top_k": 50, "top_p": 0.85, "length_penalty": 1.0},
                None,
            )
            logger.info("XTTS warmup выполнен")
        except Exception as exc:
            logger.warning("XTTS warmup не удался: %s", exc)

    def _compute_conditioning(self, speaker_wavs: list[Path]) -> tuple[Any, Any]:
        assert self._model is not None
        paths = [str(p) for p in speaker_wavs]
        logger.info(
            "Computing conditioning: %d refs, gpt_cond_len=%d, chunk=%d, max_ref=%d, norm=%s",
            len(paths), self.gpt_cond_len, self.gpt_cond_chunk_len,
            self.max_ref_len, self.sound_norm_refs,
        )
        with torch.inference_mode():
            return self._model.get_conditioning_latents(
                audio_path=paths,
                gpt_cond_len=self.gpt_cond_len,
                gpt_cond_chunk_len=self.gpt_cond_chunk_len,
                max_ref_length=self.max_ref_len,
                sound_norm_refs=self.sound_norm_refs,
            )

    def synthesize_chunk(
        self,
        text: str,
        speaker_wavs: list[Path],
        language: str,
        params: dict[str, Any],
        conditioning_cache: tuple[Any, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[Any, Any] | None]:
        assert self._model is not None

        if conditioning_cache is None:
            gpt_cond_latent, speaker_embedding = self._compute_conditioning(speaker_wavs)
            conditioning_cache = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = conditioning_cache

        temperature = float(params.get("temperature", 0.75))
        # XTTS speed даёт рябь — всегда 1.0; темп меняется в postprocess.apply_tempo
        repetition_penalty = float(params.get("repetition_penalty", 2.0))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.85))
        length_penalty = float(params.get("length_penalty", 1.0))

        use_autocast = self.use_fp16 and self.device == "cuda"
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_autocast,
            ):
                output = self._model.inference(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=temperature,
                    speed=1.0,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                    length_penalty=length_penalty,
                    enable_text_splitting=False,
                )

        wav = np.asarray(output["wav"], dtype=np.float32)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def check_vram() -> dict[str, Any]:
        info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = props.name
            info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        return info
